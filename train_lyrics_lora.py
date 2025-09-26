# -*- coding: utf-8 -*-
"""
train_lyrics_lora.py
Single-file LoRA fine-tune for GPT-2-family on a folder of JSONL shards.
Robust field detection: supports {"text": ...} or {"prompt": ..., "completion": ...}
or {"messages": [{"role": "user"/"assistant", "content": "..."} , ...]}.
Windows-friendly. No custom compute_loss. Transformers >=4.37 works.
"""

import os, glob, json, random, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import get_peft_model, LoraConfig, TaskType

def env(name, default=None, cast=str):
    v = os.environ.get(name, None)
    if v is None:
        return default
    return cast(v)

# -------------------------
# Config via ENV or defaults
# -------------------------
DATA_DIR   = env("DATA_DIR", "./datasets/hebrew_lyrics_prompting_finetune")
PATTERN    = env("DATA_GLOB", "*.jsonl")
BASE_MODEL = env("BASE_MODEL", "gpt2")
OUTPUT_DIR = env("OUTPUT_DIR", "./out_lyrics_lora")
SEED       = int(env("SEED", "1337"))
EPOCHS     = float(env("EPOCHS", "1"))
LR         = float(env("LR", "2e-4"))
BATCH_SIZE = int(env("BATCH_SIZE", "1"))
GRAD_ACC   = int(env("GRAD_ACC", "8"))
MAXLEN     = int(env("MAXLEN", "512"))
SAVE_STEPS = int(env("SAVE_STEPS", "500"))
LOG_STEPS  = int(env("LOG_STEPS", "20"))
EVAL_RATIO = float(env("EVAL_RATIO", "0.02"))  # 2% на валидацию
WARMUP     = int(env("WARMUP_STEPS", "50"))
RESUME_CKPT = env("RESUME_CKPT", None)  # путь к checkpoint-* чтобы резюмировать
LORA_R     = int(env("LORA_R", "16"))
LORA_ALPHA = int(env("LORA_ALPHA", "32"))
LORA_DROPOUT = float(env("LORA_DROPOUT", "0.05"))

random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_text(record: Dict[str, Any]) -> Optional[str]:
    # 1) plain text
    if "text" in record and isinstance(record["text"], str):
        return record["text"]
    # 2) prompt + completion/response
    if "prompt" in record and isinstance(record["prompt"], str):
        comp = record.get("completion") or record.get("response") or record.get("output")
        if isinstance(comp, str):
            return f"User: {record['prompt'].strip()}\nAssistant: {comp.strip()}"
    # 3) messages style
    if "messages" in record and isinstance(record["messages"], list):
        chunks = []
        for m in record["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            if not isinstance(content, str):
                continue
            if role == "system":
                chunks.append(f"[System] {content.strip()}")
            elif role == "assistant":
                chunks.append(f"Assistant: {content.strip()}")
            else:
                chunks.append(f"User: {content.strip()}")
        if chunks:
            return "\n".join(chunks)
    return None

def load_folder_jsonl(data_dir: str, pattern: str) -> Dataset:
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched {data_dir}/{pattern}")
    ds_list = []
    for p in paths:
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = detect_text(obj)
                if not text:
                    continue
                rows.append({"text": text})
        if rows:
            ds_list.append(Dataset.from_list(rows))
    if not ds_list:
        raise RuntimeError("No usable records in dataset after detection.")
    return concatenate_datasets(ds_list)

def tokenize_and_group(tokenizer, ds: Dataset, maxlen: int) -> Dataset:
    def tok(batch):
        return tokenizer(batch["text"])
    tokenized = ds.map(tok, batched=True, remove_columns=["text"])
    # Group contiguous token sequences into chunks of size maxlen
    def group_texts(examples):
        # Concatenate all texts.
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Drop the small remainder
        total_length = (total_length // maxlen) * maxlen
        result = {
            k: [t[i : i + maxlen] for i in range(0, total_length, maxlen)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    grouped = tokenized.map(group_texts, batched=True)
    return grouped

def main():
    print(f"[info] DATA_DIR={DATA_DIR}")
    print(f"[info] PATTERN={PATTERN}")
    print(f"[info] BASE_MODEL={BASE_MODEL}")
    print(f"[info] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[info] EPOCHS={EPOCHS}  LR={LR}  BS={BATCH_SIZE}  GA={GRAD_ACC}  MAXLEN={MAXLEN}")
    print(f"[info] SAVE_STEPS={SAVE_STEPS} LOG_STEPS={LOG_STEPS} EVAL_RATIO={EVAL_RATIO}")
    print(f"[info] LORA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")

    ds = load_folder_jsonl(DATA_DIR, PATTERN)
    # simple split
    eval_size = max(1, int(len(ds) * EVAL_RATIO))
    ds = ds.shuffle(SEED)
    ds_eval = ds.select(range(eval_size))
    ds_train = ds.select(range(eval_size, len(ds)))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = tokenize_and_group(tokenizer, ds_train, MAXLEN)
    eval_dataset  = tokenize_and_group(tokenizer, ds_eval, MAXLEN)

    # Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn","c_proj"]  # gpt2 attention/feedforward
    )
    model = get_peft_model(model, lora_cfg)

    use_fp16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        warmup_steps=WARMUP,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        bf16=False,
        fp16=use_fp16,
        dataloader_pin_memory=True,
        report_to="none",
        gradient_checkpointing=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    ckpt = RESUME_CKPT if RESUME_CKPT else None
    trainer.train(resume_from_checkpoint=ckpt)
    # Save adapter
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(OUTPUT_DIR)

    # quick metrics snapshot
    eval_metrics = trainer.evaluate()
    print("[eval]", eval_metrics)

if __name__ == "__main__":
    main()
