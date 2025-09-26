# train_lora_sft.py (v2)
# - LoRA SFT trainer with resume
# - Force-save via OUTPUT_DIR/.force_save
# - GPU autocast updated to torch.amp.autocast
# - GradScaler updated to torch.amp.GradScaler('cuda', ...)
# - Gradient checkpointing fixed with enable_input_require_grads()

import os, json, math, shutil
from contextlib import nullcontext
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    PEFT_OK = True
except Exception:
    PEFT_OK = False

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def build_text(o: dict) -> str:
    a = o.get("input", "")
    b = o.get("output", "")
    return f"Human: {a}\nAssistant: {b}\n"

class SFTDS(Dataset):
    def __init__(self, items, tok, maxlen: int):
        self.data = []
        for o in items:
            enc = tok(build_text(o), truncation=True, max_length=maxlen)
            self.data.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def guess_lora_targets(model) -> list:
    cands = ["q_proj","k_proj","v_proj","o_proj","c_attn","c_fc","c_proj","fc1","fc2","Wqkv"]
    found = set()
    for name, _ in model.named_modules():
        for c in cands:
            if c in name:
                found.add(c)
    return sorted(found)

def _safe_save(model, tok, out_dir: str, tag: str):
    path = os.path.join(out_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    print(f"[save] {path}", flush=True)

def _prune_old_checkpoints(out_dir: str, limit: int):
    if not limit or limit <= 0 or not os.path.isdir(out_dir):
        return
    cps = []
    for name in os.listdir(out_dir):
        if name.startswith("checkpoint-"):
            try:
                cps.append((int(name.split("-")[-1]), name))
            except Exception:
                pass
    cps.sort()
    while len(cps) > limit:
        _, name = cps.pop(0)
        p = os.path.join(out_dir, name)
        shutil.rmtree(p, ignore_errors=True)
        print(f"[prune] removed {p}", flush=True)

def main():
    # env
    BASE = os.environ.get("BASE_MODEL", "gpt2")
    DATA = os.environ.get("DATA_FILE", r"conversations\efm_sft_pairs.jsonl")
    OUT  = os.environ.get("OUTPUT_DIR", "out_lora")
    RESUME_ADAPTER = os.environ.get("RESUME_ADAPTER", "")   # path to checkpoint-000xxx

    BS   = int(os.environ.get("BATCH_SIZE", "2"))
    GA   = int(os.environ.get("GRAD_ACC", "8"))
    E    = int(os.environ.get("EPOCHS", "1"))
    LR   = float(os.environ.get("LR", "2e-4"))
    MAXL = int(os.environ.get("MAXLEN", "512"))
    USE_LORA = os.environ.get("USE_LORA", "1") == "1"

    SAVE_EVERY = int(os.environ.get("SAVE_EVERY_STEPS", "0"))
    SAVE_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "3"))
    USE_GC     = os.environ.get("GRAD_CHECKPOINTING", "1") == "1"  # toggle GC via env

    # resolve paths from file location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(DATA): DATA = os.path.join(BASE_DIR, DATA)
    if not os.path.isabs(OUT):  OUT  = os.path.join(BASE_DIR, OUT)

    os.makedirs(OUT, exist_ok=True)
    FORCE_FLAG = os.path.join(OUT, ".force_save")

    print(f"[info] model={BASE}")
    print(f"[info] data={DATA}")
    print(f"[info] output_dir={OUT}")
    print(f"[force] watching flag: {FORCE_FLAG}", flush=True)

    # data
    items = load_jsonl(DATA)
    print(f"[info] samples={len(items)}")
    if not items:
        raise SystemExit("No data")

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ds = SFTDS(items, tok, MAXL)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collator, num_workers=0, pin_memory=True)

    # device/dtype
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        bf16_ok  = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        dtype    = torch.bfloat16 if bf16_ok else torch.float16
        print(f"[gpu] {torch.cuda.get_device_name(0)} | dtype={dtype}", flush=True)
    else:
        dtype = torch.float32
        print("[gpu] CUDA not available, using CPU fp32", flush=True)

    # model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=(dtype if use_cuda else torch.float32), low_cpu_mem_usage=True
    )

    if USE_LORA and PEFT_OK:
        if RESUME_ADAPTER and os.path.isdir(RESUME_ADAPTER):
            print(f"[resume] loading LoRA adapter from {RESUME_ADAPTER}", flush=True)
            model = PeftModel.from_pretrained(base_model, RESUME_ADAPTER, is_trainable=True)
        else:
            tmods = guess_lora_targets(base_model)
            print(f"[info] applying new LoRA; target_modules={tmods if tmods else 'auto'}")
            cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05,
                bias="none", target_modules=(tmods if tmods else None)
            )
            model = get_peft_model(base_model, cfg)
    else:
        model = base_model

    model.to(device)

    # gradient checkpointing + input grads
    if use_cuda and USE_GC:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            model.enable_input_require_grads()
        except Exception:
            # fallback: force input embeddings to require grad
            emb = model.get_input_embeddings()
            def _force_inputs_require_grad(mod, inp, out):
                try:
                    out.requires_grad_(True)
                except Exception:
                    pass
            if emb is not None:
                emb.register_forward_hook(_force_inputs_require_grad)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    steps_per_epoch = max(1, math.ceil(len(ds) / BS / GA))
    total_updates   = steps_per_epoch * E
    warmup = max(1, min(int(0.03 * total_updates), total_updates))
    sched  = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_updates)

    scaler = torch.amp.GradScaler('cuda', enabled=(use_cuda and dtype is torch.float16))

    # train
    gstep = 0; acc_loss = 0.0; last_flag_mtime = -1
    model.train()
    try:
        for ep in range(E):
            num_batches = math.ceil(len(ds) / BS)
            for bidx, batch in enumerate(dl):
                labels = batch["labels"].to(device, non_blocking=True)
                inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in ("input_ids","attention_mask")}

                ctx = torch.amp.autocast('cuda', dtype=dtype) if use_cuda else nullcontext()
                with ctx:
                    out  = model(**inputs, labels=labels)
                    loss = out.loss / GA

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                acc_loss += float(loss.detach().item())

                is_update = ((bidx + 1) % GA == 0) or ((bidx + 1) == num_batches)
                if is_update:
                    if scaler.is_enabled():
                        scaler.step(optim); scaler.update()
                    else:
                        optim.step()
                    sched.step(); optim.zero_grad(set_to_none=True)
                    gstep += 1
                    print(f"[ep {ep+1}/{E}] step {gstep}/{total_updates}: loss={acc_loss:.4f}", flush=True)
                    acc_loss = 0.0

                    # periodic save
                    if SAVE_EVERY > 0 and (gstep % SAVE_EVERY == 0):
                        _safe_save(model, tok, OUT, f"checkpoint-{gstep:06d}")
                        _prune_old_checkpoints(OUT, SAVE_LIMIT)

                    # force-save by flag
                    if os.path.exists(FORCE_FLAG):
                        try: mtime = int(os.path.getmtime(FORCE_FLAG))
                        except OSError: mtime = -1
                        if mtime != last_flag_mtime:
                            last_flag_mtime = mtime
                            print(f"[force] hit at step {gstep}", flush=True)
                            _safe_save(model, tok, OUT, f"checkpoint-{gstep:06d}")
                            _prune_old_checkpoints(OUT, SAVE_LIMIT)

                    if gstep >= total_updates:
                        break

            _safe_save(model, tok, OUT, f"epoch_{ep+1}")
            if gstep >= total_updates:
                break

    except KeyboardInterrupt:
        _safe_save(model, tok, OUT, f"interrupt_{gstep:06d}")
        print("[panic] interrupt -> saved", flush=True)

    print("[done] training complete", flush=True)

if __name__ == "__main__":
    main()
