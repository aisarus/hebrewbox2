# train_lora_gpt2.py
import os, json, glob, argparse, sys
from typing import Any, Dict, List
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

USR = "[USR]"
AST = "[ASST]"
RLM = "\u200f"  # Right-To-Left Mark

def as_texts(rec: Dict[str,Any], rtl: bool) -> List[str]:
    out=[]
    m = rec.get("messages")
    if isinstance(m, list) and m:
        parts=[]
        for x in m:
            role = (x.get("role") or x.get("speaker") or "").lower()
            txt  = x.get("content") or x.get("text") or ""
            if not txt: 
                continue
            if role in ("assistant","bot","gpt","system"):
                if rtl: txt = RLM + txt
                parts.append(f"{AST} {txt}".strip())
            else:
                parts.append(f"{USR} {txt}".strip())
        if parts: out.append("\n".join(parts)+"\n")
        return out
    for a,b in (("input","output"),("prompt","response"),("question","answer")):
        if rec.get(a) and rec.get(b):
            txt = rec[b]
            if rtl: txt = RLM + str(txt)
            out.append(f"{USR} {rec[a]}\n{AST} {txt}\n")
            return out
    if rec.get("text"):
        t = rec["text"]
        out.append((t if t.endswith("\n") else t+"\n"))
    return out

def scan(data_dir: str, rtl: bool) -> List[str]:
    cands=[]
    for p in ("", "data", "hebrew_conversations_gpt2_v0_1"):
        base = os.path.join(data_dir, p) if p else data_dir
        cands += glob.glob(os.path.join(base, "*.jsonl"))
        cands += glob.glob(os.path.join(base, "*.json"))
        cands += glob.glob(os.path.join(base, "*.txt"))
    texts=[]
    for path in cands:
        try:
            if path.endswith(".jsonl"):
                with open(path,"r",encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        try:
                            rec=json.loads(line)
                            texts += as_texts(rec, rtl)
                        except Exception:
                            texts.append(line+"\n")
            elif path.endswith(".json"):
                with open(path,"r",encoding="utf-8") as f:
                    data=json.load(f)
                if isinstance(data,list):
                    for rec in data:
                        texts += as_texts(rec if isinstance(rec,dict) else {"text":str(rec)}, rtl)
                elif isinstance(data,dict):
                    hit=False
                    for k in ("data","samples","conversations","items"):
                        if isinstance(data.get(k),list):
                            for rec in data[k]:
                                texts += as_texts(rec if isinstance(rec,dict) else {"text":str(rec)}, rtl)
                            hit=True
                            break
                    if not hit:
                        texts.append(json.dumps(data,ensure_ascii=False)+"\n")
            else:
                with open(path,"r",encoding="utf-8") as f:
                    texts.append(f.read())
        except FileNotFoundError:
            pass
    # cleanup + dedupe
    seen=set(); uniq=[]
    for t in texts:
        t=(t.replace("\r\n","\n").strip()+"\n")
        if t and t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--output_dir", default="./lora_out")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--rtl", action="store_true", help="prefix assistant spans with RLM for RTL stability")
    ap.add_argument("--resume_from", default=None)
    args=ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[info] data_dir={args.data_dir}")
    texts = scan(args.data_dir, rtl=args.rtl)
    if not texts:
        print("[error] no samples found"); sys.exit(1)
    print(f"[info] samples={len(texts)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.add_special_tokens({"additional_special_tokens":[USR,AST]})
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    def tokfn(batch): 
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = Dataset.from_dict({"text":texts}).shuffle(seed=42).map(tokfn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))
    peft=LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["c_attn","c_proj","c_fc"])
    model = get_peft_model(model, peft)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    fp16 = torch.cuda.is_available()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=fp16, bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[]
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, tokenizer=tok, data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(args.output_dir); tok.save_pretrained(args.output_dir)
    print("[done] saved:", args.output_dir)

if __name__=="__main__":
    main()
