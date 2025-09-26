import os, json, math, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = r"C:\Users\ariel\Downloads\chat socialbox"
BASE = "gpt2"  # токенайзер и база — точно GPT-2
DATA = rf"{ROOT}\datasets\omega_books\train_part-00002.jsonl"

# выбираем последний адаптер из out_lm_omega (checkpoint-* или epoch_*)
ADIR = rf"{ROOT}\out_lm_omega"
cands = glob.glob(os.path.join(ADIR, "checkpoint-*")) + \
        [p for p in [os.path.join(ADIR,"epoch_1"), os.path.join(ADIR,"epoch_2"), os.path.join(ADIR,"epoch_3")] if os.path.isdir(p)]
if not cands:
    raise SystemExit("Не нашёл LoRA-адаптер в out_lm_omega.")
ADAPTER = max(cands, key=os.path.getmtime)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda").eval()
model = PeftModel.from_pretrained(model, ADAPTER).to("cuda").eval()

def stream(path, n=500):
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in zip(range(n), f):
            try:
                o = json.loads(ln)
                yield o["text"]
            except Exception:
                continue

tot_loss = 0.0
tot_tok  = 0
with torch.no_grad():
    for text in stream(DATA, 500):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        out = model(**enc, labels=enc["input_ids"])   # transformers сам делает shift и маску
        loss = float(out.loss)
        ntok = int(enc["input_ids"].shape[1] - 1)
        tot_loss += loss * max(1, ntok)
        tot_tok  += max(1, ntok)

ce = tot_loss / max(1, tot_tok)
print(f"[VAL] adapter={os.path.basename(ADAPTER)} | ce={ce:.3f} | ppl≈{math.exp(ce):.2f}")
