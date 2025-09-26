# -*- coding: utf-8 -*-
import os, json, glob, sys
from datasets import load_dataset

root = r"C:\Users\ariel\Downloads\chat socialbox\datasets\omega_books"
out  = os.path.join(root, "train_text.jsonl")

pqs   = glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
jsonl = glob.glob(os.path.join(root, "**", "*.jsonl"),   recursive=True)
csvs  = glob.glob(os.path.join(root, "**", "*.csv"),     recursive=True)

if pqs:
    kind, files = "parquet", pqs
elif jsonl:
    kind, files = "json", jsonl
elif csvs:
    kind, files = "csv", csvs
else:
    raise SystemExit("Не нашёл ни parquet, ни jsonl, ни csv внутри omega_books")

def load_stream():
    return load_dataset(kind, data_files=files, split="train", streaming=True)

# проба одной строки, чтобы понять имя колонки
probe = next(iter(load_stream()))
cands = ["text","content","body","raw_text","document","article","passage"]
col = next((c for c in cands if c in probe and isinstance(probe[c], str)), None)
if not col:
    raise SystemExit(f"Не нашёл текстовую колонку в примере: {list(probe.keys())}")

# второй проход уже в файл
ds = load_stream()
os.makedirs(os.path.dirname(out), exist_ok=True)
w = 0
with open(out, "w", encoding="utf-8") as f:
    for row in ds:
        t = str(row.get(col,"")).strip()
        if not t: 
            continue
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        w += 1
print(f"written={w} -> {out}")
