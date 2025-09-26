# -*- coding: utf-8 -*-
# Собираем ВЕСЬ корпус в пачки train_part-00001.jsonl, по ~100k строк (настраивается).
import os, glob, json
from datasets import load_dataset

ROOT = r"C:\Users\ariel\Downloads\chat socialbox\datasets\omega_books"
OUTD = ROOT
CHUNK = 100_000                     # строк на файл; увеличишь/уменьшишь по вкусу

files = glob.glob(os.path.join(ROOT, "**", "*.parquet"), recursive=True)
assert files, "Не нашёл parquet-файлы в omega_books"

def stream():
    return load_dataset("parquet", data_files=files, split="train", streaming=True)

probe = next(iter(stream()))
cands = ["text","content","body","raw_text","document","article","passage"]
col = next((k for k in cands if isinstance(probe.get(k), str)), None)
if not col:
    raise SystemExit(f"Не нашёл текстовую колонку в примере: {list(probe.keys())}")

part, n_in_part, total = 1, 0, 0
w = None
ds = stream()
for row in ds:
    t = (row.get(col) or "").strip()
    if not t: 
        continue
    if w is None or n_in_part >= CHUNK:
        if w: w.close()
        out = os.path.join(OUTD, f"train_part-{part:05d}.jsonl")
        w = open(out, "w", encoding="utf-8")
        print(">> new", out)
        part += 1
        n_in_part = 0
    w.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    n_in_part += 1
    total += 1
if w: w.close()
print("done, rows=", total)
