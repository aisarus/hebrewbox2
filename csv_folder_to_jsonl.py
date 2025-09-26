# csv_folder_to_jsonl.py
# Склейка всех CSV в папке SRC_DIR в формат JSONL с полями {"input","output"}.
# Автоматически ищет разумные названия колонок.

import os, csv, json, random, io

SRC_DIR   = r"C:\Users\ariel\Downloads\chat socialbox\csv"
OUT_TRAIN = r"C:\Users\ariel\Downloads\chat socialbox\conversations\efm_sft_pairs.jsonl"
OUT_VAL   = r"C:\Users\ariel\Downloads\chat socialbox\conversations\efm_sft_pairs_val.jsonl"
VAL_RATIO = 0.01

INPUT_KEYS  = {"input","instruction","prompt","question","user","query","request"}
CTX_KEYS    = {"context","system","meta"}
OUTPUT_KEYS = {"output","response","assistant","answer","completion","reply","target","label"}

def norm(s):
    return " ".join((s or "").replace("\r"," ").replace("\n"," ").split()).strip()

def pick(row, keys):
    for k in row.keys():
        lk = k.lower().strip()
        if lk in keys:
            v = norm(row[k])
            if v:
                return v
    return ""

def parse_csv(path):
    rows = []
    with io.open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(8192); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.get_dialect("excel")
        r = csv.DictReader(f, dialect=dialect)
        for raw in r:
            x = pick(raw, INPUT_KEYS)
            y = pick(raw, OUTPUT_KEYS)
            ctx = pick(raw, CTX_KEYS)
            if ctx and x: x = f"{ctx}\n\n{x}"
            if len(x) < 3 or len(y) < 1:
                continue
            rows.append({"input": x, "output": y})
    return rows

def main():
    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    all_rows = []
    for root, _, files in os.walk(SRC_DIR):
        for name in files:
            if not name.lower().endswith(".csv"): 
                continue
            p = os.path.join(root, name)
            got = parse_csv(p)
            print(f"[read] {name}: {len(got)}")
            all_rows.extend(got)
    # дедуп и перемешивание
    uniq = list({(r["input"], r["output"]): r for r in all_rows}.values())
    random.Random(42).shuffle(uniq)
    n_val = max(1000, int(len(uniq) * VAL_RATIO)) if len(uniq) > 1000 else max(1, len(uniq)//50)
    val = uniq[:n_val]
    trn = uniq[n_val:]
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in trn: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"[done] train={len(trn)}  val={len(val)}  -> {OUT_TRAIN}")

if __name__ == "__main__":
    main()
