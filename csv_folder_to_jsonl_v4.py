# csv_folder_to_jsonl_v4.py
"""
v4: Упёртый конвертер CSV -> JSONL {"input","output"} для твоих монстров.
- Поднимает лимит размера ячейки csv.
- Терпит колонки без заголовка (key=None), списки значений и прочий сюр.
- Ищет input/output по синонимам, иначе берёт две самые длинные текстовые колонки (вторая = input, первая = output).
- Добавляет context/system/meta как префикс к input.
- Печатает подробную статистику и НИКОГДА не падает из-за одной кривой строки/файла.
"""

import os, csv, json, io, re, random, sys, traceback

# Лимит размера клетки CSV (дефолт ~128К)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2_147_483_647)

SRC_DIR   = r"C:\Users\ariel\Downloads\chat socialbox\csv"
OUT_DIR   = r"C:\Users\ariel\Downloads\chat socialbox\conversations"
OUT_TRAIN = os.path.join(OUT_DIR, "efm_sft_pairs.jsonl")
OUT_VAL   = os.path.join(OUT_DIR, "efm_sft_pairs_val.jsonl")
VAL_RATIO = 0.01

ALIASES_IN  = {"input","instruction","prompt","question","user","query","request","text","source","message","utterance"}
ALIASES_CTX = {"context","system","meta","notes","history","prelude"}
ALIASES_OUT = {"output","response","assistant","answer","completion","reply","target","label","prediction"}

DROP_LIKE = re.compile(r"^(id|idx|index|row|ts|time|date|weight|score|url|link|path|filename|lang|locale)$", re.I)

def coerce_text(v):
    if v is None: return ""
    if isinstance(v, (list, tuple)):
        v = " | ".join([str(x) for x in v if x is not None])
    v = str(v).replace("\r", " ").replace("\n", " ")
    v = re.sub(r"\s+", " ", v).strip()
    return v

def safe_key(k):
    if k is None: return "_extra"
    try:
        return str(k).strip()
    except Exception:
        return "_badkey"

def guess_io(row: dict):
    # нормализуем: строки, без пустоты, вменяемые ключи
    clean = {}
    for k, v in row.items():
        key = safe_key(k)
        val = coerce_text(v)
        if val != "":
            clean[key] = val
    if not clean: return None, None, None, "empty"

    lower = {k.lower(): v for k, v in clean.items()}
    ctx = next((v for k, v in lower.items() if k in ALIASES_CTX and len(v) > 2), "")

    inp = next((v for k, v in lower.items() if k in ALIASES_IN and len(v) > 2), "")
    out = next((v for k, v in lower.items() if k in ALIASES_OUT and len(v) > 0), "")

    if inp and out:
        return inp, out, ctx, "aliases"

    # fallback: берем две самые длинные текстовые колонки, исключая откровенный мусор
    items = [(k, v) for k, v in clean.items() if not DROP_LIKE.match(k)]
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    if len(items) >= 2:
        out = items[0][1]
        inp = items[1][1]
        if len(out) >= 1 and len(inp) >= 3:
            return inp, out, ctx, "fallback"

    return None, None, None, "fail"

def parse_csv(path):
    ok = alias_hit = fallback_hit = bad_rows = 0
    rows = []
    with io.open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(262144); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except Exception:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        for idx, raw in enumerate(reader, 1):
            try:
                inp, out, ctx, how = guess_io(raw)
                if inp and out:
                    if ctx: inp = f"{ctx}\n\n{inp}"
                    rows.append({"input": inp, "output": out})
                    ok += 1
                    alias_hit += (how == "aliases")
                    fallback_hit += (how == "fallback")
            except Exception:
                bad_rows += 1
                # не падаем; продолжаем
                continue
    return rows, ok, alias_hit, fallback_hit, bad_rows

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = []
    for root, _, fs in os.walk(SRC_DIR):
        for name in fs:
            if name.lower().endswith(".csv"):
                files.append(os.path.join(root, name))
    files.sort()

    total = alias_total = fallback_total = total_bad_rows = 0
    all_rows = []
    for p in files:
        try:
            got, ok, a, fb, bad_rows = parse_csv(p)
            print(f"[read] {os.path.basename(p):40s} -> ok={ok:7d}  aliases={a:7d}  fallback={fb:7d}  bad_rows={bad_rows:6d}")
            all_rows.extend(got)
            total += ok; alias_total += a; fallback_total += fb; total_bad_rows += bad_rows
        except Exception as e:
            print(f"[skip] {os.path.basename(p)} due to error: {e}")

    # dedup + shuffle
    uniq = list({(r['input'], r['output']): r for r in all_rows}.values())
    random.Random(42).shuffle(uniq)

    # split
    n_val = max(1000, int(len(uniq)*VAL_RATIO)) if len(uniq) > 1000 else max(1, len(uniq)//50)
    val = uniq[:n_val]; trn = uniq[n_val:]

    # write
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for r in trn: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] files={len(files)} total_ok={total} dedup={len(uniq)} train={len(trn)} val={len(val)} bad_rows_total={total_bad_rows}")
    print(f"[hint] train -> {OUT_TRAIN}")

if __name__ == "__main__":
    main()
