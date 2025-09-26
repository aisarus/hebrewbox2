# convert_coursera_to_jsonl.py
import os, sys, json, argparse
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Glob to parquet files (can include **/*.parquet)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--field", default="sentence", help="Column with text")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to export (0 = all)")
    ap.add_argument("--min_len", type=int, default=0, help="Drop rows shorter than N chars")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[info] loading parquet: {args.src}")
    ds = load_dataset("parquet", data_files=args.src, split="train")

    n = 0
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in ds:
            n += 1
            s = (r.get(args.field) or "").strip()
            if not s or len(s) < args.min_len:
                continue
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break

    print(f"[done] scanned={n} wrote={kept} -> {args.out}")

if __name__ == "__main__":
    main()
