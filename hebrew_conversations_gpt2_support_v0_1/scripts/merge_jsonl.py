# merge_jsonl.py
# Merge multiple JSONL files with {"dialogue":[{role,text}...], "meta":{...}} into one, with basic dedupe.
import os, sys, json, argparse, hashlib, random

def sig(obj):
    s = " ".join(m.get("text","") for m in obj.get("dialogue",[]))
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Paths to JSONL files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    seen = set()
    out = []
    for p in args.inputs:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                h = sig(obj)
                if h in seen: 
                    continue
                seen.add(h)
                out.append(obj)
    if args.shuffle:
        random.seed(1337); random.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("[done]", len(out), "examples ->", args.out)

if __name__ == "__main__":
    main()
