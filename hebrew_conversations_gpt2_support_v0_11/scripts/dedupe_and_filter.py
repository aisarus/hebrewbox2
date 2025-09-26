# dedupe_and_filter.py
import os, sys, json, argparse, re, hashlib, random

HEB_RE = re.compile(r'[\u0590-\u05FF]')

def heb_ratio(s):
    heb = len(HEB_RE.findall(s))
    letters = sum(ch.isalpha() for ch in s)
    return (heb/letters) if letters else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_he_ratio", type=float, default=0.5)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=1200)
    ap.add_argument("--badwords", default="", help="Path to newline-separated banned terms (UTF-8).")
    args = ap.parse_args()

    banned = []
    if args.badwords and os.path.exists(args.badwords):
        banned = [w.strip() for w in open(args.badwords, "r", encoding="utf-8") if w.strip()]
    def bad(s):
        s2 = s.lower()
        return any(b in s2 for b in banned)

    seen = set(); kept = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in open(args.src, "r", encoding="utf-8"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = " ".join(m.get("text","") for m in obj.get("dialogue",[]))
            if not (args.min_len <= len(txt) <= args.max_len):
                continue
            if heb_ratio(txt) < args.min_he_ratio:
                continue
            if any(bad(m.get("text","")) for m in obj.get("dialogue",[])):
                continue
            h = hashlib.md5(txt.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.write(json.dumps(obj, ensure_ascii=False) + "\n"); kept += 1
    print("[kept]", kept, "->", args.out)

if __name__ == "__main__":
    main()
