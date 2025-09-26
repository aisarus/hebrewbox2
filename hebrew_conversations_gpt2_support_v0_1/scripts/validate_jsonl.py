# validate_jsonl.py
import json, argparse, sys
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--max_errors", type=int, default=10)
    args = ap.parse_args()
    errs = 0; n = 0
    for line in open(args.src, "r", encoding="utf-8"):
        n += 1
        try:
            obj = json.loads(line)
            assert "dialogue" in obj and isinstance(obj["dialogue"], list) and len(obj["dialogue"]) >= 2
            for m in obj["dialogue"]:
                assert isinstance(m, dict) and "role" in m and "text" in m and isinstance(m["text"], str)
        except Exception as e:
            errs += 1
            print(f"[err] line {n}: {e}")
            if errs >= args.max_errors:
                break
    print(f"[done] checked {n} lines, errors: {errs}")
if __name__ == "__main__":
    main()
