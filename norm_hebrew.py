# -*- coding: utf-8 -*-
import json, os

src = r"C:\Users\ariel\Downloads\chat socialbox\datasets\hebrew_lyrics_prompting_finetune\train.jsonl"
dst = r"C:\Users\ariel\Downloads\chat socialbox\datasets\hebrew_lyrics_prompting_finetune\train_norm.jsonl"

ok=bad=0
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for ln, line in enumerate(f, 1):
        line=line.strip()
        if not line:
            continue
        try:
            o=json.loads(line)
        except Exception:
            bad+=1; continue

        msg=None
        # 1) уже chat-формат
        if isinstance(o, dict) and isinstance(o.get("messages"), list):
            ms=[]
            for m in o["messages"]:
                if not isinstance(m, dict): continue
                r=m.get("role"); c=m.get("content")
                if r in ("user","assistant") and isinstance(c, str) and c.strip():
                    ms.append({"role": r, "content": c.strip()})
            if len(ms) >= 2 and any(m["role"]=="assistant" for m in ms):
                msg={"messages": ms}
        # 2) инструкционный формат
        elif all(k in o for k in ("prompt","response")):
            msg={"messages":[
                {"role":"user","content":str(o["prompt"]).strip()},
                {"role":"assistant","content":str(o["response"]).strip()},
            ]}
        elif all(k in o for k in ("input","output")):
            msg={"messages":[
                {"role":"user","content":str(o["input"]).strip()},
                {"role":"assistant","content":str(o["output"]).strip()},
            ]}
        elif "instruction" in o and "output" in o:
            inp = o.get("input","")
            msg={"messages":[
                {"role":"user","content":(str(o["instruction"])+"\n"+str(inp)).strip()},
                {"role":"assistant","content":str(o["output"]).strip()},
            ]}

        if msg:
            out.write(json.dumps(msg, ensure_ascii=False)+"\n"); ok+=1
        else:
            bad+=1

print(f"ok={ok} bad={bad}")
