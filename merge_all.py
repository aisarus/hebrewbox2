import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION","1")
# -*- coding: utf-8 -*-
r"""
merge_all.py
1) Finds all LoRA adapters under ROOT and merges each into BASE_MODEL (must match the base used to train the adapter).
2) Finds all full model directories (with model.safetensors or pytorch_model.bin) and averages them (model soup) into OUT_SOUP.

Usage (env vars or defaults):
    python merge_all.py

Env:
  ROOT=C:\Users\ariel\Downloads\chat socialbox
  BASE_MODEL=gpt2            # or path to a local base dir compatible with the adapters
  OUT_MERGED=%ROOT%\merged_models
  OUT_SOUP=%ROOT%\soup_all
  WEIGHTS_JSON=              # optional path to {"path_or_name": weight, ...} for weighted soup
  TOKENIZER_FROM=            # optional: path whose tokenizer files will be copied to soup
"""

import os, sys, json, math, re, shutil
from typing import List, Dict, Tuple
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file, save_file

def env(name, default=None):
    return os.environ.get(name, default)

ROOT         = env("ROOT", r"C:\Users\ariel\Downloads\chat socialbox")
BASE_MODEL   = env("BASE_MODEL", "gpt2")
OUT_MERGED   = env("OUT_MERGED", os.path.join(ROOT, "merged_models"))
OUT_SOUP     = env("OUT_SOUP", os.path.join(ROOT, "soup_all"))
WEIGHTS_JSON = env("WEIGHTS_JSON", "")
TOKENIZER_FROM = env("TOKENIZER_FROM", "")

def is_adapter_dir(path:str) -> bool:
    # Typical PEFT layout contains adapter_config.json and adapter_model.safetensors
    if not os.path.isdir(path): return False
    files = set(os.listdir(path))
    if "adapter_config.json" in files and any(f.endswith(".safetensors") for f in files):
        return True
    return False

def is_full_model_dir(path:str) -> bool:
    if not os.path.isdir(path): return False
    files = set(os.listdir(path))
    return ("model.safetensors" in files) or ("pytorch_model.bin" in files)

def walk_for_adapters(root:str) -> List[str]:
    out = []
    for dp, dn, fn in os.walk(root):
        if is_adapter_dir(dp):
            out.append(dp)
    # prefer "final_adapter" if parent contains multiple checkpoints
    # we'll later uniquify by parent run name
    return sorted(set(out))

def walk_for_full_models(root:str) -> List[str]:
    out = []
    for dp, dn, fn in os.walk(root):
        if is_full_model_dir(dp):
            out.append(dp)
    return sorted(set(out))

def safe_name(p:str) -> str:
    base = os.path.basename(p.rstrip("\\/"))
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base or "model"

def merge_adapter(base:str, adapter_dir:str, out_dir:str):
    print(f"[merge] base={base}  adapter={adapter_dir}")
    model = AutoModelForCausalLM.from_pretrained(base)
    lora  = PeftModel.from_pretrained(model, adapter_dir)
    merged = lora.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir)
    # try to copy tokenizer alongside
    try:
        tok = AutoTokenizer.from_pretrained(base)
        tok.save_pretrained(out_dir)
    except Exception as e:
        print(f"[warn] tokenizer copy failed: {e}")
    del merged; del lora; del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_safetensors(model_dir:str) -> str:
    st_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st_path):
        return st_path
    pt = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(pt):
        print(f"[convert] {pt} -> model.safetensors")
        sd = torch.load(pt, map_location="cpu")
        os.makedirs(model_dir, exist_ok=True)
        save_file(sd, st_path)
        return st_path
    raise FileNotFoundError(f"No model weights found in {model_dir}")

def average_models(model_dirs:List[str], out_dir:str, weights:Dict[str, float]=None, tokenizer_from:str=""):
    if not model_dirs:
        raise RuntimeError("No model directories to average.")
    if weights is None:
        weights = {d: 1.0 for d in model_dirs}
    # normalize weights
    total_w = sum(max(0.0, float(w)) for w in weights.values())
    if total_w <= 0:
        raise RuntimeError("Sum of weights is zero.")
    weights = {k: float(w)/total_w for k, w in weights.items()}

    print("[soup] models:")
    for d in model_dirs:
        print(f"  - {d}  (w={weights.get(d,0):.4f})")

    st_paths = [ensure_safetensors(d) for d in model_dirs]

    # build intersection of keys
    key_sets = []
    metas = []
    for p in st_paths:
        sd = load_file(p, device="cpu")
        key_sets.append(set(sd.keys()))
        metas.append(sd)
    common = set.intersection(*key_sets)
    if not common:
        raise RuntimeError("Models have no overlapping parameter keys.")
    print(f"[soup] common params: {len(common)}")

    # streaming weighted average
    acc = {}
    for d, p in zip(model_dirs, st_paths):
        w = weights.get(d, 0.0)
        sd = load_file(p, device="cpu")
        for k in common:
            t = sd[k].to(dtype=torch.float32)
            if k not in acc:
                acc[k] = t * w
            else:
                acc[k].add_(t * w)
        del sd

    os.makedirs(out_dir, exist_ok=True)
    save_file(acc, os.path.join(out_dir, "model.safetensors"))
    # copy config + tokenizer
    src_tok = tokenizer_from or model_dirs[0]
    for fname in ["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt"]:
        src = os.path.join(src_tok, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))
    print(f"[soup] saved -> {out_dir}")

def main():
    print(f"[info] ROOT={ROOT}")
    print(f"[info] BASE_MODEL={BASE_MODEL}")
    print(f"[info] OUT_MERGED={OUT_MERGED}")
    print(f"[info] OUT_SOUP={OUT_SOUP}")

    adapters = walk_for_adapters(ROOT)
    print(f"[scan] adapters found: {len(adapters)}")
    os.makedirs(OUT_MERGED, exist_ok=True)

    merged_dirs = []
    for ad in adapters:
        name = safe_name(ad)
        outd = os.path.join(OUT_MERGED, f"merged_from_{name}")
        if not os.path.exists(os.path.join(outd, "model.safetensors")):
            try:
                merge_adapter(BASE_MODEL, ad, outd)
            except Exception as e:
                print(f"[error] merge failed for {ad}: {e}")
                continue
        merged_dirs.append(outd)

    full_models = walk_for_full_models(ROOT)
    # Include newly merged models and any pre-existing full models (like out_tfm/epoch_1)
    candidates = sorted(set(full_models + merged_dirs))

    # optional weights
    weights = None
    if WEIGHTS_JSON and os.path.exists(WEIGHTS_JSON):
        try:
            raw = json.load(open(WEIGHTS_JSON, "r", encoding="utf-8"))
            # resolve to absolute paths matching candidates
            weights = {}
            for k, v in raw.items():
                # try exact match or endswith
                match = next((c for c in candidates if c == k or c.endswith(k)), None)
                if match:
                    weights[match] = float(v)
            print(f"[soup] using custom weights for {len(weights)} models.")
        except Exception as e:
            print(f"[warn] failed to load weights json: {e}")

    # Filter to GPT-2 shape-compatible only (heuristic: must have config.json with 'gpt2' arch or same hidden size)
    def is_gpt2_like(d):
        cfg = os.path.join(d, "config.json")
        if not os.path.exists(cfg): return True  # hope for the best
        try:
            j = json.load(open(cfg, "r", encoding="utf-8"))
            return j.get("_name_or_path","").startswith("gpt2") or j.get("model_type","") in ["gpt2","causal-lm","gpt_neo","gptj","phi"]
        except Exception:
            return True

    final_list = [d for d in candidates if is_gpt2_like(d)]
    final_list = sorted(set(final_list))

    if not final_list:
        print("[soup] nothing to average. Done.")
        return

    average_models(final_list, OUT_SOUP, weights=weights, tokenizer_from=TOKENIZER_FROM)

if __name__ == "__main__":
    main()
