# quick_infer.py
# Minimal generation script for any merged/soup model dir.
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = sys.argv[1] if len(sys.argv) > 1 else "./soup_all"
prompt = sys.argv[2] if len(sys.argv) > 2 else "תכתוב בית ראשון ושני על אהבה ותל אביב:\n"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=120, do_sample=True, top_p=0.9, temperature=0.9, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0], skip_special_tokens=True))
