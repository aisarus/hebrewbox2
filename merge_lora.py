import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "gpt2"                                # замени если база не gpt2
lora = rf"{r"C:\Users\ariel\Downloads\chat socialbox"}\out_lora\epoch_2"
out  = rf"{r"C:\Users\ariel\Downloads\chat socialbox"}\out_full\gpt2_merged_epoch2"
os.makedirs(out, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()             # сливаем LoRA в базу
model.save_pretrained(out, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(base)
tok.save_pretrained(out)
print("Saved:", out)
