MERGE EVERYTHING — WINDOWS QUICK GUIDE

1) Install deps (once in your venv):
   pip install -U torch transformers peft safetensors

2) Put merge_all.py into: C:\Users\ariel\Downloads\chat socialbox

3) Edit run_merge_all.ps1 if paths differ, then run:
   powershell -ExecutionPolicy Bypass -File .\run_merge_all.ps1

What happens:
- Finds all LoRA adapters (folders with adapter_config.json + adapter_model.safetensors), merges each into BASE_MODEL (gpt2).
- Finds all full model dirs (model.safetensors or pytorch_model.bin), including out_tfm\epoch_1 and the newly merged ones.
- Averages them into C:\...\soup_all (model.safetensors). Tokenizer files are copied from out_tfm\epoch_1.

Test a generation:
   py -3.11 -u .\quick_infer.py "C:\Users\ariel\Downloads\chat socialbox\soup_all" "תכתוב בית ראשון ושני על אהבה ותל אביב:"
