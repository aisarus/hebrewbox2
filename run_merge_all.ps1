# run_merge_all.ps1
# Merge all LoRA adapters under ROOT into BASE_MODEL, then average all full models into a soup.
$env:ROOT = "C:\Users\ariel\Downloads\chat socialbox"
$env:BASE_MODEL = "gpt2"                                # LoRA adapters trained on gpt2
$env:OUT_MERGED = "C:\Users\ariel\Downloads\chat socialbox\merged_models"
$env:OUT_SOUP = "C:\Users\ariel\Downloads\chat socialbox\soup_all"
# Optional: point tokenizer source for soup (will copy tokenizer files from here)
$env:TOKENIZER_FROM = "C:\Users\ariel\Downloads\chat socialbox\out_tfm\epoch_1"

$env:TRANSFORMERS_NO_TORCHVISION="1"
py -3.11 -u "C:\Users\ariel\Downloads\chat socialbox\merge_all.py"
