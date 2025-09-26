# run_train_lyrics.ps1
# Usage: Right-click → Run with PowerShell (or run from an activated venv)
# Adjust DATA_DIR if your path differs.
$env:DATA_DIR = "C:\Users\ariel\Downloads\chat socialbox\datasets\hebrew_lyrics_prompting_finetune"
$env:DATA_GLOB = "fine_tune_data_merged_lyrics_results*.jsonl"
$env:BASE_MODEL = "gpt2"
$env:OUTPUT_DIR = "C:\Users\ariel\Downloads\chat socialbox\out_lyrics_lora"
$env:EPOCHS = "1"
$env:LR = "2e-4"
$env:BATCH_SIZE = "1"
$env:GRAD_ACC = "8"
$env:MAXLEN = "512"
$env:SAVE_STEPS = "500"
$env:LOG_STEPS = "20"
$env:WARMUP_STEPS = "50"
$env:LORA_R = "16"
$env:LORA_ALPHA = "32"
$env:LORA_DROPOUT = "0.05"

py -3.11 -u "C:\Users\ariel\Downloads\chat socialbox\train_lyrics_lora.py"
