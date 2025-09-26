RUN THIS ON WINDOWS (PowerShell):

1) Create and activate venv (once):
   py -3.11 -m venv .venv
   . .venv\Scripts\Activate.ps1

2) Install libs:
   pip install -U torch transformers peft datasets safetensors protobuf

3) Copy 'train_lyrics_lora.py' into your 'C:\Users\ariel\Downloads\chat socialbox' folder.

4) Edit 'run_train_lyrics.ps1' if your paths differ, then run:
   powershell -ExecutionPolicy Bypass -File run_train_lyrics.ps1

Outputs: checkpoints and final_adapter under out_lyrics_lora.
You can resume training by setting $env:RESUME_CKPT to a checkpoint path.
