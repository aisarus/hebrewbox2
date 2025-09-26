# -*- coding: utf-8 -*-
$ErrorActionPreference='Stop'
$root = 'C:\Users\ariel\Downloads\chat socialbox'
$out  = Join-Path $root 'out_lora'
$ds   = Join-Path $root 'datasets\hebrew_lyrics_prompting_finetune\train_norm.jsonl'

Write-Host 'Ждём завершения epoch_3...' -Foreground Yellow
while (!(Test-Path (Join-Path $out 'epoch_3\adapter_model.safetensors'))) { Start-Sleep -Seconds 10 }

$ckpt = Get-ChildItem (Join-Path $out 'checkpoint-*') -Directory -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
$resume = if ($ckpt) { $ckpt.FullName } else { Join-Path $out 'epoch_3' }

[Environment]::SetEnvironmentVariable('RESUME_ADAPTER', $resume, 'Process')
[Environment]::SetEnvironmentVariable('DATA_FILE',       $ds,     'Process')
[Environment]::SetEnvironmentVariable('BATCH_SIZE',      '1',      'Process')
[Environment]::SetEnvironmentVariable('GRAD_ACC',        '16',     'Process')
[Environment]::SetEnvironmentVariable('LR',              '6e-5',   'Process')
[Environment]::SetEnvironmentVariable('EPOCHS',          '1',      'Process')
[Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES','0',  'Process')
[Environment]::SetEnvironmentVariable('SAVE_EVERY_STEPS','2000',   'Process')
[Environment]::SetEnvironmentVariable('SAVE_TOTAL_LIMIT','3',      'Process')

Remove-Item (Join-Path $out '.force_save') -ErrorAction SilentlyContinue
Write-Host 'Стартуем epoch_4 на новом датасете...' -Foreground Green
py -3.11 -u (Join-Path $root 'train_lora_sft_v2.py') 2>&1 | Tee-Object -FilePath (Join-Path $root 'train_epoch4_hebrew.log') -Append
