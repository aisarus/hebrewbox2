# -*- coding: utf-8 -*-
$ErrorActionPreference = 'Stop'

$root   = 'C:\Users\ariel\Downloads\chat socialbox'
$epoch2 = Join-Path $root 'out_lora\epoch_2'
Write-Host 'Watcher: ждём завершения epoch_2...' -ForegroundColor Yellow
while (!(Test-Path $epoch2)) { Start-Sleep -Seconds 10 }

# Резюмимся с самого свежего checkpoint-а, иначе с финала epoch_2
$ckpt = Get-ChildItem (Join-Path $root 'out_lora\checkpoint-*') -Directory -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
$resume = if ($ckpt) { $ckpt.FullName } else { $epoch2 }

# Гиперпараметры на 3-ю эпоху (легкий decay LR)
[System.Environment]::SetEnvironmentVariable('RESUME_ADAPTER',        $resume, 'Process')
[System.Environment]::SetEnvironmentVariable('BATCH_SIZE',            '1',      'Process')
[System.Environment]::SetEnvironmentVariable('GRAD_ACC',              '16',     'Process')   # «шаги летят» :)
[System.Environment]::SetEnvironmentVariable('LR',                    '8e-5',   'Process')   # чуть ниже, чем во 2-й
[System.Environment]::SetEnvironmentVariable('EPOCHS',                '1',      'Process')
[System.Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES',  '0',      'Process')
# Дополнительно: редкие сейвы
[System.Environment]::SetEnvironmentVariable('SAVE_EVERY_STEPS',      '2000',   'Process')
[System.Environment]::SetEnvironmentVariable('SAVE_TOTAL_LIMIT',      '3',      'Process')

$log = Join-Path $root 'train_epoch3.log'
Write-Host 'Стартуем 3-ю эпоху...' -ForegroundColor Green
py -3.11 (Join-Path $root 'train_lora_sft_v2.py') 2>&1 | Tee-Object -FilePath $log -Append
