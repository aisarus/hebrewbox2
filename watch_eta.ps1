param([string]$Log, [string]$OutDir)

# дождёмся строки с samples и посчитаем Total = ceil(samples/(BS*ACC))
$Total = $null
$bs  = [int]([Environment]::GetEnvironmentVariable("BATCH_SIZE","Process")  ?? "1")
$acc = [int]([Environment]::GetEnvironmentVariable("GRAD_ACC","Process")    ?? "16")
if ($bs -lt 1) { $bs = 1 }; if ($acc -lt 1) { $acc = 16 }

$start = $null
$s0 = $null

Get-Content -Path $Log -Wait -Tail 0 | ForEach-Object {

  if (-not $Total -and $_ -match 'samples=(\d+)') {
    $samples = [int]$matches[1]
    $Total = [math]::Ceiling($samples / ($bs * $acc))
    Write-Host ("[watch] total steps = {0}  (samples={1}, bs={2}, acc={3})" -f $Total,$samples,$bs,$acc) -ForegroundColor Yellow
  }

  if ($_ -match 'step\s+(\d+):.*?ce=([0-9\.]+).*?tfm=([0-9\.]+)') {
    $cur = [int]$matches[1]; $ce = [double]$matches[2]; $tfm = [double]$matches[3]

    if (-not $start) { $start = Get-Date; $s0 = $cur }
    $done = [math]::Max(1, $cur - $s0)                # сколько шагов прошло С МОМЕНТА СТАРТА вотчера
    $elapsed = (Get-Date) - $start
    $rate = $done / [math]::Max(1e-6, $elapsed.TotalSeconds)
    $ppl  = [math]::Exp($ce)
    $eta  = if ($Total) {
      [timespan]::FromSeconds([int][math]::Max(0, ($Total - $cur) / [math]::Max($rate,1e-9)))
    } else { [timespan]::Zero }
    $tps = $tfm * $rate

    if ($cur % 10 -eq 0) {
      "{0}/{1} | ce={2:N3} ppl={3:N2} | it/s={4:N2} tok/s≈{5:N0} | ETA {6:hh\:mm\:ss}" -f $cur,($Total ?? 0),$ce,$ppl,$rate,$tps,$eta
    }
    if ($cur -ne 0 -and $cur % 10000 -eq 0) {
      New-Item -ItemType File -Path (Join-Path $OutDir ".force_save") -Force | Out-Null
    }
  }
}
