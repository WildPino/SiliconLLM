# Phase 51.A - associative-store TF probe (compile, smoke, full run, read verdict).
# TF-probe-first: no DAgger, no generator. One trainer, arms run sequentially INSIDE the exe
# (substrate extracted once, shared across the 5 store-configs). STOP + report at the TF gate.
#
# Run:  .\benchmarks\phase38-42\phase51_a.ps1            # smoke then full
#       .\benchmarks\phase38-42\phase51_a.ps1 -SkipFull  # smoke only
#       .\benchmarks\phase38-42\phase51_a.ps1 -SkipSmoke # full only

param([switch]$SkipSmoke,[switch]$SkipFull)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase51_a'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$BPE_W   = $WDIR + '\bpe1024.bin'
Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
foreach ($f in @($TS_DATA,$D1_W,$BPE_W)) { if (-not (Test-Path $f)) { Write-Error "missing $f"; exit 1 } }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma')
$SRC = @('benchmarks/phase38-42/phase51_a_store.c','src/silicon_entropy.c','src/silicon_v0.c')
$EXE = $BINDIR + '\phase51_a_store.exe'

Write-Host '=== COMPILE ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS @SRC -o $EXE -lm -I .
if ($LASTEXITCODE -ne 0) { Write-Error 'compile failed'; exit 1 }
Write-Host 'compiled OK' -ForegroundColor Green

if (-not $SkipSmoke) {
    Write-Host ''; Write-Host '=== SMOKE (tiny) ===' -ForegroundColor Cyan
    & $EXE $TS_DATA $D1_W $BPE_W ($RDIR+'\smoke') --smoke
    if ($LASTEXITCODE -ne 0) { Write-Error 'smoke failed'; exit 1 }
}

if (-not $SkipFull) {
    Write-Host ''; Write-Host '=== FULL TF PROBE (Dk=1024 Dv=256 decay=0.99 epochs=6 N=400000) ===' -ForegroundColor Cyan
    & $EXE $TS_DATA $D1_W $BPE_W ($RDIR+'\tf') --len 400000 --dk 1024 --dv 256 --decay 0.99 --epochs 6 |
        Tee-Object -FilePath ($RDIR+'\tf_probe.txt')
    if ($LASTEXITCODE -ne 0) { Write-Error 'full run failed'; exit 1 }
    Write-Host ''
    Write-Host ('record -> ' + $RDIR + '\tf_probe.txt') -ForegroundColor DarkCyan
}
