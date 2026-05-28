# Phase 42 Tribunal: 42.0 (baseline) -> 42.A (pooling) -> 42.C (reservoir2)
# Run from the repo root:  .\benchmarks\phase42_run.ps1

$ROOT    = Split-Path $PSScriptRoot
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'

Set-Location $ROOT

$GCC_FLAGS = @('-O3', '-march=native', '-mavx2', '-mfma', '-lm', '-I', '.')
$SRC_CORE  = @('src/silicon_entropy.c', 'src/silicon_v0.c')

# ---- Compile ----------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan

Write-Host '  train_readout.exe ...'
& gcc @GCC_FLAGS benchmarks/train_entropy_readout.c @SRC_CORE -o "$BINDIR/train_readout.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed train_readout'; exit 1 }

Write-Host '  phase42a_pooling.exe ...'
& gcc @GCC_FLAGS benchmarks/phase42a_pooling.c @SRC_CORE -o "$BINDIR/phase42a_pooling.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed phase42a_pooling'; exit 1 }

Write-Host '  phase42c_r2.exe ...'
& gcc @GCC_FLAGS benchmarks/phase42c_reservoir2.c @SRC_CORE src/silicon_r2.c -o "$BINDIR/phase42c_r2.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed phase42c_r2'; exit 1 }

Write-Host '  phase41_generator.exe ...'
& gcc @GCC_FLAGS benchmarks/phase41_generator.c @SRC_CORE -o "$BINDIR/phase41_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed phase41_generator'; exit 1 }

Write-Host 'All binaries compiled.' -ForegroundColor Green

# ---- Phase 42.0 — TinyStories Baseline -------------------------------------
Write-Host ''
Write-Host '=== Phase 42.0 - TinyStories Baseline ===' -ForegroundColor Yellow

$TS_WEIGHTS = $WDIR + '\phase42_ts_baseline.bin'
Write-Host 'Training on TinyStories -- 50pct train, 25pct val...'
& "$BINDIR/train_readout.exe" $TS_DATA $TS_WEIGHTS
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }

Write-Host ''
Write-Host 'Audit -- val window 50-75pct...'
& "$ROOT/see.exe" audit $TS_DATA --weights $TS_WEIGHTS --blend moe --no-lz --eval-start 50 --eval-len 25

Write-Host ''
Write-Host 'Generating 2000 bytes at T=0.7 -- seed = first 512 bytes of val...'
$SEED_FILE = $ROOT + '\data\phase42_ts_seed.bin'
$bytes = [System.IO.File]::ReadAllBytes($TS_DATA)
$seed_bytes = [byte[]]$bytes[(32000000)..(32000511)]
[System.IO.File]::WriteAllBytes($SEED_FILE, $seed_bytes)
& "$BINDIR/phase41_generator.exe" $SEED_FILE $TS_WEIGHTS --gen-len 2000 --mode sample --temp 0.7 2>&1 |
    Tee-Object ($ROOT + '\results\phase42_0_generation.txt')

Write-Host ''
Write-Host ('Phase 42.0 DONE. Weights -> ' + $TS_WEIGHTS) -ForegroundColor Green

# ---- Phase 42.A — Rich Pooling Variants ------------------------------------
Write-Host ''
Write-Host '=== Phase 42.A - Rich Pooling Tribunal ===' -ForegroundColor Yellow
Write-Host 'Running 4 pooling modes: sum, max, range, threshold...'
Write-Host 'Expected time: approx 40-80 min per mode.'

$A_PREFIX = $WDIR + '\phase42a'
& "$BINDIR/phase42a_pooling.exe" $TS_DATA $A_PREFIX 2>&1 |
    Tee-Object ($ROOT + '\results\phase42_a_tribunal.txt')
if ($LASTEXITCODE -ne 0) { Write-Error 'Phase 42.A failed'; exit 1 }

Write-Host ''
Write-Host 'Phase 42.A DONE. Results -> results\phase42_a_tribunal.txt' -ForegroundColor Green

# ---- Phase 42.C — Reservoir2 -----------------------------------------------
Write-Host ''
Write-Host '=== Phase 42.C - Reservoir2 ===' -ForegroundColor Yellow
Write-Host 'Training with 256D features: 192 SEE + 64 R2, alpha=0.9...'
Write-Host 'Expected time: approx 1.5x the baseline.'

$C_WEIGHTS = $WDIR + '\phase42c_r2.bin'
& "$BINDIR/phase42c_r2.exe" $TS_DATA $C_WEIGHTS --r2-alpha 0.9 --r2-seed 123 2>&1 |
    Tee-Object ($ROOT + '\results\phase42_c_r2.txt')
if ($LASTEXITCODE -ne 0) { Write-Error 'Phase 42.C failed'; exit 1 }

Write-Host ''
Write-Host 'Phase 42.C DONE. Results -> results\phase42_c_r2.txt' -ForegroundColor Green

# ---- Summary ----------------------------------------------------------------
Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
Write-Host 'Phase 42 Tribunal -- Summary'            -ForegroundColor Cyan
Write-Host '========================================'
Write-Host '42.0 baseline:  weights\phase42_ts_baseline.bin'
Write-Host '42.A pooling:   results\phase42_a_tribunal.txt'
Write-Host '42.C reservoir: results\phase42_c_r2.txt'
Write-Host ''
Write-Host 'To audit 42.C weights manually:'
Write-Host ('  .\see.exe audit ' + $TS_DATA + ' --weights ' + $C_WEIGHTS + ' --blend moe --no-lz --eval-start 50 --eval-len 25')
Write-Host ''
Write-Host 'Key metrics:'
Write-Host '  42.0  baseline -- training-matched domain'
Write-Host '  42.A  same domain, different pooling -- is 192D the bottleneck?'
Write-Host '  42.C  same domain, +64D temporal memory -- is stationarity the bottleneck?'
