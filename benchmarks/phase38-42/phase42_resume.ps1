# Phase 42 Resume: Complete remaining tests with 5 epochs
# Run from the repo root:  .\benchmarks\phase38-42\phase42_resume.ps1

$SCRIPT_DIR = $PSScriptRoot
$ROOT    = Split-Path (Split-Path $SCRIPT_DIR)
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'

Set-Location $ROOT

$GCC_FLAGS = @('-O3', '-march=native', '-mavx2', '-mfma', '-lm', '-I', $ROOT)
$SRC_CORE  = @(($ROOT + '\src\silicon_entropy.c'), ($ROOT + '\src\silicon_v0.c'))

# ---- Recompile with 5-epoch changes -------------------------------------------
Write-Host ''
Write-Host '=== RECOMPILING WITH 5 EPOCHS ===' -ForegroundColor Cyan

Write-Host '  phase42a_pooling.exe ...'
& gcc @GCC_FLAGS ($ROOT + '\benchmarks\phase38-42\phase42a_pooling.c') @SRC_CORE -o "$BINDIR/phase42a_pooling.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed phase42a_pooling'; exit 1 }

Write-Host '  phase42c_r2.exe ...'
& gcc @GCC_FLAGS ($ROOT + '\benchmarks\phase38-42\phase42c_reservoir2.c') @SRC_CORE ($ROOT + '\src\silicon_r2.c') -o "$BINDIR/phase42c_r2.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed phase42c_r2'; exit 1 }

Write-Host 'Binaries recompiled with 5 epochs.' -ForegroundColor Green

# ---- Phase 42.A Mode 3 (thresh) only -----------------------------------------
Write-Host ''
Write-Host '=== Phase 42.A Mode 3 (thresh) - 5 epochs ===' -ForegroundColor Yellow

# Create a temporary C file that only runs mode 3
$temp_c = $ROOT + '\benchmarks\phase38-42\temp_mode3.c'
$original_c = $ROOT + '\benchmarks\phase38-42\phase42a_pooling.c'

# Read original and modify to only run mode 3
$content = Get-Content $original_c -Raw
$content = $content -replace 'for \(int mode = 0; mode < 4; mode\+\+\)', 'for (int mode = 3; mode < 4; mode++)'
$content = $content -replace 'Running 4 pooling modes', 'Running Mode 3 (thresh) only'
Set-Content -Path $temp_c -Value $content

Write-Host '  Compiling mode3-only version...'
& gcc @GCC_FLAGS $temp_c @SRC_CORE -o ($BINDIR + '\phase42a_mode3.exe')
if ($LASTEXITCODE -ne 0) { Write-Error 'Failed phase42a_mode3'; exit 1 }

$A_PREFIX = $WDIR + '\phase42a'
& "$BINDIR/phase42a_mode3.exe" $TS_DATA $A_PREFIX 2>&1 |
    Tee-Object ($ROOT + '\results\phase42_a_mode3_thresh.txt')
if ($LASTEXITCODE -ne 0) { Write-Error 'Phase 42.A Mode 3 failed'; exit 1 }

# Cleanup
Remove-Item $temp_c -Force
Remove-Item ($BINDIR + '\phase42a_mode3.exe') -Force

Write-Host ''
Write-Host 'Phase 42.A Mode 3 DONE.' -ForegroundColor Green

# ---- Phase 42.C — Reservoir2 -------------------------------------------------
Write-Host ''
Write-Host '=== Phase 42.C - Reservoir2 (5 epochs) ===' -ForegroundColor Yellow

$C_WEIGHTS = $WDIR + '\phase42c_r2.bin'
& "$BINDIR/phase42c_r2.exe" $TS_DATA $C_WEIGHTS --r2-alpha 0.9 --r2-seed 123 2>&1 |
    Tee-Object ($ROOT + '\results\phase42_c_r2.txt')
if ($LASTEXITCODE -ne 0) { Write-Error 'Phase 42.C failed'; exit 1 }

Write-Host ''
Write-Host 'Phase 42.C DONE. Results -> results\phase42_c_r2.txt' -ForegroundColor Green

# ---- Summary ----------------------------------------------------------------
Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
Write-Host 'Phase 42 Resume -- Complete'            -ForegroundColor Cyan
Write-Host '========================================'
Write-Host 'Mode 3 (thresh):  results\phase42_a_mode3_thresh.txt'
Write-Host 'Phase 42.C:       results\phase42_c_r2.txt'
