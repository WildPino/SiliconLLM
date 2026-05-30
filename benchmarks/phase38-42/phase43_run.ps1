# Phase 43.A — Multi-Timescale L1 Tribunal
# Run from repo root:  .\benchmarks\phase38-42\phase43_run.ps1

# Two levels up from phase38-42/ -> benchmarks/ -> repo root
$ROOT   = Split-Path (Split-Path $PSScriptRoot)
$BINDIR = $ROOT + '\bin'
$WDIR   = $ROOT + '\weights'
$RDIR   = $ROOT + '\results'
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'

Set-Location $ROOT

$GCC_FLAGS = @('-O3', '-march=native', '-mavx2', '-mfma', '-lm', '-I', '.')
$SRC_CORE  = @('src/silicon_entropy.c', 'src/silicon_v0.c')

# ---- Compile ----------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan

Write-Host '  phase43a_multiscale.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43a_multiscale.c' @SRC_CORE `
    -o "$BINDIR/phase43a_multiscale.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43a_multiscale'; exit 1 }

Write-Host 'Compiled OK.' -ForegroundColor Green

# ---- Verify data file -------------------------------------------------------
if (-not (Test-Path $TS_DATA)) {
    Write-Error ('TinyStories not found: ' + $TS_DATA)
    exit 1
}
Write-Host ('Data: ' + $TS_DATA)

# ---- Phase 43.A — Multi-Timescale Tribunal ----------------------------------
Write-Host ''
Write-Host '=== Phase 43.A - Multi-Timescale L1 Tribunal ===' -ForegroundColor Yellow
Write-Host '  Config 0: legacy    (chunk EMA decay=0.75, single alpha)'
Write-Host '  Config 1: ms_f0.5   (fast=0.5, mid=0.9, slow=0.99, per-byte)'
Write-Host '  Config 2: ms_f0.7   (fast=0.7, mid=0.9, slow=0.99, per-byte)'
Write-Host '  Readout:  identical (192D -> 256-class softmax residual)'
Write-Host '  Expected: approx 3x baseline time (3 configs x same data)'
Write-Host ''

$A_PREFIX = $WDIR + '\phase43a'
$A_OUT    = $RDIR + '\phase43_a_multiscale.txt'

& "$BINDIR/phase43a_multiscale.exe" $TS_DATA $A_PREFIX 2>&1 |
    Tee-Object $A_OUT

if ($LASTEXITCODE -ne 0) { Write-Error 'Phase 43.A failed'; exit 1 }

Write-Host ''
Write-Host ('Phase 43.A DONE. Results -> ' + $A_OUT) -ForegroundColor Green

# ---- Summary ----------------------------------------------------------------
Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
Write-Host 'Phase 43.A -- Summary'
Write-Host '========================================'
Write-Host ('Results file: ' + $A_OUT)
Write-Host ''
Write-Host 'Weights written:'
Write-Host ('  ' + $WDIR + '\phase43a_legacy.bin')
Write-Host ('  ' + $WDIR + '\phase43a_ms05.bin')
Write-Host ('  ' + $WDIR + '\phase43a_ms07.bin')
Write-Host ''
Write-Host 'Reading key signal from results...'
Select-String 'SIGNAL' $A_OUT | ForEach-Object { Write-Host ('  ' + $_.Line) -ForegroundColor Yellow }
Write-Host ''
Write-Host 'Interpretation guide:'
Write-Host '  delta > -0.03 : temporal scale is NOT the bottleneck -> proceed to 43.B (injection)'
Write-Host '  delta < -0.03 : temporal scale IS a bottleneck -> refine alpha grid before 43.B'
