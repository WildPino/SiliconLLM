# Phase 43 Tribunal: 43.A (done) -> 43.A2 (fast grid) -> 43.B (byte_gain) -> generation
# Run from repo root:  .\benchmarks\phase38-42\phase43_run.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results'
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'

Set-Location $ROOT

$GCC_FLAGS = @('-O3', '-march=native', '-mavx2', '-mfma', '-lm', '-I', '.')
$SRC_CORE  = @('src/silicon_entropy.c', 'src/silicon_v0.c')

if (-not (Test-Path $TS_DATA)) { Write-Error ('TinyStories not found: ' + $TS_DATA); exit 1 }

# ---- Compile ----------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan

Write-Host '  phase43a2_fast_grid.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43a2_fast_grid.c' @SRC_CORE `
    -o "$BINDIR/phase43a2_fast_grid.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43a2_fast_grid'; exit 1 }

Write-Host '  phase43b_gain.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43b_gain.c' @SRC_CORE `
    -o "$BINDIR/phase43b_gain.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43b_gain'; exit 1 }

Write-Host '  phase43_generator.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE `
    -o "$BINDIR/phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }

Write-Host 'Compiled OK.' -ForegroundColor Green

# ---- Phase 43.A2 — Fast-alpha refinement ------------------------------------
Write-Host ''
Write-Host '=== Phase 43.A2 - Fast-Alpha Grid ===' -ForegroundColor Yellow
Write-Host '  fast in 0.3, 0.5 (confirm), 0.6 -- mid=0.9 slow=0.99 fixed'
Write-Host '  43.A result: ms_f0.5=2.2757, ms_f0.7=2.2777 -> f0.5 leads'
Write-Host '  Question: does f0.3 push further?'
Write-Host ''

$A2_PREFIX = $WDIR + '\phase43a2'
$A2_OUT    = $RDIR + '\phase43_a2_fast_grid.txt'

& "$BINDIR/phase43a2_fast_grid.exe" $TS_DATA $A2_PREFIX 2>&1 |
    Tee-Object $A2_OUT

if ($LASTEXITCODE -ne 0) { Write-Error 'Phase 43.A2 failed'; exit 1 }
Write-Host ''
Write-Host ('Phase 43.A2 DONE. Results -> ' + $A2_OUT) -ForegroundColor Green
Select-String 'SIGNAL' $A2_OUT | ForEach-Object { Write-Host ('  ' + $_.Line) -ForegroundColor Yellow }

# ---- Phase 43.B — Byte gain training ----------------------------------------
Write-Host ''
Write-Host '=== Phase 43.B - Byte Gain Alternating Optimization ===' -ForegroundColor Yellow
Write-Host '  Base: ms_f0.5 (best from 43.A)'
Write-Host '  Learns byte_gain[256]: how strongly each byte writes into L1 memory'
Write-Host '  Method: 1-step gradient approximation, 3 outer rounds'
Write-Host '  Expected: 3-4x baseline time per round'
Write-Host ''

$B_WEIGHTS = $WDIR + '\phase43b_gain.bin'
$B_OUT     = $RDIR + '\phase43_b_gain.txt'

& "$BINDIR/phase43b_gain.exe" $TS_DATA $B_WEIGHTS 2>&1 |
    Tee-Object $B_OUT

if ($LASTEXITCODE -ne 0) { Write-Error 'Phase 43.B failed'; exit 1 }
Write-Host ''
Write-Host ('Phase 43.B DONE. Results -> ' + $B_OUT) -ForegroundColor Green
Select-String 'SIGNAL' $B_OUT | ForEach-Object { Write-Host ('  ' + $_.Line) -ForegroundColor Yellow }

# ---- Generation: ms_f0.5 qualitative check ----------------------------------
Write-Host ''
Write-Host '=== Generation: ms_f0.5 qualitative audit ===' -ForegroundColor Yellow
Write-Host '  phase43_generator reads 0x53454535 format with multiscale_mode=1'
Write-Host '  Seed: first 512 bytes of val split; Gen: 2000 bytes, temp=0.8'
Write-Host ''

$MS05_WEIGHTS = $WDIR + '\phase43a2_f05.bin'
$GEN_OUT      = $RDIR + '\phase43_gen_ms05.txt'

if (Test-Path $MS05_WEIGHTS) {
    Write-Host ('Generating from ' + $MS05_WEIGHTS + ' ...')
    # stdout = generated text; stderr = stats (redirected to console)
    & "$BINDIR\phase43_generator.exe" $TS_DATA $MS05_WEIGHTS --gen-len 2000 --temp 0.8 2>&1 |
        Tee-Object $GEN_OUT
    Write-Host ''
    Write-Host ('Generation saved -> ' + $GEN_OUT) -ForegroundColor Green
} else {
    Write-Host ('Weights not found: ' + $MS05_WEIGHTS) -ForegroundColor Red
}

# ---- Summary ----------------------------------------------------------------
Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
Write-Host 'Phase 43 Tribunal -- Summary'
Write-Host '========================================'
Write-Host '  43.A  (done):  legacy=2.3197, ms_f0.5=2.2757, ms_f0.7=2.2777'
Write-Host '  43.A2 results: ' + $A2_OUT
Write-Host '  43.B  results: ' + $B_OUT
Write-Host ''
Write-Host 'Decision tree after 43.A2:'
Write-Host '  f0.3 wins by >0.002 BPB -> fast wants even shorter scale -> test f0.2'
Write-Host '  f0.5 confirmed -> freeze fast=0.5'
Write-Host 'Decision tree after 43.B:'
Write-Host '  gain helps >0.01 BPB -> byte injection identity matters -> refine gain training'
Write-Host '  gain flat -> bottleneck is elsewhere -> Phase 43.C Oja or new hypothesis'
