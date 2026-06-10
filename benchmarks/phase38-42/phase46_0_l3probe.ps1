# Phase 46.0 - slow phrase/episode L3 boundary probe above D1 (diagnostic, NO training)
#
# Axis B: add a SLOWER L3 memory above the near-stable D1, refreshed RARELY (phrase/episode
# scale), not every L2 boundary. 46.0 measures candidate slow-boundary definitions on the
# real D1 trajectory (teacher-forced) and estimates the L3 update cadence, so 46.A can be
# designed. No training, low RAM, ~1-2 min per 1M-byte window. Readout stays linear; only the
# allowed signals (entropy/surprise, punct/ws byte class, L2-gate clusters, low-margin
# episodes, L2 state-change). Prints: event frequency, inter-event gap mean/p50/p90, L3
# feature stability (rel-move), cheap loss-reduction (loss after vs before), onset examples,
# and a proposed 46.A grid.
#
# Run:  .\benchmarks\phase38-42\phase46_0_l3probe.ps1 [-Len 1500000] [-Weight D1path] [-Statechg 0.5]

param(
    [long]$Len = 1500000,
    [string]$Weight = '',
    [double]$Statechg = 0.5,
    [long]$K = 8,
    [long]$Refr = 16,
    [double]$Margin = 0.10,
    [long]$MSteps = 4
)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase46_0'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
if ($Weight -eq '') { $Weight = $WDIR + '\phase44f_F0.bin' }   # D1 control (mix0.5 scale0.5)

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }
if (-not (Test-Path $Weight)) { Write-Error "Missing D1 weight: $Weight"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase46_0_l3probe.c' @SRC_CORE -o "$BINDIR\phase46_0_l3probe.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase46_0_l3probe'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

$OUT = $RDIR + '\l3events.tsv'
Write-Host ''
Write-Host ('=== Phase 46.0 L3 boundary survey (D1={0}, len={1}) ===' -f (Split-Path $Weight -Leaf),$Len) -ForegroundColor Yellow
& "$BINDIR\phase46_0_l3probe.exe" $TS_DATA $Weight --len $Len --statechg $Statechg --k $K --refr $Refr --margin $Margin --msteps $MSteps --out $OUT
if ($LASTEXITCODE -ne 0) { Write-Error '46.0 probe failed'; exit 1 }

Write-Host ''
Write-Host '  Reading: L3-scale = gap p50 in ~[20,500] bytes (phrase/episode), NOT ~5 (L2 rate) nor' -ForegroundColor Gray
Write-Host '  ~6000 (document). dLoss<0 = boundary marks a transition into an easier span (resolution);' -ForegroundColor Gray
Write-Host '  dLoss>0 = onset of a harder span (difficulty). L3rm~0 => a slow EMA(L2) is redundant with' -ForegroundColor Gray
Write-Host '  the already-frozen L2; an informative L3 must summarize the FAST SEE stream at phrase cadence.' -ForegroundColor Gray
Write-Host ''
Write-Host ('  Per-event TSV: ' + $OUT)
