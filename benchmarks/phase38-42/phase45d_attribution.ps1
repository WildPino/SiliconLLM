# Phase 45.D - per-event causal attribution of delta L2 writes (diagnostic, NO training)
#
# Question: is the USEFUL delta write signal separable by INTERNAL signals, or inseparable
# from the regime that collapses in closed-loop? Teacher-forced over a val window on the
# co-adapted C0_delta readout. Per write event: cos(write,L2_old), rel_move, write_norm,
# surprise, entropy, punct/ws, and the marginal counterfactual dloss = loss(next byte|L2_old)
# - loss(next byte|L2_new). The probe prints r(signal,dloss), best single-threshold kept-net,
# and decile means. No training, low RAM, ~1-2 min per 1M-byte window.
#
# Run:  .\benchmarks\phase38-42\phase45d_attribution.ps1 [-Len 1000000] [-Start -1]

param([long]$Len = 1000000, [long]$Start = -1)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase45d'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
$C0_W    = $WDIR + '\phase45c_C0_delta.bin'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }
if (-not (Test-Path $C0_W)) { Write-Error "Missing C0_delta (run phase45c first): $C0_W"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase45d_attribution.c' @SRC_CORE -o "$BINDIR\phase45d_attribution.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase45d_attribution'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

$OUT = $RDIR + '\events.tsv'
$startArg = if ($Start -ge 0) { @('--start', $Start) } else { @() }

Write-Host ''
Write-Host ('=== Phase 45.D attribution (C0_delta, window len={0}) ===' -f $Len) -ForegroundColor Yellow
& "$BINDIR\phase45d_attribution.exe" $TS_DATA $C0_W @startArg --len $Len --out $OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Attribution probe failed'; exit 1 }

Write-Host ''
Write-Host '  Reading: r(signal,dloss)~0 AND best kept_net ~ keep-all NET (far below oracle) for ALL' -ForegroundColor Gray
Write-Host '  signals => the useful/harmful split is INSEPARABLE by internal signals: no per-event'  -ForegroundColor Gray
Write-Host '  policy (hard or soft) can isolate the helpful writes -> delta is non-separable from the' -ForegroundColor Gray
Write-Host '  regime. If instead some signal r is large and its kept_net approaches oracle => a SOFT'   -ForegroundColor Gray
Write-Host '  policy on that signal is viable (45.E).' -ForegroundColor Gray
Write-Host ''
Write-Host ('  Per-event TSV: ' + $OUT)
