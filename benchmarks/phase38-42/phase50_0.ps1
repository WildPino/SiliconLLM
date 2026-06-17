# Phase 50.0 - Unit-Choice Cartography (the MAP). Cheap, NO substrate, NO training.
# Compiles + runs the corpus-statistics mapper and tees the table to results/phase50_0.
# Mantra-guard: this only MAPS the candidate units (byte / word / BPE-512/1024/4096 / bytepair /
# hash3gram) with a fair n-gram model in bits-per-BYTE; the silicon (Architect) decides the winner.
# No unit is wired into the readout here -> that is 50.A, only for the winner(s), after the read.
#
# Run:  .\benchmarks\phase38-42\phase50_0.ps1
#       .\benchmarks\phase38-42\phase50_0.ps1 -Smoke   (2 MB slice)

param([switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$RDIR    = $ROOT + '\results\phase50_0'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

Write-Host ''
Write-Host '=== COMPILING (phase50_0_map; no substrate, no training) ===' -ForegroundColor Cyan
& gcc -O3 -march=native ('benchmarks/phase38-42/phase50_0_map.c') -o ($BINDIR+'\phase50_0_map.exe') -lm
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

$OUT = $RDIR + '\map.txt'
$argline = @($TS_DATA)
if ($Smoke) { $argline += @('--max-bytes','2000000') }

Write-Host ''
Write-Host ('=== RUN MAP{0} ===' -f $(if ($Smoke) {' (smoke 2MB)'} else {' (full 64MB)'})) -ForegroundColor Yellow
& "$BINDIR\phase50_0_map.exe" @argline 2>&1 | Tee-Object $OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Map run failed'; exit 1 }

Write-Host ''
Write-Host '=== READ (advisory; the Architect reads) ===' -ForegroundColor Yellow
Write-Host '  A unit WINS CONSIDERATION if it: lowers-or-holds bpb-o2 (unit-invariant) AND collapses' -ForegroundColor Gray
Write-Host '  repeat-mass (dissolves the byte flood) AND shortens the sequence (b/unit > 1) AND stays' -ForegroundColor Gray
Write-Host '  feasible (vocab not absurd). LOSSY rows understate bpb (not directly promotable).' -ForegroundColor Gray
Write-Host '  Winner(s) -> 50.A: wire the unit into the readout (armB substrate BYTE-DRIVEN INVARIATO).' -ForegroundColor Gray
Write-Host ('  Table: ' + $OUT) -ForegroundColor Cyan
