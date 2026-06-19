# Phase 43 - Generator Reproducibility Check
#
# Why: 43.C3 exposed that the SEE-V2 control swung from name-ish ~7/13.5 to
# ~16.7/66.5 on IDENTICAL weights. Root cause: the sampling RNG was seeded from
# time(NULL), so every run produced different text. The word-gate was measuring
# sampling noise on top of the real signal.
#
# Fix: --rng-seed (fixed default). This script proves determinism:
#   A) 3 consecutive runs, identical args, fixed rng -> byte-identical output
#   B) 1 run with --rng-seed -1 (time-based) -> different (sanity that it varies)
#
# Run:  .\benchmarks\phase38-42\phase43_repro_check.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43c3\repro'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }

$GEN_EXE = $BINDIR + '\phase43_generator.exe'
$W       = $WDIR + '\phase43c_eta1e3.bin'   # SEE-V2
if (-not (Test-Path $GEN_EXE)) { Write-Error "Generator not built: $GEN_EXE"; exit 1 }
if (-not (Test-Path $W))       { Write-Error "Weights missing: $W"; exit 1 }

$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'
$OFF     = 16777216   # one fixed seed offset

function Run-Gen([string]$tag, [string]$extra) {
    $tf = $RDIR + '\repro_' + $tag + '.txt'
    $sf = $RDIR + '\repro_' + $tag + '_stats.txt'
    $cmd = "`"$GEN_EXE`" `"$TS_DATA`" `"$W`" --gen-len $GEN_LEN --warmup $WARMUP --temp $T --seed-start $OFF $extra > `"$tf`" 2>`"$sf`""
    cmd /c $cmd | Out-Null
    if (-not (Test-Path $tf)) { return $null }
    return (Get-FileHash $tf -Algorithm MD5).Hash
}

Write-Host ''
Write-Host '=== A) Determinism: 3 runs, fixed rng (default) ===' -ForegroundColor Yellow
$h1 = Run-Gen 'fixed_1' ''
$h2 = Run-Gen 'fixed_2' ''
$h3 = Run-Gen 'fixed_3' ''
Write-Host ('  run1 MD5: {0}' -f $h1)
Write-Host ('  run2 MD5: {0}' -f $h2)
Write-Host ('  run3 MD5: {0}' -f $h3)
$det_ok = ($h1 -eq $h2) -and ($h2 -eq $h3) -and ($null -ne $h1)
if ($det_ok) { Write-Host '  -> IDENTICAL: generator is deterministic.' -ForegroundColor Green }
else         { Write-Host '  -> DIFFER: still non-deterministic - investigate.' -ForegroundColor Red }

Write-Host ''
Write-Host '=== B) Sanity: --rng-seed -1 (time-based) should vary ===' -ForegroundColor Yellow
$t1 = Run-Gen 'time_1' '--rng-seed -1'
Start-Sleep -Milliseconds 1100   # ensure time(NULL) advances
$t2 = Run-Gen 'time_2' '--rng-seed -1'
Write-Host ('  time1 MD5: {0}' -f $t1)
Write-Host ('  time2 MD5: {0}' -f $t2)
if ($t1 -ne $t2) { Write-Host '  -> DIFFER (expected): time-based mode still available.' -ForegroundColor Green }
else             { Write-Host '  -> identical (rare but possible if same second).' -ForegroundColor Yellow }

Write-Host ''
Write-Host '=== C) Explicit --rng-seed N is reproducible ===' -ForegroundColor Yellow
$s1 = Run-Gen 'seed42_1' '--rng-seed 42'
$s2 = Run-Gen 'seed42_2' '--rng-seed 42'
Write-Host ('  seed42 run1 MD5: {0}' -f $s1)
Write-Host ('  seed42 run2 MD5: {0}' -f $s2)
if ($s1 -eq $s2) { Write-Host '  -> IDENTICAL: explicit seed reproduces.' -ForegroundColor Green }
else             { Write-Host '  -> DIFFER: bug.' -ForegroundColor Red }

Write-Host ''
if ($det_ok) {
    Write-Host 'VERDICT: reproducibility FIXED. Word-gate now measures signal, not RNG.' -ForegroundColor Green
    Write-Host '         Re-run word-gates (C2/C3) would now be stable; proceed to 43.D.' -ForegroundColor Green
} else {
    Write-Host 'VERDICT: NOT reproducible yet - do not trust word-gate deltas.' -ForegroundColor Red
}
Write-Host ('  Files: ' + $RDIR)
