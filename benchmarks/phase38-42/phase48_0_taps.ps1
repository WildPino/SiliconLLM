# Phase 48.0 - Sonda 1: TAPS (readout-side temporal taps, teacher-forced ONLY)
#
# Phase 48 = substrate frontier. Sonda 1 asks the cheapest question first: does the H32
# readout gain >= 0.02 BPB on all 3 val windows when it sees the SAME frozen SEE at
# delays 8/32/128, vs the no-tap control in the SAME pipeline? Mandatory control:
# shuffled-tap (taps from random rows) must NOT help. Ablation: leave-one-out per tap.
#
# PRE-REGISTERED (before the run):
#   PASS -> TAPS boards the 48.A DAgger run under the frozen 47 harness
#   iff [1] frozenD1 anchor |diff|<=0.005  [2] notap-taps >= 0.02 on val1/2/3
#       [3] shuftap does not beat notap by > 0.005 anywhere
# TF-only caveat: compression-positive != generative (44/45/46/47 law). No word-gate here.
#
# Run:  .\benchmarks\phase38-42\phase48_0_taps.ps1            (full, ~13 GB RAM)
#       .\benchmarks\phase38-42\phase48_0_taps.ps1 -Smoke     (tiny len, sanity only)

param([switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase48_0'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$D1_W = $WDIR + '\phase44f_F0.bin'     # D1 (mix0.5 scale0.5), frozen C2.A SEE inside
if (-not (Test-Path $D1_W)) { Write-Error "Missing weight: $D1_W"; exit 1 }
$WP = $WDIR + '\phase48_0'

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# pre-registered bars (criterion fixed BEFORE the run; tightening-only law applies)
$GAIN_MIN   = 0.02     # taps must beat notap by at least this on ALL val windows
$SHUF_TOL   = 0.005    # shuftap may not beat notap by more than this anywhere
$ANCHOR_TOL = 0.005    # frozenD1 vs BASE d1

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase48_0_taps.c' @SRC_CORE -o "$BINDIR\phase48_0_taps.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase48_0_taps'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase47a0_gauntlet','phase47b_decoder','phase47c_robust','phase47d_rollout',
              'phase47e_capacity','phase47f_tempcov','phase47g_lastmile','phase47h_h48tail',
              'phase47i_farfield','phase48_0_taps')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Aborting.')
    exit 1
}

# -------- Run -------------------------------------------------------------------
$extra = @()
if ($Smoke) { $extra = @('--len','20000','--epochs','1'); Write-Host '(SMOKE: len 20000, 1 epoch - sanity only, numbers meaningless)' -ForegroundColor DarkYellow }
$OUT = $RDIR + '\phase48_0_taps.txt'
Write-Host ''
Write-Host '=== Phase 48.0 Sonda 1: TAPS (8 probes, 4 windows, ~13 GB RAM) ===' -ForegroundColor Yellow
& "$BINDIR\phase48_0_taps.exe" $TS_DATA $D1_W $WP @extra 2>&1 | Tee-Object $OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'TAPS probe failed'; exit 1 }

# -------- Parse ------------------------------------------------------------------
$BASE = @{}
foreach ($line in Select-String '^BASE ' $OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
    $BASE[$kv['win']] = $kv
}
$PROBE = @{}
foreach ($line in Select-String '^PROBE ' $OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
    $PROBE[$kv['name']] = $kv
}
foreach ($need in @('frozenD1','notap','taps','shuftap','abl_no8','abl_no32','abl_no128')) {
    if (-not $PROBE.ContainsKey($need)) { Write-Error "Probe output incomplete: missing $need"; exit 1 }
}

# -------- Tables -------------------------------------------------------------------
Write-Host ''
Write-Host '=== Probe table ===' -ForegroundColor Yellow
Write-Host ('  {0,-11} {1,5} {2,8} {3,8} {4,8} {5,8} {6,8} {7,8}' -f 'probe','dim','val1','val2','val3','rms1','ent1','maxp1')
foreach ($n in @('frozenD1','linear','notap','taps','shuftap','abl_no8','abl_no32','abl_no128')) {
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]
    Write-Host ('  {0,-11} {1,5} {2,8} {3,8} {4,8} {5,8} {6,8} {7,8}' -f $n,$p['dim'],$p['val1'],$p['val2'],$p['val3'],$p['rms1'],$p['ent1'],$p['maxp1'])
}

Write-Host ''
Write-Host '=== Per-window gains (notap - X; positive = X better) ===' -ForegroundColor Yellow
Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f 'probe','gain_v1','gain_v2','gain_v3')
foreach ($n in @('taps','shuftap','abl_no8','abl_no32','abl_no128')) {
    $g = @()
    foreach ($wi in 1..3) { $g += [math]::Round([double]$PROBE['notap']["val$wi"] - [double]$PROBE[$n]["val$wi"],4) }
    Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f $n,$g[0],$g[1],$g[2])
}
Write-Host ''
Write-Host '=== Per-tap contribution (abl_noX - taps; positive = tap X carries info) ===' -ForegroundColor Yellow
foreach ($n in @('abl_no8','abl_no32','abl_no128')) {
    $c = @()
    foreach ($wi in 1..3) { $c += [math]::Round([double]$PROBE[$n]["val$wi"] - [double]$PROBE['taps']["val$wi"],4) }
    $tapname = $n -replace 'abl_no','tap'
    Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f $tapname,$c[0],$c[1],$c[2])
}

# -------- Pre-registered verdict ----------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 48.0 Sonda 1 (criterio pre-registrato) ===' -ForegroundColor Yellow
$checks = @()
foreach ($wi in 1..3) {
    $w = "val$wi"
    $d1 = [double]$BASE[$w]['d1']
    $fz = [double]$PROBE['frozenD1'][$w]
    $nt = [double]$PROBE['notap'][$w]
    $tp = [double]$PROBE['taps'][$w]
    $sh = [double]$PROBE['shuftap'][$w]
    $checks += [PSCustomObject]@{ name="anchor@$w";  ok=([math]::Abs($fz-$d1) -le $ANCHOR_TOL); detail=('frozenD1 {0:F4} vs d1 {1:F4}' -f $fz,$d1) }
    $checks += [PSCustomObject]@{ name="gain@$w";    ok=(($nt-$tp) -ge $GAIN_MIN);              detail=('taps {0:F4} vs notap {1:F4} (gain {2:F4}, bar {3:F3})' -f $tp,$nt,($nt-$tp),$GAIN_MIN) }
    $checks += [PSCustomObject]@{ name="shuftap@$w"; ok=(($nt-$sh) -le $SHUF_TOL);              detail=('shuftap {0:F4} vs notap {1:F4}' -f $sh,$nt) }
}
$pass = $true
foreach ($c in $checks) {
    if (-not $c.ok) { $pass = $false }
    Write-Host ('  {0,-14} {1,-6} {2}' -f $c.name,($(if($c.ok){'OK'}else{'FAIL'})),$c.detail) -ForegroundColor $(if($c.ok){'Green'}else{'Red'})
}
Write-Host ''
if ($Smoke) {
    Write-Host '  (SMOKE RUN: verdict non significativo, serve solo a validare la pipeline.)' -ForegroundColor DarkYellow
} elseif ($pass) {
    Write-Host '  -> TAPS PASS: informazione temporale reale nel readout-side, controlli puliti.' -ForegroundColor Green
    Write-Host '     Guadagna il run DAgger in 48.A (harness 47 congelato: H32+gate v2+repliche+lettura).' -ForegroundColor Green
    Write-Host '     Nota: readout-side -> stackabile con LADDER/MODOJA se vincono anche loro.' -ForegroundColor Green
    Write-Host '     TF-only: nessuna celebrazione prima di gate v2 + repliche (legge 44-47).' -ForegroundColor Green
} else {
    Write-Host '  -> TAPS FAIL sul criterio pre-registrato. Letture possibili:' -ForegroundColor Yellow
    Write-Host '     - gain < 0.02: la storia SEE a 8/32/128 byte non porta info oltre [SEE|L2] istantaneo.' -ForegroundColor Yellow
    Write-Host '     - shuftap aiuta: il gain era capacita''/leak, non informazione temporale.' -ForegroundColor Yellow
    Write-Host '     - anchor FAIL: bug di pipeline, NON interpretare nessun delta prima del fix.' -ForegroundColor Yellow
    Write-Host '     Si prosegue comunque: Sonda 2 (LADDER) e Sonda 3 (MODOJA) sono substrate-side.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
