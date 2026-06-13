# Phase 48.0 - Sonda 2 (redesigned): CAPACITY PROBE = manifold expansion (teacher-forced ONLY)
#
# TAPS (Sonda 1) came back FLAT: the reservoir already integrates recent history into SEE(t),
# and reading the substrate (Q0) showed it is a LINEAR reservoir over a mostly-linear 43-D
# input - all nonlinearity lives in the readout. So the redesigned Sonda 2 tests the real
# lever: a fixed random NONLINEAR LIFT of L0, temporally integrated as new reservoir dims.
# Two arms differ ONLY by a cos():
#   armA (control, LINEAR):    EMA(z)       - predicted FLAT (same subspace, linear)
#   armB (treatment, NONLIN):  EMA(cos z)   - the real test
# both appended to the frozen 47 base [SEE 192 | L2_D1 64]=256 (anchor stays exact).
#
# PRE-REGISTERED (before the run, by the Architetto):
#   PASS -> armB boards the 48.A DAgger run under the frozen 47 harness, iff
#     [1] frozenD1 anchor |diff|<=0.005
#     [2] armB beats BOTH notap AND armA by >= 0.015 on val1/2/3
#     [3] armB_shuf does not beat notap by > 0.005 anywhere
#   A flat + B helps = bottleneck is the lever -> 48.A. Both flat -> MODOJA. A helps = surprise.
# TF-only: compression-positive != generative (44-47 law). No word-gate here.
#
# Run:  .\benchmarks\phase38-42\phase48_0_expand.ps1            (full, ~12 GB RAM)
#       .\benchmarks\phase38-42\phase48_0_expand.ps1 -Smoke     (tiny len, sanity only)

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
$WP = $WDIR + '\phase48_0exp'

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# pre-registered bars (criterion fixed BEFORE the run; tightening-only law applies)
$GAIN_MIN   = 0.015    # armB must beat BOTH notap and armA by at least this on ALL windows
$SHUF_TOL   = 0.005    # armB_shuf may not beat notap by more than this anywhere
$ANCHOR_TOL = 0.005    # frozenD1 vs BASE d1

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase48_0_expand.c' @SRC_CORE -o "$BINDIR\phase48_0_expand.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase48_0_expand'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase47a0_gauntlet','phase47b_decoder','phase47c_robust','phase47d_rollout',
              'phase47e_capacity','phase47f_tempcov','phase47g_lastmile','phase47h_h48tail',
              'phase47i_farfield','phase48_0_taps','phase48_0_expand')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Aborting.')
    exit 1
}

# -------- Run -------------------------------------------------------------------
$extra = @()
if ($Smoke) { $extra = @('--len','20000','--epochs','1'); Write-Host '(SMOKE: len 20000, 1 epoch - sanity only, numbers meaningless)' -ForegroundColor DarkYellow }
$OUT = $RDIR + '\phase48_0_expand.txt'
Write-Host ''
Write-Host '=== Phase 48.0 Sonda 2: EXPAND (6 probes, 4 windows, ~12 GB RAM) ===' -ForegroundColor Yellow
& "$BINDIR\phase48_0_expand.exe" $TS_DATA $D1_W $WP @extra 2>&1 | Tee-Object $OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'EXPAND probe failed'; exit 1 }

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
foreach ($need in @('frozenD1','notap','armA','armB','armB_shuf')) {
    if (-not $PROBE.ContainsKey($need)) { Write-Error "Probe output incomplete: missing $need"; exit 1 }
}

# -------- Tables -------------------------------------------------------------------
Write-Host ''
Write-Host '=== Probe table ===' -ForegroundColor Yellow
Write-Host ('  {0,-11} {1,5} {2,8} {3,8} {4,8} {5,8} {6,8} {7,8}' -f 'probe','dim','val1','val2','val3','rms1','ent1','maxp1')
foreach ($n in @('frozenD1','linear','notap','armA','armB','armB_shuf')) {
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]
    Write-Host ('  {0,-11} {1,5} {2,8} {3,8} {4,8} {5,8} {6,8} {7,8}' -f $n,$p['dim'],$p['val1'],$p['val2'],$p['val3'],$p['rms1'],$p['ent1'],$p['maxp1'])
}

Write-Host ''
Write-Host '=== Per-window gains vs notap (notap - X; positive = X better) ===' -ForegroundColor Yellow
Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f 'probe','gain_v1','gain_v2','gain_v3')
foreach ($n in @('armA','armB','armB_shuf')) {
    $g = @()
    foreach ($wi in 1..3) { $g += [math]::Round([double]$PROBE['notap']["val$wi"] - [double]$PROBE[$n]["val$wi"],4) }
    Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f $n,$g[0],$g[1],$g[2])
}
Write-Host ''
Write-Host '=== armB vs armA (the nonlinearity isolate; armA - armB, positive = B better) ===' -ForegroundColor Yellow
$ba = @()
foreach ($wi in 1..3) { $ba += [math]::Round([double]$PROBE['armA']["val$wi"] - [double]$PROBE['armB']["val$wi"],4) }
Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f 'B-over-A',$ba[0],$ba[1],$ba[2])

# -------- Pre-registered verdict ----------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 48.0 Sonda 2 (criterio pre-registrato) ===' -ForegroundColor Yellow
$checks = @()
foreach ($wi in 1..3) {
    $w = "val$wi"
    $d1 = [double]$BASE[$w]['d1']
    $fz = [double]$PROBE['frozenD1'][$w]
    $nt = [double]$PROBE['notap'][$w]
    $aa = [double]$PROBE['armA'][$w]
    $bb = [double]$PROBE['armB'][$w]
    $sh = [double]$PROBE['armB_shuf'][$w]
    $checks += [PSCustomObject]@{ name="anchor@$w";   ok=([math]::Abs($fz-$d1) -le $ANCHOR_TOL); detail=('frozenD1 {0:F4} vs d1 {1:F4}' -f $fz,$d1) }
    $checks += [PSCustomObject]@{ name="B>notap@$w";  ok=(($nt-$bb) -ge $GAIN_MIN);              detail=('armB {0:F4} vs notap {1:F4} (gain {2:F4}, bar {3:F3})' -f $bb,$nt,($nt-$bb),$GAIN_MIN) }
    $checks += [PSCustomObject]@{ name="B>armA@$w";   ok=(($aa-$bb) -ge $GAIN_MIN);               detail=('armB {0:F4} vs armA {1:F4} (gain {2:F4}, bar {3:F3})' -f $bb,$aa,($aa-$bb),$GAIN_MIN) }
    $checks += [PSCustomObject]@{ name="shuf@$w";     ok=(($nt-$sh) -le $SHUF_TOL);               detail=('armB_shuf {0:F4} vs notap {1:F4}' -f $sh,$nt) }
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
    Write-Host '  -> armB PASS: l espansione NONLINEARE del manifold porta capacita'' reale, controlli puliti.' -ForegroundColor Green
    Write-Host '     Il collo e'' la larghezza/nonlinearita'' dell input (substrate-side, mantra-puro).' -ForegroundColor Green
    Write-Host '     Guadagna il run DAgger in 48.A (harness 47 congelato). Stackabile con MODOJA.' -ForegroundColor Green
    Write-Host '     TF-only: nessuna celebrazione prima di gate v2 + repliche (legge 44-47).' -ForegroundColor Green
} else {
    Write-Host '  -> armB FAIL sul criterio pre-registrato. Letture (controllo armA decisivo):' -ForegroundColor Yellow
    Write-Host '     - armA gia'' aiuta ~ armB: il guadagno e'' integrare piu'' L0 linearmente, non la nonlinearita''.' -ForegroundColor Yellow
    Write-Host '     - entrambi piatti: la famiglia di substrato a dinamica fissa e'' il ceiling -> Sonda 3 MODOJA.' -ForegroundColor Yellow
    Write-Host '     - armB_shuf aiuta: capacita''/leak, non informazione -> bocciato comunque.' -ForegroundColor Yellow
    Write-Host '     - anchor FAIL: bug pipeline, NON interpretare nessun delta prima del fix.' -ForegroundColor Yellow
    Write-Host '     Prossima sonda comunque: MODOJA (substrato che impara COSA codificare).' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
