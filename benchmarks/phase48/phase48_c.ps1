# Phase 48.C - Multiplicative-Dynamics Reservoir (TF probe)
#
# 48.B = ceiling of the STATIC map (BILIN_sp kept ~85% of BILIN -> generic quadratic, not gating).
# 48.C moves the multiplication into the DYNAMICS: a fixed-random multiplicative reservoir
# (gate integrated over time), readout-only training. Base = frozen armB (512-D). New banks append.
#   DYN     : r^d = d*r + drive (x) g, g=hardsig(B.ctx). The candidate.
#   LIN_dyn : g == 0.5 (constant gate). Isolates the multiplication from "more integrated dims".
#   DYN_st  : DYN bands from time-permuted rows (leak guard).
#   DYN_sp  : channel-permuted gate pairing (gating semantics; informative).
#
# PRE-REGISTERED (before the run): anchor==BASE; DYN beats armB >=0.015 on val1&2&3; DYN beats
# LIN_dyn >=0.008 on val1&2&3; DYN_st does not beat armB by >0.005. DYN-DYN_sp = informative.
# Gate must be live (GATE_STATS std). PASS -> DYN to 48.C.A (DAgger closed-loop, frozen harness).
# TF != generative (44-47). The script verdict is ADVISORY; the user reads the full table.
#
# Run:  .\benchmarks\phase38-42\phase48_c.ps1            (full, ~18 GB RAM)
#       .\benchmarks\phase38-42\phase48_c.ps1 -Smoke     (tiny: pipeline + anchor + gate-live)

param([switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase48c'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }
$D1_W = $WDIR + '\phase44f_F0.bin'
if (-not (Test-Path $D1_W)) { Write-Error "Missing weight: $D1_W"; exit 1 }
$WP = $WDIR + '\phase48c'

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')
$GAIN_MIN  = 0.015     # DYN must beat armB by this on all windows
$LIN_MIN   = 0.008     # DYN must beat LIN_dyn by this on all windows (multiplication-specific)
$CTRL_TOL  = 0.005     # DYN_st may not beat armB by more than this
$ANCHOR_TOL= 0.005
$GATE_STD_MIN = 0.05   # gate liveness floor

Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase48_c.c' @SRC_CORE -o "$BINDIR\phase48_c.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase48_c'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

$TRAINERS = @('phase48_0_taps','phase48_0_expand','phase48a_armb','phase48a_fix','phase48_b','phase48_c')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting.'; exit 1 }

$extra = @(); if ($Smoke) { $extra = @('--len','20000','--epochs','1'); Write-Host '(SMOKE: len 20000, 1 epoch - numbers meaningless; validates pipeline/anchor/gate-live)' -ForegroundColor DarkYellow }
$OUT = $RDIR + '\phase48_c.txt'
Write-Host ''
Write-Host '=== Phase 48.C: 5 probes on the armB baseline (~18 GB RAM) ===' -ForegroundColor Yellow
& "$BINDIR\phase48_c.exe" $TS_DATA $D1_W $WP @extra 2>&1 | Tee-Object $OUT
if ($LASTEXITCODE -ne 0) { Write-Error '48.C probe failed'; exit 1 }

$BASE = @{}
foreach ($line in Select-String '^BASE ' $OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
    if ($kv['win']) { $BASE[$kv['win']] = $kv }
}
$PROBE = @{}
foreach ($line in Select-String '^PROBE ' $OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
    if ($kv['name']) { $PROBE[$kv['name']] = $kv }
}
$gateLine = (Select-String '^GATE_STATS ' $OUT | Select-Object -Last 1)
$GS = @{}; if ($gateLine) { foreach ($tok in ($gateLine.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $GS[$p[0]]=$p[1] } } }
foreach ($need in @('frozenD1','armB','DYN','LIN_dyn','DYN_st','DYN_sp')) {
    if (-not $PROBE.ContainsKey($need)) { Write-Error "Probe output incomplete: missing $need"; exit 1 }
}

Write-Host ''
Write-Host ('=== Gate liveness: mean={0} std={1} pct_lo={2} pct_hi={3} (live if std>={4}) ===' -f `
    $GS['mean'],$GS['std'],$GS['pct_lo'],$GS['pct_hi'],$GATE_STD_MIN) -ForegroundColor Yellow
$gate_live = ([double]$GS['std'] -ge $GATE_STD_MIN)
if (-not $gate_live) {
    Write-Host '  -> GATE QUASI-MORTO (std basso o incollato): DYN~=LIN_dyn per il motivo sbagliato.' -ForegroundColor Red
    Write-Host '     Alza GATE_GAIN nel .c o standardizza B.ctx, poi ripeti. (Numeri sotto non interpretabili.)' -ForegroundColor Red
}

Write-Host ''
Write-Host '=== Probe table ===' -ForegroundColor Yellow
Write-Host ('  {0,-9} {1,5} {2,8} {3,8} {4,8} {5,7}' -f 'probe','dim','val1','val2','val3','rms1')
foreach ($n in @('frozenD1','armB','DYN','LIN_dyn','DYN_st','DYN_sp')) {
    $p = $PROBE[$n]
    Write-Host ('  {0,-9} {1,5} {2,8} {3,8} {4,8} {5,7}' -f $n,$p['dim'],$p['val1'],$p['val2'],$p['val3'],$p['rms1'])
}
Write-Host ''
Write-Host '=== Gains (positive = better than the reference) ===' -ForegroundColor Yellow
Write-Host ('  {0,-22} {1,8} {2,8} {3,8}' -f 'comparison','v1','v2','v3')
function GainRow($label,$ref,$x) {
    $g=@(); foreach ($wi in 1..3) { $g += [math]::Round([double]$PROBE[$ref]["val$wi"] - [double]$PROBE[$x]["val$wi"],4) }
    Write-Host ('  {0,-22} {1,8} {2,8} {3,8}' -f $label,$g[0],$g[1],$g[2]); return $g
}
$g_dyn_armb = GainRow 'DYN  - armB'      'armB'   'DYN'
$g_lin_armb = GainRow 'LIN_dyn - armB'   'armB'   'LIN_dyn'
$g_st_armb  = GainRow 'DYN_st - armB'    'armB'   'DYN_st'
$g_sp_armb  = GainRow 'DYN_sp - armB'    'armB'   'DYN_sp'
$g_dyn_lin  = GainRow 'DYN  - LIN_dyn'   'LIN_dyn' 'DYN'
$g_dyn_sp   = GainRow 'DYN  - DYN_sp'    'DYN_sp'  'DYN'

Write-Host ''
Write-Host '=== Verdetto 48.C (criterio pre-registrato; ADVISORY - leggi la tabella) ===' -ForegroundColor Yellow
$anchor_ok = $true
foreach ($wi in 1..3) { if ([math]::Abs([double]$PROBE['frozenD1']["val$wi"] - [double]$BASE["val$wi"]['d1']) -gt $ANCHOR_TOL) { $anchor_ok=$false } }
Write-Host ('  anchor: {0}' -f ($(if($anchor_ok){'OK'}else{'FAIL - non interpretare nulla'}))) -ForegroundColor $(if($anchor_ok){'Green'}else{'Red'})
$dyn_armb_ok = ($g_dyn_armb[0] -ge $GAIN_MIN) -and ($g_dyn_armb[1] -ge $GAIN_MIN) -and ($g_dyn_armb[2] -ge $GAIN_MIN)
$dyn_lin_ok  = ($g_dyn_lin[0]  -ge $LIN_MIN)  -and ($g_dyn_lin[1]  -ge $LIN_MIN)  -and ($g_dyn_lin[2]  -ge $LIN_MIN)
$st_ok       = ($g_st_armb[0]  -le $CTRL_TOL) -and ($g_st_armb[1]  -le $CTRL_TOL) -and ($g_st_armb[2]  -le $CTRL_TOL)
Write-Host ('  DYN > armB  >= {0}:  {1}   gains=[{2}]' -f $GAIN_MIN,($(if($dyn_armb_ok){'OK'}else{'no'})),($g_dyn_armb -join ',')) -ForegroundColor $(if($dyn_armb_ok){'Green'}else{'Gray'})
Write-Host ('  DYN > LIN   >= {0}:  {1}   gains=[{2}]  (la moltiplicazione, non le dim)' -f $LIN_MIN,($(if($dyn_lin_ok){'OK'}else{'no'})),($g_dyn_lin -join ',')) -ForegroundColor $(if($dyn_lin_ok){'Green'}else{'Gray'})
Write-Host ('  DYN_st clean (<= {0}): {1}   gains=[{2}]' -f $CTRL_TOL,($(if($st_ok){'OK'}else{'LEAK'})),($g_st_armb -join ',')) -ForegroundColor $(if($st_ok){'Green'}else{'Red'})
Write-Host ('  [info] DYN - DYN_sp (gating semantics): [{0}]' -f ($g_dyn_sp -join ',')) -ForegroundColor DarkCyan
$pass = $anchor_ok -and $gate_live -and $dyn_armb_ok -and $dyn_lin_ok -and $st_ok
Write-Host ''
if ($Smoke) {
    Write-Host '  (SMOKE: verdetto non significativo - solo pipeline/anchor/gate-live. Il test vero e 1M/6ep.)' -ForegroundColor DarkYellow
} elseif ($pass) {
    Write-Host '  -> DYN PASS: la moltiplicazione nella DINAMICA compra capacita reale oltre il kernel armB.' -ForegroundColor Green
    Write-Host '     (Checkpoint salvato magic 0x53454549.) -> 48.C.A: DAgger closed-loop sotto harness 47' -ForegroundColor Green
    Write-Host '     congelato (gate v2 + repliche + lettura umana), col determinism MD5 come HARD GATE.' -ForegroundColor Green
    Write-Host '     TF != generativo: nessuna celebrazione prima del closed-loop.' -ForegroundColor Green
} else {
    Write-Host '  -> DYN non passa il criterio pre-registrato. Letture:' -ForegroundColor Yellow
    Write-Host '     DYN~=LIN_dyn = il guadagno e piu dimensioni integrate, non il gating (gate inerte o ridondante);' -ForegroundColor Yellow
    Write-Host '     DYN~=armB = la dinamica moltiplicativa fissa-random non aggiunge oltre il kernel;' -ForegroundColor Yellow
    Write-Host '     DYN_st aiuta = leak (la struttura non e temporale).' -ForegroundColor Yellow
    Write-Host '     Se la dinamica moltiplicativa fissa e piatta: il gating deve essere APPRESO -> MODOJA.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
