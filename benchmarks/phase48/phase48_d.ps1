# Phase 48.D - MODOJA-K: error-modulated Oja tilt of the armB kernel projection (TF probe)
#
# 48.B/48.C closed the "free" directions (static map ceiling; fixed-random gating damages). 48.D
# learns the projection: the fixed Omega inside cos(GAMMA*Omega*L0norm) of armB becomes an
# error-tilted P (per-row Oja, lr modulated by the surprise of a frozen trigram). Same dims as armB
# -> zero capacity confound; the ONLY difference between probes is HOW P is obtained.
#   MODOJA-K : P tilted by error-modulated lr (m_t = trigram surprise, mean 1). The candidate.
#   PARITY   : m_t == 1 (unsupervised tilt, mean-lr matched). MODOJA-K ~= PARITY => error added nothing.
#   SHUF-MOD : m_t permuted in time (same surprise multiset, wrong positions). Leak guard.
#   armB     : P = Omega frozen, no plasticity (reproduces 2.2023/2.1839/2.1980).
#
# PRE-REGISTERED (before the run): anchor==BASE; MODOJA-K beats armB >=0.015 on val1&2&3; MODOJA-K
# beats PARITY >=0.008 on val1&2&3; SHUF-MOD does not beat armB by >0.005; P_DIVERSITY healthy
# (eff_rank not collapsed, mean|cos| not ->1). PASS -> MODOJA-K to 48.D.A (DAgger closed-loop,
# frozen harness). TF != generative (44-47). The script verdict is ADVISORY; the user reads the table.
#
# eta0: default 1e-7, validated at the FULL 1M-step count via --pdiag (plasticity-only diversity):
#   1e-7 -> eff_rank ~33/64, drift_cos ~0.974 (P tilts ~13deg, kernel stays diverse). DEFAULT.
#   2e-7 -> eff_rank ~10 (edge: more tilt, kernel degraded) - try with -Eta0 2e-7 if 1e-7 is flat.
#   3e-7+ -> collapse to rank ~1 (kernel destroyed). The smoke/run health-check rejects these.
#
# Memory: base+l0n cached once, the 4 band sets recomputed into ONE reused buffer -> peak ~9 GB
# (not ~20 GB: storing all 4 band sets at once OOM'd).
#
# Run:  .\benchmarks\phase38-42\phase48_d.ps1            (full, ~9 GB RAM, eta0=1e-7)
#       .\benchmarks\phase38-42\phase48_d.ps1 -Eta0 2e-7 (edge variant: more tilt)
#       .\benchmarks\phase38-42\phase48_d.ps1 -Smoke     (tiny: pipeline + anchor + P-alive/diverse)

param([switch]$Smoke,[double]$Eta0 = 0)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase48d'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }
$D1_W = $WDIR + '\phase44f_F0.bin'
if (-not (Test-Path $D1_W)) { Write-Error "Missing weight: $D1_W"; exit 1 }
$WP = $WDIR + '\phase48d'

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')
$GAIN_MIN   = 0.015     # MODOJA-K must beat armB by this on all windows
$PAR_MIN    = 0.008     # MODOJA-K must beat PARITY by this on all windows (error-signal-specific)
$CTRL_TOL   = 0.005     # SHUF-MOD may not beat armB by more than this
$ANCHOR_TOL = 0.005
$EFFR_MIN   = 8.0       # P collapse floor (eff_rank; random init ~40+, max 64)
$COS_MAX    = 0.60      # P collapse ceiling (mean|cos| between rows; random init ~0.10)

Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase48_d.c' @SRC_CORE -o "$BINDIR\phase48_d.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase48_d'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

$TRAINERS = @('phase48_0_taps','phase48_0_expand','phase48a_armb','phase48a_fix','phase48_b','phase48_c','phase48_d')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting.'; exit 1 }

$extra = @()
if ($Smoke) { $extra += @('--len','20000','--epochs','1'); Write-Host '(SMOKE: len 20000, 1 epoch - numbers meaningless; validates pipeline/anchor/P-alive)' -ForegroundColor DarkYellow }
if ($Eta0 -gt 0) { $extra += @('--eta0',("{0}" -f $Eta0)); Write-Host ("(eta0 override = {0})" -f $Eta0) -ForegroundColor DarkYellow }
$OUT = $RDIR + '\phase48_d.txt'
Write-Host ''
Write-Host '=== Phase 48.D: 5 probes (MODOJA-K vs armB/PARITY/SHUF-MOD) (~20 GB RAM) ===' -ForegroundColor Yellow
& "$BINDIR\phase48_d.exe" $TS_DATA $D1_W $WP @extra 2>&1 | Tee-Object $OUT
if ($LASTEXITCODE -ne 0) { Write-Error '48.D probe failed'; exit 1 }

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
$DIV = @{}
foreach ($line in Select-String '^P_DIVERSITY ' $OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
    if ($kv['tag']) { $DIV[$kv['tag']] = $kv }
}
foreach ($need in @('frozenD1','armB','MODOJA-K','PARITY','SHUF-MOD')) {
    if (-not $PROBE.ContainsKey($need)) { Write-Error "Probe output incomplete: missing $need"; exit 1 }
}

Write-Host ''
Write-Host '=== P_DIVERSITY (P must NOT collapse: eff_rank high, mean|cos| low) ===' -ForegroundColor Yellow
Write-Host ('  {0,-10} {1,9} {2,11} {3,10} {4,10}' -f 'tag','eff_rank','mean_abscos','max_abscos','drift_cos')
$div_ok = $true
foreach ($t in @('MODOJA-K','PARITY')) {
    $d = $DIV[$t]
    if ($d) {
        Write-Host ('  {0,-10} {1,9} {2,11} {3,10} {4,10}' -f $t,$d['eff_rank'],$d['mean_abscos'],$d['max_abscos'],$d['drift_cos'])
        if (([double]$d['eff_rank'] -lt $EFFR_MIN) -or ([double]$d['mean_abscos'] -gt $COS_MAX)) { $div_ok = $false }
    }
}
if (-not $div_ok) {
    Write-Host ('  -> P COLLASSATA (eff_rank < {0} o mean|cos| > {1}): Oja convergente a poche PC.' -f $EFFR_MIN,$COS_MAX) -ForegroundColor Red
    Write-Host '     Abbassa --eta0 (es. -Eta0 3e-5) o riduci i passi, poi ripeti. (Tabella sotto non fidata.)' -ForegroundColor Red
} else {
    Write-Host '  -> P viva (tilt-da-random, nessun collasso).' -ForegroundColor Green
}

Write-Host ''
Write-Host '=== Probe table ===' -ForegroundColor Yellow
Write-Host ('  {0,-9} {1,5} {2,8} {3,8} {4,8} {5,7}' -f 'probe','dim','val1','val2','val3','rms1')
foreach ($n in @('frozenD1','armB','MODOJA-K','PARITY','SHUF-MOD')) {
    $p = $PROBE[$n]
    Write-Host ('  {0,-9} {1,5} {2,8} {3,8} {4,8} {5,7}' -f $n,$p['dim'],$p['val1'],$p['val2'],$p['val3'],$p['rms1'])
}
Write-Host ''
Write-Host '=== Gains (positive = better than the reference) ===' -ForegroundColor Yellow
Write-Host ('  {0,-24} {1,8} {2,8} {3,8}' -f 'comparison','v1','v2','v3')
function GainRow($label,$ref,$x) {
    $g=@(); foreach ($wi in 1..3) { $g += [math]::Round([double]$PROBE[$ref]["val$wi"] - [double]$PROBE[$x]["val$wi"],4) }
    Write-Host ('  {0,-24} {1,8} {2,8} {3,8}' -f $label,$g[0],$g[1],$g[2]); return $g
}
$g_mk_armb  = GainRow 'MODOJA-K - armB'    'armB'   'MODOJA-K'
$g_par_armb = GainRow 'PARITY   - armB'    'armB'   'PARITY'
$g_shf_armb = GainRow 'SHUF-MOD - armB'    'armB'   'SHUF-MOD'
$g_mk_par   = GainRow 'MODOJA-K - PARITY'  'PARITY' 'MODOJA-K'

Write-Host ''
Write-Host '=== Verdetto 48.D (criterio pre-registrato; ADVISORY - leggi la tabella) ===' -ForegroundColor Yellow
$anchor_ok = $true
foreach ($wi in 1..3) { if ([math]::Abs([double]$PROBE['frozenD1']["val$wi"] - [double]$BASE["val$wi"]['d1']) -gt $ANCHOR_TOL) { $anchor_ok=$false } }
Write-Host ('  anchor: {0}' -f ($(if($anchor_ok){'OK'}else{'FAIL - non interpretare nulla'}))) -ForegroundColor $(if($anchor_ok){'Green'}else{'Red'})
$mk_armb_ok = ($g_mk_armb[0] -ge $GAIN_MIN) -and ($g_mk_armb[1] -ge $GAIN_MIN) -and ($g_mk_armb[2] -ge $GAIN_MIN)
$mk_par_ok  = ($g_mk_par[0]  -ge $PAR_MIN)  -and ($g_mk_par[1]  -ge $PAR_MIN)  -and ($g_mk_par[2]  -ge $PAR_MIN)
$shf_ok     = ($g_shf_armb[0] -le $CTRL_TOL) -and ($g_shf_armb[1] -le $CTRL_TOL) -and ($g_shf_armb[2] -le $CTRL_TOL)
Write-Host ('  MODOJA-K > armB   >= {0}:  {1}   gains=[{2}]' -f $GAIN_MIN,($(if($mk_armb_ok){'OK'}else{'no'})),($g_mk_armb -join ',')) -ForegroundColor $(if($mk_armb_ok){'Green'}else{'Gray'})
Write-Host ('  MODOJA-K > PARITY >= {0}:  {1}   gains=[{2}]  (il SEGNALE d''errore, non il tilt)' -f $PAR_MIN,($(if($mk_par_ok){'OK'}else{'no'})),($g_mk_par -join ',')) -ForegroundColor $(if($mk_par_ok){'Green'}else{'Gray'})
Write-Host ('  SHUF-MOD clean (<= {0}): {1}   gains=[{2}]' -f $CTRL_TOL,($(if($shf_ok){'OK'}else{'LEAK'})),($g_shf_armb -join ',')) -ForegroundColor $(if($shf_ok){'Green'}else{'Red'})
$pass = $anchor_ok -and $div_ok -and $mk_armb_ok -and $mk_par_ok -and $shf_ok
Write-Host ''
if ($Smoke) {
    Write-Host '  (SMOKE: verdetto non significativo - solo pipeline/anchor/P-viva. Il test vero e 1M/6ep.)' -ForegroundColor DarkYellow
} elseif ($pass) {
    Write-Host '  -> MODOJA-K PASS: il segnale d''errore sul substrato (kernel-tilt) compra capacita reale oltre armB.' -ForegroundColor Green
    Write-Host '     (Checkpoint salvato magic 0x5345454A con P appresa.) -> 48.D.A: DAgger closed-loop sotto' -ForegroundColor Green
    Write-Host '     harness 47 congelato (gate v2 + repliche + lettura umana), col determinism MD5 come HARD GATE.' -ForegroundColor Green
    Write-Host '     TF != generativo: nessuna celebrazione prima del closed-loop.' -ForegroundColor Green
} else {
    Write-Host '  -> MODOJA-K non passa il criterio pre-registrato. Letture:' -ForegroundColor Yellow
    Write-Host '     MODOJA-K~=PARITY = contava il tilt unsupervised, non il SEGNALE d''errore (la leva non e qui);' -ForegroundColor Yellow
    Write-Host '     MODOJA-K~=armB = il tilt del kernel via plasticita locale non aggiunge oltre la Omega fissa;' -ForegroundColor Yellow
    Write-Host '     SHUF-MOD aiuta = leak (il guadagno non e nell''allineamento temporale del modulatore).' -ForegroundColor Yellow
    Write-Host '     Se piatto: errore-su-substrato-via-kernel non e la leva -> ramo FORCE/RLS sul readout.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
