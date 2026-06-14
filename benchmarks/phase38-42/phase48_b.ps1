# Phase 48.B - substrate scaling law: BILIN / WAVE32 / MULTIBW on the frozen armB baseline
#
# armB = Random Fourier Features (kernel machine); more D_EXP only approximates the same
# kernel. The lever is feature CLASSES. armB sees similarities, not relations (= products).
# Three TF-only arms stacked on the frozen armB feature, each vs armB-alone:
#   BILIN   : fast_EMA . slow_EMA + slow . cur (multiplicative gating). Controls: shuffled
#             PAIRING (BILIN_sp) + shuffled-time (BILIN_st).
#   WAVE32  : integrate all 32 wave dims (21 are discarded today). Control: shuffled-time.
#   MULTIBW : cos at gamma 0.0625 + 1.0 added to armB's 0.25. Control: shuffled-time.
#
# PRE-REGISTERED (before the run): an arm earns the 48.B.A DAgger run iff it beats armB-alone
# by >= 0.015 on ALL of val1/2/3, controls don't beat armB by > 0.005, anchor exact.
# Winners are stackable. TF != generative (44-47): closed-loop gate v2 + replicas + human read.
#
# Run:  .\benchmarks\phase38-42\phase48_b.ps1            (full, ~18 GB RAM)
#       .\benchmarks\phase38-42\phase48_b.ps1 -Smoke     (tiny, pipeline check)

param([switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase48b'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }
$D1_W = $WDIR + '\phase44f_F0.bin'
if (-not (Test-Path $D1_W)) { Write-Error "Missing weight: $D1_W"; exit 1 }
$WP = $WDIR + '\phase48b'

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')
$GAIN_MIN = 0.015     # arm must beat armB by this on all windows
$CTRL_TOL = 0.005     # shuffled controls may not beat armB by more than this
$ANCHOR_TOL = 0.005

Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase48_b.c' @SRC_CORE -o "$BINDIR\phase48_b.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase48_b'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

$TRAINERS = @('phase48_0_taps','phase48_0_expand','phase48a_armb','phase48a_fix','phase48_b')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting.'; exit 1 }

$extra = @(); if ($Smoke) { $extra = @('--len','20000','--epochs','1'); Write-Host '(SMOKE: len 20000, 1 epoch - numbers meaningless)' -ForegroundColor DarkYellow }
$OUT = $RDIR + '\phase48_b.txt'
Write-Host ''
Write-Host '=== Phase 48.B: 8 probes on the armB baseline (~18 GB RAM) ===' -ForegroundColor Yellow
& "$BINDIR\phase48_b.exe" $TS_DATA $D1_W $WP @extra 2>&1 | Tee-Object $OUT
if ($LASTEXITCODE -ne 0) { Write-Error '48.B probe failed'; exit 1 }

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
foreach ($need in @('frozenD1','armB','BILIN','BILIN_sp','BILIN_st','WAVE32','WAVE32_st','MULTIBW','MULTIBW_st')) {
    if (-not $PROBE.ContainsKey($need)) { Write-Error "Probe output incomplete: missing $need"; exit 1 }
}

Write-Host ''
Write-Host '=== Probe table ===' -ForegroundColor Yellow
Write-Host ('  {0,-11} {1,5} {2,8} {3,8} {4,8}' -f 'probe','dim','val1','val2','val3')
foreach ($n in @('frozenD1','armB','BILIN','BILIN_sp','BILIN_st','WAVE32','WAVE32_st','MULTIBW','MULTIBW_st')) {
    $p = $PROBE[$n]
    Write-Host ('  {0,-11} {1,5} {2,8} {3,8} {4,8}' -f $n,$p['dim'],$p['val1'],$p['val2'],$p['val3'])
}
Write-Host ''
Write-Host '=== Gain vs armB (armB - X; positive = X better) ===' -ForegroundColor Yellow
Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f 'probe','g_v1','g_v2','g_v3')
foreach ($n in @('BILIN','BILIN_sp','BILIN_st','WAVE32','WAVE32_st','MULTIBW','MULTIBW_st')) {
    $g=@(); foreach ($wi in 1..3) { $g += [math]::Round([double]$PROBE['armB']["val$wi"] - [double]$PROBE[$n]["val$wi"],4) }
    Write-Host ('  {0,-11} {1,9} {2,9} {3,9}' -f $n,$g[0],$g[1],$g[2])
}

Write-Host ''
Write-Host '=== Verdetto 48.B (criterio pre-registrato) ===' -ForegroundColor Yellow
# anchor
$anchor_ok = $true
foreach ($wi in 1..3) { if ([math]::Abs([double]$PROBE['frozenD1']["val$wi"] - [double]$BASE["val$wi"]['d1']) -gt $ANCHOR_TOL) { $anchor_ok=$false } }
Write-Host ('  anchor: {0}' -f ($(if($anchor_ok){'OK'}else{'FAIL (base feature drift - non interpretare nulla)'}))) -ForegroundColor $(if($anchor_ok){'Green'}else{'Red'})

$arms = @(
    @{ name='BILIN';   ctrls=@('BILIN_sp','BILIN_st') },
    @{ name='WAVE32';  ctrls=@('WAVE32_st') },
    @{ name='MULTIBW'; ctrls=@('MULTIBW_st') }
)
$winners = @()
foreach ($arm in $arms) {
    $gain_ok = $true; $gains=@()
    foreach ($wi in 1..3) { $gn = [double]$PROBE['armB']["val$wi"] - [double]$PROBE[$arm.name]["val$wi"]; $gains += [math]::Round($gn,4); if ($gn -lt $GAIN_MIN) { $gain_ok=$false } }
    $ctrl_ok = $true; $ctrl_detail=@()
    foreach ($cn in $arm.ctrls) {
        foreach ($wi in 1..3) { $cg = [double]$PROBE['armB']["val$wi"] - [double]$PROBE[$cn]["val$wi"]; if ($cg -gt $CTRL_TOL) { $ctrl_ok=$false } }
        $cw = @(); foreach ($wi in 1..3) { $cw += [math]::Round([double]$PROBE['armB']["val$wi"] - [double]$PROBE[$cn]["val$wi"],4) }
        $ctrl_detail += ('{0}=[{1}]' -f $cn,($cw -join ','))
    }
    $pass = $anchor_ok -and $gain_ok -and $ctrl_ok
    if ($pass) { $winners += $arm.name }
    $col = if ($pass) {'Green'} else {'Gray'}
    Write-Host ('  {0,-9} gain=[{1}] {2}  ctrl({3}) {4}  -> {5}' -f `
        $arm.name,($gains -join ','),($(if($gain_ok){'OK'}else{'<0.015'})),($ctrl_detail -join ' '),($(if($ctrl_ok){'clean'}else{'LEAK'})),($(if($pass){'WINNER'}else{'no'}))) -ForegroundColor $col
}
Write-Host ''
if ($Smoke) {
    Write-Host '  (SMOKE: verdetto non significativo - solo validazione pipeline.)' -ForegroundColor DarkYellow
} elseif ($winners.Count -gt 0) {
    Write-Host ('  -> WINNER(S): {0}. Classi di feature reali oltre il kernel armB (stackabili).' -f ($winners -join ', ')) -ForegroundColor Green
    Write-Host '     -> 48.B.A: stack dei vincitori sotto harness 47 congelato (DAgger + gate v2 + repliche + lettura).' -ForegroundColor Green
    Write-Host '     TF != generativo: nessuna celebrazione prima del closed-loop (legge 44-47).' -ForegroundColor Green
} else {
    Write-Host '  -> Nessun braccio batte armB di 0.015 su tutte le finestre con controlli puliti. Letture:' -ForegroundColor Yellow
    Write-Host '     BILIN piatto = l interazione moltiplicativa elementwise non basta (servono prodotti cross-canale?);' -ForegroundColor Yellow
    Write-Host '     WAVE32 piatto = le 21 dim d onda scartate sono ridondanti col gia integrato;' -ForegroundColor Yellow
    Write-Host '     MULTIBW piatto = una sola lengthscale basta, il kernel e gia ben tarato.' -ForegroundColor Yellow
    Write-Host '     Se tutto piatto: armB e il ceiling di questa famiglia -> MODOJA (substrato che impara COSA codificare).' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
