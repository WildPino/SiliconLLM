# Phase 44.G - Closed-Loop L2 Telemetry (diagnostic, no fix)
#
# 44.F negative: capped-training did NOT close the hole. F1 stays topBi=10 at T=0.55
# (= inference-only) and worsens nameWst (10.6->34.7); F3 reaches topBi=9 but still
# fails and is dirty at T=0.65. Control drift: F0/D1_cap20inf do not cleanly
# reproduce 44.E's stable T=0.65 behaviour (BPB ok, internal determinism ok, but
# generation differs - the closed loop is FP-sensitive; split-dot training in 44.F
# diverged the weights from 44.E by rounding). So 44.F = negative WITH control drift.
#
# 44.G does NOT fix. It instruments the real generator (--telemetry: same code path)
# and logs per step: L2/SEE raw ratio, gamma/fire, predicted byte, L2 state norm and
# delta, boundary/update event, how much L2 changes the winning logit, and a
# byte-cycle loop detector. Then it splits steps into LOOP-ACTIVE vs CLEAN.
#
# The single question: when the loop starts, is the cap still letting too much L2
# through (ratio high, fired, L2 still flips the byte = A: more conditional gating),
# or does the loop form even with cap active and LOW ratio (= B: L2 memory/state
# dynamics dominate, fix the memory not the readout)?
#
# Configs audited (weights from the 44.F run): D1_cap20inf, F1, F3, F0.
#
# Run:  .\benchmarks\phase38-42\phase44g_telemetry.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44g'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$TEMP    = '0.55'           # the failing temperature
$RNG     = 12345
$NSEEDS  = 6                # worst-seed sweep
$GEN_LEN = 2000
$WARMUP  = 5000
$LOOP_ON   = 12            # loop_run >= this  -> loop-active step
$CLEAN_MAX = 2             # loop_run <= this  -> clean step

# label -> @{ file=...; cap=... }  (cap known from how 44.F wrote them)
$configs = @(
    @{ label='F0';          sfx='_F0.bin';          cap=0.000 },
    @{ label='D1_cap20inf'; sfx='_D1_cap20inf.bin'; cap=0.020 },
    @{ label='F1';          sfx='_F1.bin';          cap=0.020 },
    @{ label='F3';          sfx='_F3.bin';          cap=-0.020 }
)

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING (generator with --telemetry) ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# weights present?
foreach ($c in $configs) {
    $c.file = $WDIR + '\phase44f' + $c.sfx
    if (-not (Test-Path $c.file)) { Write-Error ("Missing 44.F weights: " + $c.file + " (run phase44f_captrain.ps1 first)"); exit 1 }
}

# -------- Seeds ---------------------------------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

# -------- Generate + telemetry ------------------------------------------------
Write-Host ''
Write-Host ('=== Telemetry: T={0}, rng={1}, {2} seeds/config ===' -f $TEMP,$RNG,$NSEEDS) -ForegroundColor Yellow
foreach ($c in $configs) {
    $si = 0
    foreach ($off in $SEEDS) {
        $si++
        $tf = $RDIR + '\gen_' + $c.label + '_s' + $si + '.txt'
        $tel= $RDIR + '\tel_' + $c.label + '_s' + $si + '.tsv'
        $sf = $RDIR + '\gen_' + $c.label + '_s' + $si + '_stats.txt'
        $argline = ('"{0}" "{1}" --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6} --telemetry "{7}"' -f `
                    $TS_DATA,$c.file,$GEN_LEN,$WARMUP,$TEMP,$off,$RNG,$tel)
        Write-Host ('  {0,-14} s{1} ...' -f $c.label,$si) -NoNewline
        $p = Start-Process -FilePath "$BINDIR\phase44_generator.exe" -ArgumentList $argline `
                -RedirectStandardOutput $tf -RedirectStandardError $sf -NoNewWindow -PassThru -Wait
        if ($p.ExitCode -ne 0) { Write-Host ' FAILED' -ForegroundColor Red } else { Write-Host ' OK' -ForegroundColor Green }
    }
}

# -------- Analyse -------------------------------------------------------------
# columns: 0 step 1 samp 2 predF 3 predN 4 flip 5 ratio 6 gamma 7 fired 8 l2norm
#          9 l2delta 10 gate 11 Lwin 12 loop_run 13 loop_p
function New-Acc { [PSCustomObject]@{ n=0; ratio=0.0; gamma=0.0; fired=0; flip=0; l2n=0.0; l2d=0.0; lwin=0.0; gate=0 } }
function Add-Row($a,$f) {
    $a.n++; $a.ratio+=[double]$f[5]; $a.gamma+=[double]$f[6]; $a.fired+=[int]$f[7]; $a.flip+=[int]$f[4]
    $a.l2n+=[double]$f[8]; $a.l2d+=[double]$f[9]; $a.lwin+=[math]::Abs([double]$f[11]); $a.gate+=[int]$f[10]
}
function Mean($s,$n){ if($n -le 0){return 0.0}; [math]::Round($s/$n,4) }
function Pct($s,$n){ if($n -le 0){return 0.0}; [math]::Round(100.0*$s/$n,1) }

$summary = @{}
foreach ($c in $configs) {
    $clean = New-Acc; $loop = New-Acc; $tot = 0
    $si = 0
    foreach ($off in $SEEDS) {
        $si++
        $tel = $RDIR + '\tel_' + $c.label + '_s' + $si + '.tsv'
        if (-not (Test-Path $tel)) { continue }
        foreach ($line in [System.IO.File]::ReadLines($tel)) {
            if ($line.Length -eq 0 -or $line[0] -eq '#' -or $line[0] -eq 's') { continue }
            $f = $line -split "`t"
            if ($f.Count -lt 14) { continue }
            $tot++
            $lr = [int]$f[12]
            if ($lr -ge $LOOP_ON) { Add-Row $loop $f }
            elseif ($lr -le $CLEAN_MAX) { Add-Row $clean $f }
        }
    }
    $summary[$c.label] = @{ clean=$clean; loop=$loop; tot=$tot; cap=$c.cap }
}

Write-Host ''
Write-Host '======== Per-step telemetry: CLEAN vs LOOP-ACTIVE ========' -ForegroundColor Cyan
foreach ($c in $configs) {
    $s = $summary[$c.label]; $cl=$s.clean; $lp=$s.loop
    $loopPct = Pct ($lp.n) ($s.tot)
    Write-Host ''
    Write-Host ('  {0}   cap={1:+0.000;-0.000;0.000}   loop-active {2}% of steps   (clean n={3}, loop n={4})' -f `
        $c.label,$c.cap,$loopPct,$cl.n,$lp.n) -ForegroundColor Yellow
    Write-Host ('    {0,-10} {1,9} {2,9} {3,8} {4,8} {5,9} {6,9} {7,9} {8,8}' -f 'bucket','ratio','gamma','fired%','flip%','l2norm','l2delta','|Lwin|','gate%')
    Write-Host ('    ' + '-'*80)
    if ($cl.n -gt 0) {
        Write-Host ('    {0,-10} {1,9} {2,9} {3,8} {4,8} {5,9} {6,9} {7,9} {8,8}' -f 'clean',`
            (Mean $cl.ratio $cl.n),(Mean $cl.gamma $cl.n),(Pct $cl.fired $cl.n),(Pct $cl.flip $cl.n),`
            (Mean $cl.l2n $cl.n),(Mean $cl.l2d $cl.n),(Mean $cl.lwin $cl.n),(Pct $cl.gate $cl.n))
    }
    if ($lp.n -gt 0) {
        Write-Host ('    {0,-10} {1,9} {2,9} {3,8} {4,8} {5,9} {6,9} {7,9} {8,8}' -f 'loop',`
            (Mean $lp.ratio $lp.n),(Mean $lp.gamma $lp.n),(Pct $lp.fired $lp.n),(Pct $lp.flip $lp.n),`
            (Mean $lp.l2n $lp.n),(Mean $lp.l2d $lp.n),(Mean $lp.lwin $lp.n),(Pct $lp.gate $lp.n)) -ForegroundColor Magenta
    } else {
        Write-Host '    loop       [no loop-active steps detected]' -ForegroundColor DarkGray
    }
}

# -------- Diagnostic read (A vs B) per capped config --------------------------
Write-Host ''
Write-Host '======== Diagnostic read: A (cap leaks) vs B (memory dynamics) ========' -ForegroundColor Cyan
foreach ($c in $configs) {
    if ($c.label -eq 'F0') { continue }   # F0 has no cap; reference only
    $s = $summary[$c.label]; $lp=$s.loop; $cl=$s.clean
    if ($lp.n -le 0) { Write-Host ('  {0}: nessuno step loop-active (a questo seed/temp non si incastra)' -f $c.label) -ForegroundColor Gray; continue }
    $capv = [math]::Abs($c.cap)
    $ratioLoop = (Mean $lp.ratio $lp.n); $firedLoop = (Pct $lp.fired $lp.n); $flipLoop = (Pct $lp.flip $lp.n)
    $l2nLoop = (Mean $lp.l2n $lp.n); $l2nClean = (Mean $cl.l2n $cl.n)
    $l2dLoop = (Mean $lp.l2d $lp.n); $l2dClean = (Mean $cl.l2d $cl.n)
    $msg = ''
    if (($ratioLoop -gt ($capv*1.10)) -and ($firedLoop -ge 50.0)) {
        $msg = ("A: cap ATTIVO ma INSUFFICIENTE - ratio {0} > cap {1}, fired {2}%, L2 flippa il byte {3}% -> prossimo = gating L2 piu condizionale" -f $ratioLoop,$capv,$firedLoop,$flipLoop)
        $col = 'Yellow'
    } elseif ($firedLoop -lt 25.0 -and ($flipLoop -ge 25.0 -or ($l2nClean -gt 0 -and $l2nLoop -gt ($l2nClean*1.25)))) {
        $msg = ("B: MEMORIA/STATE - cap quasi non scatta (fired {0}%, ratio {1}<=cap {2}) ma L2 domina: flip {3}%, l2norm {4} vs clean {5}, l2delta {6} vs {7} -> prossimo = homeostasis/update di L2, non readout" -f $firedLoop,$ratioLoop,$capv,$flipLoop,$l2nLoop,$l2nClean,$l2dLoop,$l2dClean)
        $col = 'Green'
    } elseif ($flipLoop -lt 15.0) {
        $msg = ("C: L2 NON decide il loop (flip {0}%): l attractor e SEE-side, non L2 -> rivedere il substrate, non L2" -f $flipLoop)
        $col = 'DarkCyan'
    } else {
        $msg = ("misto: ratioLoop {0} vs cap {1}, fired {2}%, flip {3}%, l2norm {4}/{5} - leggere i TSV grezzi" -f $ratioLoop,$capv,$firedLoop,$flipLoop,$l2nLoop,$l2nClean)
        $col = 'Gray'
    }
    Write-Host ('  {0}: {1}' -f $c.label,$msg) -ForegroundColor $col
}

Write-Host ''
Write-Host '  Note: F0 (no cap) e referenza di dinamica grezza. Confronta loop-active%,'
Write-Host '  flip% e l2norm tra F0 e le capped per vedere quanto il cap cambia davvero la'
Write-Host '  dinamica del loop (control drift 44.E/F: il loop e FP-sensibile).'
Write-Host ''
Write-Host '  Domanda 44.G: quando parte il loop, il cap lascia passare troppo L2 (A) o L2'
Write-Host '  domina via memoria/state anche sotto cap (B)? Le righe sopra rispondono per config.'
Write-Host ''
Write-Host ('  TSV grezzi per ispezione manuale: ' + $RDIR)
