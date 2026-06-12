# Phase 47.H - Validazione estesa di P_r7 (PRIMARIA) + coda H48 (SECONDARIA, tagliabile)
#
# 47.G: P_r7 (phase47g_P_h32_r7.bin) = PRIMO dual-temp full-gate PASS della storia
# (BPB 2.2497, topBi 6/7, altLp 2/2, selfBPB 1.55/1.11). Margini sottili (altLp al bar,
# BPB +0.0046, topBi 1-2 unita'), vicini falliscono (r8 fail 0.55, r9 peggio) -> rischio
# coda fortunata su worst-case-di-32. Si promuove il CHECKPOINT, non la ricetta.
#
# Componente 1 - REPLICA GATE (il cuore, regola pre-registrata):
#   Replica = una valutazione full-gate completa (32 sample/temp, ENTRAMBE le temp).
#   R1 = rng 24680+13579 x 16 seed standard
#   R2 = rng 98765+55555 x 16 seed standard
#   R3 = rng 12345+67890 x 16 seed HELD-OUT (griglia +0.25 del passo, scostata da train/val)
#   R4 = rng 12345+67890 x 16 seed HELD-OUT (griglia +0.75)
#   Decisione: 4/4 PASS -> promozione confermata. UNA replica fallisce UN componente di
#   una unita' -> MARGINALE (decide l'utente). >=2 repliche falliscono -> NON promuovibile.
# Componente 2 - STRESS ORIZZONTE: 8 seed x 2 temp a lunghezza 4x, metriche worst-window
#   su finestre di lunghezza standard (bar comparabili). Report (attrattori lenti).
# Componente 3 - CONTINUITA' TEMPERATURA: 8 seed a T0.60/T0.70, solo report (no cliff).
# Componente 4 - LETTURA UMANA: 10 sample (5/temp) copiati in results\phase47h\human\.
# Componente 5 - CODA H48 (-SkipH48Tail per tagliare): continuazione r6->r9 from-scratch,
#   prefix-MD5 r1/r5 vs checkpoint 47.G, qualify bi<=7 entrambe + selfBPB55>=0.8 +
#   BPB<=2.2543, full sui qualificati. NON entra nella decisione su P_r7.
#
# Da non fare: inseguire BPB di A_r9, P a r10+, riaprire H64, toccare gate/bar, commit.
#
# Run:  .\benchmarks\phase38-42\phase47h.ps1
#       .\benchmarks\phase38-42\phase47h.ps1 -SkipH48Tail
#       .\benchmarks\phase38-42\phase47h.ps1 -SkipTrain      # riusa train output coda H48

param([switch]$SkipTrain,[switch]$SkipH48Tail)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase47h'
$HDIR    = $RDIR + '\human'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
foreach ($d in @($RDIR,$HDIR)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$PR7_W   = $WDIR + '\phase47g_P_h32_r7.bin'    # IL candidato (checkpoint congelato)
$WP      = $WDIR + '\phase47h'
$G_H48R1 = $WDIR + '\phase47g_H48_r1.bin'
$G_H48R5 = $WDIR + '\phase47g_H48_r5.bin'
$OLD_G   = $ROOT + '\results\phase47g\phase47g_train.txt'
$C2A_BPB = 2.2593
foreach ($w in @($C2A_W,$D1_W,$PR7_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }
if (-not (Test-Path $OLD_G)) { Write-Error "Missing 47.G train output: $OLD_G"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$RNGS_ORIG = @(12345, 67890)
$RNGS_NEW_A = @(24680, 13579)     # replica R1
$RNGS_NEW_B = @(98765, 55555)     # replica R2
$TEMPS   = @('0.65', '0.55')
$GEN_LEN = 2000
$HORIZ_LEN = 8000                  # 4x
$HORIZ_SEEDS = 8
$CONT_SEEDS = 8
$CONT_TEMPS = @('0.60','0.70')
$WARMUP  = 5000
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0
$PROMO_BPB = $C2A_BPB - 0.005
$G_NAME = 20.0; $G_RUN = 5.0; $G_BI = 8.0; $G_AL = 2.0
$MINI_BI = 7.0
$THROTTLE = 12
$TRAIN_N  = 1000000                # N usato dai trainer 47.x (finestre train/val da escludere)

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47h_h48tail.c' @SRC_CORE -o "$BINDIR\phase47h_h48tail.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47h_h48tail'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47_generator.c' @SRC_CORE -o "$BINDIR\phase47_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- P_r7 valBPB (fisso, dal probe TF di 47.G) ------------------------------
function Parse-KV([string]$file,[string]$prefix) {
    $out = @{}
    foreach ($line in Select-String ('^'+$prefix+' ') $file) {
        $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
        $key = if ($prefix -eq 'BASE') { $kv['win'] } else { $kv['name'] }
        $out[$key] = $kv
    }
    return $out
}
function AvgVal($p) { return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }
$GP = Parse-KV $OLD_G 'PROBE'
if (-not $GP.ContainsKey('P_r7')) { Write-Error '47.G P_r7 probe missing'; exit 1 }
$PR7_BPB = AvgVal $GP['P_r7']
Write-Host ('P_r7 TF BPB = {0:F4} (bar {1:F4}, margine {2:F4})' -f $PR7_BPB,$PROMO_BPB,($PROMO_BPB-$PR7_BPB))
if ($PR7_BPB -gt $PROMO_BPB) { Write-Error 'P_r7 BPB sopra bar?! Train output errato.'; exit 1 }

# -------- Seed grids ------------------------------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS_STD = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS_STD += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }
# regioni usate dai trainer (train window + 3 val window, +margine warmup)
$USED_WIN = @(
    @{ lo=[long][math]::Floor($fsz/5);    hi=[long]([math]::Floor($fsz/5)+$TRAIN_N+$margin) },
    @{ lo=[long][math]::Floor($fsz/2);    hi=[long]([math]::Floor($fsz/2)+$TRAIN_N+$margin) },
    @{ lo=[long][math]::Floor(0.65*$fsz); hi=[long]([math]::Floor(0.65*$fsz)+$TRAIN_N+$margin) },
    @{ lo=[long][math]::Floor(0.80*$fsz); hi=[long]([math]::Floor(0.80*$fsz)+$TRAIN_N+$margin) }
)
function New-HeldOutSeeds([double]$frac) {
    # griglia sfalsata di +frac del passo, scostamento DETERMINISTICO fuori dalle
    # regioni usate (train/val window dei trainer); wrap se sfora il file
    $hs = @()
    for ($i = 0; $i -lt $NSEEDS; $i++) {
        $off = [long][math]::Floor(($i + $frac) * ($fsz - $margin) / $NSEEDS)
        $moved = $true
        while ($moved) {
            $moved = $false
            foreach ($uw in $USED_WIN) {
                if ($off -ge ($uw.lo - $margin) -and $off -lt $uw.hi) { $off = $uw.hi + 1048576; $moved = $true }
            }
            if ($off -gt ($fsz - $margin)) { $off = $off % ($fsz - $margin); $moved = $true }
        }
        $hs += [int]$off
    }
    return $hs
}
$SEEDS_HO_A = New-HeldOutSeeds 0.25
$SEEDS_HO_B = New-HeldOutSeeds 0.75

# -------- Word metrics (core su testo, riusato da finestre orizzonte) -------------
function Get-WordMetricsFromText([string]$text) {
    $clean = $text -replace '[^a-zA-Z]', ' '
    $toks = @($clean.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
    $nt = $toks.Count; if ($nt -lt 4) { return $null }
    $name_ct = 0; foreach ($w in $toks) { if ($NAME_SET.ContainsKey($w)) { $name_ct++ } }
    $nameish = [math]::Round($name_ct / $nt * 100.0, 1)
    $run = 1; $maxrun = 1
    for ($j = 1; $j -lt $nt; $j++) { if ($toks[$j] -eq $toks[$j-1]) { $run++ } else { $run = 1 }; if ($run -gt $maxrun) { $maxrun = $run } }
    $bf = @{}; for ($j = 0; $j -lt $nt-1; $j++) { $k = $toks[$j]+' '+$toks[$j+1]; $bf[$k] = ($bf[$k] -as [int]) + 1 }
    $top_bi = if ($bf.Count) { ($bf.Values | Measure-Object -Maximum).Maximum } else { 0 }
    $alt = 0; $altmax = 0
    for ($j = 0; $j -lt $nt-2; $j++) { if ($toks[$j] -eq $toks[$j+2]) { $alt++ } else { $alt = 0 }; if ($alt -gt $altmax) { $altmax = $alt } }
    return [PSCustomObject]@{ nameish=$nameish; maxrun=$maxrun; top_bi=$top_bi; altloop=$altmax }
}
function Get-WordMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    return Get-WordMetricsFromText ([System.Text.Encoding]::ASCII.GetString($bytes))
}
function Get-WorstWindowMetrics([string]$path,[int]$winBytes) {
    # metriche worst-window: finestre contigue di lunghezza standard dentro il sample lungo
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -lt $winBytes) { return Get-WordMetrics $path }
    $nwin = [int][math]::Floor($bytes.Length / $winBytes)
    $worst = $null
    for ($k = 0; $k -lt $nwin; $k++) {
        $chunk = [System.Text.Encoding]::ASCII.GetString($bytes, $k*$winBytes, $winBytes)
        $m = Get-WordMetricsFromText $chunk
        if ($null -eq $m) { continue }
        if ($null -eq $worst) { $worst = [PSCustomObject]@{ nameish=$m.nameish; maxrun=$m.maxrun; top_bi=$m.top_bi; altloop=$m.altloop } }
        else {
            if ($m.nameish -gt $worst.nameish) { $worst.nameish = $m.nameish }
            if ($m.maxrun  -gt $worst.maxrun)  { $worst.maxrun  = $m.maxrun }
            if ($m.top_bi  -gt $worst.top_bi)  { $worst.top_bi  = $m.top_bi }
            if ($m.altloop -gt $worst.altloop) { $worst.altloop = $m.altloop }
        }
    }
    return $worst
}
function Mx ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
function Av ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }

function New-GenTask($label,$exe,$wargs,$tp,$rng,$off,$si,$suffix,$genlen) {
    $lbl = $label+'_T'+$tp+'_r'+$rng+'_s'+$si+$suffix
    $tf  = $RDIR+'\gen_'+$lbl+'.txt'
    $sf  = $RDIR+'\gen_'+$lbl+'_stats.txt'
    $argline = ('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                $TS_DATA,$wargs,$genlen,$WARMUP,$tp,$off,$rng)
    return [PSCustomObject]@{ label=$label; tp=$tp; rng=$rng; off=$off; si=$si;
                              exe=($BINDIR+'\'+$exe); tf=$tf; sf=$sf; argline=$argline }
}
function Invoke-Throttled($tasks,$throttle) {
    $queue = New-Object System.Collections.Queue
    foreach ($t in $tasks) { [void]$queue.Enqueue($t) }
    $running = New-Object System.Collections.ArrayList
    $launched = 0; $total = $tasks.Count
    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $throttle -and $queue.Count -gt 0) {
            $t = $queue.Dequeue()
            $p = Start-Process -FilePath $t.exe -ArgumentList $t.argline `
                    -RedirectStandardOutput $t.tf -RedirectStandardError $t.sf -NoNewWindow -PassThru
            [void]$running.Add($p); $launched++
            if ($launched % 32 -eq 0 -or $launched -eq $total) {
                Write-Host ('  launched {0,4}/{1}' -f $launched,$total)
            }
        }
        Start-Sleep -Milliseconds 150
        for ($k=$running.Count-1; $k -ge 0; $k--) { if ($running[$k].HasExited) { $running.RemoveAt($k) } }
    }
}
function FileMD5($p) {
    if (-not (Test-Path $p)) { return '' }
    for ($a=0; $a -lt 15; $a++) { try { return (Get-FileHash -Algorithm MD5 -Path $p -ErrorAction Stop).Hash } catch { Start-Sleep -Milliseconds 200 } }
    return ''
}
function Get-SelfBPB([string]$statsfile) {
    $bl = Select-String 'self_BPB' $statsfile | Select-Object -Last 1
    if ($bl) { return [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    return $null
}

$PR7_ARGS = ('"'+$D1_W+'" "'+$PR7_W+'"')

# -------- Repro pre-check --------------------------------------------------------
Write-Host ''
Write-Host '=== Repro pre-check: parallel vs sequential (P_r7, 4 run) ===' -ForegroundColor Yellow
$seq_tasks=@(); $par_tasks=@()
$si=0
foreach ($off in @($SEEDS_STD | Select-Object -First 2)) { $si++
    foreach ($rng in @($RNGS_NEW_A[0],$RNGS_ORIG[0])) {
        $seq_tasks += New-GenTask 'Pr7' 'phase47_generator.exe' $PR7_ARGS '0.65' $rng $off $si '_seqchk' $GEN_LEN
        $par_tasks += New-GenTask 'Pr7' 'phase47_generator.exe' $PR7_ARGS '0.65' $rng $off $si '_parchk' $GEN_LEN
    } }
Invoke-Throttled $seq_tasks 1
Invoke-Throttled $par_tasks 4
$repro_ok = $true
for ($i=0; $i -lt $seq_tasks.Count; $i++) {
    $hs = FileMD5 $seq_tasks[$i].tf; $hp = FileMD5 $par_tasks[$i].tf
    $ok = ($hs -ne '' -and $hs -eq $hp)
    if (-not $ok) { $repro_ok = $false }
    Write-Host ('  s{0} rng{1}  {2}' -f $seq_tasks[$i].si,$seq_tasks[$i].rng,($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $repro_ok) { Write-Error 'Repro pre-check FAILED. Aborting.'; exit 1 }
Write-Host '  Repro pre-check PASSED.' -ForegroundColor Green

# ============ COMPONENTE 1: REPLICA GATE =========================================
$REPLICAS = @(
    @{ name='R1_newrngA'; rngs=$RNGS_NEW_A; seeds=$SEEDS_STD },
    @{ name='R2_newrngB'; rngs=$RNGS_NEW_B; seeds=$SEEDS_STD },
    @{ name='R3_heldoutA'; rngs=$RNGS_ORIG; seeds=$SEEDS_HO_A },
    @{ name='R4_heldoutB'; rngs=$RNGS_ORIG; seeds=$SEEDS_HO_B }
)
$rep_tasks = @()
foreach ($rep in $REPLICAS) {
    foreach ($tp in $TEMPS) { foreach ($rng in $rep.rngs) { $si=0
        foreach ($off in $rep.seeds) { $si++
            $rep_tasks += New-GenTask ($rep.name) 'phase47_generator.exe' $PR7_ARGS $tp $rng $off $si '' $GEN_LEN
        } } }
}
Write-Host ''
Write-Host ('=== COMPONENTE 1: replica gate, {0} runs (4 repliche x 32 sample x 2 temp) ===' -f $rep_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $rep_tasks $THROTTLE

$rep_agg = @{}
foreach ($rep in $REPLICAS) { $rep_agg[$rep.name] = @{}; foreach ($tp in $TEMPS) { $rep_agg[$rep.name][$tp] = @{ mets=@(); bpbs=@() } } }
foreach ($t in $rep_tasks) {
    if (-not (Test-Path $t.tf)) { Write-Host ('  MISSING ' + $t.tf) -ForegroundColor Red; continue }
    $m = Get-WordMetrics $t.tf
    if ($m) { $rep_agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf
    if ($null -ne $b) { $rep_agg[$t.label][$t.tp].bpbs += $b }
}
function WorstRep($lbl,$tp) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($m in @($rep_agg[$lbl][$tp].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
    foreach ($b in @($rep_agg[$lbl][$tp].bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni=(Mx $ni); mr=(Mx $mr); bi=(Mx $bi); al=(Mx $al); self=(Av $bs); n=@($rep_agg[$lbl][$tp].mets).Count }
}
# failure list per replica: componente + entita' dello sforamento
function RepFailures($lbl) {
    $fails = @()
    foreach ($tp in $TEMPS) {
        $w = WorstRep $lbl $tp
        if ($w.n -eq 0) { $fails += [PSCustomObject]@{ comp="nodata@$tp"; miss=99 }; continue }
        if ($w.bi -gt $G_BI)    { $fails += [PSCustomObject]@{ comp="topBi@$tp";  miss=($w.bi-$G_BI) } }
        if ($w.al -gt $G_AL)    { $fails += [PSCustomObject]@{ comp="altLp@$tp";  miss=($w.al-$G_AL) } }
        if ($w.mr -gt $G_RUN)   { $fails += [PSCustomObject]@{ comp="runWst@$tp"; miss=($w.mr-$G_RUN) } }
        if ($w.ni -gt $G_NAME)  { $fails += [PSCustomObject]@{ comp="nameWst@$tp"; miss=($w.ni-$G_NAME) } }
        if ($w.self -lt $BPB_LO) { $fails += [PSCustomObject]@{ comp="selfBPB_lo@$tp"; miss=($BPB_LO-$w.self) } }
        if ($w.self -gt $BPB_HI) { $fails += [PSCustomObject]@{ comp="selfBPB_hi@$tp"; miss=($w.self-$BPB_HI) } }
    }
    return ,$fails
}

Write-Host ''
Write-Host '======== REPLICA GATE P_r7 (worst-case per replica; bar: bi<=8 al<=2 run<=5 name<=20 self[0.8,2.0]) ========' -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,5} {2,6} {3,6} {4,6} {5,6} {6,7} {7,8}' -f 'replica','temp','topBi','altLp','runWst','nameW','selfBPB','verdict')
$rep_pass = 0; $rep_fail_detail = @{}
foreach ($rep in $REPLICAS) {
    $fails = RepFailures $rep.name
    $verdict = if (@($fails).Count -eq 0) { 'PASS' } else { 'FAIL' }
    if ($verdict -eq 'PASS') { $rep_pass++ }
    $rep_fail_detail[$rep.name] = $fails
    foreach ($tp in $TEMPS) {
        $w = WorstRep $rep.name $tp
        $vtxt = if ($tp -eq $TEMPS[0]) { $verdict } else { '' }
        $col = if ($verdict -eq 'PASS') {'Green'} else {'Red'}
        Write-Host ('  {0,-12} {1,5} {2,6} {3,6} {4,6} {5,6} {6,7} {7,8}' -f `
            $rep.name,$tp,('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F0}' -f $w.mr),('{0:F1}' -f $w.ni),('{0:F2}' -f $w.self),$vtxt) -ForegroundColor $col
    }
    if (@($fails).Count -gt 0) {
        Write-Host ('    fail: ' + (($fails | ForEach-Object { '{0} (+{1:F2})' -f $_.comp,$_.miss }) -join ', ')) -ForegroundColor Red
    }
}

# regola pre-registrata
$total_fail_components = 0
foreach ($rep in $REPLICAS) { $total_fail_components += @($rep_fail_detail[$rep.name]).Count }
$marginal = $false
if ($rep_pass -eq 3 -and $total_fail_components -eq 1) {
    $only = @($REPLICAS | Where-Object { @($rep_fail_detail[$_.name]).Count -eq 1 })[0]
    $f = @($rep_fail_detail[$only.name])[0]
    $unit = if ($f.comp -like 'selfBPB*') { 0.05 } elseif ($f.comp -like 'nameWst*') { 1.0 } else { 1.0 }
    if ($f.miss -le $unit) { $marginal = $true }
}

# ============ COMPONENTE 2: STRESS ORIZZONTE (4x, worst-window) ===================
$hor_tasks = @()
foreach ($tp in $TEMPS) { $si=0
    foreach ($off in @($SEEDS_STD | Select-Object -First $HORIZ_SEEDS)) { $si++
        $hor_tasks += New-GenTask 'HOR' 'phase47_generator.exe' $PR7_ARGS $tp $RNGS_ORIG[0] $off $si '_hor' $HORIZ_LEN
    } }
Write-Host ''
Write-Host ('=== COMPONENTE 2: stress orizzonte 4x ({0} runs, len {1}) ===' -f $hor_tasks.Count,$HORIZ_LEN) -ForegroundColor Yellow
Invoke-Throttled $hor_tasks $THROTTLE
$hor_agg = @{}
foreach ($tp in $TEMPS) { $hor_agg[$tp] = @{ mets=@(); bpbs=@() } }
foreach ($t in $hor_tasks) {
    $m = Get-WorstWindowMetrics $t.tf $GEN_LEN
    if ($m) { $hor_agg[$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf
    if ($null -ne $b) { $hor_agg[$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host '======== STRESS ORIZZONTE (worst window di lunghezza standard dentro sample 4x; REPORT, non gate) ========' -ForegroundColor Cyan
Write-Host ('  {0,5} {1,6} {2,6} {3,6} {4,6} {5,9}' -f 'temp','topBi','altLp','runWst','nameW','selfBPB*')
foreach ($tp in $TEMPS) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($m in @($hor_agg[$tp].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
    foreach ($b in @($hor_agg[$tp].bpbs)) { $bs+=[double]$b }
    $flag = if ((Mx $bi) -le $G_BI -and (Mx $al) -le $G_AL -and (Mx $mr) -le $G_RUN -and (Mx $ni) -le $G_NAME) { 'in-bar' } else { 'OLTRE-BAR' }
    $col = if ($flag -eq 'in-bar') {'Green'} else {'Yellow'}
    Write-Host ('  {0,5} {1,6} {2,6} {3,6} {4,6} {5,9}   {6}' -f $tp,('{0:F0}' -f (Mx $bi)),('{0:F0}' -f (Mx $al)),('{0:F0}' -f (Mx $mr)),('{0:F1}' -f (Mx $ni)),('{0:F2}' -f (Av $bs)),$flag) -ForegroundColor $col
}
Write-Host '  (*selfBPB = media sull''intero sample lungo, non windowed)'

# ============ COMPONENTE 3: CONTINUITA' DI TEMPERATURA ============================
$cont_tasks = @()
foreach ($tp in $CONT_TEMPS) { $si=0
    foreach ($off in @($SEEDS_STD | Select-Object -First $CONT_SEEDS)) { $si++
        $cont_tasks += New-GenTask 'CONT' 'phase47_generator.exe' $PR7_ARGS $tp $RNGS_ORIG[0] $off $si '_cont' $GEN_LEN
    } }
Write-Host ''
Write-Host ('=== COMPONENTE 3: continuita'' di temperatura T0.60/T0.70 ({0} runs) ===' -f $cont_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $cont_tasks $THROTTLE
$cont_agg = @{}
foreach ($tp in $CONT_TEMPS) { $cont_agg[$tp] = @{ mets=@(); bpbs=@() } }
foreach ($t in $cont_tasks) {
    $m = Get-WordMetrics $t.tf
    if ($m) { $cont_agg[$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf
    if ($null -ne $b) { $cont_agg[$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host '======== CONTINUITA'' TEMPERATURA (solo report: un regime vero non ha cliff tra le temp del gate) ========' -ForegroundColor Cyan
Write-Host ('  {0,5} {1,6} {2,6} {3,6} {4,6} {5,9}' -f 'temp','topBi','altLp','runWst','nameW','selfBPB')
foreach ($tp in $CONT_TEMPS) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($m in @($cont_agg[$tp].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
    foreach ($b in @($cont_agg[$tp].bpbs)) { $bs+=[double]$b }
    Write-Host ('  {0,5} {1,6} {2,6} {3,6} {4,6} {5,9}' -f $tp,('{0:F0}' -f (Mx $bi)),('{0:F0}' -f (Mx $al)),('{0:F0}' -f (Mx $mr)),('{0:F1}' -f (Mx $ni)),('{0:F2}' -f (Av $bs)))
}

# ============ COMPONENTE 4: LETTURA UMANA ========================================
Write-Host ''
Write-Host '=== COMPONENTE 4: dump per lettura umana ===' -ForegroundColor Yellow
$ncopied = 0
foreach ($tp in $TEMPS) {
    $srcs = @($rep_tasks | Where-Object { $_.label -eq 'R1_newrngA' -and $_.tp -eq $tp -and $_.rng -eq $RNGS_NEW_A[0] } | Select-Object -First 5)
    $hi = 0
    foreach ($t in $srcs) {
        $hi++
        if (Test-Path $t.tf) { Copy-Item $t.tf ($HDIR + '\Pr7_T' + $tp + '_sample' + $hi + '.txt') -Force; $ncopied++ }
    }
}
Write-Host ('  {0} sample copiati in {1} — LEGGILI prima di promuovere: il gate e'' un proxy,' -f $ncopied,$HDIR) -ForegroundColor Cyan
Write-Host '  selfBPB sano e bigram a posto non garantiscono lingua sensata. Questo passaggio e'' tuo.' -ForegroundColor Cyan

# ============ VERDETTO COMPONENTE 1 (regola pre-registrata) =======================
Write-Host ''
Write-Host '=== VERDETTO VALIDAZIONE P_r7 (regola pre-registrata) ===' -ForegroundColor Yellow
Write-Host ('  Repliche PASS: {0}/4' -f $rep_pass)
if ($rep_pass -eq 4) {
    Write-Host '  -> 4/4 PASS: PROMOZIONE CONFERMATA dalla replica statistics. Restano la lettura' -ForegroundColor Green
    Write-Host '     umana (componente 4, tua) e i report orizzonte/continuita'' sopra. Se tutto' -ForegroundColor Green
    Write-Host '     regge: P_r7 = campione Phase 47 (candidato SEE-V4), riscrittura HANDOFF.md,' -ForegroundColor Green
    Write-Host '     chiusura fase e commit SOLO a tuo ordine.' -ForegroundColor Green
} elseif ($marginal) {
    Write-Host '  -> MARGINALE: una sola replica fallisce un solo componente di <= 1 unita''.' -ForegroundColor Cyan
    Write-Host '     Decisione all''utente (regola pre-registrata). Dettaglio sopra.' -ForegroundColor Cyan
} else {
    Write-Host '  -> NON PROMUOVIBILE: >=2 repliche falliscono (o fail multipli/larghi). Il PASS' -ForegroundColor Red
    Write-Host '     di 47.G era coda fortunata; P_r7 resta il best near-pass della storia.' -ForegroundColor Red
}

# ============ COMPONENTE 5: CODA H48 (secondaria, tagliabile) =====================
if ($SkipH48Tail) {
    Write-Host ''
    Write-Host 'Coda H48 saltata (-SkipH48Tail).' -ForegroundColor Gray
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}
if (-not (Test-Path $G_H48R1) -or -not (Test-Path $G_H48R5)) {
    Write-Host 'Checkpoint H48 di 47.G mancanti: coda H48 saltata.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR); exit 0
}
# single-trainer guard
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps','phase44f_captrain','phase45a_relmove','phase45b_geometry','phase45c_gate','phase46a_l3','phase46b_l3','phase47a0_gauntlet','phase47b_decoder','phase47c_robust','phase47d_rollout','phase47e_capacity','phase47f_tempcov','phase47g_lastmile','phase47h_h48tail')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting H48 tail.'; exit 1 }

Write-Host ''
Write-Host '=== COMPONENTE 5: coda H48 r6->r9 (NON entra nella decisione P_r7) ===' -ForegroundColor Yellow
$TRAIN_OUT = $RDIR + '\phase47h_train.txt'
if ($SkipTrain -and (Test-Path $TRAIN_OUT)) {
    Write-Host ('SkipTrain: riuso ' + $TRAIN_OUT) -ForegroundColor Cyan
} else {
    & "$BINDIR\phase47h_h48tail.exe" $TS_DATA $D1_W $WP 2>&1 | Tee-Object $TRAIN_OUT
    if ($LASTEXITCODE -ne 0) { Write-Error 'H48 tail training failed'; exit 1 }
}
$HP = Parse-KV $TRAIN_OUT 'PROBE'
$HB = Parse-KV $TRAIN_OUT 'BASE'
if (-not $HP.ContainsKey('H48_r9')) { Write-Error 'H48 tail output incomplete'; exit 1 }
# anchor + prefix MD5 vs 47.G
$anchor_ok = $true
foreach ($w in @('val1','val2','val3')) {
    $d1v = [double]$HB[$w]['d1']; $i = [array]::IndexOf(@('val1','val2','val3'),$w)+1
    $fz = [double]$HP['frozenD1']["val$i"]
    if ([math]::Abs($fz-$d1v) -gt 0.005) { $anchor_ok = $false }
}
if (-not $anchor_ok) { Write-Error 'H48 tail anchor FAIL.'; exit 1 }
Write-Host '  anchor OK' -ForegroundColor Green
$prefix_ok = $true
foreach ($chk in @(@{g=($WP+'_H48_r1.bin'); f=$G_H48R1; n='r1'}, @{g=($WP+'_H48_r5.bin'); f=$G_H48R5; n='r5'})) {
    $hg = FileMD5 $chk.g; $hf = FileMD5 $chk.f
    $ok = ($hg -ne '' -and $hg -eq $hf); if (-not $ok) { $prefix_ok = $false }
    Write-Host ('  prefix@{0} {1}' -f $chk.n,($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $prefix_ok) { Write-Error 'H48 prefix property FAILED. Aborting tail.'; exit 1 }

Write-Host ''
Write-Host ('  {0,-8} {1,8} {2,8} {3,7} {4,8}' -f 'probe','avgVal','valC','roll','ent1')
foreach ($rn in 1..9) {
    $n = "H48_r$rn"
    if (-not $HP.ContainsKey($n)) { continue }
    $p = $HP[$n]; $av = AvgVal $p
    $col = if ($av -le $PROMO_BPB) {'Green'} else {'Gray'}
    Write-Host ('  {0,-8} {1,8} {2,8} {3,7} {4,8}' -f $n,('{0:F4}' -f $av),$p['valC'],$p['roll'],$p['ent1']) -ForegroundColor $col
}

# mini gate H48_r6..r9
$h48_cands = @()
foreach ($rn in 6..9) {
    $n = "H48_r$rn"; $wf = $WP + '_H48_r' + $rn + '.bin'
    if ($HP.ContainsKey($n) -and (Test-Path $wf)) {
        $h48_cands += @{ label=$n; wargs=('"'+$D1_W+'" "'+$wf+'"'); valbpb=(AvgVal $HP[$n]) }
    }
}
$SEEDS_MINI = @()
for ($i = 0; $i -lt 4; $i++) { $SEEDS_MINI += [int]([math]::Floor($i * ($fsz - $margin) / 4)) }
$mini_tasks = @()
foreach ($tp in $TEMPS) { $si=0
    foreach ($off in $SEEDS_MINI) { $si++
        foreach ($c in $h48_cands) { $mini_tasks += New-GenTask ($c.label) 'phase47_generator.exe' $c.wargs $tp $RNGS_ORIG[0] $off $si '_mini' $GEN_LEN }
    } }
Write-Host ''
Write-Host ('=== H48 mini gate ({0} runs) ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
$h48_agg = @{}
foreach ($c in $h48_cands) { $h48_agg[$c.label] = @{}; foreach ($tp in $TEMPS) { $h48_agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
foreach ($t in $mini_tasks) {
    $m = Get-WordMetrics $t.tf
    if ($m) { $h48_agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf
    if ($null -ne $b) { $h48_agg[$t.label][$t.tp].bpbs += $b }
}
function WorstH48($lbl,$tp) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($m in @($h48_agg[$lbl][$tp].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
    foreach ($b in @($h48_agg[$lbl][$tp].bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni=(Mx $ni); mr=(Mx $mr); bi=(Mx $bi); al=(Mx $al); self=(Av $bs) }
}
Write-Host ''
Write-Host ('======== H48 MINI (qualify: bi<={0} entrambe + selfBPB55>={1} + BPB<={2}) ========' -f $MINI_BI,$BPB_LO,$PROMO_BPB) -ForegroundColor Cyan
Write-Host ('  {0,-8} {1,8} {2,6} {3,6} {4,6} {5,8} {6,8} {7,7}' -f 'config','BPB','bi65','bi55','al55','self65','self55','promo')
$h48_promoted = @()
foreach ($c in $h48_cands) {
    $w65 = WorstH48 $c.label '0.65'; $w55 = WorstH48 $c.label '0.55'
    $bi_ok   = ($w65.bi -le $MINI_BI) -and ($w55.bi -le $MINI_BI)
    $self_ok = ($w55.self -ge $BPB_LO) -and ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -le $BPB_HI)
    $bpb_ok  = ($c.valbpb -le $PROMO_BPB)
    $promo = if ($bi_ok -and $self_ok -and $bpb_ok) { $h48_promoted += $c; 'FULL' } else { 'stop' }
    $col = if ($promo -eq 'FULL') {'Green'} else {'Gray'}
    Write-Host ('  {0,-8} {1,8} {2,6} {3,6} {4,6} {5,8} {6,8} {7,7}' -f `
        $c.label,('{0:F4}' -f $c.valbpb),('{0:F0}' -f $w65.bi),('{0:F0}' -f $w55.bi),('{0:F0}' -f $w55.al),('{0:F2}' -f $w65.self),('{0:F2}' -f $w55.self),$promo) -ForegroundColor $col
}

if ($h48_promoted.Count -gt 0) {
    $full_tasks = @()
    foreach ($tp in $TEMPS) { foreach ($rng in $RNGS_ORIG) { $si=0
        foreach ($off in $SEEDS_STD) { $si++
            foreach ($c in $h48_promoted) { $full_tasks += New-GenTask ($c.label) 'phase47_generator.exe' $c.wargs $tp $rng $off $si '_full' $GEN_LEN }
        } } }
    Write-Host ''
    Write-Host ('=== H48 FULL gate: {0} runs ===' -f $full_tasks.Count) -ForegroundColor Yellow
    foreach ($c in $h48_promoted) { foreach ($tp in $TEMPS) { $h48_agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
    Invoke-Throttled $full_tasks $THROTTLE
    foreach ($t in $full_tasks) {
        $m = Get-WordMetrics $t.tf
        if ($m) { $h48_agg[$t.label][$t.tp].mets += $m }
        $b = Get-SelfBPB $t.sf
        if ($null -ne $b) { $h48_agg[$t.label][$t.tp].bpbs += $b }
    }
    Write-Host ''
    Write-Host '======== H48 FULL (worst-case su 32; NON entra nella decisione P_r7) ========' -ForegroundColor Cyan
    Write-Host ('  {0,-8} {1,8} {2,5} {3,6} {4,6} {5,6} {6,6} {7,8} {8,6}' -f 'config','BPB','temp','topBi','altLp','runWst','nameW','selfBPB','gate')
    foreach ($c in $h48_promoted) {
        foreach ($tp in $TEMPS) {
            $w = WorstH48 $c.label $tp
            $pass = ($c.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
            $col = if ($pass) {'Green'} else {'Gray'}
            Write-Host ('  {0,-8} {1,8} {2,5} {3,6} {4,6} {5,6} {6,6} {7,8} {8,6}' -f `
                $c.label,('{0:F4}' -f $c.valbpb),$tp,('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F0}' -f $w.mr),('{0:F1}' -f $w.ni),('{0:F2}' -f $w.self),($(if($pass){'PASS'}else{'fail'}))) -ForegroundColor $col
        }
    }
    Write-Host ''
    Write-Host '  NB: un H48 che passa il full e'' un SECONDO candidato (dominerebbe P_r7 di ~0.04 BPB)' -ForegroundColor Cyan
    Write-Host '      ma richiederebbe la SUA validazione estesa 47.H. Decisione utente.' -ForegroundColor Cyan
} else {
    Write-Host ''
    Write-Host '  Nessun H48 qualifica il full: la coda e'' chiusa, P_r7 resta l''unico candidato.' -ForegroundColor Gray
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
