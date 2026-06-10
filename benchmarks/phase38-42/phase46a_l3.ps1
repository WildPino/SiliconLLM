# Phase 46.A - L3 phrase-memory over SEE/fa, above D1
#
# Axis B. feature = [SEE 192 | L2 64 | L3 64] = 320D. L2 = D1 (mix0.5). L3 = slow EMA of raw
# fa (SEE) refreshed only at a phrase schedule. Readout linear. No delta, no parser, no bigram
# policy. A0 D1 control (no L3), A1 PUNCT, A2 L2CLUST K=8, A3 LOWMARGIN (trigram margin<p20).
#
# Audit (per user): BPB; L3/SEE logit ratio; L3 argmax flip-rate; L3 rel-move / update freq /
# gap p50-p90; word-gate T0.65/T0.55; controls D1 ~2.2522 and C2.A ~2.2593.
# Promotion: NOT BPB alone - must pass BOTH T0.65 AND T0.55.
#
# Run:  .\benchmarks\phase38-42\phase46a_l3.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase46a'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'      # D1 control (mix0.5 scale0.5), ~2.2522
$WP      = $WDIR + '\phase46a'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
if (-not (Test-Path $C2A_W)) { Write-Error "Missing C2.A: $C2A_W"; exit 1 }
if (-not (Test-Path $D1_W))  { Write-Error "Missing D1 control: $D1_W"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$RNGS    = @(12345, 67890)
$TEMPS   = @('0.65', '0.55')
$GEN_LEN = 2000
$WARMUP  = 5000
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0
$PROMO_BPB = $C2A_BPB - 0.005   # 2.2543
$G_NAME = 20.0; $G_RUN = 5.0; $G_BI = 8.0; $G_AL = 2.0
$THROTTLE = 12
$TEL_SEEDS = 2

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase46a_l3.c' @SRC_CORE -o "$BINDIR\phase46a_l3.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase46a_l3'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase46_generator.c' @SRC_CORE -o "$BINDIR\phase46_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase46_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps','phase44f_captrain','phase45a_relmove','phase45b_geometry','phase45c_gate','phase46a_l3')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Aborting.')
    exit 1
}

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 46.A - L3 phrase-memory (A0..A3, fresh 320D extract per config) ===' -ForegroundColor Yellow
Write-Host '  320D features ~64 GB. Single trainer only. L3 schedule alters trajectory -> fresh per config.'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase46a_train.txt'
& "$BINDIR\phase46a_l3.exe" $TS_DATA $WP $C2A_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host 'Training complete.' -ForegroundColor Green

function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double]((($line.Line -replace '.*BPB=','') -replace '\s.*','').Trim()) }
    return $null
}

$sfxs = @('_A0_d1','_A1_punct','_A2_l2clust','_A3_lowmargin')
$gen_configs = @(
    @{ label='C2.A'; exe='phase43_generator.exe'; weights=$C2A_W; valbpb=$C2A_BPB; ref=$true;  l3=$false },
    @{ label='D1';   exe='phase44_generator.exe'; weights=$D1_W;  valbpb=$D1_BPB;  ref=$true;  l3=$false }
)
foreach ($s in $sfxs) {
    $gen_configs += @{ label=$s.TrimStart('_'); exe='phase46_generator.exe'; weights=$WP+$s+'.bin'; valbpb=(Get-TrainBPB ($s+'.bin')); ref=$false; l3=($s -ne '_A0_d1') }
}
$A0_BPB = Get-TrainBPB '_A0_d1.bin'

# -------- Control reproduction check ------------------------------------------
Write-Host ''
Write-Host '=== Control reproduction (A0 = D1 with dead L3 block) ===' -ForegroundColor Yellow
$a0ok = ($null -ne $A0_BPB) -and ([math]::Abs($A0_BPB-$D1_BPB) -le 0.004)
Write-Host ('  A0 D1 control: BPB={0}  (expect ~{1})  {2}' -f ('{0:F4}' -f $A0_BPB),$D1_BPB,($(if($a0ok){'OK'}else{'CHECK'}))) -ForegroundColor $(if($a0ok){'Green'}else{'Yellow'})

# -------- Word metrics --------------------------------------------------------
function Get-WordMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^a-zA-Z]', ' '
    $toks = @($text.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
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
function Mx ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
function Av ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }

# -------- Generate (parallel, deterministic-identical) ------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{}; foreach ($tp in $TEMPS) { $agg[$cfg.label][$tp] = @{ mets=@(); bpbs=@() } } }

function New-GenTask($cfg,$tp,$rng,$off,$si,$suffix) {
    $lbl = $cfg.label+'_T'+$tp+'_r'+$rng+'_s'+$si
    $tf  = $RDIR+'\gen_'+$lbl+$suffix+'.txt'
    $sf  = $RDIR+'\gen_'+$lbl+$suffix+'_stats.txt'
    $argline = ('"{0}" "{1}" --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                $TS_DATA,$cfg.weights,$GEN_LEN,$WARMUP,$tp,$off,$rng)
    return [PSCustomObject]@{ label=$cfg.label; tp=$tp; rng=$rng; off=$off; si=$si;
                              exe=($BINDIR+'\'+$cfg.exe); tf=$tf; sf=$sf; argline=$argline }
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
            Write-Host ('  launched {0,4}/{1}  {2,-12} T{3} r{4} s{5}' -f $launched,$total,$t.label,$t.tp,$t.rng,$t.si)
        }
        Start-Sleep -Milliseconds 150
        for ($k=$running.Count-1; $k -ge 0; $k--) { if ($running[$k].HasExited) { $running.RemoveAt($k) } }
    }
}
# Windows can hold a child's redirected-stdout handle briefly after HasExited; retry the hash.
function FileMD5($p) {
    if (-not (Test-Path $p)) { return '' }
    for ($a=0; $a -lt 15; $a++) { try { return (Get-FileHash -Algorithm MD5 -Path $p -ErrorAction Stop).Hash } catch { Start-Sleep -Milliseconds 200 } }
    return ''
}

# -------- Repro pre-check -----------------------------------------------------
Write-Host ''
Write-Host '=== Repro pre-check: parallel vs sequential (2 cfg x 2 seed) ===' -ForegroundColor Yellow
$chk_cfgs = @($gen_configs | Where-Object { (Test-Path $_.weights) } | Select-Object -First 2)
$chk_seeds = @($SEEDS | Select-Object -First 2)
$chk_tp = $TEMPS[0]; $chk_rng = $RNGS[0]
$seq_tasks=@(); $par_tasks=@(); $si=0
foreach ($off in $chk_seeds) { $si++; foreach ($c in $chk_cfgs) {
    $seq_tasks += New-GenTask $c $chk_tp $chk_rng $off $si '_seqchk'
    $par_tasks += New-GenTask $c $chk_tp $chk_rng $off $si '_parchk'
} }
Invoke-Throttled $seq_tasks 1
Invoke-Throttled $par_tasks ([math]::Min(4,$par_tasks.Count))
$repro_ok = $true
for ($i=0; $i -lt $seq_tasks.Count; $i++) {
    $hs = FileMD5 $seq_tasks[$i].tf; $hp = FileMD5 $par_tasks[$i].tf
    $ok = ($hs -ne '' -and $hs -eq $hp)
    if (-not $ok) { $repro_ok = $false }
    Write-Host ('  {0,-12} s{1}  seq={2}  par={3}  {4}' -f $seq_tasks[$i].label,$seq_tasks[$i].si,$hs.Substring(0,[math]::Min(8,$hs.Length)),$hp.Substring(0,[math]::Min(8,$hp.Length)),($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $repro_ok) { Write-Error 'Repro pre-check FAILED. Aborting.'; exit 1 }
Write-Host '  Repro pre-check PASSED.' -ForegroundColor Green

# -------- Full parallel word-gate ---------------------------------------------
$tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS) { $si=0
    foreach ($off in $SEEDS) { $si++
        foreach ($cfg in $gen_configs) { if (Test-Path $cfg.weights) { $tasks += New-GenTask $cfg $tp $rng $off $si '' } }
    } } }
Write-Host ''
Write-Host ('=== Word-gate: temps {0}, {1}x{2}/temp, {3} runs, throttle {4} ===' -f `
    ($TEMPS -join ','),$NSEEDS,$RNGS.Count,$tasks.Count,$THROTTLE) -ForegroundColor Yellow
Invoke-Throttled $tasks $THROTTLE

Write-Host ''
Write-Host '  Aggregating in (temp,rng,seed,config) order...' -ForegroundColor Yellow
foreach ($t in $tasks) {
    if (-not (Test-Path $t.tf)) { Write-Host ('  MISSING ' + $t.tf) -ForegroundColor Red; continue }
    $m = Get-WordMetrics $t.tf
    if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $bl = Select-String 'self_BPB' $t.sf | Select-Object -Last 1
    if ($bl) { $agg[$t.label][$t.tp].bpbs += [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
}

function Worst($lbl,$tp) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($m in @($agg[$lbl][$tp].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
    foreach ($b in @($agg[$lbl][$tp].bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni=(Mx $ni); mr=(Mx $mr); bi=(Mx $bi); al=(Mx $al); self=(Av $bs) }
}
function Passes($cfg,$tp) {
    if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { return $false }
    $w = Worst $cfg.label $tp
    return ($null -ne $cfg.valbpb) -and ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
}

# -------- Tables --------------------------------------------------------------
foreach ($tp in $TEMPS) {
    Write-Host ''
    Write-Host ('======== T={0} : worst-case over 32 samples ========' -f $tp) -ForegroundColor Cyan
    Write-Host ('  {0,-13} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f 'config','valBPB','dC2.A','nameWst','runWst','topBi','altLp','selfBPB','gate')
    Write-Host ('  ' + '-'*84)
    foreach ($cfg in $gen_configs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-13} [no data]' -f $cfg.label); continue }
        $w = Worst $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        $gate = if ($cfg.ref) {'ref'} elseif (Passes $cfg $tp) {'PASS'} else {'fail'}
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'Gray'} else {'DarkCyan'}
        Write-Host ('  {0,-13} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
}

# -------- L3 telemetry audit (per L3 config, T=0.55) --------------------------
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); [math]::Round($s[$idx],4) }
function MeanD([double[]]$a){ if($a.Count -eq 0){return 0.0}; [math]::Round(($a|Measure-Object -Average).Average,4) }

Write-Host ''
Write-Host '======== L3 telemetry audit (T=0.55) ========' -ForegroundColor Cyan
Write-Host ('  {0,-13} {1,9} {2,8} {3,9} {4,9} {5,9} {6,9}' -f 'config','L3/SEE%','L3flip%','upd_freq%','gap_p50','gap_p90','relM_p50')
Write-Host ('  ' + '-'*74)
$tel_off = @($SEEDS | Select-Object -First $TEL_SEEDS)
foreach ($cfg in $gen_configs) {
    if (-not $cfg.l3 -or -not (Test-Path $cfg.weights)) { continue }
    $ratio=@(); $rel=@(); $nFlip=0; $nWrote=0; $nStep=0; $gaps=@(); $lastw=-1
    $si=0
    foreach ($off in $tel_off) {
        $si++
        $tel = $RDIR + '\tel_' + $cfg.label + '_s' + $si + '.tsv'
        $sf  = $RDIR + '\tel_' + $cfg.label + '_s' + $si + '_stats.txt'
        $al = ('"{0}" "{1}" --gen-len {2} --warmup {3} --temp 0.55 --seed-start {4} --rng-seed {5} --telemetry "{6}"' -f `
               $TS_DATA,$cfg.weights,$GEN_LEN,$WARMUP,$off,$RNGS[0],$tel)
        Start-Process -FilePath "$BINDIR\phase46_generator.exe" -ArgumentList $al `
            -RedirectStandardOutput ($RDIR+'\_null.bin') -RedirectStandardError $sf -NoNewWindow -PassThru -Wait | Out-Null
        if (-not (Test-Path $tel)) { continue }
        $lastw=-1
        foreach ($line in [System.IO.File]::ReadLines($tel)) {
            if ($line.Length -eq 0 -or $line[0] -eq 's') { continue }
            $f = $line -split "`t"; if ($f.Count -lt 8) { continue }
            $nStep++
            $ratio += [double]$f[3]; if ([int]$f[4] -eq 1) { $nFlip++ }
            if ([int]$f[5] -eq 1) { $nWrote++; $rel += [double]$f[6]
                $step=[int]$f[0]; if ($lastw -ge 0) { $gaps += ($step-$lastw) }; $lastw=$step }
        }
    }
    $flipPct = if ($nStep -gt 0) { [math]::Round(100.0*$nFlip/$nStep,2) } else { 0 }
    $updPct  = if ($nStep -gt 0) { [math]::Round(100.0*$nWrote/$nStep,2) } else { 0 }
    Write-Host ('  {0,-13} {1,9} {2,8} {3,9} {4,9} {5,9} {6,9}' -f `
        $cfg.label,(MeanD $ratio),$flipPct,$updPct,(Pctl $gaps 0.50),(Pctl $gaps 0.90),(Pctl $rel 0.50))
}
Write-Host '  (L3/SEE ratio ~0 AND L3flip ~0 => the readout zeroed L3: dead block, = D1. Otherwise L3 is used.)' -ForegroundColor Gray

# -------- Cross-temperature stability + verdict -------------------------------
Write-Host ''
Write-Host '=== Cross-temperature stability (pass at BOTH T=0.65 AND T=0.55?) ===' -ForegroundColor Yellow
Write-Host ('  {0,-13} {1,7} {2,7} {3,9} {4,11}' -f 'config','T0.65','T0.55','valBPB','dBPB_vsD1')
$stable=@()
foreach ($cfg in $gen_configs) {
    if ($cfg.ref) { continue }
    $p65 = Passes $cfg '0.65'; $p55 = Passes $cfg '0.55'
    $dD1 = if (($null -ne $cfg.valbpb) -and ($null -ne $A0_BPB)) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$A0_BPB) } else { '-' }
    $both = $p65 -and $p55
    $col = if ($both) {'Green'} elseif ($p65) {'DarkCyan'} else {'Gray'}
    Write-Host ('  {0,-13} {1,7} {2,7} {3,9} {4,11}' -f $cfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),('{0:F4}' -f $cfg.valbpb),$dD1) -ForegroundColor $col
    if ($both -and ($cfg.label -ne 'A0_d1')) { $stable += $cfg.label }
}

Write-Host ''
Write-Host '=== Verdetto 46.A ===' -ForegroundColor Yellow
if ($stable.Count -gt 0) {
    Write-Host ('  -> CANDIDATO L3 phrase-memory: {0} passa T=0.65 E T=0.55.' -f ($stable -join ', ')) -ForegroundColor Green
    Write-Host '     Confronta col delta dC2.A: se migliora la BPB E passa il gate (L3/SEE ratio>0, L3 vivo)' -ForegroundColor Green
    Write-Host '     -> nuova memoria di frase (candidato). Se passa il gate ma perde BPB -> stabilizzatore.' -ForegroundColor Green
} else {
    Write-Host '  -> Nessun L3 passa entrambi i temp. Leggi L3/SEE ratio + BPB:' -ForegroundColor Yellow
    Write-Host '     - se ratio~0 (L3 dead) e BPB~D1: il readout lineare ignora L3-su-fa -> il substrato non' -ForegroundColor Yellow
    Write-Host '       accetta memoria di frase lineare; tornare ad asse A o C.' -ForegroundColor Yellow
    Write-Host '     - se ratio>0 ma migliora BPB e fallisce gate: stessa storia di L2 (pressione, non memoria).' -ForegroundColor Yellow
    Write-Host '     - se passa gate ma perde troppa BPB: stabilizzatore, non nuova memoria.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
