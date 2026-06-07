# Phase 44.E - Readout L2 Logit-Contribution Caps (conditional trust) Tribunal
#
# 44.D verdict: static scale helps but is not enough. D1 (mix50 scale0.50) passes
# T=0.65 keeping almost all the BPB gain, but fails T=0.55. delta archived, dropout
# rejected. 44.E tests CONDITIONAL trust: a dynamic cap on the L2 logit contribution
# that fires ONLY when L2 over-asserts (closed-loop attractor), dormant otherwise.
# Inference-only (no retrain): cap variants reuse base D1/H0 weights. Magic 0x53454540.
#
# Configs (gen): C2.A (ref), H0, mix50, D1 (baselines) + caps:
#   D1_cap20 / D1_cap15 / D1_cap10  (mix50 scale0.5 + cap 2.0/1.5/1.0%)
#   H0_cap20                        (mix0  scale1.0 + cap 2.0%)
#
# Criterion (per user): NOT max BPB. Does the T=0.65 pass EXTEND to T=0.55 without
# losing more than ~0.003-0.005 BPB vs the D1 base? Gate unchanged: BPB<=2.2543 +
# topBi<=8 + altLp<=2 + nameWst<=20 + runWst<=5 (self in [0.8,2.0]).
#
# Run:  .\benchmarks\phase38-42\phase44e_caps.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44e'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$WP      = $WDIR + '\phase44e'
$C2A_BPB = 2.2593
if (-not (Test-Path $C2A_W)) { Write-Error "Missing C2.A: $C2A_W"; exit 1 }

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
$BPB_COST  = 0.005              # max acceptable BPB loss vs D1 base
$G_NAME = 20.0; $G_RUN = 5.0; $G_BI = 8.0; $G_AL = 2.0
$THROTTLE = 12

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44e_caps.c' @SRC_CORE -o "$BINDIR\phase44e_caps.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44e_caps'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another Phase 44 trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Two trainers exceed 80 GB RAM. Aborting.')
    exit 1
}

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 44.E - Training 3 bases (H0/mix50/D1); caps derived inference-only ===' -ForegroundColor Yellow
Write-Host '  256D features ~51.5 GB + L2 canon ~13 GB. 2 cached extractions (mix0, mix0.5).'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase44e_train.txt'
& "$BINDIR\phase44e_caps.exe" $TS_DATA $WP $C2A_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host 'Training complete.' -ForegroundColor Green

function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double]((($line.Line -replace '.*BPB=','') -replace '\s.*','').Trim()) }
    return $null
}

$sfxs = @('_H0','_mix50','_D1','_D1_cap20','_D1_cap15','_D1_cap10','_H0_cap20')
$gen_configs = @( @{ label='C2.A'; exe='phase43_generator.exe'; weights=$C2A_W; valbpb=$C2A_BPB; ref=$true } )
foreach ($s in $sfxs) {
    $gen_configs += @{ label=$s.TrimStart('_'); exe='phase44_generator.exe'; weights=$WP+$s+'.bin'; valbpb=(Get-TrainBPB ($s+'.bin')); ref=$false }
}
$D1_BPB = Get-TrainBPB '_D1.bin'

# -------- Audit echo ----------------------------------------------------------
Write-Host ''
Write-Host '=== L2 audit + cap fire-rates (from train log) ===' -ForegroundColor Yellow
$auditLines = Select-String -Path $TRAIN_OUT -Pattern '======|AUDIT L2:|cap fired|cap2|cap1' | ForEach-Object { $_.Line.Trim() }
foreach ($al in $auditLines) {
    if ($al -match '^======') { Write-Host ('  ' + ($al -replace '=','').Trim()) -ForegroundColor DarkCyan }
    else { Write-Host ('    ' + $al) -ForegroundColor Gray }
}

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
            Write-Host ('  launched {0,4}/{1}  {2,-14} T{3} r{4} s{5}' -f $launched,$total,$t.label,$t.tp,$t.rng,$t.si)
        }
        Start-Sleep -Milliseconds 150
        for ($k=$running.Count-1; $k -ge 0; $k--) { if ($running[$k].HasExited) { $running.RemoveAt($k) } }
    }
}
function FileMD5($p) { if (-not (Test-Path $p)) { return '' }; (Get-FileHash -Algorithm MD5 -Path $p).Hash }

# -------- Repro pre-check: parallel == sequential (2 cfg x 2 seed) -------------
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
    Write-Host ('  {0,-14} s{1}  seq={2}  par={3}  {4}' -f $seq_tasks[$i].label,$seq_tasks[$i].si,$hs.Substring(0,[math]::Min(8,$hs.Length)),$hp.Substring(0,[math]::Min(8,$hp.Length)),($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $repro_ok) { Write-Error 'Repro pre-check FAILED: parallel output differs from sequential. Aborting.'; exit 1 }
Write-Host '  Repro pre-check PASSED: parallel is deterministic-identical.' -ForegroundColor Green

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
    Write-Host ('  {0,-12} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f 'config','valBPB','dBPB','nameWst','runWst','topBi','altLp','selfBPB','gate')
    Write-Host ('  ' + '-'*84)
    foreach ($cfg in $gen_configs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-12} [no data]' -f $cfg.label); continue }
        $w = Worst $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        $gate = if ($cfg.ref) {'ref'} elseif (Passes $cfg $tp) {'PASS'} else {'fail'}
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'Gray'} else {'DarkCyan'}
        Write-Host ('  {0,-12} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
}

# -------- Cross-temperature stability (the 44.E question) ---------------------
Write-Host ''
Write-Host '=== Cross-temperature stability (pass at BOTH T=0.65 AND T=0.55?) ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,7} {2,7} {3,9} {4,10}' -f 'config','T0.65','T0.55','valBPB','dBPB_vsD1')
Write-Host ('  ' + '-'*52)
$stable=$null; $stable_bpb=$null
foreach ($cfg in $gen_configs) {
    if ($cfg.ref) { continue }
    $p65 = Passes $cfg '0.65'; $p55 = Passes $cfg '0.55'
    $dD1 = if (($null -ne $cfg.valbpb) -and ($null -ne $D1_BPB)) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$D1_BPB) } else { '-' }
    $both = $p65 -and $p55
    $col = if ($both) {'Green'} elseif ($p65) {'DarkCyan'} else {'Gray'}
    Write-Host ('  {0,-12} {1,7} {2,7} {3,9} {4,10}' -f $cfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),('{0:F4}' -f $cfg.valbpb),$dD1) -ForegroundColor $col
    # winner = passes both temps, BPB cost vs D1 within budget, then lowest BPB
    $cost_ok = ($null -ne $cfg.valbpb) -and ($null -ne $D1_BPB) -and (($cfg.valbpb - $D1_BPB) -le $BPB_COST)
    if ($both -and $cost_ok -and ($null -eq $stable_bpb -or $cfg.valbpb -lt $stable_bpb)) { $stable_bpb=$cfg.valbpb; $stable=$cfg.label }
}

# -------- Verdict -------------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 44.E ===' -ForegroundColor Yellow
if ($null -ne $stable) {
    Write-Host ('  -> CANDIDATO: {0}  passa T=0.65 E T=0.55  valBPB {1:F4} (vs D1 {2:+0.000;-0.000;0.000}, entro {3})' -f $stable,$stable_bpb,($stable_bpb-$D1_BPB),$BPB_COST) -ForegroundColor Green
    Write-Host '     Fiducia condizionale: il cap leasha L2 solo quando esplode, estendendo la' -ForegroundColor Green
    Write-Host '     stabilita a T=0.55 a costo BPB minimo. SEE-V4 = C2.A + L2-delta(mix) memory' -ForegroundColor Green
    Write-Host '     + readout L2 logit cap. Conferma con micro-fase multi-seed prima di promuovere.' -ForegroundColor Green
} else {
    # diagnose: did any cap improve T=0.55 over D1 baseline even if not full pass?
    $d1_p55 = $false; foreach ($cfg in $gen_configs) { if ($cfg.label -eq 'D1') { $d1_p55 = Passes $cfg '0.55' } }
    Write-Host '  -> Nessun cap estende il pass a T=0.55 entro il budget BPB.' -ForegroundColor Yellow
    Write-Host '     Leggi insieme tabella + fire-rate:' -ForegroundColor Yellow
    Write-Host '     - cap abbassa topBi/altLp a T=0.55 ma BPB sfora il budget: il cap inference-only' -ForegroundColor Yellow
    Write-Host '       e troppo grezzo -> 44.F = cap CO-ADATTATO in training (stop-grad gamma).' -ForegroundColor Yellow
    Write-Host '     - cap non muove topBi/altLp (fire-rate ~0 in closed-loop): la soglia e troppo' -ForegroundColor Yellow
    Write-Host '       alta o l attractor non passa dai logit L2 -> abbassa cap o ripensa il gating.' -ForegroundColor Yellow
    Write-Host '     - cap rompe tutto (self_BPB fuori range): L2 portava strutturale, non solo loop.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host '  Domanda 44.E: L2 utile ma quanto puo pesare nei logit senza dominare il closed-loop?'
Write-Host '  Il cap dinamico e fiducia condizionale: dormiente se L2 e benigno, guinzaglio se esplode.'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
