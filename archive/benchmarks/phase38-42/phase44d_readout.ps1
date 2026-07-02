# Phase 44.D - Readout-L2 Homeostasis Tribunal
#
# 44.C verdict: delta-write is REAL information but the linear readout drinks it too
# hard -> attractor in closed-loop. D_delta BPB ~2.221 (huge) but word-gate fails;
# self_BPB collapses as alpha slows (0.78->0.21->0.06) = readout nearly copying L2
# forward. mix50 nearly passes at T=0.65 (topBi 9), fails T=0.55. So: don't change
# the memory, change HOW MUCH the readout may depend on L2 in the logits.
#
# Levers: L2 block scale (== L2-column weight decay x1/s^2; train+gen, parity easy)
#   and L2 feature dropout (different mechanism; training-only, inverted).
#   Magic 0x5345453F adds f l2_scale. Audit: L2/SEE logit-norm ratio + flip bytes.
#
# 8 configs (entropy-high, alpha0.99):
#   H0 / mix50 / delta            controls (reproduce 44.C)
#   D1 mix50 scale0.50   D2 delta scale0.25   D3 delta scale0.50
#   D4 mix50 dropout0.10 D5 delta dropout0.10
#
# Promotion gate (per user): val BPB <= 2.2543 (>=0.005 vs C2.A) AND topBi <= 8
#   AND altLp <= 2 AND nameWst <= 20 AND runWst <= 5 (self in [0.8,2.0]).
#
# Run:  .\benchmarks\phase38-42\phase44d_readout.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44d'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$WP      = $WDIR + '\phase44d'
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
$G_NAME = 20.0; $G_RUN = 5.0; $G_BI = 8.0; $G_AL = 2.0

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44d_readout.c' @SRC_CORE -o "$BINDIR\phase44d_readout.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44d_readout'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
# A Phase 44 trainer holds ~49-52 GB (features ~51.5 + L2 canon ~13). Two at once
# exceed the 80 GB budget. Abort if another Phase 44 trainer is already running.
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another Phase 44 trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Two trainers exceed 80 GB RAM. Aborting.')
    exit 1
}

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 44.D - Training 8 readout-homeostasis configs (3 cached extractions) ===' -ForegroundColor Yellow
Write-Host '  256D features ~51.5 GB + L2 canon ~13 GB. mix groups: {H0} {mix50,D1,D4} {delta,D2,D3,D5}.'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase44d_train.txt'
& "$BINDIR\phase44d_readout.exe" $TS_DATA $WP $C2A_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host 'Training complete.' -ForegroundColor Green

function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}

$sfxs = @('_H0','_mix50','_delta','_D1_mix50_s50','_D2_delta_s25','_D3_delta_s50','_D4_mix50_d10','_D5_delta_d10')
$gen_configs = @( @{ label='C2.A'; exe='phase43_generator.exe'; weights=$C2A_W; valbpb=$C2A_BPB; ref=$true } )
foreach ($s in $sfxs) {
    $gen_configs += @{ label=$s.TrimStart('_'); exe='phase44_generator.exe'; weights=$WP+$s+'.bin'; valbpb=(Get-TrainBPB ($s+'.bin')); ref=$false }
}

# -------- Audit echo (from training log) --------------------------------------
Write-Host ''
Write-Host '=== L2 readout audit (logit weight + flip bytes, from train log) ===' -ForegroundColor Yellow
$auditLines = Select-String -Path $TRAIN_OUT -Pattern '======|AUDIT L2:|top bytes' | ForEach-Object { $_.Line.Trim() }
foreach ($al in $auditLines) {
    if ($al -match '^======') { Write-Host ('  ' + ($al -replace '=','').Trim()) -ForegroundColor DarkCyan }
    elseif ($al -match 'AUDIT L2:') { Write-Host ('    ' + $al) -ForegroundColor Gray }
    else { Write-Host ('    ' + $al) -ForegroundColor DarkGray }
}

# -------- Word metrics --------------------------------------------------------
function Get-WordMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^a-zA-Z]', ' '
    $toks = @($text.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
    $nt = $toks.Count; if ($nt -lt 4) { return $null }
    $wf = @{}; foreach ($w in $toks) { $wf[$w] = ($wf[$w] -as [int]) + 1 }
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

# -------- Generate (parallel word-gate, deterministic-identical) --------------
# Each generator run is an independent process fully determined by (weights,temp,
# seed-start,rng-seed) writing to a UNIQUE file -> concurrency cannot change its
# output. We launch with a throttle, wait for all, then aggregate by iterating the
# task list in fixed (temp,rng,seed,config) order - never by completion order.
$THROTTLE = 12

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

# -------- Repro pre-check: parallel == sequential on 2 configs x 2 seeds --------
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
if (-not $repro_ok) { Write-Error 'Repro pre-check FAILED: parallel output differs from sequential. Aborting word-gate.'; exit 1 }
Write-Host '  Repro pre-check PASSED: parallel is deterministic-identical.' -ForegroundColor Green

# -------- Full parallel word-gate ---------------------------------------------
$tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS) { $si=0
    foreach ($off in $SEEDS) { $si++
        foreach ($cfg in $gen_configs) { if (Test-Path $cfg.weights) { $tasks += New-GenTask $cfg $tp $rng $off $si '' } }
    } } }

Write-Host ''
Write-Host ('=== Word-gate: temps {0}, {1}x{2}={3}/temp, {4} runs, throttle {5} ===' -f `
    ($TEMPS -join ','),$NSEEDS,$RNGS.Count,($NSEEDS*$RNGS.Count),$tasks.Count,$THROTTLE) -ForegroundColor Yellow
Invoke-Throttled $tasks $THROTTLE

# -------- Aggregate in fixed order (NOT completion order) ----------------------
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

# -------- Tables --------------------------------------------------------------
foreach ($tp in $TEMPS) {
    Write-Host ''
    Write-Host ('======== T={0} : worst-case over 32 samples ========' -f $tp) -ForegroundColor Cyan
    Write-Host ('  {0,-14} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f 'config','valBPB','dBPB','nameWst','runWst','topBi','altLp','selfBPB','gate')
    Write-Host ('  ' + '-'*86)
    foreach ($cfg in $gen_configs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-14} [no data]' -f $cfg.label); continue }
        $w = Worst $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        $gate='ref'
        if (-not $cfg.ref) {
            $pass = ($null -ne $cfg.valbpb) -and ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
            $gate = if ($pass) {'PASS'} else {'fail'}
        }
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'Gray'} else {'DarkCyan'}
        Write-Host ('  {0,-14} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
}

# -------- Control sanity (controls reproduce 44.C) ----------------------------
Write-Host ''
$h0 = Get-TrainBPB '_H0.bin'; $m50 = Get-TrainBPB '_mix50.bin'; $dl = Get-TrainBPB '_delta.bin'
if ($null -ne $h0)  { Write-Host ('  Control: H0    BPB={0:F4} (should match 44.C D_H0    ~2.2526)' -f $h0) -ForegroundColor Cyan }
if ($null -ne $m50) { Write-Host ('  Control: mix50 BPB={0:F4} (should match 44.C D_mix50 ~2.2517)' -f $m50) -ForegroundColor Cyan }
if ($null -ne $dl)  { Write-Host ('  Control: delta BPB={0:F4} (should match 44.C D_delta ~2.2214)' -f $dl) -ForegroundColor Cyan }

# -------- Verdict: best PASSING config ----------------------------------------
Write-Host ''
Write-Host '=== Verdetto 44.D ===' -ForegroundColor Yellow
$win=$null; $win_bpb=$null; $win_tp=''
foreach ($cfg in $gen_configs) {
    if ($cfg.ref -or ($null -eq $cfg.valbpb)) { continue }
    foreach ($tp in $TEMPS) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { continue }
        $w = Worst $cfg.label $tp
        $pass = ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
        if ($pass -and ($null -eq $win_bpb -or $cfg.valbpb -lt $win_bpb)) { $win_bpb=$cfg.valbpb; $win=$cfg.label; $win_tp=$tp }
    }
}

# Did any intervention keep BPB strong (<=PROMO) while cutting the loop vs delta control?
$any_strong = $false
foreach ($cfg in $gen_configs) {
    if ($cfg.ref -or ($null -eq $cfg.valbpb)) { continue }
    if ($cfg.label -notmatch 'H0|^mix50$|^delta$' -and $cfg.valbpb -le $PROMO_BPB) { $any_strong = $true }
}

if ($null -ne $win) {
    Write-Host ('  -> PROMOSSO: {0} @ T={1}  valBPB {2:F4} ({3:+0.000;-0.000;0.000} vs C2.A)' -f $win,$win_tp,$win_bpb,($win_bpb-$C2A_BPB)) -ForegroundColor Green
    Write-Host '     Omeostasi del readout: limitare il peso di L2 nei logits rompe gli attractor' -ForegroundColor Green
    Write-Host '     mantenendo il guadagno BPB. SEE-V4 = C2.A + boundary-gated L2-delta memory' -ForegroundColor Green
    Write-Host '     + readout-L2 homeostasis. Primo contesto gerarchico stabile in closed-loop.' -ForegroundColor Green
} elseif ($any_strong) {
    Write-Host '  -> Nessuna config passa il gate pieno, MA un intervento tiene BPB forte.' -ForegroundColor Yellow
    Write-Host '     Leggi insieme tabella + audit: l''intervento che abbassa il L2/SEE logit-ratio' -ForegroundColor Yellow
    Write-Host '     E il topBi/altLp senza far crollare il BPB indica la direzione. Se scale/dropout' -ForegroundColor Yellow
    Write-Host '     riducono il ratio ma non bastano: prossimo = L2 logit-contribution CAP (44.E),' -ForegroundColor Yellow
    Write-Host '     il lever non-lineare hard non ancora testato.' -ForegroundColor Yellow
} else {
    Write-Host '  -> Frenare il readout su L2 fa ricrollare il BPB sotto soglia.' -ForegroundColor Yellow
    Write-Host '     Diagnosi: il guadagno BPB di L2 e inseparabile dal suo peso eccessivo nei logits.' -ForegroundColor Yellow
    Write-Host '     Il readout lineare non puo usare L2 a meta. Prossimo: NON readout lineare su L2' -ForegroundColor Yellow
    Write-Host '     (gating non-lineare del contributo L2) oppure L2 trainable/structured projection.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host '  Domanda di fase: L2 porta contesto utile, ma quanto deve pesare nei logits'
Write-Host '  per non dominare il closed-loop? L''audit L2/SEE logit-ratio e la misura diretta.'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
