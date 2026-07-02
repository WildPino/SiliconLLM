# Phase 44.I - Endogenous L2 Trust Fatigue (inference-only) Tribunal
#
# 44.H read: L2 is a CO-CAUSE but simple "L2 logit gating" is not proven enough.
# Warnings fired on normal bigrams (diagnostic sentinels, not acceptable triggers);
# flip at birth was not high; freeze/reset/cap0.5 tied or beat l2zero. So L2 is a
# PERSISTENT CONTEXTUAL PRESSURE narrowing the basin, not just "too much voice".
#
# 44.I policy uses ONLY internal signals (no word counters): L2/SEE disagreement,
# low prob-margin, gamma saturation, L2-state stress. Inference-only first; if a
# variant passes T=0.65 AND T=0.55, only THEN train-under-fatigue (44.J).
#
# Variants (base D1 = phase44f_F0, and D1_cap20inf = phase44f_D1_cap20inf):
#   I1 flip-fatigue w16   I2 flip-fatigue w32   (reduce L2 trust after low-margin disagreement)
#   I3 low-margin cap-tighten (gamma-sat + low margin -> temporary 0.5% cap)
#   I4 state-fatigue (sustained internal stress -> brief L2 STATE damping, not zero)
#
# Gate unchanged: BPB<=2.2543 + topBi<=8 + altLp<=2 + nameWst<=20 + runWst<=5
#   (self in [0.8,2.0]). BPB is the base model's (fatigue is generation-only).
#
# Run:  .\benchmarks\phase38-42\phase44i_fatigue.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase44i'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
$F_LOG   = $ROOT + '\results\phase44f\phase44f_train.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$C2A_BPB = 2.2593
$D1_W    = $WDIR + '\phase44f_F0.bin'
$C20_W   = $WDIR + '\phase44f_D1_cap20inf.bin'
foreach ($w in @($C2A_W,$D1_W,$C20_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weights: $w (run phase44f_captrain.ps1 first)"; exit 1 } }

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

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

$I1 = '--fatigue flip --fat-window 16 --fat-trust 0.5 --fat-margin 0.15'
$I2 = '--fatigue flip --fat-window 32 --fat-trust 0.5 --fat-margin 0.15'
$I3 = '--fatigue lowmargin --fat-window 16 --fat-cap 0.005 --fat-gammalow 0.85 --fat-margin 0.15'
$I4 = '--fatigue state --fat-damp 0.85 --fat-stress 4 --fat-margin 0.15'

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING (generator with --fatigue) ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Base BPB from the 44.F train log ------------------------------------
function Get-FBPB([string]$sfx) {
    if (-not (Test-Path $F_LOG)) { return $null }
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $F_LOG | Select-Object -Last 1
    if ($line) { return [double]((($line.Line -replace '.*BPB=','') -replace '\s.*','').Trim()) }
    return $null
}
$D1_BPB  = Get-FBPB '_F0.bin'
$C20_BPB = Get-FBPB '_D1_cap20inf.bin'

# label, exe, weights, extra-args, valbpb, ref
$gen_configs = @(
    @{ label='C2.A';      exe='phase43_generator.exe'; weights=$C2A_W; extra=''; valbpb=$C2A_BPB; ref=$true },
    @{ label='D1';        exe='phase44_generator.exe'; weights=$D1_W;  extra='';   valbpb=$D1_BPB;  ref=$false },
    @{ label='D1cap20';   exe='phase44_generator.exe'; weights=$C20_W; extra='';   valbpb=$C20_BPB; ref=$false },
    @{ label='D1_I1';     exe='phase44_generator.exe'; weights=$D1_W;  extra=$I1;  valbpb=$D1_BPB;  ref=$false },
    @{ label='D1_I2';     exe='phase44_generator.exe'; weights=$D1_W;  extra=$I2;  valbpb=$D1_BPB;  ref=$false },
    @{ label='C20_I1';    exe='phase44_generator.exe'; weights=$C20_W; extra=$I1;  valbpb=$C20_BPB; ref=$false },
    @{ label='C20_I3';    exe='phase44_generator.exe'; weights=$C20_W; extra=$I3;  valbpb=$C20_BPB; ref=$false },
    @{ label='C20_I4';    exe='phase44_generator.exe'; weights=$C20_W; extra=$I4;  valbpb=$C20_BPB; ref=$false }
)

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
    if ($cfg.extra -ne '') { $argline += ' ' + $cfg.extra }
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
            Write-Host ('  launched {0,4}/{1}  {2,-10} T{3} r{4} s{5}' -f $launched,$total,$t.label,$t.tp,$t.rng,$t.si)
        }
        Start-Sleep -Milliseconds 150
        for ($k=$running.Count-1; $k -ge 0; $k--) { if ($running[$k].HasExited) { $running.RemoveAt($k) } }
    }
}
function FileMD5($p) { if (-not (Test-Path $p)) { return '' }; (Get-FileHash -Algorithm MD5 -Path $p).Hash }

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
    $ok = ($hs -ne '' -and $hs -eq $hp); if (-not $ok) { $repro_ok = $false }
    Write-Host ('  {0,-10} s{1}  seq={2}  par={3}  {4}' -f $seq_tasks[$i].label,$seq_tasks[$i].si,$hs.Substring(0,[math]::Min(8,$hs.Length)),$hp.Substring(0,[math]::Min(8,$hp.Length)),($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
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
    Write-Host ('  {0,-10} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f 'config','valBPB','dBPB','nameWst','runWst','topBi','altLp','selfBPB','gate')
    Write-Host ('  ' + '-'*82)
    foreach ($cfg in $gen_configs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-10} [no data]' -f $cfg.label); continue }
        $w = Worst $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        $gate = if ($cfg.ref) {'ref'} elseif (Passes $cfg $tp) {'PASS'} else {'fail'}
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'Gray'} else {'DarkCyan'}
        Write-Host ('  {0,-10} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
}

# -------- Cross-temperature stability + verdict -------------------------------
Write-Host ''
Write-Host '=== Cross-temperature stability (pass at BOTH T=0.65 AND T=0.55?) ===' -ForegroundColor Yellow
Write-Host ('  {0,-10} {1,7} {2,7} {3,9}' -f 'config','T0.65','T0.55','valBPB')
$stable=@()
foreach ($cfg in $gen_configs) {
    if ($cfg.ref) { continue }
    $p65 = Passes $cfg '0.65'; $p55 = Passes $cfg '0.55'
    $both = $p65 -and $p55
    $col = if ($both) {'Green'} elseif ($p65) {'DarkCyan'} else {'Gray'}
    Write-Host ('  {0,-10} {1,7} {2,7} {3,9}' -f $cfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),('{0:F4}' -f $cfg.valbpb)) -ForegroundColor $col
    if ($both -and ($cfg.label -ne 'D1') -and ($cfg.label -ne 'D1cap20')) { $stable += $cfg.label }
}

Write-Host ''
Write-Host '=== Verdetto 44.I ===' -ForegroundColor Yellow
if ($stable.Count -gt 0) {
    Write-Host ('  -> CANDIDATO inference-only: {0} passa T=0.65 E T=0.55' -f ($stable -join ', ')) -ForegroundColor Green
    Write-Host '     La fatigue endogena (segnali interni, NON bigram) stabilizza il closed-loop.' -ForegroundColor Green
    Write-Host '     SOLO ORA si fa train-under-fatigue (44.J) per consolidare il guadagno e per' -ForegroundColor Green
    Write-Host '     recuperare BPB. Conferma con micro-fase multi-seed/temp prima di promuovere.' -ForegroundColor Green
} else {
    Write-Host '  -> Nessuna fatigue inference-only passa entrambi i temp. Leggi la tabella:' -ForegroundColor Yellow
    Write-Host '     - flip (I1/I2) migliora ma non basta: il problema non e la voce istantanea L2.' -ForegroundColor Yellow
    Write-Host '     - state (I4) abbassa topBi piu di flip: confermata la pressione contestuale L2' -ForegroundColor Yellow
    Write-Host '       persistente -> 44.J = train-under-state-fatigue / damping endogeno nel substrate.' -ForegroundColor Yellow
    Write-Host '     - nessuna muove topBi a T0.55: l attrattore non passa da L2 (logit ne stato)' -ForegroundColor Yellow
    Write-Host '       -> SEE-/sampling-side, rivedere substrate o sampling, non L2.' -ForegroundColor Yellow
    Write-Host '     Scegli la variante con miglior trend e portala a train-under-fatigue mirato.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host '  Principio 44.I: la policy nasce da disagreement L2/SEE, low margin, gamma'
Write-Host '  saturation o stress di stato L2 - MAI da contatori di parole (quelli solo per verifica).'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
