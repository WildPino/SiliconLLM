# Phase 47.A0 - sanity gauntlet for the static nonlinear readout, then word-gate
#
# 47.0 found mlp-hard 2.0947 on stable [SEE|L2_D1] feats (vs D1 2.2410) - too big to trust
# without falsification. Stage 1 runs the gauntlet (phase47a0_gauntlet.exe):
#   [1] reproduce on 3 val windows  [2] linear in same pipeline ~= D1
#   [3] random-label / shuffled-time controls show NO gain  [4] SEE/L2/both ablations
#   [5] width ladder H=16/32/64/128  [6] logit-RMS / entropy / maxp telemetry
# Stage 2 (ONLY if sanity passes): closed-loop word-gate on the smallest admissible H
# (phase47_generator.exe), refs C2.A + D1. Promotion still requires BOTH T0.65 AND T0.55.
#
# Run:  .\benchmarks\phase38-42\phase47_a0.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase47a0'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'              # D1 control (mix0.5 scale0.5), ~2.2522
$DELTA_W = $WDIR + '\phase45c_C0_delta.bin'        # delta control 0x53454543 (teacher ref), ~2.2212
$B3_W    = $WDIR + '\phase46b_B3_punctlm30.bin'    # 46.B best BPB (teacher ref), ~2.2509
$WP      = $WDIR + '\phase47a0'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
foreach ($w in @($C2A_W,$D1_W,$DELTA_W,$B3_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }

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
$PROMO_BPB = $C2A_BPB - 0.005
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
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47_a0_gauntlet.c' @SRC_CORE -o "$BINDIR\phase47a0_gauntlet.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47_a0_gauntlet'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47_generator.c' @SRC_CORE -o "$BINDIR\phase47_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps','phase44f_captrain','phase45a_relmove','phase45b_geometry','phase45c_gate','phase46a_l3','phase46b_l3','phase47a0_gauntlet')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Aborting.')
    exit 1
}

# -------- Stage 1: gauntlet ----------------------------------------------------
Write-Host ''
Write-Host '=== Phase 47.A0 - sanity gauntlet (9 probes, 4 windows x 1M, ~5 GB RAM) ===' -ForegroundColor Yellow
$GAUNT_OUT = $RDIR + '\phase47a0_gauntlet.txt'
& "$BINDIR\phase47a0_gauntlet.exe" $TS_DATA $D1_W $DELTA_W $B3_W $WP 2>&1 | Tee-Object $GAUNT_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Gauntlet failed'; exit 1 }
Write-Host 'Gauntlet complete.' -ForegroundColor Green

# -------- Parse gauntlet output -------------------------------------------------
$BASE = @{}   # win -> @{tri,d1,delta,b3}
foreach ($line in Select-String '^BASE ' $GAUNT_OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
    $BASE[$kv['win']] = $kv
}
$PROBE = @{}  # name -> @{h,val1,val2,val3,rms1,ent1,maxp1}
foreach ($line in Select-String '^PROBE ' $GAUNT_OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
    $PROBE[$kv['name']] = $kv
}
if ($BASE.Count -lt 4 -or -not $PROBE.ContainsKey('mlpH128')) { Write-Error 'Gauntlet output incomplete'; exit 1 }

# -------- Sanity verdict --------------------------------------------------------
Write-Host ''
Write-Host '=== Sanity verdict ===' -ForegroundColor Yellow
$wins = @('val1','val2','val3')
$checks = @()
foreach ($w in $wins) {
    $d1 = [double]$BASE[$w]['d1']; $i = [array]::IndexOf($wins,$w)+1
    $m128 = [double]$PROBE['mlpH128']["val$i"]
    $rl   = [double]$PROBE['randlabel']["val$i"]
    $st   = [double]$PROBE['shuftime']["val$i"]
    # 47.A0b: l'ancora e' frozenD1 (D1 nel path probe), NON il linear scratch (budget-dipendente)
    if ($PROBE.ContainsKey('frozenD1')) {
        $fz = [double]$PROBE['frozenD1']["val$i"]
        $checks += [PSCustomObject]@{ name="anchor@$w"; ok=([math]::Abs($fz-$d1) -le 0.005); detail=('frozenD1 {0:F4} vs BASE d1 {1:F4}' -f $fz,$d1) }
    }
    $checks += [PSCustomObject]@{ name="reproduce@$w";  ok=($m128 -le $d1-0.05); detail=('mlpH128 {0:F4} vs d1 {1:F4}' -f $m128,$d1) }
    $checks += [PSCustomObject]@{ name="randlabel@$w";  ok=($rl -ge $d1-0.01); detail=('randlabel {0:F4} vs d1 {1:F4}' -f $rl,$d1) }
    $checks += [PSCustomObject]@{ name="shuftime@$w";   ok=($st -ge $d1-0.01); detail=('shuftime {0:F4} vs d1 {1:F4}' -f $st,$d1) }
}
$sane = $true
foreach ($c in $checks) {
    if (-not $c.ok) { $sane = $false }
    Write-Host ('  {0,-18} {1,-6} {2}' -f $c.name,($(if($c.ok){'OK'}else{'FAIL'})),$c.detail) -ForegroundColor $(if($c.ok){'Green'}else{'Red'})
}
Write-Host ''
Write-Host '=== Ablations / ladder ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f 'probe','val1','val2','val3','rms1','ent1','maxp1')
foreach ($n in @('frozenD1','linD1init','linear','ablSEE','ablL2','mlpH16','mlpH32','mlpH64','mlpH128','randlabel','shuftime')) {
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]
    Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f $n,$p['val1'],$p['val2'],$p['val3'],$p['rms1'],$p['ent1'],$p['maxp1'])
}

if (-not $sane) {
    Write-Host ''
    Write-Host '  -> SANITY FAIL: il 2.09 non sopravvive ai controlli. NO word-gate. Leggi i FAIL sopra:' -ForegroundColor Red
    Write-Host '     - reproduce FAIL: il guadagno non generalizza (era window-specific / overfit residuo).' -ForegroundColor Red
    Write-Host '     - linear FAIL: la pipeline Adam/batching non riproduce D1 -> bug di pipeline, non nonlinearita''.' -ForegroundColor Red
    Write-Host '     - randlabel/shuftime FAIL: leakage di feature/target -> il 2.09 e'' un artefatto.' -ForegroundColor Red
    exit 1
}
Write-Host ''
Write-Host '  -> SANITY PASS: nonlinearita'' statica confermata su 3 finestre, controlli puliti.' -ForegroundColor Green

# -------- Pick the smallest admissible H ----------------------------------------
$ladder = @('mlpH16','mlpH32','mlpH64','mlpH128')
$avgs = @{}
foreach ($n in $ladder) { $p=$PROBE[$n]; $avgs[$n] = ([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0 }
$best = ($avgs.Values | Measure-Object -Minimum).Minimum
$pick = $null
foreach ($n in $ladder) { if ($avgs[$n] -le $best+0.02) { $pick=$n; break } }   # smallest within 0.02
$pickH = [int]($pick -replace 'mlpH','')
$MLP_W = $WP + '_' + $pick + '.bin'
$PICK_BPB = [math]::Round($avgs[$pick],4)
Write-Host ('  Best avg BPB {0:F4}; smallest admissible: {1} (avg {2:F4}) -> word-gate' -f $best,$pick,$avgs[$pick]) -ForegroundColor Cyan
if (-not (Test-Path $MLP_W)) { Write-Error "Saved probe missing: $MLP_W"; exit 1 }

# -------- Word metrics ----------------------------------------------------------
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

# -------- Generate (parallel, deterministic-identical) --------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

# wargs = full weight argument string (phase47_generator takes TWO weight files)
$gen_configs = @(
    @{ label='C2.A';      exe='phase43_generator.exe'; wargs=('"'+$C2A_W+'"');            wfile=$C2A_W; valbpb=$C2A_BPB;  ref=$true  },
    @{ label='D1';        exe='phase44_generator.exe'; wargs=('"'+$D1_W+'"');             wfile=$D1_W;  valbpb=$D1_BPB;   ref=$true  },
    @{ label=('mlpH'+$pickH); exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$MLP_W+'"'); wfile=$MLP_W; valbpb=$PICK_BPB; ref=$false }
)

$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{}; foreach ($tp in $TEMPS) { $agg[$cfg.label][$tp] = @{ mets=@(); bpbs=@() } } }

function New-GenTask($cfg,$tp,$rng,$off,$si,$suffix) {
    $lbl = $cfg.label+'_T'+$tp+'_r'+$rng+'_s'+$si
    $tf  = $RDIR+'\gen_'+$lbl+$suffix+'.txt'
    $sf  = $RDIR+'\gen_'+$lbl+$suffix+'_stats.txt'
    $argline = ('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                $TS_DATA,$cfg.wargs,$GEN_LEN,$WARMUP,$tp,$off,$rng)
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

# -------- Repro pre-check -------------------------------------------------------
Write-Host ''
Write-Host '=== Repro pre-check: parallel vs sequential (2 cfg x 2 seed) ===' -ForegroundColor Yellow
$chk_cfgs = @($gen_configs | Where-Object { (Test-Path $_.wfile) } | Select-Object -Last 2)   # D1 + mlp
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

# -------- Full parallel word-gate ------------------------------------------------
$tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS) { $si=0
    foreach ($off in $SEEDS) { $si++
        foreach ($cfg in $gen_configs) { if (Test-Path $cfg.wfile) { $tasks += New-GenTask $cfg $tp $rng $off $si '' } }
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

# -------- Tables ------------------------------------------------------------------
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

# -------- Closed-loop telemetry audit (mlp config, T=0.55) -------------------------
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); [math]::Round($s[$idx],4) }
function MeanD([double[]]$a){ if($a.Count -eq 0){return 0.0}; [math]::Round(($a|Measure-Object -Average).Average,4) }

Write-Host ''
Write-Host '======== Closed-loop telemetry (mlp, T=0.55): attractor signature check ========' -ForegroundColor Cyan
$mcfg = $gen_configs | Where-Object { -not $_.ref } | Select-Object -First 1
$rms=@(); $ent=@(); $mxp=@()
$tel_off = @($SEEDS | Select-Object -First $TEL_SEEDS)
$si=0
foreach ($off in $tel_off) {
    $si++
    $tel = $RDIR + '\tel_' + $mcfg.label + '_s' + $si + '.tsv'
    $sf  = $RDIR + '\tel_' + $mcfg.label + '_s' + $si + '_stats.txt'
    $al = ('"{0}" {1} --gen-len {2} --warmup {3} --temp 0.55 --seed-start {4} --rng-seed {5} --telemetry "{6}"' -f `
           $TS_DATA,$mcfg.wargs,$GEN_LEN,$WARMUP,$off,$RNGS[0],$tel)
    Start-Process -FilePath "$BINDIR\phase47_generator.exe" -ArgumentList $al `
        -RedirectStandardOutput ($RDIR+'\_null.bin') -RedirectStandardError $sf -NoNewWindow -PassThru -Wait | Out-Null
    if (-not (Test-Path $tel)) { continue }
    foreach ($line in [System.IO.File]::ReadLines($tel)) {
        if ($line.Length -eq 0 -or $line[0] -eq 's') { continue }
        $f = $line -split "`t"; if ($f.Count -lt 7) { continue }
        $rms += [double]$f[3]; $ent += [double]$f[4]; $mxp += [double]$f[5]
    }
}
Write-Host ('  mlp logitRMS  p50={0} p90={1}   (gauntlet TF rms1: {2})' -f (Pctl $rms 0.5),(Pctl $rms 0.9),$PROBE[$pick]['rms1'])
Write-Host ('  out entropy   p50={0} p10={1}   (gauntlet TF ent1: {2})' -f (Pctl $ent 0.5),(Pctl $ent 0.1),$PROBE[$pick]['ent1'])
Write-Host ('  max prob      p50={0} p90={1}   (gauntlet TF maxp1: {2})' -f (Pctl $mxp 0.5),(Pctl $mxp 0.9),$PROBE[$pick]['maxp1'])
Write-Host '  (closed-loop RMS >> TF RMS o entropy p10 ~ 0 = readout overconfident fuori distribuzione -> attrattore.)' -ForegroundColor Gray

# -------- Verdict -------------------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 47.A0 ===' -ForegroundColor Yellow
$p65 = Passes $mcfg '0.65'; $p55 = Passes $mcfg '0.55'
Write-Host ('  {0}: T0.65={1}  T0.55={2}  valBPB={3:F4} (D1 {4:F4}, delta {5})' -f $mcfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),$mcfg.valbpb,$D1_BPB,'2.2212')
if ($p65 -and $p55) {
    Write-Host '  -> Il primo readout NONLINEARE STATICO del progetto e'' AMMISSIBILE: comprime ~2.09 e' -ForegroundColor Green
    Write-Host '     genera stabile a entrambe le temperature. La nonlinearita'' statica rompe il muro' -ForegroundColor Green
    Write-Host '     compressione-vs-generazione. -> aprire 47.A (promozione formale + regularization study).' -ForegroundColor Green
} else {
    Write-Host '  -> Comprime ma non genera stabile: stessa parete di 44/45/46, ora SENZA memoria volatile.' -ForegroundColor Yellow
    Write-Host '     Leggi la telemetry closed-loop: se RMS esplode / entropy collassa fuori distribuzione,' -ForegroundColor Yellow
    Write-Host '     il decoder e'' overconfident OOD -> 47.B regularized static decoder (weight decay,' -ForegroundColor Yellow
    Write-Host '     logit/spectral norm, label smoothing, dropout train-time). NON tornare a L2/L3.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
