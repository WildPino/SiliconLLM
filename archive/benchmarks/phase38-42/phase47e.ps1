# Phase 47.E - Capacity Transfer della ricetta D16 (H64)
#
# 47.D = PRIMO breach del muro strutturale: D16_h32/D16nc_h32 prime config di sempre sotto
# topBi<=8 a entrambi i temp con selfBPB sano; il fail si e' SPOSTATO da struttura a BPB
# (+1 altLp). K=16 punto dolce (8 no rientro, 32 degenera), lamC=0.02 assicurazione gratis.
# Perche' H64: aritmetica - H32 clean ceiling 2.2360 vs bar 2.2543 con costo rollout ~+0.04
# = no headroom; H64 clean 2.1627 = ~0.09 margine. LA domanda: il fix strutturale
# sopravvive alla capacita' maggiore o H64 "compartimentalizza" la lezione del rollout?
#
# 4 config: D0_h32/D0_h64 (clean 6ep, de-confound budget, TF-only NO word-gate),
# D16_h64 (primaria, ricetta identica), D16r3_h64 (asse round).
# Mini gate: topBi(T0.55) < mlpH64 ref E selfBPB sano a entrambi i temp -> FULL.
# Ref full: C2.A, D1, mlpH64, D16_h32 (cross-capacita' = il dato chiave).
# Da NON fare: K>16, mix 30%, altri sweep H32, L2/L3, commit.
#
# Run:  .\benchmarks\phase38-42\phase47e.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase47e'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$WP      = $WDIR + '\phase47e'
$A0_H64  = $WDIR + '\phase47a0_mlpH64.bin'      # unregularized clean ref @ H64
$D16_W   = $WDIR + '\phase47d_D16_h32.bin'      # 47.D structural-fix ref (cross-capacity)
$OLD_A0  = $ROOT + '\results\phase47a0\phase47a0_gauntlet.txt'
$OLD_D   = $ROOT + '\results\phase47d\phase47d_train.txt'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
foreach ($w in @($C2A_W,$D1_W,$A0_H64,$D16_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }
if (-not (Test-Path $OLD_A0)) { Write-Error "Missing A0 gauntlet output: $OLD_A0"; exit 1 }
if (-not (Test-Path $OLD_D))  { Write-Error "Missing 47.D train output: $OLD_D"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS_MINI = 4
$RNGS_MINI   = @(12345)
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
$TFCFGS  = @('D0_h32','D0_h64','D16_h64','D16r3_h64')   # tabella TF (tutte)
$GENCFGS = @('D16_h64','D16r3_h64')                     # word-gate (D0 = TF-only)

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47e_capacity.c' @SRC_CORE -o "$BINDIR\phase47e_capacity.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47e_capacity'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47_generator.c' @SRC_CORE -o "$BINDIR\phase47_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps','phase44f_captrain','phase45a_relmove','phase45b_geometry','phase45c_gate','phase46a_l3','phase46b_l3','phase47a0_gauntlet','phase47b_decoder','phase47c_robust','phase47d_rollout','phase47e_capacity')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Aborting.')
    exit 1
}

# -------- Stage 1: capacity-transfer training (4 config) -------------------------
Write-Host ''
Write-Host '=== Phase 47.E - capacity transfer ricetta D16 (D0_h32/D0_h64 clean + D16_h64 + D16r3_h64) ===' -ForegroundColor Yellow
$TRAIN_OUT = $RDIR + '\phase47e_train.txt'
& "$BINDIR\phase47e_capacity.exe" $TS_DATA $D1_W $WP 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host 'Training complete.' -ForegroundColor Green

# -------- Parse ---------------------------------------------------------------------
function Parse-KV([string]$file,[string]$prefix) {
    $out = @{}
    foreach ($line in Select-String ('^'+$prefix+' ') $file) {
        $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
        $key = if ($prefix -eq 'BASE') { $kv['win'] } else { $kv['name'] }
        $out[$key] = $kv
    }
    return $out
}
$BASE  = Parse-KV $TRAIN_OUT 'BASE'
$PROBE = Parse-KV $TRAIN_OUT 'PROBE'
$A0P   = Parse-KV $OLD_A0  'PROBE'
$DP    = Parse-KV $OLD_D   'PROBE'
if (-not $PROBE.ContainsKey('frozenD1') -or -not $PROBE.ContainsKey('D16r3_h64')) { Write-Error 'Train output incomplete'; exit 1 }
if (-not $DP.ContainsKey('D16_h32')) { Write-Error '47.D D16_h32 probe missing from old output'; exit 1 }

# -------- Anchor check ----------------------------------------------------------------
Write-Host ''
Write-Host '=== Anchor (frozenD1 nel path probe) ===' -ForegroundColor Yellow
$wins = @('val1','val2','val3'); $anchor_ok = $true
foreach ($w in $wins) {
    $d1 = [double]$BASE[$w]['d1']; $i = [array]::IndexOf($wins,$w)+1
    $fz = [double]$PROBE['frozenD1']["val$i"]
    $ok = ([math]::Abs($fz-$d1) -le 0.005); if (-not $ok) { $anchor_ok = $false }
    Write-Host ('  anchor@{0}  {1}  frozenD1 {2:F4} vs BASE d1 {3:F4}' -f $w,($(if($ok){'OK'}else{'FAIL'})),$fz,$d1) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $anchor_ok) { Write-Error 'Anchor FAIL: path probe non integro. Aborting.'; exit 1 }

# -------- TF table + rollout diagnostic + de-confound ------------------------------------
Write-Host ''
Write-Host '=== TF probes + roll per round (trend verso 1.5-2 = meno degenerativo; in discesa = collasso) ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,7} {6,7} {7,7} {8,8} {9,8}' -f 'probe','val1','val2','val3','valC','roll1','roll2','roll3','rms1','ent1')
foreach ($n in (@('frozenD1') + $TFCFGS)) {
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]
    $r1 = if ($p.ContainsKey('roll1')) { $p['roll1'] } else { '-' }
    $r2 = if ($p.ContainsKey('roll2')) { $p['roll2'] } else { '-' }
    $r3 = if ($p.ContainsKey('roll3')) { $p['roll3'] } else { '-' }
    Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,7} {6,7} {7,7} {8,8} {9,8}' -f $n,$p['val1'],$p['val2'],$p['val3'],$p['valC'],$r1,$r2,$r3,$p['rms1'],$p['ent1'])
}
function AvgVal($p) { return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }
Write-Host ''
Write-Host '=== De-confound retroattivo 47.D + costo rollout H64 ===' -ForegroundColor Yellow
$d16h32_tf = AvgVal $DP['D16_h32']
$d0h32_tf  = AvgVal $PROBE['D0_h32']
$d0h64_tf  = AvgVal $PROBE['D0_h64']
$d16h64_tf = AvgVal $PROBE['D16_h64']
Write-Host ('  H32: D0 clean 6ep {0:F4} vs 47.D D16 {1:F4}  -> costo rollout H32 = {2:+0.0000;-0.0000;0.0000}' -f $d0h32_tf,$d16h32_tf,($d16h32_tf-$d0h32_tf))
Write-Host ('  H64: D0 clean 6ep {0:F4} vs D16_h64  {1:F4}  -> costo rollout H64 = {2:+0.0000;-0.0000;0.0000}  (bar BPB={3:F4})' -f $d0h64_tf,$d16h64_tf,($d16h64_tf-$d0h64_tf),$PROMO_BPB)

# -------- Gen configs --------------------------------------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

$gen_configs = @(
    @{ label='C2.A'; exe='phase43_generator.exe'; wargs=('"'+$C2A_W+'"'); wfile=$C2A_W; valbpb=$C2A_BPB; ref=$true },
    @{ label='D1';   exe='phase44_generator.exe'; wargs=('"'+$D1_W+'"');  wfile=$D1_W;  valbpb=$D1_BPB;  ref=$true },
    @{ label='mlpH64'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$A0_H64+'"'); wfile=$A0_H64; valbpb=(AvgVal $A0P['mlpH64']); ref=$true },
    @{ label='D16_h32'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$D16_W+'"'); wfile=$D16_W; valbpb=$d16h32_tf; ref=$true }
)
foreach ($n in $GENCFGS) {
    if (-not $PROBE.ContainsKey($n)) { Write-Error "Probe $n missing"; exit 1 }
    $wf = $WP + '_' + $n + '.bin'
    if (-not (Test-Path $wf)) { Write-Error "Saved probe missing: $wf"; exit 1 }
    $gen_configs += @{ label=$n; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$wf+'"'); wfile=$wf; valbpb=(AvgVal $PROBE[$n]); ref=$false }
}

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
function FileMD5($p) {
    if (-not (Test-Path $p)) { return '' }
    for ($a=0; $a -lt 15; $a++) { try { return (Get-FileHash -Algorithm MD5 -Path $p -ErrorAction Stop).Hash } catch { Start-Sleep -Milliseconds 200 } }
    return ''
}
function Aggregate($tasks,$agg) {
    foreach ($t in $tasks) {
        if (-not (Test-Path $t.tf)) { Write-Host ('  MISSING ' + $t.tf) -ForegroundColor Red; continue }
        $m = Get-WordMetrics $t.tf
        if ($m) { $agg[$t.label][$t.tp].mets += $m }
        $bl = Select-String 'self_BPB' $t.sf | Select-Object -Last 1
        if ($bl) { $agg[$t.label][$t.tp].bpbs += [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    }
}
function Worst($agg,$lbl,$tp) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $bs=@()
    foreach ($m in @($agg[$lbl][$tp].mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop }
    foreach ($b in @($agg[$lbl][$tp].bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni=(Mx $ni); mr=(Mx $mr); bi=(Mx $bi); al=(Mx $al); self=(Av $bs) }
}

# -------- Repro pre-check -------------------------------------------------------
Write-Host ''
Write-Host '=== Repro pre-check: parallel vs sequential (2 cfg x 2 seed) ===' -ForegroundColor Yellow
$chk_cfgs = @(($gen_configs | Where-Object { -not $_.ref } | Select-Object -First 1), ($gen_configs | Where-Object { $_.label -eq 'D1' }))
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

# -------- Stage 2: MINI word-gate ------------------------------------------------------
$SEEDS_MINI = @()
for ($i = 0; $i -lt $NSEEDS_MINI; $i++) { $SEEDS_MINI += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS_MINI)) }
$mini_agg = @{}
foreach ($cfg in $gen_configs) { $mini_agg[$cfg.label] = @{}; foreach ($tp in $TEMPS) { $mini_agg[$cfg.label][$tp] = @{ mets=@(); bpbs=@() } } }
$mini_tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS_MINI) { $si=0
    foreach ($off in $SEEDS_MINI) { $si++
        foreach ($cfg in $gen_configs) { $mini_tasks += New-GenTask $cfg $tp $rng $off $si '_mini' }
    } } }
Write-Host ''
Write-Host ('=== MINI word-gate: {0} runs ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
Aggregate $mini_tasks $mini_agg

Write-Host ''
Write-Host '======== MINI (criterio esplicito: topBi(T0.55) < mlpH64 ref E selfBPB sano a entrambi i temp) ========' -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,7} {2,7} {3,7} {4,7} {5,8} {6,8} {7,7}' -f 'config','bi65','bi55','al55','ni55','self65','self55','promo')
$promoted = @()
foreach ($cfg in $gen_configs) {
    $w65 = Worst $mini_agg $cfg.label '0.65'; $w55 = Worst $mini_agg $cfg.label '0.55'
    $promo = '-'
    if (-not $cfg.ref) {
        $r65 = Worst $mini_agg 'mlpH64' '0.65'; $r55 = Worst $mini_agg 'mlpH64' '0.55'
        $self_ok = ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -ge $BPB_LO) -and ($w55.self -le $BPB_HI)
        $bi_ok   = ($w55.bi -lt $r55.bi) -and ($w65.bi -le $r65.bi)
        if ($self_ok -and $bi_ok) { $promo = 'FULL'; $promoted += $cfg } else { $promo = 'stop' }
    }
    $col = if ($promo -eq 'FULL') {'Green'} elseif ($promo -eq 'stop') {'Gray'} else {'DarkCyan'}
    Write-Host ('  {0,-12} {1,7} {2,7} {3,7} {4,7} {5,8} {6,8} {7,7}' -f `
        $cfg.label,('{0:F0}' -f $w65.bi),('{0:F0}' -f $w55.bi),('{0:F0}' -f $w55.al),('{0:F1}' -f $w55.ni),('{0:F2}' -f $w65.self),('{0:F2}' -f $w55.self),$promo) -ForegroundColor $col
}

if ($promoted.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 47.E (mini) ===' -ForegroundColor Yellow
    Write-Host '  -> La struttura NON trasferisce a H64 al mini gate (ramo 3 albero pre-registrato):' -ForegroundColor Yellow
    Write-Host '     insight = la capacita'' COMPARTIMENTALIZZA la lezione del rollout.' -ForegroundColor Yellow
    Write-Host '     Fallback (decisione utente) = recovery BPB su H32: piu'' round / anneal mix' -ForegroundColor Yellow
    Write-Host '     20->10% / piu'' epoche pretrain clean. Phrase-scale SOLO se muore anche quello.' -ForegroundColor Yellow
    Write-Host '     NON fare: K>16, mix 30%, riaprire L2/L3.' -ForegroundColor Yellow
    Write-Host ''
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}

# -------- Stage 3: FULL word-gate --------------------------------------------------------
$full_cfgs = @($gen_configs | Where-Object { $_.ref }) + $promoted
$agg = @{}
foreach ($cfg in $full_cfgs) { $agg[$cfg.label] = @{}; foreach ($tp in $TEMPS) { $agg[$cfg.label][$tp] = @{ mets=@(); bpbs=@() } } }
$tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS) { $si=0
    foreach ($off in $SEEDS) { $si++
        foreach ($cfg in $full_cfgs) { $tasks += New-GenTask $cfg $tp $rng $off $si '' }
    } } }
Write-Host ''
Write-Host ('=== FULL word-gate: {0} promosse ({1}), {2} runs ===' -f $promoted.Count,(($promoted | ForEach-Object { $_.label }) -join ', '),$tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $tasks $THROTTLE
Aggregate $tasks $agg

function Passes($cfg,$tp) {
    if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { return $false }
    $w = Worst $agg $cfg.label $tp
    return ($null -ne $cfg.valbpb) -and ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
}

foreach ($tp in $TEMPS) {
    Write-Host ''
    Write-Host ('======== T={0} : worst-case over 32 samples (D16_h32 ref = confronto cross-capacita'') ========' -f $tp) -ForegroundColor Cyan
    Write-Host ('  {0,-12} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f 'config','valBPB','dC2.A','nameWst','runWst','topBi','altLp','selfBPB','gate')
    Write-Host ('  ' + '-'*84)
    foreach ($cfg in $full_cfgs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-12} [no data]' -f $cfg.label); continue }
        $w = Worst $agg $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        $gate = if ($cfg.ref) {'ref'} elseif (Passes $cfg $tp) {'PASS'} else {'fail'}
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'Gray'} else {'DarkCyan'}
        Write-Host ('  {0,-12} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
}

# -------- Closed-loop telemetry (promosse, T=0.55) -------------------------------------
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); [math]::Round($s[$idx],4) }

Write-Host ''
Write-Host '======== Closed-loop telemetry (promosse, T=0.55; sorvegliare ent_p10/maxp_p90 = storia overconfidence H64) ========' -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,9} {2,9} {3,9} {4,9} {5,9}   TF(rms/ent)' -f 'config','rms_p50','rms_p90','ent_p50','ent_p10','maxp_p90')
$tel_off = @($SEEDS | Select-Object -First $TEL_SEEDS)
foreach ($mcfg in $promoted) {
    $rms=@(); $ent=@(); $mxp=@(); $si=0
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
    $p = $PROBE[$mcfg.label]
    Write-Host ('  {0,-12} {1,9} {2,9} {3,9} {4,9} {5,9}   {6}/{7}' -f `
        $mcfg.label,(Pctl $rms 0.5),(Pctl $rms 0.9),(Pctl $ent 0.5),(Pctl $ent 0.1),(Pctl $mxp 0.9),$p['rms1'],$p['ent1'])
}

# -------- Verdict (albero pre-registrato) ------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 47.E ===' -ForegroundColor Yellow
$winners = @()
foreach ($mcfg in $promoted) {
    $p65 = Passes $mcfg '0.65'; $p55 = Passes $mcfg '0.55'
    $col = if ($p65 -and $p55) {'Green'} elseif ($p65) {'DarkCyan'} else {'Gray'}
    Write-Host ('  {0,-12} T0.65={1}  T0.55={2}  valBPB={3:F4}' -f $mcfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),$mcfg.valbpb)  -ForegroundColor $col
    if ($p65 -and $p55) { $winners += $mcfg.label }
}
if ($winners.Count -gt 0) {
    Write-Host ''
    Write-Host ('  -> RAMO 1: {0} passa il FULL gate a entrambi i temp = PRIMO candidato a promozione' -f ($winners -join ', ')) -ForegroundColor Green
    Write-Host '     nella storia del progetto. PRIMA del verdetto: validazione estesa (terzo rng,' -ForegroundColor Green
    Write-Host '     piu'' sample). Commit del blocco 47 SOLO a ordine utente.' -ForegroundColor Green
} else {
    Write-Host ''
    Write-Host '  -> Nessun PASS pieno. Albero pre-registrato:' -ForegroundColor Yellow
    Write-Host '     RAMO 2 (topBi<=8 + selfBPB sano ma BPB/altLp last-mile): micro-iterazione =' -ForegroundColor Yellow
    Write-Host '       D16r3 (gia'' in tabella) o mix 15%. K resta 16.' -ForegroundColor Yellow
    Write-Host '     RAMO 3 (topBi torna ~12+): la capacita'' compartimentalizza -> fallback recovery' -ForegroundColor Yellow
    Write-Host '       BPB su H32 (piu'' round / anneal mix 20->10% / piu'' pretrain clean).' -ForegroundColor Yellow
    Write-Host '     RAMO 4 (selfBPB55<0.8 a H64): scendi a mix 15%, NON toccare K.' -ForegroundColor Yellow
    Write-Host '     Decisione utente in ogni ramo. NON fare: K>16, mix30, L2/L3, commit.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
