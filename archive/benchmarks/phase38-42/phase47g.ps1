# Phase 47.G - Ultimo miglio H32 (esecuzione ramo 4 di 47.F)
#
# 47.F: ipotesi temperatura FALSIFICATA (r3Tmix vs r3: topBi55 13=13 con 50% pool a 0.55);
# asse round ROVESCIATO a H64 (topBi65 9->8->11, r3 ottimo strutturale); via H64 chiusa.
# D16r5_h32 = quasi-PASS storico: topBi 5/5, manca 0.0072 di BPB + 1 unita' altLp55.
# Watch: quiet-down coi round (selfBPB55 0.84->0.81 floor 0.8, ent_p50 0.378->0.351).
#
# 47.G: passo 0 = retro-gate checkpoint r3/r4 di 47.F (gratis, --eval nel trainer per BPB
# esatto); passo 1 = continuazione PLAIN r6->r9 mix 20% (from-scratch con prefisso r1-r5
# BIT-IDENTICO a 47.F, verificato MD5 — i checkpoint non hanno i momenti Adam, resume vero
# impossibile); passo 2 = ANNEAL mix 10% da r6 (branch in-memory dallo stato r5 completo);
# passo 3 = sonda H48 (opzionale, individua la soglia di formazione bacini 32-64).
# Mini-gate per checkpoint: topBi<=7 a ENTRAMBE le temp + selfBPB55>=0.8 + BPB<=2.2543
# -> FULL solo sui qualificati. Ref: C2.A/D1/D16_h32/D16r5_h32/D16r3_h64.
# Ramo 5 esplicito: altLp55 inchiodato a 3 con tutto verde -> riportato, gate NON toccato.
# Da non fare: K!=16, mix 30%, sweep quota 0.55, r7 H64, label smoothing, L2/L3, commit.
#
# Run:  .\benchmarks\phase38-42\phase47g.ps1
#       .\benchmarks\phase38-42\phase47g.ps1 -SkipTrain   # riusa train output + pesi
#       .\benchmarks\phase38-42\phase47g.ps1 -SkipH48     # taglia la sonda H48

param([switch]$SkipTrain,[switch]$SkipH48)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase47g'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$WP      = $WDIR + '\phase47g'
$D16_W   = $WDIR + '\phase47d_D16_h32.bin'         # 47.D breach ref
$R5H32_W = $WDIR + '\phase47f_D16r5_h32.bin'       # 47.F near-pass ref (= r5)
$R5H32_R5= $WDIR + '\phase47f_D16r5_h32_r5.bin'    # prefix-property MD5 target
$R5H32_R3= $WDIR + '\phase47f_D16r5_h32_r3.bin'    # retro-gate step 0
$R5H32_R4= $WDIR + '\phase47f_D16r5_h32_r4.bin'    # retro-gate step 0
$R3H64_W = $WDIR + '\phase47e_D16r3_h64.bin'       # 47.E PASS-T0.65 ref
$OLD_D   = $ROOT + '\results\phase47d\phase47d_train.txt'
$OLD_E   = $ROOT + '\results\phase47e\phase47e_train.txt'
$OLD_F   = $ROOT + '\results\phase47f\phase47f_train.txt'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
foreach ($w in @($C2A_W,$D1_W,$D16_W,$R5H32_W,$R5H32_R5,$R5H32_R3,$R5H32_R4,$R3H64_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }
foreach ($f in @($OLD_D,$OLD_E,$OLD_F)) { if (-not (Test-Path $f)) { Write-Error "Missing prior output: $f"; exit 1 } }

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
$MINI_BI = 7.0     # mini qualify: topBi <= 7 a ENTRAMBE le temp
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
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47g_lastmile.c' @SRC_CORE -o "$BINDIR\phase47g_lastmile.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47g_lastmile'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47_generator.c' @SRC_CORE -o "$BINDIR\phase47_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps','phase44f_captrain','phase45a_relmove','phase45b_geometry','phase45c_gate','phase46a_l3','phase46b_l3','phase47a0_gauntlet','phase47b_decoder','phase47c_robust','phase47d_rollout','phase47e_capacity','phase47f_tempcov','phase47g_lastmile')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Aborting.')
    exit 1
}

# -------- Stage 1: last-mile training -------------------------------------------
Write-Host ''
Write-Host '=== Phase 47.G - last mile H32 (P r9 plain + A anneal branch + retro-eval r3/r4 + H48) ===' -ForegroundColor Yellow
$TRAIN_OUT = $RDIR + '\phase47g_train.txt'
if ($SkipTrain -and (Test-Path $TRAIN_OUT)) {
    Write-Host ('SkipTrain: riuso ' + $TRAIN_OUT + ' e i pesi phase47g_* gia'' su disco.') -ForegroundColor Cyan
} else {
    $targs = @($TS_DATA,$D1_W,$WP,'--eval','F_r3',$R5H32_R3,'--eval','F_r4',$R5H32_R4)
    if ($SkipH48) { $targs += '--skip-h48' }
    & "$BINDIR\phase47g_lastmile.exe" @targs 2>&1 | Tee-Object $TRAIN_OUT
    if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
    Write-Host 'Training complete.' -ForegroundColor Green
}

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
$SAVED = Parse-KV $TRAIN_OUT 'SAVED'
$DP    = Parse-KV $OLD_D 'PROBE'
$EPb   = Parse-KV $OLD_E 'PROBE'
$FP    = Parse-KV $OLD_F 'PROBE'
if (-not $PROBE.ContainsKey('frozenD1') -or -not $PROBE.ContainsKey('P_r9')) { Write-Error 'Train output incomplete (P_r9 missing)'; exit 1 }

# -------- Anchor check ----------------------------------------------------------------
Write-Host ''
Write-Host '=== Anchor (frozenD1 nel path probe) ===' -ForegroundColor Yellow
$wins = @('val1','val2','val3'); $anchor_ok = $true
foreach ($w in $wins) {
    $d1v = [double]$BASE[$w]['d1']; $i = [array]::IndexOf($wins,$w)+1
    $fz = [double]$PROBE['frozenD1']["val$i"]
    $ok = ([math]::Abs($fz-$d1v) -le 0.005); if (-not $ok) { $anchor_ok = $false }
    Write-Host ('  anchor@{0}  {1}  frozenD1 {2:F4} vs BASE d1 {3:F4}' -f $w,($(if($ok){'OK'}else{'FAIL'})),$fz,$d1v) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $anchor_ok) { Write-Error 'Anchor FAIL: path probe non integro. Aborting.'; exit 1 }

# -------- Prefix property: P_r5 deve essere BIT-IDENTICO al checkpoint r5 di 47.F ------
function FileMD5($p) {
    if (-not (Test-Path $p)) { return '' }
    for ($a=0; $a -lt 15; $a++) { try { return (Get-FileHash -Algorithm MD5 -Path $p -ErrorAction Stop).Hash } catch { Start-Sleep -Milliseconds 200 } }
    return ''
}
Write-Host ''
Write-Host '=== Prefix property (P r1-r5 == 47.F D16r5_h32, MD5 sui checkpoint) ===' -ForegroundColor Yellow
$prefix_ok = $true
foreach ($chk in @(@{g=($WP+'_P_h32_r1.bin'); f=($WDIR+'\phase47f_D16r5_h32_r1.bin'); n='r1'},
                   @{g=($WP+'_P_h32_r5.bin'); f=$R5H32_R5; n='r5'})) {
    $hg = FileMD5 $chk.g; $hf = FileMD5 $chk.f
    $ok = ($hg -ne '' -and $hg -eq $hf); if (-not $ok) { $prefix_ok = $false }
    Write-Host ('  prefix@{0}  {1}  47G={2}  47F={3}' -f $chk.n,($(if($ok){'MATCH'}else{'MISMATCH'})),$hg.Substring(0,[math]::Min(8,$hg.Length)),$hf.Substring(0,[math]::Min(8,$hf.Length))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $prefix_ok) { Write-Error 'Prefix property FAILED: la continuazione non e'' bit-identica a 47.F. Aborting.'; exit 1 }

# -------- TF table (mappa struttura-vs-round) --------------------------------------------
function AvgVal($p) { return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }
$ALLROWS = @('F_r3','F_r4') + (1..9 | ForEach-Object { "P_r$_" }) + (6..9 | ForEach-Object { "A_r$_" }) + (1..5 | ForEach-Object { "H48_r$_" })
Write-Host ''
Write-Host ('=== TF probes per round (bar BPB={0:F4}; watch ent1 in discesa = quiet-down) ===' -f $PROMO_BPB) -ForegroundColor Yellow
Write-Host ('  {0,-8} {1,8} {2,8} {3,8} {4,8} {5,8} {6,7} {7,8} {8,8}' -f 'probe','avgVal','val1','val2','val3','valC','roll','rms1','ent1')
foreach ($n in (@('frozenD1') + $ALLROWS)) {
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]
    $rl = if ($p.ContainsKey('roll')) { $p['roll'] } else { '-' }
    $av = AvgVal $p
    $col = if ($n -like 'P_*' -or $n -like 'A_*' -or $n -like 'H48_*') { if ($av -le $PROMO_BPB) {'Green'} else {'Gray'} } else {'DarkCyan'}
    Write-Host ('  {0,-8} {1,8} {2,8} {3,8} {4,8} {5,8} {6,7} {7,8} {8,8}' -f $n,('{0:F4}' -f $av),$p['val1'],$p['val2'],$p['val3'],$p['valC'],$rl,$p['rms1'],$p['ent1']) -ForegroundColor $col
}

# -------- Gen configs --------------------------------------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

$gen_configs = @(
    @{ label='C2.A'; exe='phase43_generator.exe'; wargs=('"'+$C2A_W+'"'); wfile=$C2A_W; valbpb=$C2A_BPB; ref=$true },
    @{ label='D1';   exe='phase44_generator.exe'; wargs=('"'+$D1_W+'"');  wfile=$D1_W;  valbpb=$D1_BPB;  ref=$true },
    @{ label='D16_h32'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$D16_W+'"'); wfile=$D16_W; valbpb=(AvgVal $DP['D16_h32']); ref=$true },
    @{ label='D16r5_h32'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$R5H32_W+'"'); wfile=$R5H32_W; valbpb=(AvgVal $FP['D16r5_h32']); ref=$true },
    @{ label='D16r3_h64'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$R3H64_W+'"'); wfile=$R3H64_W; valbpb=(AvgVal $EPb['D16r3_h64']); ref=$true }
)
# Candidati word-gate: retro (F_r3/F_r4 = checkpoint 47.F su disco), P_r6-r9, A_r6-r9, H48_r1-r5.
# P_r1-r5 = solo TF (r2~47.D, r5=47.F gia' full-gatati).
$cand_map = @(
    @{ n='F_r3'; wf=$R5H32_R3 }, @{ n='F_r4'; wf=$R5H32_R4 }
)
foreach ($ri in 6..9) { $cand_map += @{ n="P_r$ri"; wf=($WP+'_P_h32_r'+$ri+'.bin') } }
foreach ($ri in 6..9) { $cand_map += @{ n="A_r$ri"; wf=($WP+'_A_h32_r'+$ri+'.bin') } }
foreach ($ri in 1..5) { $cand_map += @{ n="H48_r$ri"; wf=($WP+'_H48_r'+$ri+'.bin') } }
foreach ($c in $cand_map) {
    if (-not $PROBE.ContainsKey($c.n)) { continue }   # H48 tagliata o run parziale
    if (-not (Test-Path $c.wf)) { Write-Error ("Probe presente ma peso mancante: " + $c.wf); exit 1 }
    $gen_configs += @{ label=$c.n; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$c.wf+'"'); wfile=$c.wf; valbpb=(AvgVal $PROBE[$c.n]); ref=$false }
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
            Write-Host ('  launched {0,4}/{1}  {2,-10} T{3} r{4} s{5}' -f $launched,$total,$t.label,$t.tp,$t.rng,$t.si)
        }
        Start-Sleep -Milliseconds 150
        for ($k=$running.Count-1; $k -ge 0; $k--) { if ($running[$k].HasExited) { $running.RemoveAt($k) } }
    }
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
    Write-Host ('  {0,-10} s{1}  seq={2}  par={3}  {4}' -f $seq_tasks[$i].label,$seq_tasks[$i].si,$hs.Substring(0,[math]::Min(8,$hs.Length)),$hp.Substring(0,[math]::Min(8,$hp.Length)),($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $repro_ok) { Write-Error 'Repro pre-check FAILED. Aborting.'; exit 1 }
Write-Host '  Repro pre-check PASSED.' -ForegroundColor Green

# -------- Stage 2: MINI word-gate su tutti i checkpoint candidati ------------------------
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
Write-Host ('=== MINI word-gate per checkpoint: {0} runs ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
Aggregate $mini_tasks $mini_agg

Write-Host ''
Write-Host ('======== MINI (qualifica FULL: topBi<={0} a ENTRAMBE le temp + selfBPB55>={1} + BPB<={2}) ========' -f $MINI_BI,$BPB_LO,$PROMO_BPB) -ForegroundColor Cyan
Write-Host ('  {0,-10} {1,8} {2,7} {3,7} {4,7} {5,8} {6,8} {7,7}' -f 'config','BPB','bi65','bi55','al55','self65','self55','promo')
$promoted = @()
foreach ($cfg in $gen_configs) {
    $w65 = Worst $mini_agg $cfg.label '0.65'; $w55 = Worst $mini_agg $cfg.label '0.55'
    $promo = '-'
    if (-not $cfg.ref) {
        $bi_ok   = ($w65.bi -le $MINI_BI) -and ($w55.bi -le $MINI_BI)
        $self_ok = ($w55.self -ge $BPB_LO) -and ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -le $BPB_HI)
        $bpb_ok  = ($cfg.valbpb -le $PROMO_BPB)
        if ($bi_ok -and $self_ok -and $bpb_ok) { $promo = 'FULL'; $promoted += $cfg } else { $promo = 'stop' }
    }
    $col = if ($promo -eq 'FULL') {'Green'} elseif ($promo -eq 'stop') {'Gray'} else {'DarkCyan'}
    Write-Host ('  {0,-10} {1,8} {2,7} {3,7} {4,7} {5,8} {6,8} {7,7}' -f `
        $cfg.label,('{0:F4}' -f $cfg.valbpb),('{0:F0}' -f $w65.bi),('{0:F0}' -f $w55.bi),('{0:F0}' -f $w55.al),('{0:F2}' -f $w65.self),('{0:F2}' -f $w55.self),$promo) -ForegroundColor $col
}

if ($promoted.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 47.G (mini) ===' -ForegroundColor Yellow
    Write-Host '  -> Nessun checkpoint qualifica il FULL. Leggere la mappa TF per separare i rami:' -ForegroundColor Yellow
    Write-Host '     RAMO 2: struttura ok ma BPB mai sotto bar entro r9 -> lever round saturo, decide' -ForegroundColor Yellow
    Write-Host '       l''anneal; se corto anche quello, ceiling H32 reale -> sonda H48 primaria.' -ForegroundColor Yellow
    Write-Host '     RAMO 3: selfBPB55<0.8 o entropia collassa lungo la plain -> vale l''anneal.' -ForegroundColor Yellow
    Write-Host '     RAMO 4: struttura si rovescia coi round anche a H32 -> r5_h32 resta il miglior' -ForegroundColor Yellow
    Write-Host '       near-pass; restano anneal e H48; se muoiono entrambe -> domanda phrase-scale' -ForegroundColor Yellow
    Write-Host '       all''utente con tutti i dati in mano.' -ForegroundColor Yellow
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
Write-Host ('=== FULL word-gate: {0} promossi ({1}), {2} runs ===' -f $promoted.Count,(($promoted | ForEach-Object { $_.label }) -join ', '),$tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $tasks $THROTTLE
Aggregate $tasks $agg

function Passes($cfg,$tp) {
    if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { return $false }
    $w = Worst $agg $cfg.label $tp
    return ($null -ne $cfg.valbpb) -and ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -le $G_AL) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
}
# ramo 5: tutto verde TRANNE altLp55==3
function AltLpOnly($cfg) {
    if (-not (Passes $cfg '0.65')) { return $false }
    if (@($agg[$cfg.label]['0.55'].mets).Count -eq 0) { return $false }
    $w = Worst $agg $cfg.label '0.55'
    return ($cfg.valbpb -le $PROMO_BPB) -and ($w.ni -le $G_NAME) -and ($w.mr -le $G_RUN) -and ($w.bi -le $G_BI) -and ($w.al -eq 3) -and ($w.self -ge $BPB_LO) -and ($w.self -le $BPB_HI)
}

foreach ($tp in $TEMPS) {
    Write-Host ''
    Write-Host ('======== T={0} : worst-case over 32 samples ========' -f $tp) -ForegroundColor Cyan
    Write-Host ('  {0,-10} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f 'config','valBPB','dC2.A','nameWst','runWst','topBi','altLp','selfBPB','gate')
    Write-Host ('  ' + '-'*82)
    foreach ($cfg in $full_cfgs) {
        if (@($agg[$cfg.label][$tp].mets).Count -eq 0) { Write-Host ('  {0,-10} [no data]' -f $cfg.label); continue }
        $w = Worst $agg $cfg.label $tp
        $vb = if ($null -ne $cfg.valbpb) { '{0:F4}' -f $cfg.valbpb } else { 'N/A' }
        $db = if ($null -ne $cfg.valbpb) { '{0:+0.000;-0.000;0.000}' -f ($cfg.valbpb-$C2A_BPB) } else { '-' }
        $gate = if ($cfg.ref) {'ref'} elseif (Passes $cfg $tp) {'PASS'} else {'fail'}
        $col = if ($gate -eq 'PASS') {'Green'} elseif ($gate -eq 'fail') {'Gray'} else {'DarkCyan'}
        Write-Host ('  {0,-10} {1,9} {2,8} {3,7} {4,7} {5,7} {6,7} {7,8} {8,6}' -f `
            $cfg.label,$vb,$db,('{0:F1}' -f $w.ni),('{0:F0}' -f $w.mr),('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F2}' -f $w.self),$gate) -ForegroundColor $col
    }
}

# -------- Closed-loop telemetry (promossi, T=0.55; watch ent_p50/p10 = quiet-down) -------
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); [math]::Round($s[$idx],4) }

Write-Host ''
Write-Host '======== Closed-loop telemetry (promossi, T=0.55; ent_p50/p10 per round = misura del quiet-down) ========' -ForegroundColor Cyan
Write-Host ('  {0,-10} {1,9} {2,9} {3,9} {4,9} {5,9}   TF(rms/ent)' -f 'config','rms_p50','rms_p90','ent_p50','ent_p10','maxp_p90')
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
    Write-Host ('  {0,-10} {1,9} {2,9} {3,9} {4,9} {5,9}   {6}/{7}' -f `
        $mcfg.label,(Pctl $rms 0.5),(Pctl $rms 0.9),(Pctl $ent 0.5),(Pctl $ent 0.1),(Pctl $mxp 0.9),$p['rms1'],$p['ent1'])
}

# -------- Verdict (albero pre-registrato 1-5) ---------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 47.G ===' -ForegroundColor Yellow
$winners = @(); $ramo5 = @()
foreach ($mcfg in $promoted) {
    $p65 = Passes $mcfg '0.65'; $p55 = Passes $mcfg '0.55'
    $col = if ($p65 -and $p55) {'Green'} elseif ($p65) {'DarkCyan'} else {'Gray'}
    Write-Host ('  {0,-10} T0.65={1}  T0.55={2}  valBPB={3:F4}' -f $mcfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),$mcfg.valbpb)  -ForegroundColor $col
    if ($p65 -and $p55) { $winners += $mcfg.label }
    elseif (AltLpOnly $mcfg) { $ramo5 += $mcfg.label }
}
if ($winners.Count -gt 0) {
    Write-Host ''
    Write-Host ('  -> RAMO 1: {0} passa TUTTO a entrambe le temperature = primo candidato a' -f ($winners -join ', ')) -ForegroundColor Green
    Write-Host '     promozione nella storia del progetto. PRIMA del verdetto: validazione estesa' -ForegroundColor Green
    Write-Host '     (terzo rng, piu'' sample). Chiusura fase e commit SOLO a ordine utente.' -ForegroundColor Green
} elseif ($ramo5.Count -gt 0) {
    Write-Host ''
    Write-Host ('  -> RAMO 5: {0} = tutto verde TRANNE altLp55=3 (gate 2). Decisione sul gate' -f ($ramo5 -join ', ')) -ForegroundColor Cyan
    Write-Host '     all''utente: il gate NON si tocca da harness. Dati completi in tabella.' -ForegroundColor Cyan
} else {
    Write-Host ''
    Write-Host '  -> Nessun PASS pieno. Rami pre-registrati: 2 (BPB saturo entro r9 -> decide' -ForegroundColor Yellow
    Write-Host '     anneal, poi H48 primaria), 3 (quiet-down sulla plain -> vale anneal),' -ForegroundColor Yellow
    Write-Host '     4 (struttura si rovescia anche a H32 -> r5_h32 resta il near-pass; se muoiono' -ForegroundColor Yellow
    Write-Host '     anneal e H48 -> domanda phrase-scale all''utente con tutti i dati).' -ForegroundColor Yellow
    Write-Host '     NON fare: K!=16, mix30, sweep 0.55, r7 H64, label smoothing, L2/L3, commit.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
