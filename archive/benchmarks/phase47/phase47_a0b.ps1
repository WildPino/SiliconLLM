# Phase 47.A0b - Linear Anchor Repair (banco prova, non nuova fase concettuale)
#
# La 47.A0 e' INCONCLUSIVA, non negativa: mlpH128 riproduce su 3 finestre, randlabel/shuftime
# puliti (no leakage), ma il probe lineare scratch NON atterra su D1 -> l'ancora era il punto
# debole, non il MLP. A0b ripara il banco:
#   1. frozenD1: il readout D1 originale spinto nello STESSO path di eval dei probe
#      -> deve riprodurre BASE d1 (integrita' feature/normalizzazione/trigram/offset)
#   2. linD1init: probe lineare INIZIALIZZATO dai pesi D1, poi allenato -> Adam-su-finestra
#      migliora o degrada la soluzione globale?
#   3. linear / linearLong (3x epoche): budget story dello scratch
# Se frozenD1 matcha -> il path e' integro, il criterio "linear ~= D1 in 6ep" era mal
# calibrato -> ladder/controls della run A0 restano VALIDI (riusati da phase47a0_gauntlet.txt)
# -> word-gate su H32 / H64 / H128 (H128 compression champion ma piu' overconfident;
# H64 primo candidato serio per generazione).
#
# Run:  .\benchmarks\phase38-42\phase47_a0b.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase47a0'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$DELTA_W = $WDIR + '\phase45c_C0_delta.bin'
$B3_W    = $WDIR + '\phase46b_B3_punctlm30.bin'
$WP      = $WDIR + '\phase47a0'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
$OLD_OUT = $RDIR + '\phase47a0_gauntlet.txt'   # full A0 run: ladder/controls/ablations reused
foreach ($w in @($C2A_W,$D1_W,$DELTA_W,$B3_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }
if (-not (Test-Path $OLD_OUT)) { Write-Error "Missing A0 gauntlet output: $OLD_OUT (run phase47_a0.ps1 first)"; exit 1 }

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
$GATE_HS = @(32, 64, 128)    # word-gate candidates (user: H64 primo candidato serio)

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

# -------- Stage 1: anchor-only gauntlet -----------------------------------------
Write-Host ''
Write-Host '=== Phase 47.A0b - linear anchor repair (frozenD1 / linD1init / linear / linearLong) ===' -ForegroundColor Yellow
$ANCH_OUT = $RDIR + '\phase47a0b_anchor.txt'
& "$BINDIR\phase47a0_gauntlet.exe" $TS_DATA $D1_W $DELTA_W $B3_W $WP --anchor-only 2>&1 | Tee-Object $ANCH_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Anchor run failed'; exit 1 }
Write-Host 'Anchor run complete.' -ForegroundColor Green

# -------- Parse: anchor file (BASE + anchor probes) + old A0 file (ladder/controls) ----
function Parse-KV([string]$file,[string]$prefix) {
    $out = @{}
    foreach ($line in Select-String ('^'+$prefix+' ') $file) {
        $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
        $key = if ($prefix -eq 'BASE') { $kv['win'] } else { $kv['name'] }
        $out[$key] = $kv
    }
    return $out
}
$BASE  = Parse-KV $ANCH_OUT 'BASE'
$PROBE = Parse-KV $OLD_OUT 'PROBE'                     # ladder/controls/ablations from A0
$ANCH  = Parse-KV $ANCH_OUT 'PROBE'
foreach ($k in $ANCH.Keys) { $PROBE[$k] = $ANCH[$k] }  # anchor probes (and fresh linear) win
if ($BASE.Count -lt 4 -or -not $PROBE.ContainsKey('frozenD1') -or -not $PROBE.ContainsKey('mlpH128')) {
    Write-Error 'Output incomplete (need BASE x4 + frozenD1 + mlpH128)'; exit 1
}

# -------- Anchor + sanity verdict -------------------------------------------------
Write-Host ''
Write-Host '=== Anchor verdict (47.A0b) ===' -ForegroundColor Yellow
$wins = @('val1','val2','val3')
$checks = @()
foreach ($w in $wins) {
    $d1 = [double]$BASE[$w]['d1']; $i = [array]::IndexOf($wins,$w)+1
    $fz  = [double]$PROBE['frozenD1']["val$i"]
    $m128= [double]$PROBE['mlpH128']["val$i"]
    $rl  = [double]$PROBE['randlabel']["val$i"]
    $st  = [double]$PROBE['shuftime']["val$i"]
    $checks += [PSCustomObject]@{ name="anchor@$w";    ok=([math]::Abs($fz-$d1) -le 0.005); detail=('frozenD1 {0:F4} vs BASE d1 {1:F4}' -f $fz,$d1) }
    $checks += [PSCustomObject]@{ name="reproduce@$w"; ok=($m128 -le $d1-0.05); detail=('mlpH128 {0:F4} vs d1 {1:F4}' -f $m128,$d1) }
    $checks += [PSCustomObject]@{ name="randlabel@$w"; ok=($rl -ge $d1-0.01); detail=('randlabel {0:F4} vs d1 {1:F4}' -f $rl,$d1) }
    $checks += [PSCustomObject]@{ name="shuftime@$w";  ok=($st -ge $d1-0.01); detail=('shuftime {0:F4} vs d1 {1:F4}' -f $st,$d1) }
}
$sane = $true
foreach ($c in $checks) {
    if (-not $c.ok) { $sane = $false }
    Write-Host ('  {0,-18} {1,-6} {2}' -f $c.name,($(if($c.ok){'OK'}else{'FAIL'})),$c.detail) -ForegroundColor $(if($c.ok){'Green'}else{'Red'})
}

# -------- Linear story (diagnostica, NON gate) -------------------------------------
Write-Host ''
Write-Host '=== Linear story (diagnostica budget, non gate) ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f 'probe','val1','val2','val3','rms1','ent1','maxp1')
foreach ($n in @('frozenD1','linD1init','linear','linearLong','ablSEE','ablL2','mlpH16','mlpH32','mlpH64','mlpH128','randlabel','shuftime')) {
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]
    Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f $n,$p['val1'],$p['val2'],$p['val3'],$p['rms1'],$p['ent1'],$p['maxp1'])
}
if ($PROBE.ContainsKey('linD1init')) {
    $fz1 = [double]$PROBE['frozenD1']['val1']; $li1 = [double]$PROBE['linD1init']['val1']
    if ($li1 -gt $fz1 + 0.005) {
        Write-Host ('  -> linD1init PEGGIORA frozenD1 ({0:F4} -> {1:F4}): Adam-su-finestra OVERFITTA la soluzione' -f $fz1,$li1) -ForegroundColor Gray
        Write-Host '     globale. Il criterio "linear scratch ~= D1 in 6ep" era mal calibrato: D1 e'' allenato sul corpus.' -ForegroundColor Gray
    } elseif ($li1 -lt $fz1 - 0.005) {
        Write-Host ('  -> linD1init MIGLIORA frozenD1 ({0:F4} -> {1:F4}): il lineare scratch era solo sotto-allenato.' -f $fz1,$li1) -ForegroundColor Gray
    } else {
        Write-Host ('  -> linD1init ~= frozenD1 ({0:F4}): D1 e'' al tetto lineare di queste feature; lo scratch in 6ep non lo raggiunge.' -f $fz1) -ForegroundColor Gray
    }
}

if (-not $sane) {
    Write-Host ''
    Write-Host '  -> ANCHOR/SANITY FAIL. Se fallisce anchor@*: il path probe NON riproduce D1 ->' -ForegroundColor Red
    Write-Host '     disallineamento feature/normalizzazione/trigram/offset, il 2.09 resta sospetto. NO word-gate.' -ForegroundColor Red
    exit 1
}
Write-Host ''
Write-Host '  -> ANCHOR PASS: path probe integro (frozenD1 == BASE d1), riproduzione e controlli OK.' -ForegroundColor Green
Write-Host '     Il fail "linear" della A0 era criterio mal calibrato, non bug. Ladder A0 VALIDA.' -ForegroundColor Green

# -------- Word-gate candidates: H32 / H64 / H128 -------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

$gen_configs = @(
    @{ label='C2.A'; exe='phase43_generator.exe'; wargs=('"'+$C2A_W+'"'); wfile=$C2A_W; valbpb=$C2A_BPB; ref=$true },
    @{ label='D1';   exe='phase44_generator.exe'; wargs=('"'+$D1_W+'"');  wfile=$D1_W;  valbpb=$D1_BPB;  ref=$true }
)
foreach ($h in $GATE_HS) {
    $n = "mlpH$h"; $wf = $WP + '_' + $n + '.bin'
    if (-not (Test-Path $wf)) { Write-Error "Saved probe missing: $wf"; exit 1 }
    if (-not $PROBE.ContainsKey($n)) { Write-Error "Probe $n missing from gauntlet output"; exit 1 }
    $p = $PROBE[$n]
    $avg = ([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0
    $gen_configs += @{ label=$n; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$wf+'"'); wfile=$wf; valbpb=[math]::Round($avg,4); ref=$false }
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
function FileMD5($p) {
    if (-not (Test-Path $p)) { return '' }
    for ($a=0; $a -lt 15; $a++) { try { return (Get-FileHash -Algorithm MD5 -Path $p -ErrorAction Stop).Hash } catch { Start-Sleep -Milliseconds 200 } }
    return ''
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

# -------- Closed-loop telemetry audit (each mlp config, T=0.55) ---------------------
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); [math]::Round($s[$idx],4) }

Write-Host ''
Write-Host '======== Closed-loop telemetry (T=0.55): attractor signature check ========' -ForegroundColor Cyan
Write-Host ('  {0,-10} {1,9} {2,9} {3,9} {4,9} {5,9} {6,9}   (TF rms1/ent1/maxp1 dal gauntlet)' -f 'config','rms_p50','rms_p90','ent_p50','ent_p10','maxp_p50','maxp_p90')
$tel_off = @($SEEDS | Select-Object -First $TEL_SEEDS)
foreach ($mcfg in ($gen_configs | Where-Object { -not $_.ref })) {
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
    Write-Host ('  {0,-10} {1,9} {2,9} {3,9} {4,9} {5,9} {6,9}   TF: {7}/{8}/{9}' -f `
        $mcfg.label,(Pctl $rms 0.5),(Pctl $rms 0.9),(Pctl $ent 0.5),(Pctl $ent 0.1),(Pctl $mxp 0.5),(Pctl $mxp 0.9),$p['rms1'],$p['ent1'],$p['maxp1'])
}
Write-Host '  (rms closed-loop >> rms TF, o ent_p10 ~ 0 = decoder overconfident fuori distribuzione -> attrattore.)' -ForegroundColor Gray

# -------- Verdict -------------------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 47.A0b ===' -ForegroundColor Yellow
$winners = @()
foreach ($mcfg in ($gen_configs | Where-Object { -not $_.ref })) {
    $p65 = Passes $mcfg '0.65'; $p55 = Passes $mcfg '0.55'
    $col = if ($p65 -and $p55) {'Green'} elseif ($p65) {'DarkCyan'} else {'Gray'}
    Write-Host ('  {0,-10} T0.65={1}  T0.55={2}  valBPB={3:F4}' -f $mcfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),$mcfg.valbpb) -ForegroundColor $col
    if ($p65 -and $p55) { $winners += $mcfg.label }
}
if ($winners.Count -gt 0) {
    Write-Host ''
    Write-Host ('  -> AMMISSIBILE: {0} passa ENTRAMBI i temp. Il primo readout nonlineare statico' -f ($winners -join ', ')) -ForegroundColor Green
    Write-Host '     del progetto comprime ~2.1 E genera stabile: la nonlinearita'' statica rompe il muro' -ForegroundColor Green
    Write-Host '     compressione-vs-generazione. -> 47.A promozione formale (candidato = H minima che passa).' -ForegroundColor Green
} else {
    Write-Host ''
    Write-Host '  -> Nessuna H passa entrambi i temp: comprime ma non genera, stessa parete di 44/45/46' -ForegroundColor Yellow
    Write-Host '     ma ora SENZA memoria volatile. Leggi la telemetry: se rms esplode / ent_p10 collassa' -ForegroundColor Yellow
    Write-Host '     -> 47.B regularized static decoder (weight decay, logit/spectral norm, label smoothing,' -ForegroundColor Yellow
    Write-Host '     dropout train-time come proprieta'' addestrata). NON tornare a L2/L3.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
