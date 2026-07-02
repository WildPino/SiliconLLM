# Phase 47.B - Regularized Static Decoder (H32/H64 matrix), word-gate is the verdict
#
# 47.A0b: il decoder statico nonlineare cura nameWst/runWst/altLp ma fallisce su topBi +
# overconfidence OOD (ent_p10~0, rms closed-loop >> TF) -> parete DIVERSA da L2/L3: il
# decoder non distrugge struttura, diventa troppo certo. 47.B = proprieta' ADDESTRATE:
#   B1 label smoothing 0.10 | B2 logit-RMS penalty (tau4.5 lam0.02) | B3 wd 1e-3 (H64)
#   B4 hidden dropout 0.2 train-time | B5 combined (ls0.05+rms+wd3e-4+drop0.1)
# Niente L2/L3, niente cap/hack a inferenza. Successo (utente): NON inseguire 2.08 -
# anche 2.23-2.24 vince se topBi<=8, altLp<=2 e name/run restano bassi. Chiudere il loop.
#
# Run:  .\benchmarks\phase38-42\phase47b.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase47b'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$WP      = $WDIR + '\phase47b'
$A0_H32  = $WDIR + '\phase47a0_mlpH32.bin'   # unregularized controls from A0
$A0_H64  = $WDIR + '\phase47a0_mlpH64.bin'
$OLD_OUT = $ROOT + '\results\phase47a0\phase47a0_gauntlet.txt'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
foreach ($w in @($C2A_W,$D1_W,$A0_H32,$A0_H64)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }
if (-not (Test-Path $OLD_OUT)) { Write-Error "Missing A0 gauntlet output: $OLD_OUT"; exit 1 }

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
$BCFGS = @('B1ls_h32','B1ls_h64','B2rms_h32','B2rms_h64','B3wd_h64','B4drop_h32','B4drop_h64','B5all_h32','B5all_h64')

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47b_decoder.c' @SRC_CORE -o "$BINDIR\phase47b_decoder.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47b_decoder'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47_generator.c' @SRC_CORE -o "$BINDIR\phase47_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase44_generator.c' @SRC_CORE -o "$BINDIR\phase44_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase44_generator'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Single-trainer guard ------------------------------------------------
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps','phase44f_captrain','phase45a_relmove','phase45b_geometry','phase45c_gate','phase46a_l3','phase46b_l3','phase47a0_gauntlet','phase47b_decoder')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) {
    Write-Error ('Another trainer is running (' + (($busy.ProcessName | Select-Object -Unique) -join ',') + '). Aborting.')
    exit 1
}

# -------- Stage 1: train the matrix ----------------------------------------------
Write-Host ''
Write-Host '=== Phase 47.B - regularized decoder matrix (9 configs, 4 windows x 1M, ~5 GB RAM) ===' -ForegroundColor Yellow
$TRAIN_OUT = $RDIR + '\phase47b_train.txt'
& "$BINDIR\phase47b_decoder.exe" $TS_DATA $D1_W $WP 2>&1 | Tee-Object $TRAIN_OUT
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
$A0P   = Parse-KV $OLD_OUT  'PROBE'    # unregularized A0 references (TF telemetry comparison)
if ($BASE.Count -lt 4 -or -not $PROBE.ContainsKey('frozenD1')) { Write-Error 'Train output incomplete'; exit 1 }

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
if (-not $anchor_ok) { Write-Error 'Anchor FAIL: path probe non integro. Aborting (no word-gate).'; exit 1 }

# -------- TF table: regularization effect vs unregularized A0 -------------------------
Write-Host ''
Write-Host '=== TF probes (vs A0 unregularized: mlpH32 rms 4.11/ent 2.16, mlpH64 rms 4.80/ent 2.11) ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f 'probe','val1','val2','val3','rms1','ent1','maxp1')
foreach ($n in (@('frozenD1') + $BCFGS)) {
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]
    Write-Host ('  {0,-12} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f $n,$p['val1'],$p['val2'],$p['val3'],$p['rms1'],$p['ent1'],$p['maxp1'])
}

# -------- Word-gate configs --------------------------------------------------------------
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

function AvgVal($p) { return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }

$gen_configs = @(
    @{ label='C2.A'; exe='phase43_generator.exe'; wargs=('"'+$C2A_W+'"'); wfile=$C2A_W; valbpb=$C2A_BPB; ref=$true },
    @{ label='D1';   exe='phase44_generator.exe'; wargs=('"'+$D1_W+'"');  wfile=$D1_W;  valbpb=$D1_BPB;  ref=$true },
    @{ label='mlpH32'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$A0_H32+'"'); wfile=$A0_H32; valbpb=(AvgVal $A0P['mlpH32']); ref=$true },
    @{ label='mlpH64'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$A0_H64+'"'); wfile=$A0_H64; valbpb=(AvgVal $A0P['mlpH64']); ref=$true }
)
foreach ($n in $BCFGS) {
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

# -------- Closed-loop telemetry (T=0.55): did regularization kill the OOD overconfidence? ----
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); [math]::Round($s[$idx],4) }

Write-Host ''
Write-Host '======== Closed-loop telemetry (T=0.55) ========' -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,9} {2,9} {3,9} {4,9} {5,9}   TF(rms/ent)' -f 'config','rms_p50','rms_p90','ent_p50','ent_p10','maxp_p90')
$tel_off = @($SEEDS | Select-Object -First $TEL_SEEDS)
foreach ($mcfg in ($gen_configs | Where-Object { $_.exe -eq 'phase47_generator.exe' })) {
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
    $p = if ($PROBE.ContainsKey($mcfg.label)) { $PROBE[$mcfg.label] } else { $A0P[$mcfg.label] }
    Write-Host ('  {0,-12} {1,9} {2,9} {3,9} {4,9} {5,9}   {6}/{7}' -f `
        $mcfg.label,(Pctl $rms 0.5),(Pctl $rms 0.9),(Pctl $ent 0.5),(Pctl $ent 0.1),(Pctl $mxp 0.9),$p['rms1'],$p['ent1'])
}
Write-Host '  (vittoria della regularizzazione: rms_p90 ~ TF rms e ent_p10 lontana da 0, SENZA cap a inferenza.)' -ForegroundColor Gray

# -------- Verdict -------------------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 47.B ===' -ForegroundColor Yellow
$winners = @()
foreach ($mcfg in ($gen_configs | Where-Object { -not $_.ref })) {
    $p65 = Passes $mcfg '0.65'; $p55 = Passes $mcfg '0.55'
    $col = if ($p65 -and $p55) {'Green'} elseif ($p65) {'DarkCyan'} else {'Gray'}
    Write-Host ('  {0,-12} T0.65={1}  T0.55={2}  valBPB={3:F4}' -f $mcfg.label,($(if($p65){'PASS'}else{'fail'})),($(if($p55){'PASS'}else{'fail'})),$mcfg.valbpb) -ForegroundColor $col
    if ($p65 -and $p55) { $winners += $mcfg.label }
}
if ($winners.Count -gt 0) {
    Write-Host ''
    Write-Host ('  -> LOOP CHIUSO: {0} passa ENTRAMBI i temp. Primo decoder nonlineare statico' -f ($winners -join ', ')) -ForegroundColor Green
    Write-Host '     regolarizzato che comprime sotto gate E genera stabile, senza memoria volatile' -ForegroundColor Green
    Write-Host '     e senza cap a inferenza. -> promozione formale 47 (candidato SEE-V4 readout).' -ForegroundColor Green
} else {
    Write-Host ''
    Write-Host '  -> Nessuna config chiude il loop. Leggi la telemetry: se la regularizzazione ha' -ForegroundColor Yellow
    Write-Host '     normalizzato rms/ent ma topBi resta alto -> il problema non e'' overconfidence ma' -ForegroundColor Yellow
    Write-Host '     mancanza di contesto phrase-scale nel decoder statico (parete informativa, non' -ForegroundColor Yellow
    Write-Host '     di calibrazione). Se rms/ent restano OOD -> dosi piu'' forti o spectral norm.' -ForegroundColor Yellow
}
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
