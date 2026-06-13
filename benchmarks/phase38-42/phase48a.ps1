# Phase 48.A - armB-expanded substrate under the 47.I-final DAgger recipe, full gate v2
#
# 48.0 EXPAND PASSED (+0.04 TF, controls clean): the nonlinear lift re-injects the substrate's
# discarded nonlinearity. 48.A tests the closed loop. Pipeline:
#   - train armB (phase48a_armb): H32, 9 rounds, r1-5 K16, r6-9 K16/K128 far-field mix.
#   - anchor check (frozenD1 base-256 readout == BASE d1) + TF-per-round table.
#   - DETERMINISM PRE-CHECK (mandatory, user): the new armB substrate path must be byte-
#     deterministic closed-loop (MD5 parallel vs sequential) BEFORE any gate. A bug here
#     invalidates every downstream number.
#   - gate v2 (word bars + corpus-calibrated byte-guards) at two temps: mini (I_r6..r9) ->
#     full (promoted) -> replica protocol (4 reps, 47.H rule) -> human dump.
#   - references C2.A / D1 / P_r7 generated fresh under the SAME gate (isolates how much the
#     substrate, not the recipe, contributes to the closed loop). chRun column = the 47.I
#     char-flood residual watch (does armB dissolve / leave / move it?).
# TF-only != generative (44-47 law): no celebration before gate v2 + replicas + human read.
#
# Run:  .\benchmarks\phase38-42\phase48a.ps1
#       .\benchmarks\phase38-42\phase48a.ps1 -SkipTrain     # reuse train output
#       .\benchmarks\phase38-42\phase48a.ps1 -Smoke         # tiny end-to-end pipeline check

param([switch]$SkipTrain,[switch]$Smoke,[switch]$CharFlood)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase48a'
$HDIR    = $RDIR + '\human'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
foreach ($d in @($RDIR,$HDIR)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$PR7_W   = $WDIR + '\phase47g_P_h32_r7.bin'
$WP      = if ($CharFlood) { $WDIR + '\phase48afix' } else { $WDIR + '\phase48a' }   # separate prefix: fix never clobbers the base run
$C2A_BPB = 2.2593
$PREMISE_CF_THR = 5.0   # pre-registered: proceed with char-flood coverage only if stayPct >= this (else seeding teaches nothing)
$D1_BPB  = 2.2522
foreach ($w in @($C2A_W,$D1_W,$PR7_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = if ($Smoke) { 4 } else { 16 }
$RNGS_ORIG  = @(12345, 67890)
$RNGS_NEW_A = @(24680, 13579)
$RNGS_NEW_B = @(98765, 55555)
$TEMPS   = @('0.65', '0.55')
$GEN_LEN = 2000
$WARMUP  = 5000
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0
$PROMO_BPB = $C2A_BPB - 0.005
$G_NAME = 20.0; $G_RUN = 5.0; $G_BI = 8.0; $G_AL = 2.0
$MINI_BI = 7.0
$THROTTLE = 12
$CAL_WINDOWS = if ($Smoke) { 200 } else { 1000 }
$CAL_WBYTES  = 2048

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
foreach ($pair in @(
    @{src='phase48a_armb.c';      exe='phase48a_armb.exe'},
    @{src='phase48a_fix.c';       exe='phase48a_fix.exe'},
    @{src='phase48a_generator.c'; exe='phase48a_generator.exe'},
    @{src='phase47_generator.c';  exe='phase47_generator.exe'},
    @{src='phase44_generator.c';  exe='phase44_generator.exe'},
    @{src='phase43_generator.c';  exe='phase43_generator.exe'})) {
    & gcc @GCC_FLAGS ('benchmarks/phase38-42/'+$pair.src) @SRC_CORE -o ($BINDIR+'\'+$pair.exe)
    if ($LASTEXITCODE -ne 0) { Write-Error ('Compile failed: '+$pair.src); exit 1 }
}
Write-Host 'Compiled OK.' -ForegroundColor Green

# ============ Gate v2 helpers (byte-guard + word; from phase47i) =====================
function Get-ByteGuardFromBytes([byte[]]$bytes,[int]$off,[int]$len) {
    $maxWs=0; $maxCh=0; $ws=0; $ch=0; $wsCount=0; $nonPrint=0; $prev=-1
    $end = $off + $len
    for ($bi=$off; $bi -lt $end; $bi++) {
        $b = $bytes[$bi]
        if ($b -eq 32 -or $b -eq 9 -or $b -eq 10 -or $b -eq 13) {
            $wsCount++; $ws++; if ($ws -gt $maxWs) { $maxWs=$ws }; $ch=0
        } else {
            $ws=0
            if ($b -eq $prev) { $ch++ } else { $ch=1 }
            if ($ch -gt $maxCh) { $maxCh=$ch }
            if ($b -lt 32 -or $b -gt 126) { $nonPrint++ }
        }
        $prev = $b
    }
    return [PSCustomObject]@{ wsRun=$maxWs; chRun=$maxCh; wsFrac=[math]::Round($wsCount/$len,4); nonPrint=$nonPrint }
}
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); $s[$idx] }
function Mx2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
function Av2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }

# ============ STEP 1: byte-guard calibration (corpus) -> docs/gatev2_bars.json ========
Write-Host ''
Write-Host ('=== STEP 1: calibrazione gate v2 su corpus ({0} finestre da {1}B) ===' -f $CAL_WINDOWS,$CAL_WBYTES) -ForegroundColor Yellow
$corpus = [System.IO.File]::ReadAllBytes($TS_DATA)
$stride = [long][math]::Floor(($corpus.Length - $CAL_WBYTES) / $CAL_WINDOWS)
$cal_ws=@(); $cal_ch=@(); $cal_wf=@(); $cal_np=@()
for ($wi=0; $wi -lt $CAL_WINDOWS; $wi++) {
    $g = Get-ByteGuardFromBytes $corpus ([long]$wi*$stride) $CAL_WBYTES
    $cal_ws += [double]$g.wsRun; $cal_ch += [double]$g.chRun; $cal_wf += [double]$g.wsFrac; $cal_np += [double]$g.nonPrint
}
$corpus = $null
$V2 = [PSCustomObject]@{
    wsRun    = [int]((Mx2 $cal_ws) + 2)
    chRun    = [int]((Mx2 $cal_ch) + 2)
    wsFrac   = [math]::Round((Mx2 $cal_wf) + 0.03, 4)
    nonPrint = [int]((Mx2 $cal_np) + 2)
}
Write-Host ('  corpus: wsRun max={0} | chRun max={1} | wsFrac max={2:F4} | nonPrint max={3}' -f `
    (Mx2 $cal_ws),(Mx2 $cal_ch),(Mx2 $cal_wf),(Mx2 $cal_np))
Write-Host ('  GATE V2 BARS: wsRun<={0}  chRun<={1}  wsFrac<={2:F4}  nonPrint<={3}' -f $V2.wsRun,$V2.chRun,$V2.wsFrac,$V2.nonPrint) -ForegroundColor Cyan
# tightening-only: existing docs bars are the contract. Report drift; do NOT overwrite on a re-run.
$BARS_DOC = $ROOT + '\docs\gatev2_bars.json'
if (Test-Path $BARS_DOC) {
    $existing = Get-Content $BARS_DOC -Raw | ConvertFrom-Json
    Write-Host ('  docs/gatev2_bars.json (contract): wsRun<={0} chRun<={1} wsFrac<={2:F4} nonPrint<={3} (using THESE)' -f `
        $existing.wsRun,$existing.chRun,$existing.wsFrac,$existing.nonPrint) -ForegroundColor DarkCyan
    $V2 = $existing   # the committed bars are the law; calibration here only reports
} else {
    $V2 | ConvertTo-Json | Out-File $BARS_DOC -Encoding utf8
}
function GateV2Fails($w,$tp) {
    $fails=@()
    if ($w.n -eq 0)         { return ,@("nodata@$tp") }
    if ($w.bi -gt $G_BI)    { $fails += "topBi@$tp" }
    if ($w.al -gt $G_AL)    { $fails += "altLp@$tp" }
    if ($w.mr -gt $G_RUN)   { $fails += "runWst@$tp" }
    if ($w.ni -gt $G_NAME)  { $fails += "nameWst@$tp" }
    if ($w.self -lt $BPB_LO -or $w.self -gt $BPB_HI) { $fails += "selfBPB@$tp" }
    if ($w.wsRun -gt $V2.wsRun)       { $fails += "wsRun@$tp" }
    if ($w.chRun -gt $V2.chRun)       { $fails += "chRun@$tp" }
    if ($w.wsFrac -gt $V2.wsFrac)     { $fails += "wsFrac@$tp" }
    if ($w.nonPrint -gt $V2.nonPrint) { $fails += "nonPrint@$tp" }
    return ,$fails
}
function PrintGateRow($name,$tp,$w,$verdict,$col) {
    Write-Host ('  {0,-10} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f `
        $name,$tp,('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F0}' -f $w.mr),('{0:F1}' -f $w.ni),('{0:F2}' -f $w.self),`
        ('{0:F0}' -f $w.wsRun),('{0:F0}' -f $w.chRun),('{0:F2}' -f $w.wsFrac),('{0:F0}' -f $w.nonPrint),$verdict) -ForegroundColor $col
}
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
function Get-AllMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    $wm = Get-WordMetricsFromText ([System.Text.Encoding]::ASCII.GetString($bytes))
    if ($null -eq $wm) { return $null }
    $bg = Get-ByteGuardFromBytes $bytes 0 $bytes.Length
    return [PSCustomObject]@{ nameish=$wm.nameish; maxrun=$wm.maxrun; top_bi=$wm.top_bi; altloop=$wm.altloop;
                              wsRun=$bg.wsRun; chRun=$bg.chRun; wsFrac=$bg.wsFrac; nonPrint=$bg.nonPrint }
}
function Get-SelfBPB([string]$statsfile) {
    $bl = Select-String 'self_BPB' $statsfile | Select-Object -Last 1
    if ($bl) { return [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    return $null
}
function WorstOf($mets,$bpbs) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $ws=@(); $ch=@(); $wf=@(); $np=@(); $bs=@()
    foreach ($m in @($mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop
        $ws+=[double]$m.wsRun; $ch+=[double]$m.chRun; $wf+=[double]$m.wsFrac; $np+=[double]$m.nonPrint }
    foreach ($b in @($bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni=(Mx2 $ni); mr=(Mx2 $mr); bi=(Mx2 $bi); al=(Mx2 $al);
                              wsRun=(Mx2 $ws); chRun=(Mx2 $ch); wsFrac=(Mx2 $wf); nonPrint=(Mx2 $np);
                              self=(Av2 $bs); n=@($mets).Count }
}
# per-config generation task (exe + wargs vary by reference vs candidate)
function New-GenTask($label,$exe,$wargs,$tp,$rng,$off,$si,$suffix) {
    $lbl = $label+'_T'+$tp+'_r'+$rng+'_s'+$si+$suffix
    $tf  = $RDIR+'\gen_'+$lbl+'.txt'
    $sf  = $RDIR+'\gen_'+$lbl+'_stats.txt'
    $argline = ('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                $TS_DATA,$wargs,$GEN_LEN,$WARMUP,$tp,$off,$rng)
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
            if ($launched % 32 -eq 0 -or $launched -eq $total) { Write-Host ('  launched {0,4}/{1}' -f $launched,$total) }
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

# ============ STEP 2: train armB ====================================================
$TRAINERS = @('phase47i_farfield','phase48_0_taps','phase48_0_expand','phase48a_armb','phase48a_fix')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting.'; exit 1 }

$TRAINER_EXE = if ($CharFlood) { 'phase48a_fix.exe' } else { 'phase48a_armb.exe' }
Write-Host ''
Write-Host ('=== STEP 2: train {0} (9 rounds, r1-5 K16, r6-9 {1}) ===' -f ($(if($CharFlood){'armB + char-flood coverage'}else{'armB'}),$(if($CharFlood){'K16/K128ws/CF'}else{'K16/K128'})) ) -ForegroundColor Yellow
$TRAIN_OUT = if ($CharFlood) { $RDIR + '\phase48afix_train.txt' } else { $RDIR + '\phase48a_train.txt' }
$train_extra = if ($Smoke) { @('--len','20000') } else { @() }
if ($SkipTrain -and (Test-Path $TRAIN_OUT)) {
    Write-Host ('SkipTrain: riuso ' + $TRAIN_OUT) -ForegroundColor Cyan
} else {
    & "$BINDIR\$TRAINER_EXE" $TS_DATA $D1_W $WP @train_extra 2>&1 | Tee-Object $TRAIN_OUT
    if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
}
function Parse-KV([string]$file,[string]$prefix) {
    $out = @{}
    foreach ($line in Select-String ('^'+$prefix+' ') $file) {
        $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
        $key = if ($prefix -eq 'BASE') { $kv['win'] } else { $kv['name'] }
        if ($null -ne $key -and $key -ne '') { $out[$key] = $kv }   # skip prose lines that happen to start with BASE/PROBE
    }
    return $out
}
function AvgVal($p) { return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }
$BASE  = Parse-KV $TRAIN_OUT 'BASE'
$PROBE = Parse-KV $TRAIN_OUT 'PROBE'
if (-not $PROBE.ContainsKey('frozenD1') -or -not $PROBE.ContainsKey('I_r9')) { Write-Error 'Train output incomplete'; exit 1 }
# anchor: base-256 readout through armB rows must == BASE d1
$anchor_ok = $true
foreach ($w in @('val1','val2','val3')) {
    $d1v = [double]$BASE[$w]['d1']; $i = [array]::IndexOf(@('val1','val2','val3'),$w)+1
    if ([math]::Abs([double]$PROBE['frozenD1']["val$i"]-$d1v) -gt 0.005) { $anchor_ok = $false }
}
if (-not $anchor_ok) { Write-Error 'Anchor FAIL (base feature/norm/trigram drift).'; exit 1 }
Write-Host '  anchor OK (base-256 readout reproduces BASE d1)' -ForegroundColor Green

Write-Host ''
Write-Host '=== TF per round (armB feature; rollK128 = whitespace-far, rollCF = char-flood-far) ===' -ForegroundColor Yellow
Write-Host ('  {0,-6} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f 'probe','avgVal','valC','rollK16','rollK128','rollCF','ent1')
foreach ($rn in 1..9) {
    $n = "I_r$rn"; if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]; $av = AvgVal $p
    $r16 = if ($p.ContainsKey('rollK16')) { $p['rollK16'] } elseif ($p.ContainsKey('roll')) { $p['roll'] } else { '-' }
    $r128 = if ($p.ContainsKey('rollK128')) { $p['rollK128'] } else { '-' }
    $rcf = if ($p.ContainsKey('rollCF')) { $p['rollCF'] } else { '-' }
    $col = if ($av -le $PROMO_BPB) {'Green'} else {'Gray'}
    Write-Host ('  {0,-6} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}' -f $n,('{0:F4}' -f $av),$p['valC'],$r16,$r128,$rcf,$p['ent1']) -ForegroundColor $col
}

# ---- char-flood premise gate (only -CharFlood): seeding must actually reach the flood basin ----
if ($CharFlood) {
    Write-Host ''
    Write-Host '=== Premise char-flood (no training, r5 decoder): seeding raggiunge il bacino? ===' -ForegroundColor Yellow
    $prem = (Select-String '^PREMISE_CF ' $TRAIN_OUT | Select-Object -Last 1)
    if (-not $prem) { Write-Error 'PREMISE_CF line missing from training output.'; exit 1 }
    $pk = @{}; foreach ($tok in ($prem.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $kp=$tok -split '=',2; if ($kp.Count -eq 2) { $pk[$kp[0]]=$kp[1] } }
    $stayPct = [double]$pk['stayPct']; $meanCont = [double]$pk['meanCont']
    Write-Host ('  stayPct={0:F2}%% (soglia pre-registrata >= {1:F1}%%)  meanCont={2:F3}  n={3}' -f $stayPct,$PREMISE_CF_THR,$meanCont,$pk['nprobe']) -ForegroundColor Cyan
    if ($stayPct -lt $PREMISE_CF_THR) {
        Write-Host ''
        Write-Host '  -> PREMISE FAIL: il decoder seeded si auto-corregge subito (non resta nel char-flood).' -ForegroundColor Yellow
        Write-Host '     Il seeding NON insegna nulla: il flood libero nasce da un percorso diverso, non da' -ForegroundColor Yellow
        Write-Host '     stati char-ripetuti raggiungibili per seeding. -> RIPORTA: serve un altro meccanismo' -ForegroundColor Yellow
        Write-Host '     di copertura (branch 3). 48.A si chiude su questo, hard cap esaurito.' -ForegroundColor Yellow
        Write-Host ('  Files: ' + $RDIR)
        exit 0
    }
    Write-Host '  -> PREMISE OK: il bacino char-flood e'' raggiungibile per seeding; la copertura puo'' insegnare l''uscita.' -ForegroundColor Green
}

# ============ STEP 3: CLOSED-LOOP DETERMINISM PRE-CHECK (mandatory) ==================
# A new substrate path in the generator: prove it is byte-deterministic (parallel vs
# sequential, same seeds/temp/rng) BEFORE spending the gate. A bug here voids everything.
Write-Host ''
Write-Host '=== STEP 3: determinism pre-check (armB generator, MD5 parallel vs sequential) ===' -ForegroundColor Yellow
$I_R9 = $WP + '_I_h32_r9.bin'
if (-not (Test-Path $I_R9)) { Write-Error "Missing armB checkpoint: $I_R9"; exit 1 }
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS_STD = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS_STD += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }
$chk_seeds = @($SEEDS_STD | Select-Object -First 2)
$armb_wargs = ('"'+$D1_W+'" "'+$I_R9+'"')
$seq=@(); $par=@(); $si=0
foreach ($off in $chk_seeds) { $si++
    $seq += New-GenTask 'I_r9' 'phase48a_generator.exe' $armb_wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_seqchk'
    $par += New-GenTask 'I_r9' 'phase48a_generator.exe' $armb_wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_parchk'
}
Invoke-Throttled $seq 1
Invoke-Throttled $par 4
$det_ok = $true
for ($i=0; $i -lt $seq.Count; $i++) {
    $hs = FileMD5 $seq[$i].tf; $hp = FileMD5 $par[$i].tf
    $ok = ($hs -ne '' -and $hs -eq $hp); if (-not $ok) { $det_ok = $false }
    Write-Host ('  s{0}  seq={1}  par={2}  {3}' -f $seq[$i].si,$hs.Substring(0,[math]::Min(8,$hs.Length)),$hp.Substring(0,[math]::Min(8,$hp.Length)),($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $det_ok) { Write-Error 'Determinism pre-check FAILED. The armB substrate path is non-deterministic. Aborting before gate.'; exit 1 }
Write-Host '  Determinism pre-check PASSED. armB closed-loop path is byte-reproducible.' -ForegroundColor Green

# ============ STEP 4: MINI gate v2 (I_r6..r9, 4 seeds) ===============================
$SEEDS_MINI = @()
for ($i = 0; $i -lt 4; $i++) { $SEEDS_MINI += [int]([math]::Floor($i * ($fsz - $margin) / 4)) }
$cands = @()
foreach ($rn in 6..9) {
    $n = "I_r$rn"; $wf = $WP + '_I_h32_r' + $rn + '.bin'
    if ($PROBE.ContainsKey($n) -and (Test-Path $wf)) {
        $cands += @{ label=$n; exe='phase48a_generator.exe'; wargs=('"'+$D1_W+'" "'+$wf+'"'); valbpb=(AvgVal $PROBE[$n]) }
    }
}
$mini_tasks = @()
foreach ($tp in $TEMPS) { $si=0
    foreach ($off in $SEEDS_MINI) { $si++
        foreach ($c in $cands) { $mini_tasks += New-GenTask ($c.label) $c.exe $c.wargs $tp $RNGS_ORIG[0] $off $si '_mini' }
    } }
Write-Host ''
Write-Host ('=== STEP 4: MINI gate v2 ({0} runs) ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
$agg = @{}
foreach ($c in $cands) { $agg[$c.label] = @{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
foreach ($t in $mini_tasks) {
    $m = Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf;    if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host ('======== MINI v2 (qualify: bi<={0} entrambe + selfBPB sano + BPB<={1} + byte puliti) ========' -f $MINI_BI,$PROMO_BPB) -ForegroundColor Cyan
Write-Host ('  {0,-10} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','promo')
$promoted = @()
foreach ($c in $cands) {
    $w65 = WorstOf $agg[$c.label]['0.65'].mets $agg[$c.label]['0.65'].bpbs
    $w55 = WorstOf $agg[$c.label]['0.55'].mets $agg[$c.label]['0.55'].bpbs
    $bi_ok   = ($w65.bi -le $MINI_BI) -and ($w55.bi -le $MINI_BI)
    $self_ok = ($w55.self -ge $BPB_LO) -and ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -le $BPB_HI)
    $bpb_ok  = ($c.valbpb -le $PROMO_BPB)
    $gf65 = GateV2Fails $w65 '0.65'; $gf55 = GateV2Fails $w55 '0.55'
    $bf65 = @($gf65 | Where-Object { $_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*' })
    $bf55 = @($gf55 | Where-Object { $_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*' })
    $byte_ok = ($bf65.Count -eq 0) -and ($bf55.Count -eq 0)
    $promo = if ($bi_ok -and $self_ok -and $bpb_ok -and $byte_ok) { $promoted += $c; 'FULL' } else { 'stop' }
    $col = if ($promo -eq 'FULL') {'Green'} else {'Gray'}
    PrintGateRow $c.label '0.65' $w65 '' $col
    PrintGateRow $c.label '0.55' $w55 $promo $col
}
if ($promoted.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 48.A (mini v2) ===' -ForegroundColor Yellow
    Write-Host '  -> Nessun checkpoint armB qualifica sotto gate v2 (mini). Leggi le colonne:' -ForegroundColor Yellow
    Write-Host '     topBi alto = il substrato ricco apre piu'' bacini (trappola H64/47); byte (ws/ch)' -ForegroundColor Yellow
    Write-Host '     = wasteland; BPB>bar = costo rollout sopra il margine. Riporta all utente.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}

# ============ STEP 5: FULL gate v2 (promoted) + references C2.A/D1/P_r7 ==============
$ref_configs = @(
    @{ label='C2.A'; exe='phase43_generator.exe'; wargs=('"'+$C2A_W+'"');            valbpb=$C2A_BPB },
    @{ label='D1';   exe='phase44_generator.exe'; wargs=('"'+$D1_W+'"');             valbpb=$D1_BPB  },
    @{ label='P_r7'; exe='phase47_generator.exe'; wargs=('"'+$D1_W+'" "'+$PR7_W+'"'); valbpb=2.2497  }
)
$full_set = @(); foreach ($c in $promoted) { $full_set += $c }; foreach ($r in $ref_configs) { $full_set += $r }
$full_tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS_ORIG) { $si=0
    foreach ($off in $SEEDS_STD) { $si++
        foreach ($c in $full_set) { $full_tasks += New-GenTask ($c.label) $c.exe $c.wargs $tp $rng $off $si '_full' }
    } } }
Write-Host ''
Write-Host ('=== STEP 5: FULL gate v2: {0} armB promossi + {1} reference, {2} runs ===' -f $promoted.Count,$ref_configs.Count,$full_tasks.Count) -ForegroundColor Yellow
foreach ($c in $full_set) { $agg[$c.label] = @{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
Invoke-Throttled $full_tasks $THROTTLE
foreach ($t in $full_tasks) {
    $m = Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf;    if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host '======== FULL gate v2 (worst-of-32, word + byte; ref = context) ========' -ForegroundColor Cyan
Write-Host ('  {0,-10} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','gate')
$full_pass = @()
foreach ($c in $full_set) {
    $isref = ($ref_configs | Where-Object { $_.label -eq $c.label }).Count -gt 0
    $allfails = @()
    foreach ($tp in $TEMPS) {
        $w = WorstOf $agg[$c.label][$tp].mets $agg[$c.label][$tp].bpbs
        $tf2 = GateV2Fails $w $tp; $allfails += $tf2
        $v = if ($isref) { 'ref' } elseif (@($tf2).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) { 'PASS' } else { 'fail' }
        $col = if ($v -eq 'PASS') {'Green'} elseif ($v -eq 'ref') {'DarkCyan'} else {'Gray'}
        PrintGateRow $c.label $tp $w $v $col
        if (-not $isref -and @($tf2).Count -gt 0) { Write-Host ('    fail: ' + ($tf2 -join ', ')) -ForegroundColor Red }
    }
    if (-not $isref -and @($allfails).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) { $full_pass += $c }
}
if ($full_pass.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 48.A (full v2) ===' -ForegroundColor Yellow
    Write-Host '  -> Promossi al full ma nessun PASS pieno sotto gate v2. Confronta le righe armB con' -ForegroundColor Yellow
    Write-Host '     i reference C2.A/D1/P_r7: se armB e'' byte-piu''-pulito ma fallisce su altro, il' -ForegroundColor Yellow
    Write-Host '     substrato ha spostato la parete; se peggiora i byte, la ricchezza apre bacini.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}

# ============ STEP 6: REPLICA PROTOCOL on every full-PASS ============================
$USED_WIN = @(
    @{ lo=[long][math]::Floor($fsz/5);    hi=[long]([math]::Floor($fsz/5)+1000000+$margin) },
    @{ lo=[long][math]::Floor($fsz/2);    hi=[long]([math]::Floor($fsz/2)+1000000+$margin) },
    @{ lo=[long][math]::Floor(0.65*$fsz); hi=[long]([math]::Floor(0.65*$fsz)+1000000+$margin) },
    @{ lo=[long][math]::Floor(0.80*$fsz); hi=[long]([math]::Floor(0.80*$fsz)+1000000+$margin) }
)
function New-HeldOutSeeds([double]$frac) {
    $hs = @()
    for ($i = 0; $i -lt $NSEEDS; $i++) {
        $off = [long][math]::Floor(($i + $frac) * ($fsz - $margin) / $NSEEDS)
        $moved = $true
        while ($moved) {
            $moved = $false
            foreach ($uw in $USED_WIN) { if ($off -ge ($uw.lo - $margin) -and $off -lt $uw.hi) { $off = $uw.hi + 1048576; $moved = $true } }
            if ($off -gt ($fsz - $margin)) { $off = $off % ($fsz - $margin); $moved = $true }
        }
        $hs += [int]$off
    }
    return $hs
}
$REPLICAS = @(
    @{ name='R1'; rngs=$RNGS_NEW_A; seeds=$SEEDS_STD },
    @{ name='R2'; rngs=$RNGS_NEW_B; seeds=$SEEDS_STD },
    @{ name='R3'; rngs=$RNGS_ORIG; seeds=(New-HeldOutSeeds 0.25) },
    @{ name='R4'; rngs=$RNGS_ORIG; seeds=(New-HeldOutSeeds 0.75) }
)
foreach ($c in $full_pass) {
    Write-Host ''
    Write-Host ('=== STEP 6: REPLICA PROTOCOL (gate v2) su {0}: 4 repliche x {1} x 2 temp ===' -f $c.label,$NSEEDS) -ForegroundColor Yellow
    $rep_tasks = @()
    foreach ($rep in $REPLICAS) {
        foreach ($tp in $TEMPS) { foreach ($rng in $rep.rngs) { $si=0
            foreach ($off in $rep.seeds) { $si++
                $rep_tasks += New-GenTask ($c.label+'_'+$rep.name) $c.exe $c.wargs $tp $rng $off $si ''
            } } }
    }
    Invoke-Throttled $rep_tasks $THROTTLE
    $rep_agg = @{}
    foreach ($rep in $REPLICAS) { $rep_agg[$rep.name] = @{}; foreach ($tp in $TEMPS) { $rep_agg[$rep.name][$tp] = @{ mets=@(); bpbs=@() } } }
    foreach ($t in $rep_tasks) {
        $repname = ($t.label -split '_')[-1]
        $m = Get-AllMetrics $t.tf; if ($m) { $rep_agg[$repname][$t.tp].mets += $m }
        $b = Get-SelfBPB $t.sf;    if ($null -ne $b) { $rep_agg[$repname][$t.tp].bpbs += $b }
    }
    $rep_pass = 0
    Write-Host ('  {0,-10} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'replica','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','verdict')
    foreach ($rep in $REPLICAS) {
        $rfails = @()
        foreach ($tp in $TEMPS) { $rfails += (GateV2Fails (WorstOf $rep_agg[$rep.name][$tp].mets $rep_agg[$rep.name][$tp].bpbs) $tp) }
        $v = if (@($rfails).Count -eq 0) { $rep_pass++; 'PASS' } else { 'FAIL' }
        foreach ($tp in $TEMPS) {
            $w = WorstOf $rep_agg[$rep.name][$tp].mets $rep_agg[$rep.name][$tp].bpbs
            PrintGateRow $rep.name $tp $w ($(if($tp -eq $TEMPS[0]){$v}else{''})) ($(if($v -eq 'PASS'){'Green'}else{'Red'}))
        }
        if (@($rfails).Count -gt 0) { Write-Host ('    fail: ' + ($rfails -join ', ')) -ForegroundColor Red }
    }
    # human dump: 5 sample/temp from replica R1
    $ncopied = 0
    foreach ($tp in $TEMPS) {
        $srcs = @($rep_tasks | Where-Object { $_.label -eq ($c.label+'_R1') -and $_.tp -eq $tp -and $_.rng -eq $RNGS_NEW_A[0] } | Select-Object -First 5)
        $hi2 = 0
        foreach ($t in $srcs) { $hi2++
            if (Test-Path $t.tf) { Copy-Item $t.tf ($HDIR+'\'+$c.label+'_T'+$tp+'_sample'+$hi2+'.txt') -Force; $ncopied++ } }
    }
    Write-Host ''
    Write-Host ('  {0}: repliche PASS {1}/4. Sample per lettura umana ({2}) in {3}' -f $c.label,$rep_pass,$ncopied,$HDIR) -ForegroundColor Cyan
    if ($rep_pass -eq 4) {
        Write-Host ('  -> {0} = 4/4 sotto GATE V2: primo generatore stabile della storia del progetto?' -f $c.label) -ForegroundColor Green
        Write-Host '     Resta la LETTURA UMANA (livello permanente del gate) e la decisione utente.' -ForegroundColor Green
        Write-Host '     Aspettativa a verbale: anche un pass a ~2.20 legge come word-salad strutturato;' -ForegroundColor Green
        Write-Host '     la promozione misura la stabilita'', la lingua e'' substrato. Candidato SEE-V5.' -ForegroundColor Green
    } else {
        Write-Host ('  -> {0} non regge le repliche sotto gate v2 (coda fortunata sul full).' -f $c.label) -ForegroundColor Yellow
    }
}
Write-Host ''
Write-Host 'STEP 7 (fuori harness): lettura umana dei sample da parte dell utente; poi validazione estesa /' -ForegroundColor Yellow
Write-Host 'MODOJA stack / chiusura, a decisione utente. Nessuna promozione senza la lettura umana.' -ForegroundColor Yellow
Write-Host ('  Files: ' + $RDIR + '  |  human: ' + $HDIR)
