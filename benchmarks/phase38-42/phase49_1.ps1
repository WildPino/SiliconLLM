# Phase 49.1 - Stabilized / negative feedback (ERR + ADAPT), closed-loop, anti-runaway + gate v2
#
# 49.0 found output-feedback LOAD-BEARING but MOMENTUM-pathological (POS deepens the floods;
# chRun/wsRun climb monotonically r6->r9). 49.1 keeps the SAME fixed reservoir and changes only the
# feedback CONTENT:
#   ERR   = W_fb.(onehot(b)-p)  : negative/predictive-coding feedback (reinject only the surprise).
#   ADAPT = W_fb.onehot(b) - G_adapt*c : positive feedback + spike-frequency adaptation (damper).
# POS and NO-FB weights are REUSED from 49.0 (not retrained) as references.
#
# Pre-registered (2-D: keep the signal AND kill the runaway):
#   (1) anti-runaway PRIMARY: ERR/ADAPT chRun/wsRun do NOT rise with the DAgger rounds (~= NO-FB),
#       unlike POS -> the FLOOD SCAN (closed-loop generation per round) is the read.
#   (2) keep-the-signal: TF avgVal stays BELOW NO-FB (keeps the predictive gain).
#   (3) WIN: a checkpoint that keeps the TF gain AND matches NO-FB on floods -> mini -> full gate v2
#       + structure (quote/paren/sentLen) + replicas + human reading.
# Discriminants: ERR/ADAPT-vs-POS on floods (must be much better), ERR/ADAPT-vs-NO-FB on TF (keep
# the gain). Verdict ADVISORY; the Architect reads the tables + samples.
#
# Run:  .\benchmarks\phase38-42\phase49_1.ps1
#       .\benchmarks\phase38-42\phase49_1.ps1 -SkipTrain
#       .\benchmarks\phase38-42\phase49_1.ps1 -Smoke

param([switch]$SkipTrain,[switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase49_1'
$HDIR    = $RDIR + '\human'
$R0DIR   = $ROOT + '\results\phase49_0'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
foreach ($d in @($RDIR,$HDIR)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$PR7_W   = $WDIR + '\phase47g_P_h32_r7.bin'
$ARMB_R7 = $WDIR + '\phase48a_I_h32_r7.bin'
$WP_POS  = $WDIR + '\phase49force'    # 49.0 POS (reused)
$WP_NOFB = $WDIR + '\phase49nofb'     # 49.0 NO-FB (reused)
$WP_ERR  = $WDIR + '\phase49err'
$WP_ADP  = $WDIR + '\phase49adapt'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
$ECHO_MAX = 0.05
foreach ($w in @($C2A_W,$D1_W,$PR7_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }
foreach ($r in 6..9) {
    if (-not (Test-Path ($WP_POS+'_I_h32_r'+$r+'.bin')))  { Write-Error "Missing 49.0 POS  weight r$r (run 49.0 first)";  exit 1 }
    if (-not (Test-Path ($WP_NOFB+'_I_h32_r'+$r+'.bin'))) { Write-Error "Missing 49.0 NO-FB weight r$r (run 49.0 first)"; exit 1 }
}

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

$NSEEDS  = if ($Smoke) { 4 } else { 16 }
$SCANSEEDS = if ($Smoke) { 4 } else { 8 }
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

$NAME_WORDS = @('lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy')
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
foreach ($pair in @(
    @{src='phase49_1_force.c';     exe='phase49_1_force.exe'},
    @{src='phase49_1_generator.c'; exe='phase49_1_generator.exe'},
    @{src='phase49_0_generator.c'; exe='phase49_0_generator.exe'},
    @{src='phase48a_generator.c';  exe='phase48a_generator.exe'},
    @{src='phase47_generator.c';   exe='phase47_generator.exe'},
    @{src='phase44_generator.c';   exe='phase44_generator.exe'},
    @{src='phase43_generator.c';   exe='phase43_generator.exe'})) {
    & gcc @GCC_FLAGS ('benchmarks/phase38-42/'+$pair.src) @SRC_CORE -o ($BINDIR+'\'+$pair.exe)
    if ($LASTEXITCODE -ne 0) { Write-Error ('Compile failed: '+$pair.src); exit 1 }
}
Write-Host 'Compiled OK.' -ForegroundColor Green

# ============ helpers (gate v2 byte+word; from 49.0) =================================
function Get-ByteGuardFromBytes([byte[]]$bytes,[int]$off,[int]$len) {
    $maxWs=0; $maxCh=0; $ws=0; $ch=0; $wsCount=0; $nonPrint=0; $prev=-1
    $end = $off + $len
    for ($bi=$off; $bi -lt $end; $bi++) {
        $b = $bytes[$bi]
        if ($b -eq 32 -or $b -eq 9 -or $b -eq 10 -or $b -eq 13) { $wsCount++; $ws++; if ($ws -gt $maxWs) { $maxWs=$ws }; $ch=0 }
        else { $ws=0; if ($b -eq $prev) { $ch++ } else { $ch=1 }; if ($ch -gt $maxCh) { $maxCh=$ch }; if ($b -lt 32 -or $b -gt 126) { $nonPrint++ } }
        $prev = $b
    }
    return [PSCustomObject]@{ wsRun=$maxWs; chRun=$maxCh; wsFrac=[math]::Round($wsCount/$len,4); nonPrint=$nonPrint }
}
function Mx2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
function Av2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }
function Get-StructFromText([string]$text) {
    $q = ([regex]::Matches($text,'"')).Count
    $op = ([regex]::Matches($text,'[\(\[]')).Count; $cl = ([regex]::Matches($text,'[\)\]]')).Count
    $sent = @($text -split '[.!?]+' | Where-Object { $_.Trim().Length -gt 0 })
    $slen = if ($sent.Count -gt 0) { [math]::Round((($sent | ForEach-Object { ($_ -split '\s+').Count } | Measure-Object -Average).Average),1) } else { 0 }
    return [PSCustomObject]@{ quoteParity=($q % 2); parenImbal=[math]::Abs($op-$cl); sentLen=$slen }
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
    $bytes = [System.IO.File]::ReadAllBytes($path); if ($bytes.Length -eq 0) { return $null }
    $txt = [System.Text.Encoding]::ASCII.GetString($bytes)
    $wm = Get-WordMetricsFromText $txt; if ($null -eq $wm) { return $null }
    $bg = Get-ByteGuardFromBytes $bytes 0 $bytes.Length
    $st = Get-StructFromText $txt
    return [PSCustomObject]@{ nameish=$wm.nameish; maxrun=$wm.maxrun; top_bi=$wm.top_bi; altloop=$wm.altloop;
                              wsRun=$bg.wsRun; chRun=$bg.chRun; wsFrac=$bg.wsFrac; nonPrint=$bg.nonPrint;
                              quoteParity=$st.quoteParity; parenImbal=$st.parenImbal; sentLen=$st.sentLen }
}
function Get-SelfBPB([string]$statsfile) {
    $bl = Select-String 'self_BPB' $statsfile | Select-Object -Last 1
    if ($bl) { return [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    return $null
}
function WorstOf($mets,$bpbs) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $ws=@(); $ch=@(); $wf=@(); $np=@(); $bs=@(); $qp=@(); $pi=@(); $sl=@()
    foreach ($m in @($mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop
        $ws+=[double]$m.wsRun; $ch+=[double]$m.chRun; $wf+=[double]$m.wsFrac; $np+=[double]$m.nonPrint
        $qp+=[double]$m.quoteParity; $pi+=[double]$m.parenImbal; $sl+=[double]$m.sentLen }
    foreach ($b in @($bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni=(Mx2 $ni); mr=(Mx2 $mr); bi=(Mx2 $bi); al=(Mx2 $al);
                              wsRun=(Mx2 $ws); chRun=(Mx2 $ch); wsFrac=(Mx2 $wf); nonPrint=(Mx2 $np);
                              quoteParity=(Av2 $qp); parenImbal=(Av2 $pi); sentLen=(Av2 $sl); self=(Av2 $bs); n=@($mets).Count }
}
function New-GenTask($label,$exe,$wargs,$tp,$rng,$off,$si,$suffix) {
    $lbl = $label+'_T'+$tp+'_r'+$rng+'_s'+$si+$suffix
    $tf  = $RDIR+'\gen_'+$lbl+'.txt'; $sf  = $RDIR+'\gen_'+$lbl+'_stats.txt'
    $argline = ('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                $TS_DATA,$wargs,$GEN_LEN,$WARMUP,$tp,$off,$rng)
    return [PSCustomObject]@{ label=$label; tp=$tp; rng=$rng; off=$off; si=$si; exe=($BINDIR+'\'+$exe); tf=$tf; sf=$sf; argline=$argline }
}
function Invoke-Throttled($tasks,$throttle) {
    $queue = New-Object System.Collections.Queue
    foreach ($t in $tasks) { [void]$queue.Enqueue($t) }
    $running = New-Object System.Collections.ArrayList; $launched = 0; $total = $tasks.Count
    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $throttle -and $queue.Count -gt 0) {
            $t = $queue.Dequeue()
            $p = Start-Process -FilePath $t.exe -ArgumentList $t.argline -RedirectStandardOutput $t.tf -RedirectStandardError $t.sf -NoNewWindow -PassThru
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
function Parse-KV([string]$file,[string]$prefix) {
    $out = @{}
    if (-not (Test-Path $file)) { return $out }
    foreach ($line in Select-String ('^'+$prefix+' ') $file) {
        $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
        $key = if ($prefix -eq 'BASE') { $kv['win'] } else { $kv['name'] }
        if ($null -ne $key -and $key -ne '') { $out[$key] = $kv }
    }
    return $out
}
function AvgVal($p) { if ($null -eq $p) { return 0 }; return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }
function Get-Echo([string]$file) {
    $l = (Select-String '^ECHO_STATE ' $file | Select-Object -Last 1)
    if (-not $l) { return $null }
    $kv=@{}; foreach ($tok in ($l.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
    return $kv
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
    Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f `
        $name,$tp,('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F0}' -f $w.mr),('{0:F1}' -f $w.ni),('{0:F2}' -f $w.self),`
        ('{0:F0}' -f $w.wsRun),('{0:F0}' -f $w.chRun),('{0:F2}' -f $w.wsFrac),('{0:F0}' -f $w.nonPrint),$verdict) -ForegroundColor $col
}

# byte-guard bars (contract)
$corpus = [System.IO.File]::ReadAllBytes($TS_DATA)
$stride = [long][math]::Floor(($corpus.Length - $CAL_WBYTES) / $CAL_WINDOWS)
$cal_ws=@(); $cal_ch=@(); $cal_wf=@(); $cal_np=@()
for ($wi=0; $wi -lt $CAL_WINDOWS; $wi++) { $g = Get-ByteGuardFromBytes $corpus ([long]$wi*$stride) $CAL_WBYTES
    $cal_ws += [double]$g.wsRun; $cal_ch += [double]$g.chRun; $cal_wf += [double]$g.wsFrac; $cal_np += [double]$g.nonPrint }
$corpus = $null
$BARS_DOC = $ROOT + '\docs\gatev2_bars.json'
$V2 = if (Test-Path $BARS_DOC) { Get-Content $BARS_DOC -Raw | ConvertFrom-Json } else {
    [PSCustomObject]@{ wsRun=[int]((Mx2 $cal_ws)+2); chRun=[int]((Mx2 $cal_ch)+2); wsFrac=[math]::Round((Mx2 $cal_wf)+0.03,4); nonPrint=[int]((Mx2 $cal_np)+2) } }
Write-Host ('  GATE V2 BARS (contract): wsRun<={0} chRun<={1} wsFrac<={2:F4} nonPrint<={3}' -f $V2.wsRun,$V2.chRun,$V2.wsFrac,$V2.nonPrint) -ForegroundColor DarkCyan

# ============ STEP 2: train ERR + ADAPT (POS/NO-FB reused from 49.0) =================
$TRAINERS = @('phase49_1_force','phase49_0_force','phase48a_armb','phase48a_fix')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting.'; exit 1 }
$train_extra = if ($Smoke) { @('--len','4000') } else { @() }   # ERR re-extracts each round -> keep smoke len small
$TRAIN_E = $RDIR + '\phase49err_train.txt'
$TRAIN_A = $RDIR + '\phase49adapt_train.txt'
Write-Host ''
Write-Host '=== STEP 2: train ERR then ADAPT (POS/NO-FB reused from 49.0) ===' -ForegroundColor Yellow
if ($SkipTrain -and (Test-Path $TRAIN_E) -and (Test-Path $TRAIN_A)) {
    Write-Host '  SkipTrain: reuse ERR+ADAPT train outputs' -ForegroundColor Cyan
} else {
    & "$BINDIR\phase49_1_force.exe" $TS_DATA $D1_W $WP_ERR --ftype err   @train_extra 2>&1 | Tee-Object $TRAIN_E
    if ($LASTEXITCODE -ne 0) { Write-Error 'ERR training failed'; exit 1 }
    & "$BINDIR\phase49_1_force.exe" $TS_DATA $D1_W $WP_ADP --ftype adapt @train_extra 2>&1 | Tee-Object $TRAIN_A
    if ($LASTEXITCODE -ne 0) { Write-Error 'ADAPT training failed'; exit 1 }
}

# echo-state gate
Write-Host ''
Write-Host '=== Echo-state (ESP): ratio must be small for ERR and ADAPT ===' -ForegroundColor Yellow
$echo_ok = $true
foreach ($pair in @(@{n='ERR';f=$TRAIN_E}, @{n='ADAPT';f=$TRAIN_A})) {
    $e = Get-Echo $pair.f; if (-not $e) { Write-Error ("ECHO_STATE missing from "+$pair.n); exit 1 }
    $ratio = [double]$e['ratio']; $col = if ($ratio -le $ECHO_MAX) {'Green'} else {'Red'}
    Write-Host ('  {0,-6} ratio={1:E3} (<= {2}) beta={3} gadapt={4}' -f $pair.n,$ratio,$ECHO_MAX,$e['beta'],$e['gadapt']) -ForegroundColor $col
    if ($ratio -gt $ECHO_MAX) { $echo_ok = $false }
}
if (-not $echo_ok) { Write-Host '  -> ECHO-STATE FAIL: abbassa gadapt/rho/alpha nel .c. Stop.' -ForegroundColor Red; exit 1 }
Write-Host '  -> ESP OK (ERR, ADAPT).' -ForegroundColor Green

# anchor + TF tables
$BASE_E=Parse-KV $TRAIN_E 'BASE'; $PROBE_E=Parse-KV $TRAIN_E 'PROBE'
$BASE_A=Parse-KV $TRAIN_A 'BASE'; $PROBE_A=Parse-KV $TRAIN_A 'PROBE'
$PROBE_POS=Parse-KV ($R0DIR+'\phase49force_train.txt') 'PROBE'
$PROBE_NOF=Parse-KV ($R0DIR+'\phase49nofb_train.txt') 'PROBE'
$anchor_ok = $true
foreach ($w in @('val1','val2','val3')) { $d1v=[double]$BASE_E[$w]['d1']; $i=[array]::IndexOf(@('val1','val2','val3'),$w)+1
    if ([math]::Abs([double]$PROBE_E['frozenD1']["val$i"]-$d1v) -gt 0.005) { $anchor_ok=$false } }
if (-not $anchor_ok) { Write-Error 'Anchor FAIL'; exit 1 }
Write-Host '  anchor OK' -ForegroundColor Green

Write-Host ''
Write-Host '=== TF keep-the-signal: avgVal per round (lower=better; ERR/ADAPT must stay <= NO-FB) ===' -ForegroundColor Yellow
Write-Host ('  {0,-6} {1,9} {2,9} {3,9} {4,9}' -f 'round','ERR','ADAPT','POS','NO-FB')
foreach ($rn in 1..9) {
    $n="I_r$rn"
    $e=AvgVal $PROBE_E[$n]; $a=AvgVal $PROBE_A[$n]; $p=AvgVal $PROBE_POS[$n]; $nf=AvgVal $PROBE_NOF[$n]
    Write-Host ('  {0,-6} {1,9} {2,9} {3,9} {4,9}' -f "r$rn",('{0:F4}' -f $e),('{0:F4}' -f $a),('{0:F4}' -f $p),('{0:F4}' -f $nf))
}

# ============ STEP 3: determinism (ERR r9 + ADAPT r9) ===============================
Write-Host ''
Write-Host '=== STEP 3: determinism (ERR + ADAPT generators, MD5 par vs seq) ===' -ForegroundColor Yellow
$fsz = (Get-Item $TS_DATA).Length; $margin = $WARMUP + 1024
$SEEDS_STD = @(); for ($i=0;$i -lt $NSEEDS;$i++){ $SEEDS_STD += [int]([math]::Floor($i*($fsz-$margin)/$NSEEDS)) }
$chk_seeds = @($SEEDS_STD | Select-Object -First 2)
$det_ok = $true
foreach ($v in @(@{n='ERR';w=($WP_ERR+'_I_h32_r9.bin')}, @{n='ADAPT';w=($WP_ADP+'_I_h32_r9.bin')})) {
    if (-not (Test-Path $v.w)) { Write-Error ("Missing "+$v.n+" r9"); exit 1 }
    $wargs=('"'+$D1_W+'" "'+$v.w+'"'); $seq=@(); $par=@(); $si=0
    foreach ($off in $chk_seeds) { $si++
        $seq += New-GenTask ($v.n) 'phase49_1_generator.exe' $wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_seqchk'
        $par += New-GenTask ($v.n) 'phase49_1_generator.exe' $wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_parchk' }
    Invoke-Throttled $seq 1; Invoke-Throttled $par 4
    for ($i=0;$i -lt $seq.Count;$i++){ $hs=FileMD5 $seq[$i].tf; $hp=FileMD5 $par[$i].tf; $ok=($hs -ne '' -and $hs -eq $hp); if(-not $ok){$det_ok=$false}
        Write-Host ('  {0} s{1}  {2}' -f $v.n,$seq[$i].si,($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'}) }
}
if (-not $det_ok) { Write-Error 'Determinism FAILED. Aborting.'; exit 1 }
Write-Host '  Determinism PASSED (ERR, ADAPT byte-reproducible).' -ForegroundColor Green

# ============ STEP 4: FLOOD SCAN per round (anti-runaway primary) ====================
# closed-loop generation from each round's checkpoint; worst chRun/wsRun @0.55 (floods worst there).
# POS should CLIMB with rounds; ERR/ADAPT should stay flat ~= NO-FB.
$SCAN_OFF = @(); for ($i=0;$i -lt $SCANSEEDS;$i++){ $SCAN_OFF += [int]([math]::Floor($i*($fsz-$margin)/$SCANSEEDS)) }
$variants = @(
    @{ n='ERR';   pfx=$WP_ERR;  exe='phase49_1_generator.exe' },
    @{ n='ADAPT'; pfx=$WP_ADP;  exe='phase49_1_generator.exe' },
    @{ n='POS';   pfx=$WP_POS;  exe='phase49_0_generator.exe' },
    @{ n='NO-FB'; pfx=$WP_NOFB; exe='phase49_0_generator.exe' }
)
$scan_tasks = @()
foreach ($v in $variants) { foreach ($rn in 1..9) { $wf=$v.pfx+'_I_h32_r'+$rn+'.bin'; if (-not (Test-Path $wf)) { continue }
    $si=0; foreach ($off in $SCAN_OFF) { $si++
        $scan_tasks += New-GenTask ($v.n+'#'+$rn) $v.exe ('"'+$D1_W+'" "'+$wf+'"') '0.55' $RNGS_ORIG[0] $off $si '_scan' } } }
Write-Host ''
Write-Host ('=== STEP 4: FLOOD SCAN ({0} runs, @0.55, worst chRun/wsRun per round) ===' -f $scan_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $scan_tasks $THROTTLE
$scan = @{}
foreach ($t in $scan_tasks) { $m=Get-AllMetrics $t.tf; if ($m) { if (-not $scan.ContainsKey($t.label)) { $scan[$t.label]=@{ch=@();ws=@()} }
    $scan[$t.label].ch += [double]$m.chRun; $scan[$t.label].ws += [double]$m.wsRun } }
Write-Host ''
Write-Host '  chRun (worst per round; POS climbs, ERR/ADAPT should stay flat ~= NO-FB):' -ForegroundColor Cyan
Write-Host ('  {0,-6} {1,6} {2,6} {3,6} {4,6} {5,6} {6,6} {7,6} {8,6} {9,6}' -f 'var','r1','r2','r3','r4','r5','r6','r7','r8','r9')
foreach ($v in $variants) {
    $row=@(); foreach ($rn in 1..9) { $k=$v.n+'#'+$rn; $row += $(if ($scan.ContainsKey($k)) { '{0:F0}' -f (Mx2 $scan[$k].ch) } else { '-' }) }
    Write-Host ('  {0,-6} {1,6} {2,6} {3,6} {4,6} {5,6} {6,6} {7,6} {8,6} {9,6}' -f $v.n,$row[0],$row[1],$row[2],$row[3],$row[4],$row[5],$row[6],$row[7],$row[8])
}
Write-Host '  wsRun (worst per round):' -ForegroundColor Cyan
Write-Host ('  {0,-6} {1,6} {2,6} {3,6} {4,6} {5,6} {6,6} {7,6} {8,6} {9,6}' -f 'var','r1','r2','r3','r4','r5','r6','r7','r8','r9')
foreach ($v in $variants) {
    $row=@(); foreach ($rn in 1..9) { $k=$v.n+'#'+$rn; $row += $(if ($scan.ContainsKey($k)) { '{0:F0}' -f (Mx2 $scan[$k].ws) } else { '-' }) }
    Write-Host ('  {0,-6} {1,6} {2,6} {3,6} {4,6} {5,6} {6,6} {7,6} {8,6} {9,6}' -f $v.n,$row[0],$row[1],$row[2],$row[3],$row[4],$row[5],$row[6],$row[7],$row[8])
}
# anti-runaway advisory: compare r9 flood vs NO-FB r9
$nf_ch = if ($scan.ContainsKey('NO-FB#9')) { Mx2 $scan['NO-FB#9'].ch } else { 0 }
$nf_ws = if ($scan.ContainsKey('NO-FB#9')) { Mx2 $scan['NO-FB#9'].ws } else { 0 }
Write-Host ''
Write-Host ('  anti-runaway read: NO-FB r9 chRun={0:F0} wsRun={1:F0} (the flat reference); a stabilized arm' -f $nf_ch,$nf_ws) -ForegroundColor Yellow
Write-Host '  keeps r9 floods near this AND avgVal below NO-FB (lettura Architetto, advisory).' -ForegroundColor Yellow

# ============ STEP 5: MINI gate v2 (ERR + ADAPT r6..r9) + discriminant ===============
$SEEDS_MINI = @(); for ($i=0;$i -lt 4;$i++){ $SEEDS_MINI += [int]([math]::Floor($i*($fsz-$margin)/4)) }
$mini_set = @()
foreach ($rn in 6..9) {
    foreach ($v in @(@{n='E_r';pfx=$WP_ERR;P=$PROBE_E},@{n='A_r';pfx=$WP_ADP;P=$PROBE_A})) {
        $wf=$v.pfx+'_I_h32_r'+$rn+'.bin'; if (Test-Path $wf) { $mini_set += @{ label=($v.n+$rn); wargs=('"'+$D1_W+'" "'+$wf+'"'); valbpb=(AvgVal $v.P["I_r$rn"]) } }
    }
}
$mini_tasks = @()
foreach ($tp in $TEMPS) { $si=0; foreach ($off in $SEEDS_MINI) { $si++
    foreach ($c in $mini_set) { $mini_tasks += New-GenTask ($c.label) 'phase49_1_generator.exe' $c.wargs $tp $RNGS_ORIG[0] $off $si '_mini' } } }
Write-Host ''
Write-Host ('=== STEP 5: MINI gate v2 ({0} runs) ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
$agg = @{}
foreach ($c in $mini_set) { $agg[$c.label]=@{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp]=@{mets=@();bpbs=@()} } }
foreach ($t in $mini_tasks) { $m=Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b=Get-SelfBPB $t.sf; if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b } }
Write-Host ''
Write-Host ('======== MINI v2 (qualify: bi<={0} entrambe + selfBPB sano + BPB<={1} + byte puliti) ========' -f $MINI_BI,$PROMO_BPB) -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','promo')
$promoted = @()
foreach ($c in $mini_set) {
    $w65=WorstOf $agg[$c.label]['0.65'].mets $agg[$c.label]['0.65'].bpbs
    $w55=WorstOf $agg[$c.label]['0.55'].mets $agg[$c.label]['0.55'].bpbs
    $bi_ok=($w65.bi -le $MINI_BI) -and ($w55.bi -le $MINI_BI)
    $self_ok=($w55.self -ge $BPB_LO) -and ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -le $BPB_HI)
    $bpb_ok=($c.valbpb -le $PROMO_BPB)
    $gf65=GateV2Fails $w65 '0.65'; $gf55=GateV2Fails $w55 '0.55'
    $bf65=@($gf65|Where-Object{$_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*'}); $bf55=@($gf55|Where-Object{$_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*'})
    $byte_ok=($bf65.Count -eq 0) -and ($bf55.Count -eq 0)
    $promo=if ($bi_ok -and $self_ok -and $bpb_ok -and $byte_ok) { $promoted += @{ label=$c.label; exe='phase49_1_generator.exe'; wargs=$c.wargs; valbpb=$c.valbpb }; 'FULL' } else { 'stop' }
    $col=if ($promo -eq 'FULL') {'Green'} else {'Gray'}
    PrintGateRow $c.label '0.65' $w65 '' $col; PrintGateRow $c.label '0.55' $w55 $promo $col
}
if ($promoted.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 49.1 (mini v2) ===' -ForegroundColor Yellow
    Write-Host '  -> Nessun ERR/ADAPT qualifica al full. Leggi la FLOOD SCAN + TF keep-signal:' -ForegroundColor Yellow
    Write-Host '     anti-runaway domato (chRun/wsRun piatti vs POS) + avgVal<NO-FB = memoria stabilizzata' -ForegroundColor Yellow
    Write-Host '     (anche se non passa la bar = vittoria parziale, ramo 1/3). Se ERR~=NO-FB su TF = segnale' -ForegroundColor Yellow
    Write-Host '     ucciso col runaway (inseparabili, ramo 2). Lettura Architetto.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}

# ============ STEP 6: FULL gate v2 + refs + structure ===============================
$ref_configs = @(
    @{ label='NO-FB'; exe='phase49_0_generator.exe'; wargs=('"'+$D1_W+'" "'+$WP_NOFB+'_I_h32_r9.bin"'); valbpb=2.24 },
    @{ label='POS';   exe='phase49_0_generator.exe'; wargs=('"'+$D1_W+'" "'+$WP_POS+'_I_h32_r9.bin"');  valbpb=2.24 },
    @{ label='P_r7';  exe='phase47_generator.exe';   wargs=('"'+$D1_W+'" "'+$PR7_W+'"'); valbpb=2.2497 },
    @{ label='C2.A';  exe='phase43_generator.exe';   wargs=('"'+$C2A_W+'"'); valbpb=$C2A_BPB }
)
if (Test-Path $ARMB_R7) { $ref_configs += @{ label='armB_r7'; exe='phase48a_generator.exe'; wargs=('"'+$D1_W+'" "'+$ARMB_R7+'"'); valbpb=2.230 } }
$full_set=@(); foreach ($c in $promoted) { $full_set += $c }; foreach ($r in $ref_configs) { $full_set += $r }
$full_tasks=@()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS_ORIG) { $si=0; foreach ($off in $SEEDS_STD) { $si++
    foreach ($c in $full_set) { $full_tasks += New-GenTask ($c.label) $c.exe $c.wargs $tp $rng $off $si '_full' } } } }
Write-Host ''
Write-Host ('=== STEP 6: FULL gate v2: {0} promossi + {1} ref, {2} runs ===' -f $promoted.Count,$ref_configs.Count,$full_tasks.Count) -ForegroundColor Yellow
foreach ($c in $full_set) { $agg[$c.label]=@{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp]=@{mets=@();bpbs=@()} } }
Invoke-Throttled $full_tasks $THROTTLE
foreach ($t in $full_tasks) { $m=Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b=Get-SelfBPB $t.sf; if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b } }
Write-Host ''
Write-Host '======== FULL gate v2 (worst-of-32; ref = context) ========' -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','gate')
$full_pass=@(); $fw=@{}
foreach ($c in $full_set) {
    $isref=($ref_configs|Where-Object{$_.label -eq $c.label}).Count -gt 0; $allfails=@(); $fw[$c.label]=@{}
    foreach ($tp in $TEMPS) {
        $w=WorstOf $agg[$c.label][$tp].mets $agg[$c.label][$tp].bpbs; $fw[$c.label][$tp]=$w
        $tf2=GateV2Fails $w $tp; $allfails += $tf2
        $v=if ($isref) {'ref'} elseif (@($tf2).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) {'PASS'} else {'fail'}
        $col=if ($v -eq 'PASS') {'Green'} elseif ($v -eq 'ref') {'DarkCyan'} else {'Gray'}
        PrintGateRow $c.label $tp $w $v $col
        if (-not $isref -and @($tf2).Count -gt 0) { Write-Host ('    fail: '+($tf2 -join ', ')) -ForegroundColor Red }
    }
    if (-not $isref -and @($allfails).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) { $full_pass += $c }
}
Write-Host ''
Write-Host '=== STRUCTURE advisory (@0.65 worst; quoteParity 0=bilanciato, parenImbal basso, sentLen ~corpus) ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,12} {2,12} {3,10}' -f 'config','quoteParity','parenImbal','sentLen')
foreach ($c in $full_set) { $w=$fw[$c.label]['0.65']; Write-Host ('  {0,-12} {1,12} {2,12} {3,10}' -f $c.label,$w.quoteParity,$w.parenImbal,$w.sentLen) }
if ($full_pass.Count -eq 0) {
    Write-Host ''
    Write-Host '  -> Nessun PASS pieno. Confronta ERR/ADAPT con NO-FB/POS/armB: stabilita'' tenuta + struttura' -ForegroundColor Yellow
    Write-Host '     migliore = memoria stabilizzata (ramo 1, escalation RLS); identico a NO-FB = feedback inerte.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR); exit 0
}

# ============ STEP 7: REPLICA PROTOCOL on every full-PASS ============================
$USED_WIN = @(
    @{ lo=[long][math]::Floor($fsz/5);    hi=[long]([math]::Floor($fsz/5)+1000000+$margin) },
    @{ lo=[long][math]::Floor($fsz/2);    hi=[long]([math]::Floor($fsz/2)+1000000+$margin) },
    @{ lo=[long][math]::Floor(0.65*$fsz); hi=[long]([math]::Floor(0.65*$fsz)+1000000+$margin) },
    @{ lo=[long][math]::Floor(0.80*$fsz); hi=[long]([math]::Floor(0.80*$fsz)+1000000+$margin) }
)
function New-HeldOutSeeds([double]$frac) {
    $hs=@(); for ($i=0;$i -lt $NSEEDS;$i++){ $off=[long][math]::Floor(($i+$frac)*($fsz-$margin)/$NSEEDS); $moved=$true
        while ($moved) { $moved=$false; foreach ($uw in $USED_WIN) { if ($off -ge ($uw.lo-$margin) -and $off -lt $uw.hi) { $off=$uw.hi+1048576; $moved=$true } }
            if ($off -gt ($fsz-$margin)) { $off=$off % ($fsz-$margin); $moved=$true } }
        $hs += [int]$off }
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
    Write-Host ('=== STEP 7: REPLICA PROTOCOL su {0}: 4 repliche x {1} x 2 temp ===' -f $c.label,$NSEEDS) -ForegroundColor Yellow
    $rep_tasks=@()
    foreach ($rep in $REPLICAS) { foreach ($tp in $TEMPS) { foreach ($rng in $rep.rngs) { $si=0
        foreach ($off in $rep.seeds) { $si++; $rep_tasks += New-GenTask ($c.label+'_'+$rep.name) $c.exe $c.wargs $tp $rng $off $si '' } } } }
    Invoke-Throttled $rep_tasks $THROTTLE
    $rep_agg=@{}; foreach ($rep in $REPLICAS) { $rep_agg[$rep.name]=@{}; foreach ($tp in $TEMPS) { $rep_agg[$rep.name][$tp]=@{mets=@();bpbs=@()} } }
    foreach ($t in $rep_tasks) { $repname=($t.label -split '_')[-1]; $m=Get-AllMetrics $t.tf; if ($m) { $rep_agg[$repname][$t.tp].mets += $m }
        $b=Get-SelfBPB $t.sf; if ($null -ne $b) { $rep_agg[$repname][$t.tp].bpbs += $b } }
    $rep_pass=0
    Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'replica','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','verdict')
    foreach ($rep in $REPLICAS) {
        $rfails=@(); foreach ($tp in $TEMPS) { $rfails += (GateV2Fails (WorstOf $rep_agg[$rep.name][$tp].mets $rep_agg[$rep.name][$tp].bpbs) $tp) }
        $v=if (@($rfails).Count -eq 0) { $rep_pass++; 'PASS' } else { 'FAIL' }
        foreach ($tp in $TEMPS) { $w=WorstOf $rep_agg[$rep.name][$tp].mets $rep_agg[$rep.name][$tp].bpbs
            PrintGateRow $rep.name $tp $w ($(if($tp -eq $TEMPS[0]){$v}else{''})) ($(if($v -eq 'PASS'){'Green'}else{'Red'})) }
        if (@($rfails).Count -gt 0) { Write-Host ('    fail: '+($rfails -join ', ')) -ForegroundColor Red }
    }
    $ncopied=0
    foreach ($tp in $TEMPS) { $srcs=@($rep_tasks|Where-Object{$_.label -eq ($c.label+'_R1') -and $_.tp -eq $tp -and $_.rng -eq $RNGS_NEW_A[0]}|Select-Object -First 5); $hi2=0
        foreach ($t in $srcs) { $hi2++; if (Test-Path $t.tf) { Copy-Item $t.tf ($HDIR+'\'+$c.label+'_T'+$tp+'_sample'+$hi2+'.txt') -Force; $ncopied++ } } }
    Write-Host ''
    Write-Host ('  {0}: repliche PASS {1}/4. Sample ({2}) in {3}' -f $c.label,$rep_pass,$ncopied,$HDIR) -ForegroundColor Cyan
    if ($rep_pass -eq 4) { Write-Host ('  -> {0} = 4/4 GATE V2: prima MEMORIA STABILIZZATA? Resta lettura umana + struttura.' -f $c.label) -ForegroundColor Green }
    else { Write-Host ('  -> {0} non regge le repliche.' -f $c.label) -ForegroundColor Yellow }
}
Write-Host ''
Write-Host 'STEP 8 (fuori harness): lettura umana + struttura long-range, decisione utente.' -ForegroundColor Yellow
Write-Host ('  Files: ' + $RDIR + '  |  human: ' + $HDIR)
