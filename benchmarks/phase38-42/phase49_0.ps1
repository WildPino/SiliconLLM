# Phase 49.0 - FORCE-flavored output-feedback feasibility (closed-loop), full gate v2 + discriminant
#
# Phase 48 closed the static per-step axis. 49.0 pivots to GENERATION DYNAMICS: a FIXED feedback
# reservoir h_t is appended to the armB feature; in DAgger rollout bursts the feedback byte is the
# model's OWN sample (on-policy). The readout trains on the loop with stop-gradient (h is a stored
# input -> no BPTT, no backprop into the substrate).
#
# Pipeline (reuses the 47/48 DAgger + gate v2 machinery):
#   - ECHO-STATE gate: the reservoir must forget its init (ESP) for FORCE and NO-FB.
#   - train FORCE (use_fb=1) AND the NO-FB control (use_fb=0, W_fb=0) - the discriminant.
#   - anchor (frozenD1 base-256 readout == BASE d1) + TF-per-round, for both.
#   - DETERMINISM PRE-CHECK (mandatory): the new feedback path must be byte-reproducible closed-loop
#     (MD5 parallel vs sequential) BEFORE any gate.
#   - MINI gate v2 over FORCE r6..r9 (qualify) + the FORCE-vs-NO-FB discriminant on the degeneracy
#     channels (topBi / altLp / chRun / selfBPB): the feedback must be LOAD-BEARING.
#   - FULL gate v2 + replicas: refs C2.A / D1 / P_r7 / armB I_r7 / armB I_r8 (FORCE >= armB-DAgger
#     on byte-guards at comparable BPB) + a long-range STRUCTURE advisory (quote/paren balance,
#     sentence length) + human dump.
# TF != generative (44-47): no celebration before gate v2 + replicas + human reading. The script
# verdict is ADVISORY; the Architect reads the table + samples.
#
# Run:  .\benchmarks\phase38-42\phase49_0.ps1
#       .\benchmarks\phase38-42\phase49_0.ps1 -SkipTrain
#       .\benchmarks\phase38-42\phase49_0.ps1 -Smoke

param([switch]$SkipTrain,[switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase49_0'
$HDIR    = $RDIR + '\human'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
foreach ($d in @($RDIR,$HDIR)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$PR7_W   = $WDIR + '\phase47g_P_h32_r7.bin'
$ARMB_R7 = $WDIR + '\phase48a_I_h32_r7.bin'
$ARMB_R8 = $WDIR + '\phase48a_I_h32_r8.bin'
$WP_F    = $WDIR + '\phase49force'
$WP_N    = $WDIR + '\phase49nofb'
$C2A_BPB = 2.2593
$D1_BPB  = 2.2522
$ECHO_MAX = 0.05    # echo-state: ||dh||_T / ||dh||_0 must be below this (state forgets init)
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
    @{src='phase49_0_force.c';     exe='phase49_0_force.exe'},
    @{src='phase49_0_generator.c'; exe='phase49_0_generator.exe'},
    @{src='phase48a_generator.c';  exe='phase48a_generator.exe'},
    @{src='phase47_generator.c';   exe='phase47_generator.exe'},
    @{src='phase44_generator.c';   exe='phase44_generator.exe'},
    @{src='phase43_generator.c';   exe='phase43_generator.exe'})) {
    & gcc @GCC_FLAGS ('benchmarks/phase38-42/'+$pair.src) @SRC_CORE -o ($BINDIR+'\'+$pair.exe)
    if ($LASTEXITCODE -ne 0) { Write-Error ('Compile failed: '+$pair.src); exit 1 }
}
Write-Host 'Compiled OK.' -ForegroundColor Green

# ============ Gate v2 helpers (byte-guard + word; from phase47i/48a) ==================
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
function Mx2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
function Av2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }

# long-range structure advisory: double-quote parity, net paren imbalance, mean sentence length
function Get-StructFromText([string]$text) {
    $q = ([regex]::Matches($text,'"')).Count
    $op = ([regex]::Matches($text,'[\(\[]')).Count; $cl = ([regex]::Matches($text,'[\)\]]')).Count
    $sent = @($text -split '[.!?]+' | Where-Object { $_.Trim().Length -gt 0 })
    $slen = if ($sent.Count -gt 0) { [math]::Round((($sent | ForEach-Object { ($_ -split '\s+').Count } | Measure-Object -Average).Average),1) } else { 0 }
    return [PSCustomObject]@{ quoteParity=($q % 2); parenImbal=[math]::Abs($op-$cl); sentLen=$slen }
}

# ============ STEP 1: byte-guard calibration (uses committed docs/gatev2_bars.json) ====
Write-Host ''
Write-Host ('=== STEP 1: gate v2 bars (corpus {0}x{1}B; contract = docs/gatev2_bars.json) ===' -f $CAL_WINDOWS,$CAL_WBYTES) -ForegroundColor Yellow
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
$BARS_DOC = $ROOT + '\docs\gatev2_bars.json'
if (Test-Path $BARS_DOC) {
    $existing = Get-Content $BARS_DOC -Raw | ConvertFrom-Json
    Write-Host ('  docs/gatev2_bars.json (contract): wsRun<={0} chRun<={1} wsFrac<={2:F4} nonPrint<={3} (using THESE)' -f `
        $existing.wsRun,$existing.chRun,$existing.wsFrac,$existing.nonPrint) -ForegroundColor DarkCyan
    $V2 = $existing
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
    Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f `
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
    $txt = [System.Text.Encoding]::ASCII.GetString($bytes)
    $wm = Get-WordMetricsFromText $txt
    if ($null -eq $wm) { return $null }
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
                              quoteParity=(Av2 $qp); parenImbal=(Av2 $pi); sentLen=(Av2 $sl);
                              self=(Av2 $bs); n=@($mets).Count }
}
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
function Parse-KV([string]$file,[string]$prefix) {
    $out = @{}
    foreach ($line in Select-String ('^'+$prefix+' ') $file) {
        $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
        $key = if ($prefix -eq 'BASE') { $kv['win'] } else { $kv['name'] }
        if ($null -ne $key -and $key -ne '') { $out[$key] = $kv }
    }
    return $out
}
function AvgVal($p) { return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }
function Get-Echo([string]$file) {
    $l = (Select-String '^ECHO_STATE ' $file | Select-Object -Last 1)
    if (-not $l) { return $null }
    $kv=@{}; foreach ($tok in ($l.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
    return $kv
}

# ============ STEP 2: train FORCE + NO-FB ==========================================
$TRAINERS = @('phase49_0_force','phase48a_armb','phase48a_fix')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting.'; exit 1 }

$train_extra = if ($Smoke) { @('--len','20000') } else { @() }
$TRAIN_F = $RDIR + '\phase49force_train.txt'
$TRAIN_N = $RDIR + '\phase49nofb_train.txt'
Write-Host ''
Write-Host '=== STEP 2: train FORCE (use_fb=1) then NO-FB control (use_fb=0) ===' -ForegroundColor Yellow
if ($SkipTrain -and (Test-Path $TRAIN_F) -and (Test-Path $TRAIN_N)) {
    Write-Host '  SkipTrain: reuse both train outputs' -ForegroundColor Cyan
} else {
    & "$BINDIR\phase49_0_force.exe" $TS_DATA $D1_W $WP_F @train_extra 2>&1 | Tee-Object $TRAIN_F
    if ($LASTEXITCODE -ne 0) { Write-Error 'FORCE training failed'; exit 1 }
    & "$BINDIR\phase49_0_force.exe" $TS_DATA $D1_W $WP_N @train_extra --nofb 2>&1 | Tee-Object $TRAIN_N
    if ($LASTEXITCODE -ne 0) { Write-Error 'NO-FB training failed'; exit 1 }
}

# ---- echo-state gate (both must forget init) ----
Write-Host ''
Write-Host '=== Echo-state (ESP): ||dh||_T/||dh||_0 must be small (state forgets init) ===' -ForegroundColor Yellow
$echo_ok = $true
foreach ($pair in @(@{n='FORCE';f=$TRAIN_F}, @{n='NO-FB';f=$TRAIN_N})) {
    $e = Get-Echo $pair.f
    if (-not $e) { Write-Error ("ECHO_STATE line missing from "+$pair.n); exit 1 }
    $ratio = [double]$e['ratio']
    $col = if ($ratio -le $ECHO_MAX) {'Green'} else {'Red'}
    Write-Host ('  {0,-6} ratio={1:E3} d0={2} dT={3} (<= {4})' -f $pair.n,$ratio,$e['d0'],$e['dT'],$ECHO_MAX) -ForegroundColor $col
    if ($ratio -gt $ECHO_MAX) { $echo_ok = $false }
}
if (-not $echo_ok) {
    Write-Host '  -> ECHO-STATE FAIL: il reservoir non dimentica l''init (rho troppo alto / alpha). La' -ForegroundColor Red
    Write-Host '     dinamica non e'' contrattiva: abbassa RHO o ALPHA_LEAK nel .c. Stop prima del gate.' -ForegroundColor Red
    exit 1
}
Write-Host '  -> ESP OK (entrambi): il feedback reservoir e'' echo-state.' -ForegroundColor Green

# ---- anchor + TF-per-round (both) ----
$BASE_F  = Parse-KV $TRAIN_F 'BASE';  $PROBE_F = Parse-KV $TRAIN_F 'PROBE'
$BASE_N  = Parse-KV $TRAIN_N 'BASE';  $PROBE_N = Parse-KV $TRAIN_N 'PROBE'
if (-not $PROBE_F.ContainsKey('frozenD1') -or -not $PROBE_F.ContainsKey('I_r9')) { Write-Error 'FORCE train output incomplete'; exit 1 }
if (-not $PROBE_N.ContainsKey('I_r9')) { Write-Error 'NO-FB train output incomplete'; exit 1 }
$anchor_ok = $true
foreach ($w in @('val1','val2','val3')) {
    $d1v = [double]$BASE_F[$w]['d1']; $i = [array]::IndexOf(@('val1','val2','val3'),$w)+1
    if ([math]::Abs([double]$PROBE_F['frozenD1']["val$i"]-$d1v) -gt 0.005) { $anchor_ok = $false }
}
if (-not $anchor_ok) { Write-Error 'Anchor FAIL (base feature/norm/trigram drift).'; exit 1 }
Write-Host '  anchor OK (base-256 readout reproduces BASE d1)' -ForegroundColor Green

Write-Host ''
Write-Host '=== TF per round (FORCE vs NO-FB; avgVal/valC/rollK16/rollK128/rollCF) ===' -ForegroundColor Yellow
Write-Host ('  {0,-8} {1,8} {2,8} {3,8} {4,8} {5,8}' -f 'probe','F.avgVal','N.avgVal','F.rollCF','N.rollCF','F.valC')
foreach ($rn in 1..9) {
    $n = "I_r$rn"; if (-not $PROBE_F.ContainsKey($n)) { continue }
    $pf = $PROBE_F[$n]; $pn = $PROBE_N[$n]
    $avf = AvgVal $pf; $avn = if ($pn) { AvgVal $pn } else { 0 }
    $rcf_f = if ($pf.ContainsKey('rollCF')) { $pf['rollCF'] } else { '-' }
    $rcf_n = if ($pn -and $pn.ContainsKey('rollCF')) { $pn['rollCF'] } else { '-' }
    Write-Host ('  {0,-8} {1,8} {2,8} {3,8} {4,8} {5,8}' -f $n,('{0:F4}' -f $avf),('{0:F4}' -f $avn),$rcf_f,$rcf_n,$pf['valC'])
}

# ============ STEP 3: determinism pre-check (FORCE generator, MD5 par vs seq) ========
Write-Host ''
Write-Host '=== STEP 3: determinism pre-check (FORCE feedback generator) ===' -ForegroundColor Yellow
$I_R9_F = $WP_F + '_I_h32_r9.bin'
if (-not (Test-Path $I_R9_F)) { Write-Error "Missing FORCE checkpoint: $I_R9_F"; exit 1 }
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS_STD = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS_STD += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }
$chk_seeds = @($SEEDS_STD | Select-Object -First 2)
$f_wargs = ('"'+$D1_W+'" "'+$I_R9_F+'"')
$seq=@(); $par=@(); $si=0
foreach ($off in $chk_seeds) { $si++
    $seq += New-GenTask 'F_r9' 'phase49_0_generator.exe' $f_wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_seqchk'
    $par += New-GenTask 'F_r9' 'phase49_0_generator.exe' $f_wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_parchk'
}
Invoke-Throttled $seq 1
Invoke-Throttled $par 4
$det_ok = $true
for ($i=0; $i -lt $seq.Count; $i++) {
    $hs = FileMD5 $seq[$i].tf; $hp = FileMD5 $par[$i].tf
    $ok = ($hs -ne '' -and $hs -eq $hp); if (-not $ok) { $det_ok = $false }
    Write-Host ('  s{0}  seq={1}  par={2}  {3}' -f $seq[$i].si,$hs.Substring(0,[math]::Min(8,$hs.Length)),$hp.Substring(0,[math]::Min(8,$hp.Length)),($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $det_ok) { Write-Error 'Determinism pre-check FAILED. The feedback path is non-deterministic. Aborting.'; exit 1 }
Write-Host '  Determinism pre-check PASSED. Feedback closed-loop path is byte-reproducible.' -ForegroundColor Green

# ============ STEP 4: MINI gate v2 (FORCE + NO-FB r6..r9) + discriminant =============
$SEEDS_MINI = @()
for ($i = 0; $i -lt 4; $i++) { $SEEDS_MINI += [int]([math]::Floor($i * ($fsz - $margin) / 4)) }
$mini_set = @()
foreach ($rn in 6..9) {
    $wf = $WP_F + '_I_h32_r' + $rn + '.bin'; $wn = $WP_N + '_I_h32_r' + $rn + '.bin'
    if (Test-Path $wf) { $mini_set += @{ label=("F_r$rn"); kind='FORCE'; round=$rn; wargs=('"'+$D1_W+'" "'+$wf+'"'); valbpb=(AvgVal $PROBE_F["I_r$rn"]) } }
    if (Test-Path $wn) { $mini_set += @{ label=("N_r$rn"); kind='NOFB';  round=$rn; wargs=('"'+$D1_W+'" "'+$wn+'"'); valbpb=(AvgVal $PROBE_N["I_r$rn"]) } }
}
$mini_tasks = @()
foreach ($tp in $TEMPS) { $si=0
    foreach ($off in $SEEDS_MINI) { $si++
        foreach ($c in $mini_set) { $mini_tasks += New-GenTask ($c.label) 'phase49_0_generator.exe' $c.wargs $tp $RNGS_ORIG[0] $off $si '_mini' }
    } }
Write-Host ''
Write-Host ('=== STEP 4: MINI gate v2 ({0} runs: FORCE+NO-FB r6..r9) ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
$agg = @{}
foreach ($c in $mini_set) { $agg[$c.label] = @{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
foreach ($t in $mini_tasks) {
    $m = Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf;    if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host ('======== MINI v2 (qualify FORCE: bi<={0} entrambe + selfBPB sano + BPB<={1} + byte puliti) ========' -f $MINI_BI,$PROMO_BPB) -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','promo')
$promoted = @()
$worst = @{}
foreach ($c in $mini_set) {
    $w65 = WorstOf $agg[$c.label]['0.65'].mets $agg[$c.label]['0.65'].bpbs
    $w55 = WorstOf $agg[$c.label]['0.55'].mets $agg[$c.label]['0.55'].bpbs
    $worst[$c.label] = @{ '0.65'=$w65; '0.55'=$w55; kind=$c.kind; round=$c.round; valbpb=$c.valbpb; wargs=$c.wargs }
    $promo = 'stop'
    if ($c.kind -eq 'FORCE') {
        $bi_ok   = ($w65.bi -le $MINI_BI) -and ($w55.bi -le $MINI_BI)
        $self_ok = ($w55.self -ge $BPB_LO) -and ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -le $BPB_HI)
        $bpb_ok  = ($c.valbpb -le $PROMO_BPB)
        $gf65 = GateV2Fails $w65 '0.65'; $gf55 = GateV2Fails $w55 '0.55'
        $bf65 = @($gf65 | Where-Object { $_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*' })
        $bf55 = @($gf55 | Where-Object { $_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*' })
        $byte_ok = ($bf65.Count -eq 0) -and ($bf55.Count -eq 0)
        if ($bi_ok -and $self_ok -and $bpb_ok -and $byte_ok) { $promoted += @{ label=$c.label; exe='phase49_0_generator.exe'; wargs=$c.wargs; valbpb=$c.valbpb }; $promo='FULL' }
    }
    $col = if ($promo -eq 'FULL') {'Green'} elseif ($c.kind -eq 'NOFB') {'DarkGray'} else {'Gray'}
    PrintGateRow $c.label '0.65' $w65 '' $col
    PrintGateRow $c.label '0.55' $w55 $promo $col
}
# ---- DISCRIMINANT: FORCE vs NO-FB per round on the degeneracy channels (feedback load-bearing?) ----
Write-Host ''
Write-Host '=== DISCRIMINANT FORCE-vs-NO-FB (worst@0.55; negativo = FORCE meglio sul canale) ===' -ForegroundColor Yellow
Write-Host ('  {0,-6} {1,10} {2,10} {3,10} {4,10}' -f 'round','d.topBi','d.altLp','d.chRun','d.selfBPB')
$disc_force_better = 0; $disc_n = 0
foreach ($rn in 6..9) {
    $fl="F_r$rn"; $nl="N_r$rn"
    if (-not ($worst.ContainsKey($fl) -and $worst.ContainsKey($nl))) { continue }
    $f=$worst[$fl]['0.55']; $n=$worst[$nl]['0.55']
    $dbi=[math]::Round($f.bi-$n.bi,1); $dal=[math]::Round($f.al-$n.al,1); $dch=[math]::Round($f.chRun-$n.chRun,1); $dsb=[math]::Round($f.self-$n.self,3)
    Write-Host ('  {0,-6} {1,10} {2,10} {3,10} {4,10}' -f "r$rn",$dbi,$dal,$dch,$dsb)
    $disc_n++; if (($dbi + $dal + $dch) -lt 0) { $disc_force_better++ }
}
Write-Host ('  -> FORCE batte NO-FB sui canali di degenerazione in {0}/{1} round (advisory; lettura Architetto).' -f $disc_force_better,$disc_n) -ForegroundColor Cyan

if ($promoted.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 49.0 (mini v2) ===' -ForegroundColor Yellow
    Write-Host '  -> Nessun checkpoint FORCE qualifica al full. Leggi il DISCRIMINANT: se FORCE ~= NO-FB,' -ForegroundColor Yellow
    Write-Host '     il feedback e'' inerte (la dinamica del reservoir e'' troppo debole) -> ramo 2 (arricchire' -ForegroundColor Yellow
    Write-Host '     la ricorrenza / unit-choice). Se FORCE migliora ma non passa la bar -> vittoria parziale.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}

# ============ STEP 5: FULL gate v2 (promoted FORCE) + refs ===========================
$ref_configs = @(
    @{ label='C2.A';   exe='phase43_generator.exe';   wargs=('"'+$C2A_W+'"');             valbpb=$C2A_BPB },
    @{ label='D1';     exe='phase44_generator.exe';   wargs=('"'+$D1_W+'"');              valbpb=$D1_BPB  },
    @{ label='P_r7';   exe='phase47_generator.exe';   wargs=('"'+$D1_W+'" "'+$PR7_W+'"'); valbpb=2.2497  }
)
if (Test-Path $ARMB_R7) { $ref_configs += @{ label='armB_r7'; exe='phase48a_generator.exe'; wargs=('"'+$D1_W+'" "'+$ARMB_R7+'"'); valbpb=2.230 } }
if (Test-Path $ARMB_R8) { $ref_configs += @{ label='armB_r8'; exe='phase48a_generator.exe'; wargs=('"'+$D1_W+'" "'+$ARMB_R8+'"'); valbpb=2.230 } }
$full_set = @(); foreach ($c in $promoted) { $full_set += $c }; foreach ($r in $ref_configs) { $full_set += $r }
$full_tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS_ORIG) { $si=0
    foreach ($off in $SEEDS_STD) { $si++
        foreach ($c in $full_set) { $full_tasks += New-GenTask ($c.label) $c.exe $c.wargs $tp $rng $off $si '_full' }
    } } }
Write-Host ''
Write-Host ('=== STEP 5: FULL gate v2: {0} FORCE promossi + {1} reference, {2} runs ===' -f $promoted.Count,$ref_configs.Count,$full_tasks.Count) -ForegroundColor Yellow
foreach ($c in $full_set) { $agg[$c.label] = @{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
Invoke-Throttled $full_tasks $THROTTLE
foreach ($t in $full_tasks) {
    $m = Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf;    if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host '======== FULL gate v2 (worst-of-32, word + byte; ref = context) ========' -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','gate')
$full_pass = @()
$full_worst = @{}
foreach ($c in $full_set) {
    $isref = ($ref_configs | Where-Object { $_.label -eq $c.label }).Count -gt 0
    $allfails = @()
    $full_worst[$c.label] = @{}
    foreach ($tp in $TEMPS) {
        $w = WorstOf $agg[$c.label][$tp].mets $agg[$c.label][$tp].bpbs
        $full_worst[$c.label][$tp] = $w
        $tf2 = GateV2Fails $w $tp; $allfails += $tf2
        $v = if ($isref) { 'ref' } elseif (@($tf2).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) { 'PASS' } else { 'fail' }
        $col = if ($v -eq 'PASS') {'Green'} elseif ($v -eq 'ref') {'DarkCyan'} else {'Gray'}
        PrintGateRow $c.label $tp $w $v $col
        if (-not $isref -and @($tf2).Count -gt 0) { Write-Host ('    fail: ' + ($tf2 -join ', ')) -ForegroundColor Red }
    }
    if (-not $isref -and @($allfails).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) { $full_pass += $c }
}
# ---- long-range STRUCTURE advisory: FORCE promossi vs armB refs ----
Write-Host ''
Write-Host '=== STRUCTURE advisory (long-range; @0.65 worst; quoteParity 0=bilanciato, parenImbal basso, sentLen ~corpus) ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,12} {2,12} {3,10}' -f 'config','quoteParity','parenImbal','sentLen')
foreach ($c in $full_set) {
    $w = $full_worst[$c.label]['0.65']
    Write-Host ('  {0,-12} {1,12} {2,12} {3,10}' -f $c.label,$w.quoteParity,$w.parenImbal,$w.sentLen)
}

if ($full_pass.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 49.0 (full v2) ===' -ForegroundColor Yellow
    Write-Host '  -> FORCE promosso ma nessun PASS pieno. Confronta con armB_r7/r8: se FORCE e'' byte-pari' -ForegroundColor Yellow
    Write-Host '     ma struttura migliore (quote/paren/sentLen) = il feedback porta dinamica (ramo 1/3);' -ForegroundColor Yellow
    Write-Host '     se identico ad armB = feedback inerte (ramo 2). Lettura Architetto.' -ForegroundColor Yellow
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
    Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'replica','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','verdict')
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
        Write-Host ('  -> {0} = 4/4 sotto GATE V2: primo generatore stabile con feedback?' -f $c.label) -ForegroundColor Green
        Write-Host '     Resta la LETTURA UMANA (gate finale) + struttura long-range. Decisione utente.' -ForegroundColor Green
    } else {
        Write-Host ('  -> {0} non regge le repliche (coda fortunata sul full).' -f $c.label) -ForegroundColor Yellow
    }
}
Write-Host ''
Write-Host 'STEP 7 (fuori harness): lettura umana dei sample + struttura long-range, decisione utente.' -ForegroundColor Yellow
Write-Host ('  Files: ' + $RDIR + '  |  human: ' + $HDIR)
