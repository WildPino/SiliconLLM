# Phase 50.A - BPE-1024 token readout over the byte-driven armB substrate: closed-loop gate.
#
# 50.0 picked BPE-1024. 50.A wires it into the readout (substrate INVARIATO) and asks the gate
# whether emitting whole tokens makes the byte-degeneration channels (chRun/wsRun/...) structurally
# impossible -> first promoted generator. The output is BYTES (decoded text) so gate v2 applies
# UNCHANGED: byte-guards from docs/gatev2_bars.json + word/struct guards (topBi/altLp/...) on the
# decoded text + BPB in bits-per-BYTE (bar 2.2543) + structure + replicas + human reading.
#
# Tree: (1) byte-guards pass (now easy) + topBi clean + BPB<bar + reads cleaner -> FIRST PROMOTED
# generator (byte-fidelity solved; long-range structure = charter-question for later). (2) byte-
# guards pass but reads like armB word-salad (just cleaner tokens) -> byte-fidelity solved, structure
# = charter-question; honest partial win. (3) token-flood (repeats valid tokens) -> diagnose.
#
# Run:  .\benchmarks\phase38-42\phase50_a.ps1
#       .\benchmarks\phase38-42\phase50_a.ps1 -SkipTrain
#       .\benchmarks\phase38-42\phase50_a.ps1 -Smoke

param([switch]$SkipTrain,[switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase50_a'
$HDIR    = $RDIR + '\human'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
foreach ($d in @($RDIR,$HDIR)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$D1_W    = $WDIR + '\phase44f_F0.bin'
$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$ARMB_R7 = $WDIR + '\phase48afix_I_h32_r7.bin'
$BPE_W   = $WDIR + '\bpe1024.bin'
$WP_TOK  = $WDIR + '\phase50a_tok'
$C2A_BPB = 2.2593
foreach ($w in @($D1_W,$C2A_W)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

$NSEEDS   = if ($Smoke) { 4 } else { 16 }
$RNGS_ORIG  = @(12345, 67890)
$RNGS_NEW_A = @(24680, 13579)
$RNGS_NEW_B = @(98765, 55555)
$TEMPS   = @('0.65','0.55')
$GEN_LEN = 2000
$WARMUP  = 5000
$MINLEN  = 2
$BPB_LO  = 0.6   # token self-BPB runs lower than byte models (more confident); widened floor
$BPB_HI  = 2.0
$PROMO_BPB = 2.2543
$G_NAME = 20.0; $G_RUN = 5.0; $G_BI = 8.0; $G_AL = 2.0
$MINI_BI = 7.0
$THROTTLE = 12
$TRAIN_LEN = if ($Smoke) { '40000' } else { '1000000' }
$CAL_WINDOWS = if ($Smoke) { 200 } else { 1000 }
$CAL_WBYTES  = 2048

$NAME_WORDS = @('lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy')
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- helpers (gate v2 byte+word; from 49.x) -----------------------------
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
    Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,6} {9,7} {10,7} {11,8}' -f `
        $name,$tp,('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F0}' -f $w.mr),('{0:F1}' -f $w.ni),('{0:F2}' -f $w.self),`
        ('{0:F0}' -f $w.wsRun),('{0:F0}' -f $w.chRun),('{0:F3}' -f $w.wsFrac),('{0:F0}' -f $w.nonPrint),$verdict) -ForegroundColor $col
}
function Parse-Probe([string]$file) {
    $out=@{}
    foreach ($line in Select-String '^PROBE name=I_r' $file) {
        $kv=@{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; if ($p.Count -eq 2) { $kv[$p[0]]=$p[1] } }
        if ($kv['name']) { $out[$kv['name']]=$kv }
    }
    return $out
}
function ValBPB($p) { if ($null -eq $p) { return 9 }; return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }

# byte-guard bars (contract)
$BARS_DOC = $ROOT + '\docs\gatev2_bars.json'
$V2 = if (Test-Path $BARS_DOC) { Get-Content $BARS_DOC -Raw | ConvertFrom-Json } else {
    [PSCustomObject]@{ wsRun=6; chRun=8; wsFrac=0.2502; nonPrint=135 } }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
foreach ($pair in @(
    @{src='phase50_0_map.c';       exe='phase50_0_map.exe';       core=$false},
    @{src='phase50_a_token.c';     exe='phase50_a_token.exe';     core=$true},
    @{src='phase50_a_generator.c'; exe='phase50_a_generator.exe'; core=$true},
    @{src='phase48a_generator.c';  exe='phase48a_generator.exe';  core=$true},
    @{src='phase43_generator.c';   exe='phase43_generator.exe';   core=$true})) {
    if ($pair.core) { & gcc @GCC_FLAGS ('benchmarks/phase38-42/'+$pair.src) @SRC_CORE -o ($BINDIR+'\'+$pair.exe) }
    else            { & gcc -O3 -march=native ('benchmarks/phase38-42/'+$pair.src) -o ($BINDIR+'\'+$pair.exe) -lm }
    if ($LASTEXITCODE -ne 0) { Write-Error ('Compile failed: '+$pair.src); exit 1 }
}
Write-Host 'Compiled OK.' -ForegroundColor Green
Write-Host ('  GATE V2 BARS: wsRun<={0} chRun<={1} wsFrac<={2:F4} nonPrint<={3}  BPB bar<={4}' -f $V2.wsRun,$V2.chRun,$V2.wsFrac,$V2.nonPrint,$PROMO_BPB) -ForegroundColor DarkCyan

# -------- STEP 0: save BPE-1024 merges (full corpus, deterministic) -----------
$mbflag = if ($Smoke) { @('--max-bytes','2000000') } else { @() }
if (-not (Test-Path $BPE_W) -or -not $SkipTrain) {
    Write-Host ''
    Write-Host '=== STEP 0: save BPE-1024 merges (full corpus) ===' -ForegroundColor Yellow
    & "$BINDIR\phase50_0_map.exe" $TS_DATA @mbflag --bpe-only --save-bpe $BPE_W --bpe-vocab 1024 2>&1 | Tee-Object ($RDIR+'\bpe_save.txt') | Select-Object -Last 3
    if (-not (Test-Path $BPE_W)) { Write-Error 'BPE save failed'; exit 1 }
}

# -------- STEP 1: train -------------------------------------------------------
$TRAIN_OUT = $RDIR + '\phase50a_train.txt'
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq 'phase50_a_token' })
if ($busy.Count -gt 0) { Write-Error 'A trainer is already running. Aborting.'; exit 1 }
Write-Host ''
Write-Host '=== STEP 1: train token readout (DAgger, 9 rounds) ===' -ForegroundColor Yellow
if ($SkipTrain -and (Test-Path $TRAIN_OUT)) {
    Write-Host '  SkipTrain: reuse train output' -ForegroundColor Cyan
} else {
    $tmax = if ($Smoke) { @('--max-bytes','4000000') } else { @() }
    & "$BINDIR\phase50_a_token.exe" $TS_DATA $D1_W $BPE_W $WP_TOK --len $TRAIN_LEN @tmax 2>&1 | Tee-Object $TRAIN_OUT | Select-String '^PROBE|^BASE|round '
    if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
}
$PROBE = Parse-Probe $TRAIN_OUT
Write-Host ''
Write-Host '=== TF val bits/byte per round (bar 2.2543) ===' -ForegroundColor Yellow
Write-Host ('  {0,-6} {1,9} {2,9} {3,9} {4,9}' -f 'round','val1','val2','val3','mean')
foreach ($rn in 1..9) { $p=$PROBE["I_r$rn"]; if ($p) {
    Write-Host ('  {0,-6} {1,9} {2,9} {3,9} {4,9}' -f "r$rn",$p['val1'],$p['val2'],$p['val3'],('{0:F4}' -f (ValBPB $p))) } }

# -------- STEP 2: determinism (token generator) -------------------------------
Write-Host ''
Write-Host '=== STEP 2: determinism (token gen, MD5 par vs seq) ===' -ForegroundColor Yellow
$fsz = (Get-Item $TS_DATA).Length; $margin = $WARMUP + 1024
$SEEDS_STD = @(); for ($i=0;$i -lt $NSEEDS;$i++){ $SEEDS_STD += [int]([math]::Floor($i*($fsz-$margin)/$NSEEDS)) }
$chk_seeds = @($SEEDS_STD | Select-Object -First 2)
$tokw = ($WP_TOK+'_I_h32_r9.bin'); if (-not (Test-Path $tokw)) { Write-Error 'Missing r9 checkpoint'; exit 1 }
$wargs=('"'+$D1_W+'" "'+$tokw+'"'); $seq=@(); $par=@(); $si=0; $det_ok=$true
foreach ($off in $chk_seeds) { $si++
    $seq += New-GenTask 'TOK' 'phase50_a_generator.exe' $wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_seqchk'
    $par += New-GenTask 'TOK' 'phase50_a_generator.exe' $wargs $TEMPS[0] $RNGS_ORIG[0] $off $si '_parchk' }
Invoke-Throttled $seq 1; Invoke-Throttled $par 4
for ($i=0;$i -lt $seq.Count;$i++){ $hs=FileMD5 $seq[$i].tf; $hp=FileMD5 $par[$i].tf; $ok=($hs -ne '' -and $hs -eq $hp); if(-not $ok){$det_ok=$false}
    Write-Host ('  TOK s{0}  {1}' -f $seq[$i].si,($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'}) }
if (-not $det_ok) { Write-Error 'Determinism FAILED'; exit 1 }
Write-Host '  Determinism PASSED.' -ForegroundColor Green

# -------- STEP 3: BYTE-GUARD vs armB (the 50.A headline) -----------------------
# closed-loop @0.55 worst byte-guards per round; tokens should make chRun/wsRun structurally tiny.
$SCANSEEDS = if ($Smoke) { 4 } else { 8 }
$SCAN_OFF = @(); for ($i=0;$i -lt $SCANSEEDS;$i++){ $SCAN_OFF += [int]([math]::Floor($i*($fsz-$margin)/$SCANSEEDS)) }
$scan_tasks=@()
foreach ($rn in 6..9) { $wf=$WP_TOK+'_I_h32_r'+$rn+'.bin'; if (-not (Test-Path $wf)) { continue }
    $si=0; foreach ($off in $SCAN_OFF) { $si++
        $scan_tasks += New-GenTask ('TOK#'+$rn) 'phase50_a_generator.exe' ('"'+$D1_W+'" "'+$wf+'"') '0.55' $RNGS_ORIG[0] $off $si '_scan' } }
if (Test-Path $ARMB_R7) { $si=0; foreach ($off in $SCAN_OFF) { $si++
        $scan_tasks += New-GenTask 'armB#7' 'phase48a_generator.exe' ('"'+$D1_W+'" "'+$ARMB_R7+'"') '0.55' $RNGS_ORIG[0] $off $si '_scan' } }
Write-Host ''
Write-Host ('=== STEP 3: BYTE-GUARD scan ({0} runs, @0.55, worst per group) ===' -f $scan_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $scan_tasks $THROTTLE
$scan=@{}
foreach ($t in $scan_tasks) { $m=Get-AllMetrics $t.tf; if ($m) { if (-not $scan.ContainsKey($t.label)) { $scan[$t.label]=@{ch=@();ws=@();wf=@();np=@()} }
    $scan[$t.label].ch+=[double]$m.chRun; $scan[$t.label].ws+=[double]$m.wsRun; $scan[$t.label].wf+=[double]$m.wsFrac; $scan[$t.label].np+=[double]$m.nonPrint } }
Write-Host ('  {0,-9} {1,8} {2,8} {3,8} {4,8}' -f 'group','chRun','wsRun','wsFrac','nonPrint')
foreach ($k in @('TOK#6','TOK#7','TOK#8','TOK#9','armB#7')) { if ($scan.ContainsKey($k)) { $s=$scan[$k]
    Write-Host ('  {0,-9} {1,8} {2,8} {3,8:F3} {4,8}' -f $k,('{0:F0}' -f (Mx2 $s.ch)),('{0:F0}' -f (Mx2 $s.ws)),(Mx2 $s.wf),('{0:F0}' -f (Mx2 $s.np))) `
        -ForegroundColor $(if($k -like 'armB*'){'DarkCyan'}else{'Gray'}) } }
Write-Host ('  bars: chRun<={0} wsRun<={1} wsFrac<={2:F4} nonPrint<={3}  (tokens should sit FAR under armB)' -f $V2.chRun,$V2.wsRun,$V2.wsFrac,$V2.nonPrint) -ForegroundColor DarkCyan

# -------- STEP 4: MINI gate v2 (r6-9 both temps) ------------------------------
$SEEDS_MINI = @(); for ($i=0;$i -lt 4;$i++){ $SEEDS_MINI += [int]([math]::Floor($i*($fsz-$margin)/4)) }
$mini_set=@()
foreach ($rn in 6..9) { $wf=$WP_TOK+'_I_h32_r'+$rn+'.bin'; if (Test-Path $wf) {
    $mini_set += @{ label=("r$rn"); wargs=('"'+$D1_W+'" "'+$wf+'"'); valbpb=(ValBPB $PROBE["I_r$rn"]) } } }
$mini_tasks=@()
foreach ($tp in $TEMPS) { $si=0; foreach ($off in $SEEDS_MINI) { $si++
    foreach ($c in $mini_set) { $mini_tasks += New-GenTask ($c.label) 'phase50_a_generator.exe' $c.wargs $tp $RNGS_ORIG[0] $off $si '_mini' } } }
Write-Host ''
Write-Host ('=== STEP 4: MINI gate v2 ({0} runs) ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
$agg=@{}
foreach ($c in $mini_set) { $agg[$c.label]=@{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp]=@{mets=@();bpbs=@()} } }
foreach ($t in $mini_tasks) { $m=Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b=Get-SelfBPB $t.sf; if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b } }
Write-Host ('======== MINI v2 (qualify: bi<={0} both + selfBPB sane + valBPB<={1} + byte clean) ========' -f $MINI_BI,$PROMO_BPB) -ForegroundColor Cyan
Write-Host '  (0.65 row promo col shows valBPB; 0.55 row promo col shows FULL/stop)' -ForegroundColor DarkGray
Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,6} {9,7} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','promo')
$promoted=@()
foreach ($c in $mini_set) {
    $w65=WorstOf $agg[$c.label]['0.65'].mets $agg[$c.label]['0.65'].bpbs
    $w55=WorstOf $agg[$c.label]['0.55'].mets $agg[$c.label]['0.55'].bpbs
    $bi_ok=($w65.bi -le $MINI_BI) -and ($w55.bi -le $MINI_BI)
    $self_ok=($w55.self -ge $BPB_LO) -and ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -le $BPB_HI)
    $bpb_ok=($c.valbpb -le $PROMO_BPB)
    $gf65=GateV2Fails $w65 '0.65'; $gf55=GateV2Fails $w55 '0.55'
    $bf65=@($gf65|Where-Object{$_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*'}); $bf55=@($gf55|Where-Object{$_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*'})
    $byte_ok=($bf65.Count -eq 0) -and ($bf55.Count -eq 0)
    $promo=if ($bi_ok -and $self_ok -and $bpb_ok -and $byte_ok) { $promoted += @{ label=$c.label; wargs=$c.wargs; valbpb=$c.valbpb }; 'FULL' } else { 'stop' }
    $col=if ($promo -eq 'FULL') {'Green'} else {'Gray'}
    PrintGateRow $c.label '0.65' $w65 ('{0:F3}' -f $c.valbpb) $col; PrintGateRow $c.label '0.55' $w55 $promo $col
}
if ($promoted.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 50.A (mini) ===' -ForegroundColor Yellow
    Write-Host '  -> Nessuno qualifica al full. Leggi la BYTE-GUARD scan (tokens vs armB) + i sample umani:' -ForegroundColor Yellow
    Write-Host '     se byte-guards crollano vs armB ma topBi/struttura restano = byte-fidelity risolta,' -ForegroundColor Yellow
    Write-Host '     struttura = charter-question (ramo 2). Se token-flood (bi alto) = ramo 3. Lettura Architetto.' -ForegroundColor Yellow
    # dump a few human samples anyway
    $hs=@(); $si=0; foreach ($off in ($SEEDS_MINI|Select-Object -First 3)) { $si++
        foreach ($tp in $TEMPS) { foreach ($rn in @(8,9)) { $wf=$WP_TOK+'_I_h32_r'+$rn+'.bin'
            $hs += New-GenTask ("r$rn") 'phase50_a_generator.exe' ('"'+$D1_W+'" "'+$wf+'"') $tp $RNGS_ORIG[0] $off $si '_hum' } } }
    Invoke-Throttled $hs $THROTTLE
    foreach ($t in $hs) { if (Test-Path $t.tf) { Copy-Item $t.tf ($HDIR+'\'+$t.label+'_T'+$t.tp+'_s'+$t.si+'.txt') -Force } }
    Write-Host ('  Human samples in: ' + $HDIR) -ForegroundColor Cyan
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}

# -------- STEP 5: FULL gate v2 + refs + structure -----------------------------
$ref_configs=@(
    @{ label='C2.A'; exe='phase43_generator.exe'; wargs=('"'+$C2A_W+'"'); valbpb=$C2A_BPB })
if (Test-Path $ARMB_R7) { $ref_configs += @{ label='armB_r7'; exe='phase48a_generator.exe'; wargs=('"'+$D1_W+'" "'+$ARMB_R7+'"'); valbpb=2.230 } }
$full_set=@(); foreach ($c in $promoted) { $c.exe='phase50_a_generator.exe'; $full_set += $c }; foreach ($r in $ref_configs) { $full_set += $r }
$full_tasks=@()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS_ORIG) { $si=0; foreach ($off in $SEEDS_STD) { $si++
    foreach ($c in $full_set) { $full_tasks += New-GenTask ($c.label) $c.exe $c.wargs $tp $rng $off $si '_full' } } } }
Write-Host ''
Write-Host ('=== STEP 5: FULL gate v2: {0} promossi + {1} ref, {2} runs ===' -f $promoted.Count,$ref_configs.Count,$full_tasks.Count) -ForegroundColor Yellow
foreach ($c in $full_set) { $agg[$c.label]=@{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp]=@{mets=@();bpbs=@()} } }
Invoke-Throttled $full_tasks $THROTTLE
foreach ($t in $full_tasks) { $m=Get-AllMetrics $t.tf; if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b=Get-SelfBPB $t.sf; if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b } }
Write-Host '======== FULL gate v2 (worst-of-32; ref = context) ========' -ForegroundColor Cyan
Write-Host ('  {0,-12} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,6} {9,7} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','gate')
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
Write-Host '=== STRUCTURE advisory (@0.65 worst) ===' -ForegroundColor Yellow
Write-Host ('  {0,-12} {1,12} {2,12} {3,10}' -f 'config','quoteParity','parenImbal','sentLen')
foreach ($c in $full_set) { $w=$fw[$c.label]['0.65']; Write-Host ('  {0,-12} {1,12} {2,12} {3,10}' -f $c.label,$w.quoteParity,$w.parenImbal,$w.sentLen) }

# -------- STEP 6: replicas + human on every full-PASS -------------------------
$REPLICAS=@(
    @{ name='R1'; rngs=$RNGS_NEW_A; seeds=$SEEDS_STD },
    @{ name='R2'; rngs=$RNGS_NEW_B; seeds=$SEEDS_STD })
foreach ($c in $full_pass) {
    Write-Host ''
    Write-Host ('=== STEP 6: REPLICA su {0} ===' -f $c.label) -ForegroundColor Yellow
    $rep_tasks=@()
    foreach ($rep in $REPLICAS) { foreach ($tp in $TEMPS) { foreach ($rng in $rep.rngs) { $si=0
        foreach ($off in $rep.seeds) { $si++; $rep_tasks += New-GenTask ($c.label+'_'+$rep.name) $c.exe $c.wargs $tp $rng $off $si '' } } } }
    Invoke-Throttled $rep_tasks $THROTTLE
    $rep_agg=@{}; foreach ($rep in $REPLICAS) { $rep_agg[$rep.name]=@{}; foreach ($tp in $TEMPS) { $rep_agg[$rep.name][$tp]=@{mets=@();bpbs=@()} } }
    foreach ($t in $rep_tasks) { $repname=($t.label -split '_')[-1]; $m=Get-AllMetrics $t.tf; if ($m) { $rep_agg[$repname][$t.tp].mets += $m }
        $b=Get-SelfBPB $t.sf; if ($null -ne $b) { $rep_agg[$repname][$t.tp].bpbs += $b } }
    $rep_pass=0
    foreach ($rep in $REPLICAS) {
        $rfails=@(); foreach ($tp in $TEMPS) { $rfails += (GateV2Fails (WorstOf $rep_agg[$rep.name][$tp].mets $rep_agg[$rep.name][$tp].bpbs) $tp) }
        $v=if (@($rfails).Count -eq 0) { $rep_pass++; 'PASS' } else { 'FAIL' }
        foreach ($tp in $TEMPS) { $w=WorstOf $rep_agg[$rep.name][$tp].mets $rep_agg[$rep.name][$tp].bpbs
            PrintGateRow $rep.name $tp $w ($(if($tp -eq $TEMPS[0]){$v}else{''})) ($(if($v -eq 'PASS'){'Green'}else{'Red'})) }
        if (@($rfails).Count -gt 0) { Write-Host ('    fail: '+($rfails -join ', ')) -ForegroundColor Red }
    }
    $ncopied=0
    foreach ($tp in $TEMPS) { $srcs=@($rep_tasks|Where-Object{$_.label -eq ($c.label+'_R1') -and $_.tp -eq $tp -and $_.rng -eq $RNGS_NEW_A[0]}|Select-Object -First 5)
        $hi2=0; foreach ($t in $srcs) { $hi2++; if (Test-Path $t.tf) { Copy-Item $t.tf ($HDIR+'\'+$c.label+'_T'+$tp+'_sample'+$hi2+'.txt') -Force; $ncopied++ } } }
    Write-Host ('  {0}: repliche PASS {1}/2. Sample ({2}) in {3}' -f $c.label,$rep_pass,$ncopied,$HDIR) -ForegroundColor Cyan
    if ($rep_pass -eq 2) { Write-Host ('  -> {0} = PRIMO GENERATORE PROMOSSO? Resta lettura umana (struttura long-range).' -f $c.label) -ForegroundColor Green }
}
Write-Host ''
Write-Host 'STEP 7 (fuori harness): lettura umana + struttura long-range, decisione utente.' -ForegroundColor Yellow
Write-Host ('  Files: ' + $RDIR + '  |  human: ' + $HDIR)
