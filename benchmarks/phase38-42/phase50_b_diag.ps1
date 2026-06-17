# Phase 50.B-diag - classify topBi (corpus calibration + re-posed premise). NO training, NO hack.
#
# The single-token premise falsified "topBi = coverable token-flood" (seeded~=natural). Two cheap
# measures now decide WHAT topBi is:
#   MISURA A (corpus calibration): the word-guards (topBi/altLp/runWst/nameWst) on N held-out windows
#     of REAL TinyStories, in the gate's identical window (GEN_LEN bytes ~ 350 words, same bigram
#     count) + the generator's current topBi -> if generator ~= corpus, the bar 7 is mis-calibrated
#     for the token stream; recalibrate to corpus-p90 (documented, like gatev2_bars.json).
#   MISURA B (re-posed premise, r5 AND deployed r9): seed the REAL topBi modes (alternation "A B A B",
#     content-word dup) -> seeded >> natural = coverable attractor; seeded ~= natural = not coverable.
#
# Tree (pre-registered): (1) generator topBi ~= corpus -> bar-artifact -> recalibrate -> 50.A PASS
# (first promoted generator; "loses the thread" = charter-question, separate). (2) topBi >> corpus AND
# alt/dup coverable -> 50.C targeted coverage. (3) topBi >> corpus AND not coverable -> template-reset
# long-range = charter-question -> 50.A+B = branch-2 milestone to commit, then the charter-question.
#
# Run:  .\benchmarks\phase38-42\phase50_b_diag.ps1
#       .\benchmarks\phase38-42\phase50_b_diag.ps1 -Smoke

param([switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase50_b_diag'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$D1_W   = $WDIR + '\phase44f_F0.bin'
$BPE_W  = $WDIR + '\bpe1024.bin'
$R5     = $WDIR + '\phase50a_tok_I_h32_r5.bin'
$R9     = $WDIR + '\phase50a_tok_I_h32_r9.bin'
foreach ($w in @($D1_W,$BPE_W,$R5,$R9)) { if (-not (Test-Path $w)) { Write-Error "Missing: $w"; exit 1 } }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

$GEN_LEN = 2000
$WARMUP  = 5000
$MINLEN  = 2
$TEMPS   = @('0.65','0.55')
$NSEEDS  = if ($Smoke) { 4 } else { 16 }
$NCORP   = if ($Smoke) { 60 } else { 400 }
$THROTTLE = 12
$MAXB    = if ($Smoke) { @('--max-bytes','8000000') } else { @() }

$NAME_WORDS = @('lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy')
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

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
    return [PSCustomObject]@{ nameish=$nameish; maxrun=$maxrun; top_bi=$top_bi; altloop=$altmax; ntok=$nt }
}
function Pctl([double[]]$a,[double]$p) {
    if ($a.Count -eq 0) { return 0 }
    $s = $a | Sort-Object; $idx = [int][math]::Floor($p*($s.Count-1)); return $s[$idx]
}
function Dist([double[]]$a) {
    return [PSCustomObject]@{ p50=(Pctl $a 0.50); p90=(Pctl $a 0.90); max=(($a|Measure-Object -Maximum).Maximum); mean=[math]::Round(($a|Measure-Object -Average).Average,2) }
}
function New-GenTask($label,$exe,$wargs,$tp,$rng,$off,$si) {
    $lbl = $label+'_T'+$tp+'_s'+$si
    $tf  = $RDIR+'\gen_'+$lbl+'.txt'; $sf  = $RDIR+'\gen_'+$lbl+'_stats.txt'
    $argline = ('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                $TS_DATA,$wargs,$GEN_LEN,$WARMUP,$tp,$off,$rng)
    return [PSCustomObject]@{ label=$label; tp=$tp; si=$si; exe=($BINDIR+'\'+$exe); tf=$tf; sf=$sf; argline=$argline }
}
function Invoke-Throttled($tasks,$throttle) {
    $queue = New-Object System.Collections.Queue; foreach ($t in $tasks) { [void]$queue.Enqueue($t) }
    $running = New-Object System.Collections.ArrayList; $launched=0; $total=$tasks.Count
    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $throttle -and $queue.Count -gt 0) {
            $t = $queue.Dequeue()
            $p = Start-Process -FilePath $t.exe -ArgumentList $t.argline -RedirectStandardOutput $t.tf -RedirectStandardError $t.sf -NoNewWindow -PassThru
            [void]$running.Add($p); $launched++
        }
        Start-Sleep -Milliseconds 150
        for ($k=$running.Count-1; $k -ge 0; $k--) { if ($running[$k].HasExited) { $running.RemoveAt($k) } }
    }
}
function Read-BytesRetry($p) { for ($a=0;$a -lt 20;$a++){ try { return [System.IO.File]::ReadAllBytes($p) } catch { Start-Sleep -Milliseconds 200 } }; return $null }

# -------- Compile --------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS ('benchmarks/phase38-42/phase50_b_token.c') @SRC_CORE -o ($BINDIR+'\phase50_b_token.exe')
if ($LASTEXITCODE -ne 0) { Write-Error 'compile phase50_b_token failed'; exit 1 }
& gcc @GCC_FLAGS ('benchmarks/phase38-42/phase50_a_generator.c') @SRC_CORE -o ($BINDIR+'\phase50_a_generator.exe')
if ($LASTEXITCODE -ne 0) { Write-Error 'compile generator failed'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# ================= MISURA A: corpus calibration =================================
Write-Host ''
Write-Host ('=== MISURA A: corpus word-guards on {0} held-out windows ({1} bytes ~ gate window) ===' -f $NCORP,$GEN_LEN) -ForegroundColor Yellow
$corpus = [System.IO.File]::ReadAllBytes($TS_DATA); $fsz=$corpus.Length
$held_lo = [long][math]::Floor(0.90*$fsz); $held_hi = $fsz - $GEN_LEN - 8
$stride = [long][math]::Floor(($held_hi-$held_lo)/$NCORP)
$cTopBi=@(); $cAlt=@(); $cRun=@(); $cName=@()
for ($wi=0; $wi -lt $NCORP; $wi++) {
    $off=[long]($held_lo + $wi*$stride)
    $txt=[System.Text.Encoding]::ASCII.GetString($corpus,[int]$off,$GEN_LEN)
    $m=Get-WordMetricsFromText $txt; if ($null -ne $m) { $cTopBi+=[double]$m.top_bi; $cAlt+=[double]$m.altloop; $cRun+=[double]$m.maxrun; $cName+=[double]$m.nameish }
}
$corpus=$null
$dTop=Dist $cTopBi; $dAlt=Dist $cAlt; $dRun=Dist $cRun; $dName=Dist $cName
Write-Host ('  {0,-10} {1,8} {2,8} {3,8} {4,8}' -f 'guard','p50','p90','max','mean')
Write-Host ('  {0,-10} {1,8} {2,8} {3,8} {4,8}' -f 'topBi',$dTop.p50,$dTop.p90,$dTop.max,$dTop.mean)
Write-Host ('  {0,-10} {1,8} {2,8} {3,8} {4,8}' -f 'altLp',$dAlt.p50,$dAlt.p90,$dAlt.max,$dAlt.mean)
Write-Host ('  {0,-10} {1,8} {2,8} {3,8} {4,8}' -f 'runWst',$dRun.p50,$dRun.p90,$dRun.max,$dRun.mean)
Write-Host ('  {0,-10} {1,8} {2,8} {3,8} {4,8}' -f 'nameWst',$dName.p50,$dName.p90,$dName.max,$dName.mean)
$cmp = if ($dTop.p90 -gt 7) { 'ABOVE' } else { 'at/below' }
Write-Host ('  NOTE: corpus topBi p90={0} max={1} is {2} the current gate bar 7 = REAL TinyStories itself' -f $dTop.p90,$dTop.max,$cmp) -ForegroundColor DarkCyan
Write-Host ('        exceeds topBi<=7 in held-out windows -> the bar is stricter than ground truth.' ) -ForegroundColor DarkCyan

# generator current topBi (deployed r9, both temps)
Write-Host ''
Write-Host ('=== MISURA A: generator topBi (r9 deployed, {0} seeds x 2 temp) ===' -f $NSEEDS) -ForegroundColor Yellow
$margin=$WARMUP+1024; $SEEDS=@(); for ($i=0;$i -lt $NSEEDS;$i++){ $SEEDS += [int]([math]::Floor($i*($fsz-$margin)/$NSEEDS)) }
$gtasks=@()
foreach ($tp in $TEMPS) { $si=0; foreach ($off in $SEEDS) { $si++
    $gtasks += New-GenTask 'r9' 'phase50_a_generator.exe' ('"'+$D1_W+'" "'+$R9+'"') $tp 12345 $off $si } }
Invoke-Throttled $gtasks $THROTTLE
$gen=@{}; foreach ($tp in $TEMPS) { $gen[$tp]=@{top=@();alt=@();run=@()} }
foreach ($t in $gtasks) { $b=Read-BytesRetry $t.tf; if ($null -eq $b -or $b.Length -eq 0) { continue }
    $m=Get-WordMetricsFromText ([System.Text.Encoding]::ASCII.GetString($b)); if ($m) { $gen[$t.tp].top+=[double]$m.top_bi; $gen[$t.tp].alt+=[double]$m.altloop; $gen[$t.tp].run+=[double]$m.maxrun } }
Write-Host ('  {0,-8} {1,8} {2,8} {3,8} {4,8}' -f 'temp','top.p50','top.p90','top.max','alt.max')
foreach ($tp in $TEMPS) { $d=Dist $gen[$tp].top; $da=Dist $gen[$tp].alt
    Write-Host ('  {0,-8} {1,8} {2,8} {3,8} {4,8}' -f $tp,$d.p50,$d.p90,$d.max,$da.max) }

# verdict A
$gen_top_p90 = [math]::Max((Dist $gen['0.65'].top).p90, (Dist $gen['0.55'].top).p90)
$artifact = ($gen_top_p90 -le ($dTop.p90 + 2))
Write-Host ''
if ($artifact) {
    Write-Host ('  -> MISURA A: generator topBi.p90={0} ~< corpus topBi.p90={1} = BAR-ARTIFACT (function-word freq).' -f $gen_top_p90,$dTop.p90) -ForegroundColor Green
    $newbar=[int]([math]::Ceiling($dTop.p90))+1
    Write-Host ('     Suggested corpus-truth topBi bar = corpus-p90+1 = {0} (vs old 7). altLp bar = {1}.' -f $newbar,([int]([math]::Ceiling($dAlt.p90))+1)) -ForegroundColor Green
    $sug = [PSCustomObject]@{ topBi=$newbar; altLp=([int]([math]::Ceiling($dAlt.p90))+1); runWst=([int]([math]::Ceiling($dRun.p90))+1); nameWst=([int]([math]::Ceiling($dName.p90))+2); source='corpus-p90+margin (phase50_b_diag)' }
    $sug | ConvertTo-Json | Set-Content ($RDIR+'\suggested_word_bars.json') -Encoding utf8
    Write-Host ('     Written: ' + $RDIR + '\suggested_word_bars.json (advisory, not applied)') -ForegroundColor DarkCyan
} else {
    Write-Host ('  -> MISURA A: generator topBi.p90={0} >> corpus topBi.p90={1} = REAL EXCESS (not just bar).' -f $gen_top_p90,$dTop.p90) -ForegroundColor Yellow
}

# ================= MISURA B: re-posed premise (r5 + r9) =========================
Write-Host ''
Write-Host '=== MISURA B: re-posed premise (single/alt/dup) on r5 and deployed r9 ===' -ForegroundColor Yellow
foreach ($ck in @(@{n='r5';w=$R5}, @{n='r9';w=$R9})) {
    Write-Host ('  -- decoder ' + $ck.n + ' --') -ForegroundColor Cyan
    & "$BINDIR\phase50_b_token.exe" $TS_DATA $D1_W $BPE_W ($RDIR+'\btmp') --premise $ck.w --len 1000000 @MAXB 2>&1 |
        Select-String '^PREMISE mode=' | ForEach-Object { Write-Host ('    ' + $_.Line) }
}
Write-Host ''
Write-Host '  read: ratio>=2 AND seeded>=0.30 => that mode is a coverable attractor (-> 50.C). Else the' -ForegroundColor DarkGray
Write-Host '        alternation is natural function-word frequency (bar-artifact) and the residual is' -ForegroundColor DarkGray
Write-Host '        long-range template-reset = charter-question (selective memory).' -ForegroundColor DarkGray

# ================= pre-registered tree (advisory) ==============================
Write-Host ''
Write-Host '=== TREE (advisory; Architect reads A + B together) ===' -ForegroundColor Yellow
Write-Host '  1. gen topBi ~= corpus + no mode coverable  -> bar-artifact -> recalibrate bar to corpus-truth'
Write-Host '     -> re-gate 50.A with new word-bars = FIRST PROMOTED GENERATOR (charter-question separate).'
Write-Host '  2. gen topBi >> corpus AND alt/dup coverable -> 50.C targeted coverage (one adjustment).'
Write-Host '  3. gen topBi >> corpus AND not coverable     -> template-reset = charter-question; 50.A+B'
Write-Host '     = branch-2 milestone to commit, then the charter-question.'
Write-Host ('  Files: ' + $RDIR)
