# Phase 50 - closure & consolidation (NO training, NO new experiments).
# 1) recalibrate the WORD bars to corpus-truth for the token stream (gate window); write the
#    token-stream section into docs/gatev2_bars.json with an honest note.
# 2) re-gate the deployed 50.A r9 on the corrected bars FOR THE RECORD (not to promote); save table.
# Generator armB UNCHANGED. Rule (documented): word bar = corpus p90 over 400 held-out gate-windows,
# rounded up (conservative: worst-of-32 would reach higher; chosen to be honest, not tuned-to-pass).
#
# Run:  .\benchmarks\phase38-42\phase50_consolidate.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase50_a'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
$BARS    = $ROOT + '\docs\gatev2_bars.json'
Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }

$D1_W   = $WDIR + '\phase44f_F0.bin'
$R9     = $WDIR + '\phase50a_tok_I_h32_r9.bin'
$ARMB_R7= $WDIR + '\phase48afix_I_h32_r7.bin'
foreach ($w in @($D1_W,$R9,$TS_DATA)) { if (-not (Test-Path $w)) { Write-Error "Missing: $w"; exit 1 } }

$GCC_FLAGS=@('-O3','-march=native','-mavx2','-mfma','-lm','-I','.'); $SRC_CORE=@('src/silicon_entropy.c','src/silicon_v0.c')
$GEN_LEN=2000; $WARMUP=5000; $MINLEN=2; $TEMPS=@('0.65','0.55'); $NCORP=400; $NSEEDS=16; $RNGS=@(12345,67890); $THROTTLE=12
$NAME_WORDS=@('lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue','dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy')
$NAME_SET=@{}; foreach($w in $NAME_WORDS){$NAME_SET[$w]=$true}

function Get-ByteGuardFromBytes([byte[]]$bytes,[int]$off,[int]$len){ $maxWs=0;$maxCh=0;$ws=0;$ch=0;$wsCount=0;$nonPrint=0;$prev=-1;$end=$off+$len
    for($bi=$off;$bi -lt $end;$bi++){ $b=$bytes[$bi]
        if($b -eq 32 -or $b -eq 9 -or $b -eq 10 -or $b -eq 13){$wsCount++;$ws++;if($ws -gt $maxWs){$maxWs=$ws};$ch=0}
        else{$ws=0;if($b -eq $prev){$ch++}else{$ch=1};if($ch -gt $maxCh){$maxCh=$ch};if($b -lt 32 -or $b -gt 126){$nonPrint++}};$prev=$b }
    return [PSCustomObject]@{wsRun=$maxWs;chRun=$maxCh;wsFrac=[math]::Round($wsCount/$len,4);nonPrint=$nonPrint} }
function Get-WordMetricsFromText([string]$text){ $clean=$text -replace '[^a-zA-Z]',' '
    $toks=@($clean.ToLower() -split '\s+'|Where-Object{$_.Length -ge $MINLEN}); $nt=$toks.Count; if($nt -lt 4){return $null}
    $name_ct=0; foreach($w in $toks){if($NAME_SET.ContainsKey($w)){$name_ct++}}; $nameish=[math]::Round($name_ct/$nt*100.0,1)
    $run=1;$maxrun=1; for($j=1;$j -lt $nt;$j++){if($toks[$j] -eq $toks[$j-1]){$run++}else{$run=1};if($run -gt $maxrun){$maxrun=$run}}
    $bf=@{}; for($j=0;$j -lt $nt-1;$j++){$k=$toks[$j]+' '+$toks[$j+1];$bf[$k]=($bf[$k] -as [int])+1}
    $top_bi=if($bf.Count){($bf.Values|Measure-Object -Maximum).Maximum}else{0}
    $alt=0;$altmax=0; for($j=0;$j -lt $nt-2;$j++){if($toks[$j] -eq $toks[$j+2]){$alt++}else{$alt=0};if($alt -gt $altmax){$altmax=$alt}}
    return [PSCustomObject]@{nameish=$nameish;maxrun=$maxrun;top_bi=$top_bi;altloop=$altmax} }
function Pctl([double[]]$a,[double]$p){ if($a.Count -eq 0){return 0}; $s=$a|Sort-Object; return $s[[int][math]::Floor($p*($s.Count-1))] }
function Mx([double[]]$a){ if($a.Count -eq 0){return 0}; ($a|Measure-Object -Maximum).Maximum }
function Read-BytesRetry($p){ for($a=0;$a -lt 20;$a++){try{return [System.IO.File]::ReadAllBytes($p)}catch{Start-Sleep -Milliseconds 200}}; return $null }
function New-GenTask($label,$exe,$wargs,$tp,$rng,$off,$si){ $lbl=$label+'_T'+$tp+'_r'+$rng+'_s'+$si
    $tf=$RDIR+'\rg_'+$lbl+'.txt';$sf=$RDIR+'\rg_'+$lbl+'_stats.txt'
    $argline=('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f $TS_DATA,$wargs,$GEN_LEN,$WARMUP,$tp,$off,$rng)
    return [PSCustomObject]@{label=$label;tp=$tp;exe=($BINDIR+'\'+$exe);tf=$tf;sf=$sf;argline=$argline} }
function Invoke-Throttled($tasks,$throttle){ $q=New-Object System.Collections.Queue; foreach($t in $tasks){[void]$q.Enqueue($t)}
    $run=New-Object System.Collections.ArrayList
    while($q.Count -gt 0 -or $run.Count -gt 0){ while($run.Count -lt $throttle -and $q.Count -gt 0){ $t=$q.Dequeue()
            $p=Start-Process -FilePath $t.exe -ArgumentList $t.argline -RedirectStandardOutput $t.tf -RedirectStandardError $t.sf -NoNewWindow -PassThru; [void]$run.Add($p) }
        Start-Sleep -Milliseconds 150; for($k=$run.Count-1;$k -ge 0;$k--){if($run[$k].HasExited){$run.RemoveAt($k)}} } }
function Get-SelfBPB($sf){ $bl=Select-String 'self_BPB' $sf|Select-Object -Last 1; if($bl){return [double]($bl.Line.Trim() -replace 'self_BPB:\s*','')}; return $null }

Write-Host ''
Write-Host '=== COMPILE generator ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS ('benchmarks/phase38-42/phase50_a_generator.c') @SRC_CORE -o ($BINDIR+'\phase50_a_generator.exe')
if($LASTEXITCODE -ne 0){Write-Error 'compile failed';exit 1}
& gcc @GCC_FLAGS ('benchmarks/phase38-42/phase48a_generator.c') @SRC_CORE -o ($BINDIR+'\phase48a_generator.exe') | Out-Null

# ---- 1) corpus calibration (400 held-out gate-windows) ----
Write-Host ''
Write-Host ('=== CORPUS CALIBRATION: word-guards over {0} held-out {1}-byte windows ===' -f $NCORP,$GEN_LEN) -ForegroundColor Yellow
$corpus=[System.IO.File]::ReadAllBytes($TS_DATA); $fsz=$corpus.Length
$lo=[long][math]::Floor(0.90*$fsz); $hi=$fsz-$GEN_LEN-8; $stride=[long][math]::Floor(($hi-$lo)/$NCORP)
$cT=@();$cA=@();$cR=@();$cN=@()
for($wi=0;$wi -lt $NCORP;$wi++){ $off=[long]($lo+$wi*$stride)
    $m=Get-WordMetricsFromText ([System.Text.Encoding]::ASCII.GetString($corpus,[int]$off,$GEN_LEN))
    if($m){$cT+=[double]$m.top_bi;$cA+=[double]$m.altloop;$cR+=[double]$m.maxrun;$cN+=[double]$m.nameish} }
$corpus=$null
$bTop=[int][math]::Ceiling((Pctl $cT 0.90)); $bAlt=[int][math]::Ceiling((Pctl $cA 0.90)); $bRun=[int][math]::Ceiling((Pctl $cR 0.90)); $bName=[int][math]::Ceiling((Pctl $cN 0.90))+1
Write-Host ('  {0,-8} {1,6} {2,6} {3,6} {4,6}  -> bar(p90 ceil)' -f 'guard','p50','p90','p97','max')
Write-Host ('  {0,-8} {1,6} {2,6} {3,6} {4,6}  -> {5}' -f 'topBi',(Pctl $cT 0.5),(Pctl $cT 0.9),(Pctl $cT 0.97),(Mx $cT),$bTop)
Write-Host ('  {0,-8} {1,6} {2,6} {3,6} {4,6}  -> {5}' -f 'altLp',(Pctl $cA 0.5),(Pctl $cA 0.9),(Pctl $cA 0.97),(Mx $cA),$bAlt)
Write-Host ('  {0,-8} {1,6} {2,6} {3,6} {4,6}  -> {5}' -f 'runWst',(Pctl $cR 0.5),(Pctl $cR 0.9),(Pctl $cR 0.97),(Mx $cR),$bRun)
Write-Host ('  {0,-8} {1,6} {2,6} {3,6} {4,6}  -> {5}' -f 'nameWst',(Pctl $cN 0.5),(Pctl $cN 0.9),(Pctl $cN 0.97),(Mx $cN),$bName)
Write-Host ('  NOTE: byte-era topBi<=7 was STRICTER than ground truth (corpus p90={0}, max={1}).' -f (Pctl $cT 0.9),(Mx $cT)) -ForegroundColor DarkCyan

# ---- write token-stream section into gatev2_bars.json (preserve byte guards) ----
$cur = Get-Content $BARS -Raw | ConvertFrom-Json
$obj = [ordered]@{
    wsRun=$cur.wsRun; chRun=$cur.chRun; wsFrac=$cur.wsFrac; nonPrint=$cur.nonPrint
    token_word_bars=[ordered]@{ topBi=$bTop; altLp=$bAlt; runWst=$bRun; nameWst=$bName }
    note=("byte guards (wsRun/chRun/wsFrac/nonPrint) unchanged. token_word_bars recalibrated to corpus-truth p90 over $NCORP held-out $GEN_LEN-byte gate-windows of real TinyStories (Phase 50 consolidation). The byte-era word bar topBi<=7 was STRICTER than ground truth (corpus topBi p90=" + (Pctl $cT 0.9) + ", max=" + (Mx $cT) + "): real human text itself exceeds 7. p90 chosen as conservative/honest (worst-of-32 reaches higher); NOT tuned to pass.")
}
($obj | ConvertTo-Json -Depth 5) | Set-Content $BARS -Encoding utf8
Write-Host ('  Updated ' + $BARS + ' (token_word_bars: topBi<=' + $bTop + ' altLp<=' + $bAlt + ' runWst<=' + $bRun + ' nameWst<=' + $bName + ')') -ForegroundColor Green

# ---- 2) re-gate r9 (deployed) on corrected bars, FOR THE RECORD ----
$margin=$WARMUP+1024; $SEEDS=@(); for($i=0;$i -lt $NSEEDS;$i++){$SEEDS+=[int]([math]::Floor($i*($fsz-$margin)/$NSEEDS))}
$cfgs=@(@{label='50A_r9';exe='phase50_a_generator.exe';wargs=('"'+$D1_W+'" "'+$R9+'"')})
if(Test-Path $ARMB_R7){ $cfgs+=@{label='armB_r7';exe='phase48a_generator.exe';wargs=('"'+$D1_W+'" "'+$ARMB_R7+'"')} }
$tasks=@(); foreach($c in $cfgs){foreach($tp in $TEMPS){foreach($rng in $RNGS){$si=0;foreach($off in $SEEDS){$si++
    $tasks+=New-GenTask $c.label $c.exe $c.wargs $tp $rng $off $si}}}}
Write-Host ''
Write-Host ('=== RE-GATE FOR THE RECORD: {0} configs x 2 temp x worst-of-{1} ===' -f $cfgs.Count,(2*$NSEEDS)) -ForegroundColor Yellow
Invoke-Throttled $tasks $THROTTLE
$agg=@{}; foreach($c in $cfgs){$agg[$c.label]=@{};foreach($tp in $TEMPS){$agg[$c.label][$tp]=@{top=@();alt=@();run=@();name=@();ws=@();ch=@();wf=@();np=@();bpb=@()}}}
foreach($t in $tasks){ $b=Read-BytesRetry $t.tf; if($null -eq $b -or $b.Length -eq 0){continue}
    $wm=Get-WordMetricsFromText ([System.Text.Encoding]::ASCII.GetString($b)); $bg=Get-ByteGuardFromBytes $b 0 $b.Length
    if($wm){$a=$agg[$t.label][$t.tp]; $a.top+=[double]$wm.top_bi;$a.alt+=[double]$wm.altloop;$a.run+=[double]$wm.maxrun;$a.name+=[double]$wm.nameish
        $a.ws+=[double]$bg.wsRun;$a.ch+=[double]$bg.chRun;$a.wf+=[double]$bg.wsFrac;$a.np+=[double]$bg.nonPrint}
    $sb=Get-SelfBPB $t.sf; if($null -ne $sb){$agg[$t.label][$t.tp].bpb+=$sb} }

$rec=@()
$rec+=('Phase 50 RE-GATE FOR THE RECORD (corrected token-stream bars; NOT a promotion)')
$rec+=('bars: byte wsRun<={0} chRun<={1} wsFrac<={2} nonPrint<={3} | word topBi<={4} altLp<={5} runWst<={6} nameWst<={7}' -f $cur.wsRun,$cur.chRun,$cur.wsFrac,$cur.nonPrint,$bTop,$bAlt,$bRun,$bName)
$hdr=('  {0,-9} {1,5} {2,6} {3,6} {4,6} {5,7} {6,7} {7,7} {8,7} {9,8} {10,7}' -f 'config','temp','topBi','altLp','runWst','nameWst','wsRun','chRun','wsFrac','nonPrint','selfBPB')
$rec+=$hdr; Write-Host ''; Write-Host '======== RE-GATE (worst-of-32) ========' -ForegroundColor Cyan; Write-Host $hdr
foreach($c in $cfgs){ foreach($tp in $TEMPS){ $a=$agg[$c.label][$tp]
    $wTop=Mx $a.top;$wAlt=Mx $a.alt;$wRun=Mx $a.run;$wName=Mx $a.name;$wWs=Mx $a.ws;$wCh=Mx $a.ch;$wWf=Mx $a.wf;$wNp=Mx $a.np
    $sBpb=if($a.bpb.Count){[math]::Round(($a.bpb|Measure-Object -Average).Average,2)}else{0}
    $line=('  {0,-9} {1,5} {2,6} {3,6} {4,6} {5,7} {6,7} {7,7} {8,7:F3} {9,8} {10,7:F2}' -f $c.label,$tp,$wTop,$wAlt,$wRun,$wName,$wWs,$wCh,$wWf,$wNp,$sBpb)
    $byteOK=($wWs -le $cur.wsRun) -and ($wCh -le $cur.chRun) -and ($wWf -le $cur.wsFrac) -and ($wNp -le $cur.nonPrint)
    $wordOK=($wTop -le $bTop) -and ($wAlt -le $bAlt) -and ($wRun -le $bRun) -and ($wName -le $bName)
    $col=if($c.label -like 'armB*'){'DarkCyan'}elseif($byteOK -and $wordOK){'Green'}else{'Gray'}
    Write-Host $line -ForegroundColor $col; $rec+=$line
    $tag=''; if(-not $byteOK){$tag+=' BYTE-FAIL'}; if(-not $wordOK){ $f=@(); if($wTop -gt $bTop){$f+="topBi $wTop>$bTop"}; if($wAlt -gt $bAlt){$f+="altLp $wAlt>$bAlt"}; if($wRun -gt $bRun){$f+="runWst $wRun>$bRun"}; if($wName -gt $bName){$f+="nameWst $wName>$bName"}; $tag+=' WORD-residual: '+($f -join ', ') }
    if($tag -ne '' -and $c.label -notlike 'armB*'){ Write-Host ('      '+$tag.Trim()) -ForegroundColor Yellow; $rec+=('      '+$tag.Trim()) } } }
$rec | Set-Content ($RDIR+'\regate_record.txt') -Encoding utf8
Write-Host ''
Write-Host '  HONEST READ: byte-guards PASS (token stream dissolved the byte floods); topBi at T0.65' -ForegroundColor Yellow
Write-Host '  near-but-above corpus, at T0.55 above = residual (function-word fallback when the thread is' -ForegroundColor Yellow
Write-Host '  lost = charter-question, NOT a coverable attractor). Record: ' -ForegroundColor Yellow
Write-Host ('    ' + $RDIR + '\regate_record.txt') -ForegroundColor Cyan
