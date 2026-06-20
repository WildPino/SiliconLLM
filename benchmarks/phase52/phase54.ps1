# Phase 54 - GLOBAL-k3 generator prior: two-sided protocol (TF-BPB compression + closed-loop read).
#   Compression side: TF-eval held-out BPB sweep over lambda_k3 x count-thr, stratified by distance.
#   Read side: closed-loop @T0.55/T0.65 for top lambdas, gate-v2 word + Phase-47 byte-guards, save samples.
# PROMOTES (architect reads, no auto-verdict) iff TF-BPB drops AND closed-loop read does NOT regress.
# Substrate + token MLP UNCHANGED. k3 = counts only (deterministic). No commit until promoted.
#
# Usage:
#   .\benchmarks\phase52\phase54.ps1                 # build + TINY deterministic smoke (mechanics)
#   .\benchmarks\phase52\phase54.ps1 -Real           # FULL TF sweep + closed-loop gate (the long run; user launches)
param([switch]$Real,[switch]$SkipBuild)

$ErrorActionPreference='Continue'   # native exe stderr must not abort (PS 5.1); exit codes checked explicitly
$ROOT   = Split-Path (Split-Path $PSScriptRoot)
$BIN    = "$ROOT\bin"; $W="$ROOT\weights"; $R="$ROOT\results\phase54"; $H="$R\human"
$TS     = "$ROOT\data\corpora\tinystories_64mb.txt"
$D1     = "$W\phase44f_F0.bin"
$TOK    = "$W\phase50a_tok_I_h32_r9.bin"
$EXE    = "$BIN\phase54_generator.exe"
$K3CACHE= "$R\k3_global.bin"
$SRC    = "benchmarks/phase52/phase54_generator.c"
Set-Location $ROOT
foreach($d in @($R,$H)){ if(-not(Test-Path $d)){ New-Item -ItemType Directory $d|Out-Null } }
foreach($f in @($TS,$D1,$TOK)){ if(-not(Test-Path $f)){ Write-Error "missing $f"; exit 1 } }

$V2 = Get-Content "$ROOT\docs\gatev2_bars.json" -Raw | ConvertFrom-Json
$WB = $V2.token_word_bars
$BPB_LO=0.6; $BPB_HI=2.0
$NAME_WORDS=@('lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue','dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy')
$NAME_SET=@{}; foreach($w in $NAME_WORDS){ $NAME_SET[$w]=$true }
$MINLEN=2

function ByteGuard([byte[]]$b,[int]$off,[int]$len){ $mWs=0;$mCh=0;$ws=0;$ch=0;$wc=0;$np=0;$p=-1; $e=$off+$len
  for($i=$off;$i -lt $e;$i++){ $x=$b[$i]
    if($x -eq 32 -or $x -eq 9 -or $x -eq 10 -or $x -eq 13){$wc++;$ws++;if($ws -gt $mWs){$mWs=$ws};$ch=0}
    else{$ws=0;if($x -eq $p){$ch++}else{$ch=1};if($ch -gt $mCh){$mCh=$ch};if($x -lt 32 -or $x -gt 126){$np++}}; $p=$x }
  [PSCustomObject]@{wsRun=$mWs;chRun=$mCh;wsFrac=[math]::Round($wc/$len,4);nonPrint=$np} }
function WordMetrics([string]$t){ $c=$t -replace '[^a-zA-Z]',' '; $tk=@($c.ToLower() -split '\s+'|Where-Object{$_.Length -ge $MINLEN})
  $n=$tk.Count; if($n -lt 4){return $null}
  $nc=0; foreach($w in $tk){if($NAME_SET.ContainsKey($w)){$nc++}}; $ni=[math]::Round($nc/$n*100,1)
  $run=1;$mr=1; for($j=1;$j -lt $n;$j++){ if($tk[$j] -eq $tk[$j-1]){$run++}else{$run=1}; if($run -gt $mr){$mr=$run} }
  $bf=@{}; for($j=0;$j -lt $n-1;$j++){ $k=$tk[$j]+' '+$tk[$j+1]; $bf[$k]=($bf[$k] -as [int])+1 }
  $bi=if($bf.Count){($bf.Values|Measure-Object -Maximum).Maximum}else{0}
  $al=0;$am=0; for($j=0;$j -lt $n-2;$j++){ if($tk[$j] -eq $tk[$j+2]){$al++}else{$al=0}; if($al -gt $am){$am=$al} }
  [PSCustomObject]@{nameish=$ni;maxrun=$mr;top_bi=$bi;altloop=$am} }
function AllMetrics([string]$p){ if(-not(Test-Path $p)){return $null}
  $b=[System.IO.File]::ReadAllBytes($p); if($b.Length -eq 0){return $null}
  $t=[System.Text.Encoding]::ASCII.GetString($b); $wm=WordMetrics $t; if($null -eq $wm){return $null}
  $bg=ByteGuard $b 0 $b.Length
  [PSCustomObject]@{nameish=$wm.nameish;maxrun=$wm.maxrun;top_bi=$wm.top_bi;altloop=$wm.altloop;wsRun=$bg.wsRun;chRun=$bg.chRun;wsFrac=$bg.wsFrac;nonPrint=$bg.nonPrint} }
function SelfBPB([string]$sf){ $l=Select-String 'self_BPB' $sf|Select-Object -Last 1; if($l){return [double]($l.Line.Trim() -replace 'self_BPB:\s*','')}; return $null }
function Mx([double[]]$a){ if($a.Count -eq 0){return 0.0}; ($a|Measure-Object -Maximum).Maximum }
function Av([double[]]$a){ if($a.Count -eq 0){return 0.0}; [math]::Round(($a|Measure-Object -Average).Average,3) }
function WorstOf($mets,$bpbs){ $bi=@();$al=@();$mr=@();$ni=@();$ws=@();$ch=@();$wf=@();$np=@();$bs=@()
  foreach($m in @($mets)){ $bi+=[double]$m.top_bi;$al+=[double]$m.altloop;$mr+=[double]$m.maxrun;$ni+=[double]$m.nameish;$ws+=[double]$m.wsRun;$ch+=[double]$m.chRun;$wf+=[double]$m.wsFrac;$np+=[double]$m.nonPrint }
  foreach($b in @($bpbs)){ $bs+=[double]$b }
  [PSCustomObject]@{bi=(Mx $bi);al=(Mx $al);mr=(Mx $mr);ni=(Mx $ni);wsRun=(Mx $ws);chRun=(Mx $ch);wsFrac=(Mx $wf);nonPrint=(Mx $np);self=(Av $bs);n=@($mets).Count} }
function GuardFails($w){ $f=@()
  if($w.n -eq 0){return ,@('nodata')}
  if($w.bi -gt $WB.topBi){$f+="topBi($($w.bi)>$($WB.topBi))"}
  if($w.al -gt $WB.altLp){$f+="altLp($($w.al)>$($WB.altLp))"}
  if($w.mr -gt $WB.runWst){$f+="runWst($($w.mr)>$($WB.runWst))"}
  if($w.ni -gt $WB.nameWst){$f+="nameWst($($w.ni)>$($WB.nameWst))"}
  if($w.self -lt $BPB_LO -or $w.self -gt $BPB_HI){$f+="selfBPB($($w.self))"}
  if($w.wsRun -gt $V2.wsRun){$f+="wsRun($($w.wsRun)>$($V2.wsRun))"}
  if($w.chRun -gt $V2.chRun){$f+="chRun($($w.chRun)>$($V2.chRun))"}
  if($w.wsFrac -gt $V2.wsFrac){$f+="wsFrac($($w.wsFrac)>$($V2.wsFrac))"}
  if($w.nonPrint -gt $V2.nonPrint){$f+="nonPrint($($w.nonPrint)>$($V2.nonPrint))"}
  return ,$f }
function Row($name,$tp,$w){ Write-Host ('  {0,-14} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,6} {9,7} {10,7}' -f `
    $name,$tp,('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F0}' -f $w.mr),('{0:F1}' -f $w.ni),('{0:F2}' -f $w.self),('{0:F0}' -f $w.wsRun),('{0:F0}' -f $w.chRun),('{0:F3}' -f $w.wsFrac),('{0:F0}' -f $w.nonPrint)) }

if(-not $SkipBuild){
  Write-Host '== build ==' -ForegroundColor Cyan
  & gcc -O3 -march=native -mavx2 -mfma $SRC src/silicon_entropy.c src/silicon_v0.c -o $EXE -lm -I .
  if($LASTEXITCODE -ne 0){ throw 'build failed' }
  Write-Host 'build OK' -ForegroundColor Green
}
Write-Host ("gate-v2 bars: wsRun<={0} chRun<={1} wsFrac<={2} nonPrint<={3} | topBi<={4} altLp<={5} runWst<={6} nameWst<={7}" -f $V2.wsRun,$V2.chRun,$V2.wsFrac,$V2.nonPrint,$WB.topBi,$WB.altLp,$WB.runWst,$WB.nameWst) -ForegroundColor DarkCyan

$fsz=(Get-Item $TS).Length

if(-not $Real){
  Write-Host ''
  Write-Host '== TINY SMOKE (mechanics only, NOT the verdict) ==' -ForegroundColor Cyan
  Write-Host '-- TF-eval (capped 6MB, tf-len 8000) --'
  & $EXE $TS $D1 $TOK --tf-eval --tf-len 8000 --k3-max-bytes 6000000 --tf-start-frac 0.85 2>"$R\smoke_tf.err" | Select-String 'lam=0.00|lam=0.50'
  Write-Host '-- closed-loop @0.65, lambda 0 (baseline) vs 0.5 (det check) --'
  & $EXE $TS $D1 $TOK --k3-save $K3CACHE --k3-max-bytes 6000000 2>"$R\smoke_k3save.err" | Out-Null
  $b0="$R\smoke_l0.txt"; $b5="$R\smoke_l5.txt"
  # NOTE: must use Start-Process -RedirectStandardOutput (byte-clean). PS '>' re-encodes raw bytes as UTF-16 -> corrupt.
  $a0=('"{0}" "{1}" "{2}" --gen-len 1500 --warmup 3000 --temp 0.65 --rng-seed 12345' -f $TS,$D1,$TOK)
  $a5=('"{0}" "{1}" "{2}" --gen-len 1500 --warmup 3000 --temp 0.65 --rng-seed 12345 --lambda-k3 0.5 --k3-thr 2 --k3-load "{3}"' -f $TS,$D1,$TOK,$K3CACHE)
  (Start-Process -FilePath $EXE -ArgumentList $a0 -RedirectStandardOutput $b0 -RedirectStandardError "$R\smoke_l0.err" -NoNewWindow -PassThru -Wait)|Out-Null
  (Start-Process -FilePath $EXE -ArgumentList $a5 -RedirectStandardOutput $b5 -RedirectStandardError "$R\smoke_l5.err" -NoNewWindow -PassThru -Wait)|Out-Null
  foreach($pair in @(@{n='lam0';f=$b0;e="$R\smoke_l0.err"},@{n='lam0.5';f=$b5;e="$R\smoke_l5.err"})){
    $m=AllMetrics $pair.f; $bp=SelfBPB $pair.e
    Write-Host ("  {0,-7} self_BPB={1} topBi={2} altLp={3} runWst={4} | wsRun={5} chRun={6} wsFrac={7} nonPrint={8}" -f $pair.n,$bp,$m.top_bi,$m.altloop,$m.maxrun,$m.wsRun,$m.chRun,$m.wsFrac,$m.nonPrint) }
  Write-Host ''
  Write-Host 'SMOKE sanity: TF-eval shows k3 lowers BPB at lam=0.5; closed-loop deterministic & byte-guards printed.' -ForegroundColor Green
  Write-Host 'Next (long run, user launches):  .\benchmarks\phase52\phase54.ps1 -Real' -ForegroundColor Yellow
  return
}

# ============================== REAL RUN ==============================
Write-Host ''
Write-Host '=== STEP 1: build+save GLOBAL-k3 (full corpus) + TF-EVAL sweep (compression side) ===' -ForegroundColor Yellow
& $EXE $TS $D1 $TOK --tf-eval --tf-len 120000 --tf-start-frac 0.93 --k3-train-frac 0.90 --k3-save $K3CACHE 2>"$R\tf_eval.err" | Tee-Object "$R\tf_eval.out"
Write-Host ("  (k3 cache: {0})  TF curve saved -> {1}" -f $K3CACHE,"$R\tf_eval.out") -ForegroundColor DarkCyan

Write-Host ''
Write-Host '=== STEP 2: closed-loop READ (gate-v2 word + byte-guards, worst-of-N) ===' -ForegroundColor Yellow
$LAMS=@(0.0,0.25,0.5,1.0); $TEMPS=@('0.65','0.55'); $NSEED=8
$margin=3000+1024; $SEEDS=@(); for($i=0;$i -lt $NSEED;$i++){ $SEEDS+=[int][math]::Floor($i*($fsz-$margin)/$NSEED) }
$RNG=12345
$agg=@{}; foreach($L in $LAMS){ foreach($tp in $TEMPS){ $agg["$L|$tp"]=@{mets=@();bpbs=@()} } }
$tasks=@()
foreach($L in $LAMS){ foreach($tp in $TEMPS){ $si=0; foreach($off in $SEEDS){ $si++
  $lab="l$($L)_T$($tp)_s$si"; $tf="$R\gen_$lab.txt"; $sf="$R\gen_$lab.err"
  $k3args=if($L -gt 0){"--lambda-k3 $L --k3-thr 2 --k3-load `"$K3CACHE`""}else{''}
  $al=('"{0}" "{1}" "{2}" --gen-len 2000 --warmup 3000 --temp {3} --seed-start {4} --rng-seed {5} {6}' -f $TS,$D1,$TOK,$tp,$off,$RNG,$k3args)
  $tasks+=[PSCustomObject]@{key="$L|$tp";exe=$EXE;tf=$tf;sf=$sf;al=$al} } } }
Write-Host ("  launching {0} closed-loop gens (throttle 12)..." -f $tasks.Count)
$q=New-Object System.Collections.Queue; foreach($t in $tasks){[void]$q.Enqueue($t)}
$runp=New-Object System.Collections.ArrayList
while($q.Count -gt 0 -or $runp.Count -gt 0){
  while($runp.Count -lt 12 -and $q.Count -gt 0){ $t=$q.Dequeue()
    $p=Start-Process -FilePath $t.exe -ArgumentList $t.al -RedirectStandardOutput $t.tf -RedirectStandardError $t.sf -NoNewWindow -PassThru
    [void]$runp.Add(@{p=$p;t=$t}) }
  Start-Sleep -Milliseconds 150
  for($k=$runp.Count-1;$k -ge 0;$k--){ if($runp[$k].p.HasExited){ $runp.RemoveAt($k) } } }
foreach($t in $tasks){ $m=AllMetrics $t.tf; if($m){$agg[$t.key].mets+=$m}; $b=SelfBPB $t.sf; if($null -ne $b){$agg[$t.key].bpbs+=$b} }

Write-Host ''
Write-Host '======== closed-loop gate-v2 (worst-of-N; lambda 0 = bigram-only baseline) ========' -ForegroundColor Cyan
Write-Host ('  {0,-14} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,6} {9,7} {10,7}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr')
foreach($L in $LAMS){ foreach($tp in $TEMPS){ $w=WorstOf $agg["$L|$tp"].mets $agg["$L|$tp"].bpbs
  Row ("lam=$L") $tp $w
  $f=GuardFails $w; if(@($f).Count -gt 0){ Write-Host ('    guard-break: '+($f -join ', ')) -ForegroundColor Red } } }

# save 8-10 human samples for the best non-baseline lambdas (0.25,0.5)
foreach($L in @(0.25,0.5)){ $n=0; foreach($tp in $TEMPS){ $si=0; foreach($off in $SEEDS){ $si++
  if($n -ge 10){break}; $src="$R\gen_l$($L)_T$($tp)_s$si.txt"; if(Test-Path $src){ Copy-Item $src "$H\lam$($L)_T$($tp)_s$si.txt" -Force; $n++ } } } }
Write-Host ''
Write-Host ("Samples in $H. Raw numbers above; architect reads the verdict:") -ForegroundColor Cyan
Write-Host '  PROMOTE iff TF-BPB drops (STEP 1 curve) AND closed-loop read does NOT regress (no guard-break vs lam=0 baseline).' -ForegroundColor Cyan
Write-Host '  BPB-down via flood/repetition (guard-break) = n-gram TRAP, reject. Honest negative is a valid result.' -ForegroundColor Cyan
