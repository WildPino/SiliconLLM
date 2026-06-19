# Phase 49.1b - structure & human read on the best stabilized checkpoints (NO training).
#
# 49.1 produced the first positive (stabilized memory: keeps the signal, breaks the runaway) but the
# run stopped at the mini-gate -> structure-advisory + human reading are missing. This script ONLY
# generates from the saved weights and computes structure metrics. It decides nothing.
#
#   ADAPT : A_r8, A_r9   (phase49adapt_I_h32_r8/r9.bin   via phase49_1_generator, 0x53454551)
#   ERR   : E_r7, E_r8   (phase49err_I_h32_r7/r8.bin     via phase49_1_generator, 0x53454551)
#   armB  : armB_r7/r8   (phase48afix_I_h32_r7/r8.bin     via phase48a_generator,  0x53454548) = ref
#
# LONG horizon on purpose (~3500 byte): see if coherence holds in time or the bounded floods re-emerge.
# Structure-advisory: quoteParity, parenImbal, sentLen-vs-corpus, + a repetition/diversity metric
# (unique word-bigram/trigram fraction, unique char-4gram fraction, longest word-run). Verdict ADVISORY.
#
# Run:  .\benchmarks\phase38-42\phase49_1b.ps1
#       .\benchmarks\phase38-42\phase49_1b.ps1 -Smoke   (1 seed, len 1500)

param([switch]$Smoke)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase49_1b'
$HDIR    = $RDIR + '\human'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
foreach ($d in @($RDIR,$HDIR)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$D1_W = $WDIR + '\phase44f_F0.bin'
if (-not (Test-Path $D1_W)) { Write-Error "Missing D1 weight: $D1_W"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

$GEN_LEN = if ($Smoke) { 1500 } else { 3500 }
$WARMUP  = 5000
$TEMPS   = @('0.65','0.55')
$RNG     = 12345
$NSEEDS  = if ($Smoke) { 1 } else { 4 }
$THROTTLE = 12
$MINLEN  = 2

# checkpoints: {label, gen exe, weight prefix file, arm}
$CKPTS = @(
    @{ label='A_r8';    arm='ADAPT'; exe='phase49_1_generator.exe'; w=($WDIR+'\phase49adapt_I_h32_r8.bin') },
    @{ label='A_r9';    arm='ADAPT'; exe='phase49_1_generator.exe'; w=($WDIR+'\phase49adapt_I_h32_r9.bin') },
    @{ label='E_r7';    arm='ERR';   exe='phase49_1_generator.exe'; w=($WDIR+'\phase49err_I_h32_r7.bin')   },
    @{ label='E_r8';    arm='ERR';   exe='phase49_1_generator.exe'; w=($WDIR+'\phase49err_I_h32_r8.bin')   },
    @{ label='armB_r7'; arm='armB';  exe='phase48a_generator.exe';  w=($WDIR+'\phase48afix_I_h32_r7.bin')  },
    @{ label='armB_r8'; arm='armB';  exe='phase48a_generator.exe';  w=($WDIR+'\phase48afix_I_h32_r8.bin')  }
)
foreach ($c in $CKPTS) { if (-not (Test-Path $c.w)) { Write-Error ('Missing weight: '+$c.w); exit 1 } }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING (generators only, no training) ===' -ForegroundColor Cyan
foreach ($pair in @(
    @{src='phase49_1_generator.c'; exe='phase49_1_generator.exe'},
    @{src='phase48a_generator.c';  exe='phase48a_generator.exe'})) {
    & gcc @GCC_FLAGS ('benchmarks/phase38-42/'+$pair.src) @SRC_CORE -o ($BINDIR+'\'+$pair.exe)
    if ($LASTEXITCODE -ne 0) { Write-Error ('Compile failed: '+$pair.src); exit 1 }
}
Write-Host 'Compiled OK.' -ForegroundColor Green

# ============ metric helpers (structure + diversity) ================================
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
function Get-StructFromText([string]$text) {
    $q = ([regex]::Matches($text,'"')).Count
    $op = ([regex]::Matches($text,'[\(\[]')).Count; $cl = ([regex]::Matches($text,'[\)\]]')).Count
    $sent = @($text -split '[.!?]+' | Where-Object { $_.Trim().Length -gt 0 })
    $slen = if ($sent.Count -gt 0) { [math]::Round((($sent | ForEach-Object { ($_ -split '\s+').Count } | Measure-Object -Average).Average),1) } else { 0 }
    return [PSCustomObject]@{ quoteParity=($q % 2); parenImbal=[math]::Abs($op-$cl); sentLen=$slen }
}
function Get-DiversityFromText([string]$text) {
    # word-level n-gram uniqueness (1.0 = no repetition; low = repetitive/word-salad-loop)
    $clean = $text -replace '[^a-zA-Z]', ' '
    $toks = @($clean.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
    $nt = $toks.Count
    $run=1; $maxrun=1; for ($j=1; $j -lt $nt; $j++) { if ($toks[$j] -eq $toks[$j-1]) { $run++ } else { $run=1 }; if ($run -gt $maxrun) { $maxrun=$run } }
    $uniBi=1.0; if ($nt -ge 2) { $s=@{}; for ($j=0;$j -lt $nt-1;$j++){ $s[$toks[$j]+' '+$toks[$j+1]]=$true }; $uniBi=[math]::Round($s.Count/($nt-1),3) }
    $uniTri=1.0; if ($nt -ge 3) { $s=@{}; for ($j=0;$j -lt $nt-2;$j++){ $s[$toks[$j]+' '+$toks[$j+1]+' '+$toks[$j+2]]=$true }; $uniTri=[math]::Round($s.Count/($nt-2),3) }
    # char 4-gram uniqueness over printable text
    $uniC4=1.0; $L=$text.Length; if ($L -ge 4) { $s=@{}; for ($j=0;$j -lt $L-3;$j++){ $s[$text.Substring($j,4)]=$true }; $uniC4=[math]::Round($s.Count/($L-3),3) }
    return [PSCustomObject]@{ uniBi=$uniBi; uniTri=$uniTri; uniC4=$uniC4; maxrun=$maxrun; ntok=$nt }
}
function Get-SelfBPB([string]$statsfile) {
    $bl = Select-String 'self_BPB' $statsfile | Select-Object -Last 1
    if ($bl) { return [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    return $null
}
function Read-BytesRetry($p) {
    for ($a=0; $a -lt 20; $a++) { try { return [System.IO.File]::ReadAllBytes($p) } catch { Start-Sleep -Milliseconds 200 } }
    return $null
}
function Av([double[]]$a){ if($a.Count -eq 0){return 0.0}; [math]::Round(($a|Measure-Object -Average).Average,3) }
function Mn([double[]]$a){ if($a.Count -eq 0){return 0.0}; ($a|Measure-Object -Minimum).Minimum }
function Mx([double[]]$a){ if($a.Count -eq 0){return 0.0}; ($a|Measure-Object -Maximum).Maximum }
function Invoke-Throttled($tasks,$throttle) {
    $queue = New-Object System.Collections.Queue
    foreach ($t in $tasks) { [void]$queue.Enqueue($t) }
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

# corpus sentLen reference
$corpus = [System.IO.File]::ReadAllBytes($TS_DATA)
$fsz = $corpus.Length
$ctxt = [System.Text.Encoding]::ASCII.GetString($corpus, [int][math]::Floor($fsz/3), 200000)
$CORP_SENTLEN = (Get-StructFromText $ctxt).sentLen
$corpus = $null; $ctxt = $null
Write-Host ('  corpus sentLen ref ~ {0} words/sentence' -f $CORP_SENTLEN) -ForegroundColor DarkCyan

# ============ generate ==============================================================
$margin = $WARMUP + 1024
$SEEDS = @(); for ($i=0;$i -lt $NSEEDS;$i++){ $SEEDS += [int]([math]::Floor($i*($fsz-$margin)/$NSEEDS)) }
$tasks=@()
foreach ($c in $CKPTS) {
    $wargs = '"'+$D1_W+'" "'+$c.w+'"'
    foreach ($tp in $TEMPS) { $si=0
        foreach ($off in $SEEDS) { $si++
            $lbl = $c.label+'_T'+$tp+'_s'+$si
            $tf = $RDIR+'\gen_'+$lbl+'.txt'; $sf = $RDIR+'\gen_'+$lbl+'_stats.txt'
            $argline = ('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                        $TS_DATA,$wargs,$GEN_LEN,$WARMUP,$tp,$off,$RNG)
            $tasks += [PSCustomObject]@{ label=$c.label; arm=$c.arm; tp=$tp; si=$si; lbl=$lbl;
                                         exe=($BINDIR+'\'+$c.exe); tf=$tf; sf=$sf; argline=$argline }
        }
    }
}
Write-Host ''
Write-Host ('=== GENERATE: {0} samples (len {1}, {2} seeds x 2 temp x {3} ckpts) ===' -f $tasks.Count,$GEN_LEN,$NSEEDS,$CKPTS.Count) -ForegroundColor Yellow
Invoke-Throttled $tasks $THROTTLE

# ============ measure + dump =========================================================
$rows=@{}
foreach ($c in $CKPTS) { foreach ($tp in $TEMPS) { $rows[$c.label+'@'+$tp]=@{ st=@(); div=@(); bg=@(); bpb=@() } } }
foreach ($t in $tasks) {
    if (-not (Test-Path $t.tf)) { continue }
    $bytes = Read-BytesRetry $t.tf; if ($null -eq $bytes -or $bytes.Length -eq 0) { continue }
    $txt = [System.Text.Encoding]::ASCII.GetString($bytes)
    $st = Get-StructFromText $txt; $dv = Get-DiversityFromText $txt; $bg = Get-ByteGuardFromBytes $bytes 0 $bytes.Length
    $bpb = Get-SelfBPB $t.sf
    $key = $t.label+'@'+$t.tp
    $rows[$key].st += $st; $rows[$key].div += $dv; $rows[$key].bg += $bg; if ($null -ne $bpb) { $rows[$key].bpb += $bpb }
    # dump to human dir with requested name {arm}_{rN}_T{temp}_s{seed}.txt
    $rN = ($t.label -replace '^[A-Za-z]+_','')   # r7/r8/r9
    $dst = $HDIR+'\'+$t.arm+'_'+$rN+'_T'+$t.tp+'_s'+$t.si+'.txt'
    Copy-Item $t.tf $dst -Force
}

Write-Host ''
Write-Host '======== STRUCTURE + DIVERSITY advisory (per checkpoint x temp; avg over seeds) ========' -ForegroundColor Cyan
Write-Host ('  quoteParity: avg of (#quotes mod 2)  | parenImbal: avg |open-close|  | sentLen vs corpus {0}' -f $CORP_SENTLEN) -ForegroundColor DarkGray
Write-Host '  uniBi/uniTri/uniC4: n-gram uniqueness (1=no repeat, low=loop)  | maxRun: longest word repeat  | selfBPB' -ForegroundColor DarkGray
Write-Host ('  {0,-9} {1,5} {2,7} {3,7} {4,8} {5,7} {6,7} {7,7} {8,7} {9,7}' -f `
    'config','temp','qParity','parImb','sentLen','uniBi','uniTri','uniC4','maxRun','selfBPB')
foreach ($c in $CKPTS) {
    foreach ($tp in $TEMPS) {
        $r = $rows[$c.label+'@'+$tp]
        $qp = Av (@($r.st | ForEach-Object { [double]$_.quoteParity }))
        $pi = Av (@($r.st | ForEach-Object { [double]$_.parenImbal }))
        $sl = Av (@($r.st | ForEach-Object { [double]$_.sentLen }))
        $ub = Mn (@($r.div | ForEach-Object { [double]$_.uniBi }))    # worst (min) uniqueness
        $ut = Mn (@($r.div | ForEach-Object { [double]$_.uniTri }))
        $uc = Mn (@($r.div | ForEach-Object { [double]$_.uniC4 }))
        $mr = Mx (@($r.div | ForEach-Object { [double]$_.maxrun }))   # worst (max) run
        $bp = Av (@($r.bpb | ForEach-Object { [double]$_ }))
        $col = if ($c.arm -eq 'armB') { 'DarkCyan' } else { 'Gray' }
        Write-Host ('  {0,-9} {1,5} {2,7} {3,7} {4,8} {5,7} {6,7} {7,7} {8,7} {9,7}' -f `
            $c.label,$tp,$qp,$pi,$sl,$ub,$ut,$uc,$mr,$bp) -ForegroundColor $col
    }
}
Write-Host ''
Write-Host '  Read: ADAPT/ERR vs armB. Higher uniTri/uniC4 + lower maxRun + sentLen near corpus + low' -ForegroundColor Yellow
Write-Host '  parenImbal/quoteParity = more structure. If ADAPT/ERR >= armB on these at LONG horizon = the' -ForegroundColor Yellow
Write-Host '  stabilized feedback buys structure, not just lower BPB (escalation signal). ADVISORY.' -ForegroundColor Yellow
Write-Host ''
Write-Host ('  Samples ({0}) for human reading in: {1}' -f $tasks.Count,$HDIR) -ForegroundColor Cyan
