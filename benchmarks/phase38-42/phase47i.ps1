# Phase 47.I - Gate V2 (tightening) + premise probe + far-field rollout (re-scoped)
#
# Lettura umana dei sample P_r7 (utente, 2026-06-12) = GOODHART: tre patologie byte-level
# invisibili alle metriche word-level del gate — whitespace flood (10-50+ spazi), il
# "wasteland" (attrattore diffuso a frammenti-template con variazione di superficie:
# nessun bigram domina, topBi muto), char flood ("Jaaaa...", una sola parola strana) +
# byte non stampabili. Pattern temporale: seconda meta' dei sample (~500-1000 byte di
# self-generazione) = attrattore FAR-FIELD mai visitato dai burst K=16 ancorati.
# DECISIONE (delega esercitata): P_r7 NON promuovibile (Goodhart, non statistica).
# Gate V2 = TIGHTENING: altLp resta com'e'; si AGGIUNGONO guard byte-level oggettivi
# (maxCharRun, maxWsRun, wsFrac, nonPrintable) con bar CALIBRATE sul corpus (~1000
# finestre 2KB; bar = max osservato/p99.9 + margine, stampate). Stringere puo' solo
# bocciare, mai promuovere -> nessuna corruzione post-hoc.
#
# Step 1: calibrazione gate v2 + RE-SCORE di P_r7 (sample R1-R4 salvati) e ref C2.A/D1
#         (full 47.G salvati) -> il wasteland e' MLP-specifico o c'era sempre stato?
# Step 2a: PREMISE PROBE senza training (minuti): P_r7 con burst K=128 vs K=16 control;
#         wasteland-entry del tail. Regola pre-registrata: si traina SOLO se K128 >= 1%.
# Step 2b: 47.I re-scoped (solo se premise regge): skeleton 47.G P-branch (prefisso r1-r5
#         bit-identico, MD5), round 6-9 burst mix K16 87.5% / K128 12.5% per indice burst,
#         roll per subset K, mini -> full gate v2 -> REPLICA PROTOCOL su ogni full-PASS.
#         HARD CAP: un tentativo, max un aggiustamento di quota (utente), poi stop.
# Step 3 (fuori harness): chiusura Phase 47 comunque vada, HANDOFF, commit a ordine.
#
# DA NON FARE: allentare bar, filtri/penalita' in inferenza, promuovere "con riserva".
#
# Run:  .\benchmarks\phase38-42\phase47i.ps1
#       .\benchmarks\phase38-42\phase47i.ps1 -RescoreOnly    # solo step 1 + 2a
#       .\benchmarks\phase38-42\phase47i.ps1 -SkipTrain      # riusa train output

param([switch]$SkipTrain,[switch]$RescoreOnly)

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase47i'
$HDIR    = $RDIR + '\human'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
foreach ($d in @($RDIR,$HDIR)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$C2A_W   = $WDIR + '\phase43c2_C2A.bin'
$D1_W    = $WDIR + '\phase44f_F0.bin'
$PR7_W   = $WDIR + '\phase47g_P_h32_r7.bin'
$WP      = $WDIR + '\phase47i'
$G_PR1   = $WDIR + '\phase47g_P_h32_r1.bin'
$G_PR5   = $WDIR + '\phase47g_P_h32_r5.bin'
$OLD_G   = $ROOT + '\results\phase47g\phase47g_train.txt'
$H_RES   = $ROOT + '\results\phase47h'
$G_RES   = $ROOT + '\results\phase47g'
$C2A_BPB = 2.2593
foreach ($w in @($C2A_W,$D1_W,$PR7_W,$G_PR1,$G_PR5)) { if (-not (Test-Path $w)) { Write-Error "Missing weight: $w"; exit 1 } }
if (-not (Test-Path $OLD_G)) { Write-Error "Missing 47.G output: $OLD_G"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
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
$TRAIN_N  = 1000000
$CAL_WINDOWS = 1000
$CAL_WBYTES  = 2048
$PREMISE_THR = 1.0      # % wasteland-entry K128: si traina solo se >= (pre-registrato)

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47i_premise.c' @SRC_CORE -o "$BINDIR\phase47i_premise.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47i_premise'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47i_farfield.c' @SRC_CORE -o "$BINDIR\phase47i_farfield.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47i_farfield'; exit 1 }
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase47_generator.c' @SRC_CORE -o "$BINDIR\phase47_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase47_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# ============ STEP 1a: GATE V2 - byte-guard + calibrazione sul corpus ===============
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
function Get-ByteGuard([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    return Get-ByteGuardFromBytes $bytes 0 $bytes.Length
}
function Pctl([double[]]$a,[double]$q){ if($a.Count -eq 0){return 0.0}; $s=@($a|Sort-Object); $idx=[int][math]::Floor($q*($s.Count-1)); $s[$idx] }

Write-Host ''
Write-Host ('=== STEP 1a: calibrazione gate v2 su corpus ({0} finestre da {1}B) ===' -f $CAL_WINDOWS,$CAL_WBYTES) -ForegroundColor Yellow
$corpus = [System.IO.File]::ReadAllBytes($TS_DATA)
$stride = [long][math]::Floor(($corpus.Length - $CAL_WBYTES) / $CAL_WINDOWS)
$cal_ws=@(); $cal_ch=@(); $cal_wf=@(); $cal_np=@()
for ($wi=0; $wi -lt $CAL_WINDOWS; $wi++) {
    $g = Get-ByteGuardFromBytes $corpus ([long]$wi*$stride) $CAL_WBYTES
    $cal_ws += [double]$g.wsRun; $cal_ch += [double]$g.chRun; $cal_wf += [double]$g.wsFrac; $cal_np += [double]$g.nonPrint
}
$corpus = $null
function Mx2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
# bar = massimo osservato + margine (interi: +2; wsFrac: +0.03; nonPrint: +2); p99.9 stampato accanto
$V2 = [PSCustomObject]@{
    wsRun    = [int]((Mx2 $cal_ws) + 2)
    chRun    = [int]((Mx2 $cal_ch) + 2)
    wsFrac   = [math]::Round((Mx2 $cal_wf) + 0.03, 4)
    nonPrint = [int]((Mx2 $cal_np) + 2)
}
Write-Host ('  corpus: wsRun max={0} p99.9={1} | chRun max={2} p99.9={3} | wsFrac max={4:F4} p99.9={5:F4} | nonPrint max={6} p99.9={7}' -f `
    (Mx2 $cal_ws),(Pctl $cal_ws 0.999),(Mx2 $cal_ch),(Pctl $cal_ch 0.999),(Mx2 $cal_wf),(Pctl $cal_wf 0.999),(Mx2 $cal_np),(Pctl $cal_np 0.999))
Write-Host ('  GATE V2 BARS (max osservato + margine): wsRun<={0}  chRun<={1}  wsFrac<={2:F4}  nonPrint<={3}' -f $V2.wsRun,$V2.chRun,$V2.wsFrac,$V2.nonPrint) -ForegroundColor Cyan
$V2 | ConvertTo-Json | Out-File ($RDIR+'\gatev2_bars.json') -Encoding utf8
$V2 | ConvertTo-Json | Out-File ($ROOT+'\docs\gatev2_bars.json') -Encoding utf8   # copia TRACCIATA (results/ e' gitignored)

function ByteFails($g) {
    $fails=@()
    if ($g.wsRun  -gt $V2.wsRun)    { $fails += ('wsRun {0}>{1}' -f $g.wsRun,$V2.wsRun) }
    if ($g.chRun  -gt $V2.chRun)    { $fails += ('chRun {0}>{1}' -f $g.chRun,$V2.chRun) }
    if ($g.wsFrac -gt $V2.wsFrac)   { $fails += ('wsFrac {0:F3}>{1:F3}' -f $g.wsFrac,$V2.wsFrac) }
    if ($g.nonPrint -gt $V2.nonPrint) { $fails += ('nonPrint {0}>{1}' -f $g.nonPrint,$V2.nonPrint) }
    return ,$fails
}

# ============ STEP 1b: RE-SCORE sotto gate v2 (sample gia' su disco) =================
Write-Host ''
Write-Host '=== STEP 1b: re-score byte-guard di P_r7 (R1-R4, 47.H) e ref C2.A/D1 (full 47.G) ===' -ForegroundColor Yellow
$rescore_groups = @()
foreach ($rep in @('R1_newrngA','R2_newrngB','R3_heldoutA','R4_heldoutB')) {
    foreach ($tp in $TEMPS) {
        $files = @(Get-ChildItem ($H_RES+'\gen_'+$rep+'_T'+$tp+'_*.txt') -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*stats*' })
        if ($files.Count -gt 0) { $rescore_groups += @{ name=('Pr7_'+$rep); tp=$tp; files=$files } }
    }
}
foreach ($refn in @('C2.A','D1')) {
    foreach ($tp in $TEMPS) {
        $files = @(Get-ChildItem ($G_RES+'\gen_'+$refn+'_T'+$tp+'_*.txt') -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike '*stats*' -and $_.Name -notlike '*chk*' -and $_.Name -notlike '*mini*' })
        if ($files.Count -gt 0) { $rescore_groups += @{ name=$refn; tp=$tp; files=$files } }
    }
}
Write-Host ('  {0,-16} {1,5} {2,4} {3,7} {4,7} {5,8} {6,9} {7,7} {8,8}' -f 'gruppo','temp','n','wsRunW','chRunW','wsFracW','nonPrintW','#fail','verdict')
$rescore_summary = @{}
foreach ($grp in $rescore_groups) {
    $ws=@(); $ch=@(); $wf=@(); $np=@(); $nfail=0
    foreach ($f in $grp.files) {
        $g = Get-ByteGuard $f.FullName
        if ($null -eq $g) { continue }
        $ws+=[double]$g.wsRun; $ch+=[double]$g.chRun; $wf+=[double]$g.wsFrac; $np+=[double]$g.nonPrint
        $ff = ByteFails $g   # NB: assegnare PRIMA di contare (return ,$arr + @(call) diretto conta 1 su vuoto)
        if (@($ff).Count -gt 0) { $nfail++ }
    }
    $worst = [PSCustomObject]@{ wsRun=(Mx2 $ws); chRun=(Mx2 $ch); wsFrac=(Mx2 $wf); nonPrint=(Mx2 $np) }
    $gfails = ByteFails $worst
    $verdict = if (@($gfails).Count -eq 0) { 'PASS' } else { 'FAIL' }
    $key = $grp.name + '@' + $grp.tp
    $rescore_summary[$key] = @{ verdict=$verdict; nfail=$nfail; n=$grp.files.Count }
    $col = if ($verdict -eq 'PASS') {'Green'} else {'Red'}
    Write-Host ('  {0,-16} {1,5} {2,4} {3,7} {4,7} {5,8} {6,9} {7,7} {8,8}' -f `
        $grp.name,$grp.tp,$grp.files.Count,('{0:F0}' -f $worst.wsRun),('{0:F0}' -f $worst.chRun),('{0:F3}' -f $worst.wsFrac),('{0:F0}' -f $worst.nonPrint),$nfail,$verdict) -ForegroundColor $col
}
Write-Host ''
Write-Host '  Lettura: se C2.A/D1 FALLISCONO anch''essi il byte-guard, il wasteland non e''' -ForegroundColor Cyan
Write-Host '  MLP-specifico — c''era sempre stato, invisibile alle metriche word-level.' -ForegroundColor Cyan

# ============ STEP 2a: PREMISE PROBE (no training, minuti) ==========================
Write-Host ''
Write-Host '=== STEP 2a: premise probe (P_r7, burst K=128 vs K=16, wasteland-entry del tail) ===' -ForegroundColor Yellow
$PREM_OUT = $RDIR + '\phase47i_premise.txt'
& "$BINDIR\phase47i_premise.exe" $TS_DATA $D1_W $PR7_W 2>&1 | Tee-Object $PREM_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Premise probe failed'; exit 1 }
$k128_rates = @()
foreach ($line in Select-String '^PREMISE ' $PREM_OUT) {
    $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
    if ($kv['K'] -eq '128') { $k128_rates += [double]$kv['waste%'] }
}
$k128_max = Mx2 $k128_rates
Write-Host ''
Write-Host ('  K128 wasteland-entry max sulle finestre: {0:F2}%% (soglia pre-registrata: >= {1:F1}%%)' -f $k128_max,$PREMISE_THR) -ForegroundColor Cyan

if ($RescoreOnly) {
    Write-Host ''
    Write-Host 'RescoreOnly: stop dopo step 1 + 2a.' -ForegroundColor Gray
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}
if ($k128_max -lt $PREMISE_THR) {
    Write-Host ''
    Write-Host '=== Verdetto 47.I (premise) ===' -ForegroundColor Yellow
    Write-Host ('  -> I burst K=128 NON entrano nel wasteland ({0:F2}%% < {1:F1}%%): l''ipotesi di' -f $k128_max,$PREMISE_THR) -ForegroundColor Yellow
    Write-Host '     copertura MUORE QUI, senza spendere il training. Il far-field non e''' -ForegroundColor Yellow
    Write-Host '     raggiungibile da burst ancorati nemmeno a 128 byte.' -ForegroundColor Yellow
    Write-Host '     -> STEP 3: chiusura Phase 47 (verdetto onesto: stack di stabilita'' validato,' -ForegroundColor Yellow
    Write-Host '        generatore non promosso, P_r7 best near-pass entro base metrica word-level).' -ForegroundColor Yellow
    Write-Host '        HANDOFF + commit a ordine utente. Phase 48 = substrate scaling, harness congelato.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}
Write-Host '  -> Premise REGGE: si procede col training far-field (hard cap: UN tentativo).' -ForegroundColor Green

# ============ STEP 2b: far-field training ==========================================
$TRAINERS = @('phase44a_boundary','phase44b_homeostasis','phase44c_delta','phase44d_readout','phase44e_caps','phase44f_captrain','phase45a_relmove','phase45b_geometry','phase45c_gate','phase46a_l3','phase46b_l3','phase47a0_gauntlet','phase47b_decoder','phase47c_robust','phase47d_rollout','phase47e_capacity','phase47f_tempcov','phase47g_lastmile','phase47h_h48tail','phase47i_farfield')
$busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $TRAINERS -contains $_.ProcessName })
if ($busy.Count -gt 0) { Write-Error 'Another trainer is running. Aborting.'; exit 1 }

Write-Host ''
Write-Host '=== STEP 2b: far-field rollout training (prefisso r1-r5 = 47.G, round 6-9 K16/K128) ===' -ForegroundColor Yellow
$TRAIN_OUT = $RDIR + '\phase47i_train.txt'
if ($SkipTrain -and (Test-Path $TRAIN_OUT)) {
    Write-Host ('SkipTrain: riuso ' + $TRAIN_OUT) -ForegroundColor Cyan
} else {
    & "$BINDIR\phase47i_farfield.exe" $TS_DATA $D1_W $WP 2>&1 | Tee-Object $TRAIN_OUT
    if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
}
function Parse-KV([string]$file,[string]$prefix) {
    $out = @{}
    foreach ($line in Select-String ('^'+$prefix+' ') $file) {
        $kv = @{}; foreach ($tok in ($line.Line.Trim() -split '\s+' | Select-Object -Skip 1)) { $p=$tok -split '=',2; $kv[$p[0]]=$p[1] }
        $key = if ($prefix -eq 'BASE') { $kv['win'] } else { $kv['name'] }
        $out[$key] = $kv
    }
    return $out
}
function AvgVal($p) { return [math]::Round((([double]$p['val1']+[double]$p['val2']+[double]$p['val3'])/3.0),4) }
$BASE  = Parse-KV $TRAIN_OUT 'BASE'
$PROBE = Parse-KV $TRAIN_OUT 'PROBE'
if (-not $PROBE.ContainsKey('frozenD1') -or -not $PROBE.ContainsKey('I_r9')) { Write-Error 'Train output incomplete'; exit 1 }
# anchor
$anchor_ok = $true
foreach ($w in @('val1','val2','val3')) {
    $d1v = [double]$BASE[$w]['d1']; $i = [array]::IndexOf(@('val1','val2','val3'),$w)+1
    if ([math]::Abs([double]$PROBE['frozenD1']["val$i"]-$d1v) -gt 0.005) { $anchor_ok = $false }
}
if (-not $anchor_ok) { Write-Error 'Anchor FAIL.'; exit 1 }
Write-Host '  anchor OK' -ForegroundColor Green
# prefix MD5 r1/r5 vs 47.G
function FileMD5($p) {
    if (-not (Test-Path $p)) { return '' }
    for ($a=0; $a -lt 15; $a++) { try { return (Get-FileHash -Algorithm MD5 -Path $p -ErrorAction Stop).Hash } catch { Start-Sleep -Milliseconds 200 } }
    return ''
}
$prefix_ok = $true
foreach ($chk in @(@{g=($WP+'_I_h32_r1.bin'); f=$G_PR1; n='r1'}, @{g=($WP+'_I_h32_r5.bin'); f=$G_PR5; n='r5'})) {
    $hg = FileMD5 $chk.g; $hf = FileMD5 $chk.f
    $ok = ($hg -ne '' -and $hg -eq $hf); if (-not $ok) { $prefix_ok = $false }
    Write-Host ('  prefix@{0} {1}' -f $chk.n,($(if($ok){'MATCH'}else{'MISMATCH'}))) -ForegroundColor $(if($ok){'Green'}else{'Red'})
}
if (-not $prefix_ok) { Write-Error 'Prefix property FAILED. Aborting.'; exit 1 }

Write-Host ''
Write-Host '=== TF per round (rollK128 = diagnostica collasso far-field; sano se sale verso la banda K16) ===' -ForegroundColor Yellow
Write-Host ('  {0,-6} {1,8} {2,8} {3,8} {4,8} {5,8}' -f 'probe','avgVal','valC','rollK16','rollK128','ent1')
foreach ($rn in 1..9) {
    $n = "I_r$rn"
    if (-not $PROBE.ContainsKey($n)) { continue }
    $p = $PROBE[$n]; $av = AvgVal $p
    $r16 = if ($p.ContainsKey('rollK16')) { $p['rollK16'] } elseif ($p.ContainsKey('roll')) { $p['roll'] } else { '-' }
    $r128 = if ($p.ContainsKey('rollK128')) { $p['rollK128'] } else { '-' }
    $col = if ($av -le $PROMO_BPB) {'Green'} else {'Gray'}
    Write-Host ('  {0,-6} {1,8} {2,8} {3,8} {4,8} {5,8}' -f $n,('{0:F4}' -f $av),$p['valC'],$r16,$r128,$p['ent1']) -ForegroundColor $col
}

# ============ Gate V2 machinery (word + byte) =======================================
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
function Mx2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }
function Av2 ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }
function Get-SelfBPB([string]$statsfile) {
    $bl = Select-String 'self_BPB' $statsfile | Select-Object -Last 1
    if ($bl) { return [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    return $null
}
function New-GenTask($label,$wargs,$tp,$rng,$off,$si,$suffix) {
    $lbl = $label+'_T'+$tp+'_r'+$rng+'_s'+$si+$suffix
    $tf  = $RDIR+'\gen_'+$lbl+'.txt'
    $sf  = $RDIR+'\gen_'+$lbl+'_stats.txt'
    $argline = ('"{0}" {1} --gen-len {2} --warmup {3} --temp {4} --seed-start {5} --rng-seed {6}' -f `
                $TS_DATA,$wargs,$GEN_LEN,$WARMUP,$tp,$off,$rng)
    return [PSCustomObject]@{ label=$label; tp=$tp; rng=$rng; off=$off; si=$si;
                              exe=($BINDIR+'\phase47_generator.exe'); tf=$tf; sf=$sf; argline=$argline }
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
function WorstOf($mets,$bpbs) {
    $ni=@(); $mr=@(); $bi=@(); $al=@(); $ws=@(); $ch=@(); $wf=@(); $np=@(); $bs=@()
    foreach ($m in @($mets)) { $ni+=[double]$m.nameish; $mr+=[double]$m.maxrun; $bi+=[double]$m.top_bi; $al+=[double]$m.altloop
        $ws+=[double]$m.wsRun; $ch+=[double]$m.chRun; $wf+=[double]$m.wsFrac; $np+=[double]$m.nonPrint }
    foreach ($b in @($bpbs)) { $bs+=[double]$b }
    return [PSCustomObject]@{ ni=(Mx2 $ni); mr=(Mx2 $mr); bi=(Mx2 $bi); al=(Mx2 $al);
                              wsRun=(Mx2 $ws); chRun=(Mx2 $ch); wsFrac=(Mx2 $wf); nonPrint=(Mx2 $np);
                              self=(Av2 $bs); n=@($mets).Count }
}
# gate v2 completo: word bars + byte bars + selfBPB band (+ BPB a monte)
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
    Write-Host ('  {0,-8} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f `
        $name,$tp,('{0:F0}' -f $w.bi),('{0:F0}' -f $w.al),('{0:F0}' -f $w.mr),('{0:F1}' -f $w.ni),('{0:F2}' -f $w.self),`
        ('{0:F0}' -f $w.wsRun),('{0:F0}' -f $w.chRun),('{0:F2}' -f $w.wsFrac),('{0:F0}' -f $w.nonPrint),$verdict) -ForegroundColor $col
}

# ============ Mini gate v2 sui checkpoint I_r6..I_r9 ==============================
$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS_STD = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS_STD += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }
$SEEDS_MINI = @()
for ($i = 0; $i -lt 4; $i++) { $SEEDS_MINI += [int]([math]::Floor($i * ($fsz - $margin) / 4)) }

$cands = @()
foreach ($rn in 6..9) {
    $n = "I_r$rn"; $wf = $WP + '_I_h32_r' + $rn + '.bin'
    if ($PROBE.ContainsKey($n) -and (Test-Path $wf)) {
        $cands += @{ label=$n; wargs=('"'+$D1_W+'" "'+$wf+'"'); valbpb=(AvgVal $PROBE[$n]) }
    }
}
$mini_tasks = @()
foreach ($tp in $TEMPS) { $si=0
    foreach ($off in $SEEDS_MINI) { $si++
        foreach ($c in $cands) { $mini_tasks += New-GenTask ($c.label) $c.wargs $tp $RNGS_ORIG[0] $off $si '_mini' }
    } }
Write-Host ''
Write-Host ('=== MINI gate v2 ({0} runs) ===' -f $mini_tasks.Count) -ForegroundColor Yellow
Invoke-Throttled $mini_tasks $THROTTLE
$agg = @{}
foreach ($c in $cands) { $agg[$c.label] = @{}; foreach ($tp in $TEMPS) { $agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
foreach ($t in $mini_tasks) {
    $m = Get-AllMetrics $t.tf
    if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf
    if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host ('======== MINI v2 (qualify: bi<={0} entrambe + selfBPB55>={1} + BPB<={2} + byte-guard puliti) ========' -f $MINI_BI,$BPB_LO,$PROMO_BPB) -ForegroundColor Cyan
Write-Host ('  {0,-8} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','promo')
$promoted = @()
foreach ($c in $cands) {
    $w65 = WorstOf $agg[$c.label]['0.65'].mets $agg[$c.label]['0.65'].bpbs
    $w55 = WorstOf $agg[$c.label]['0.55'].mets $agg[$c.label]['0.55'].bpbs
    $bi_ok   = ($w65.bi -le $MINI_BI) -and ($w55.bi -le $MINI_BI)
    $self_ok = ($w55.self -ge $BPB_LO) -and ($w65.self -ge $BPB_LO) -and ($w65.self -le $BPB_HI) -and ($w55.self -le $BPB_HI)
    $bpb_ok  = ($c.valbpb -le $PROMO_BPB)
    $gf65 = GateV2Fails $w65 '0.65'; $gf55 = GateV2Fails $w55 '0.55'
    $byte_ok = (@($gf65 | Where-Object { $_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*' }).Count -eq 0) -and `
               (@($gf55 | Where-Object { $_ -like 'ws*' -or $_ -like 'ch*' -or $_ -like 'nonPrint*' }).Count -eq 0)
    $promo = if ($bi_ok -and $self_ok -and $bpb_ok -and $byte_ok) { $promoted += $c; 'FULL' } else { 'stop' }
    $col = if ($promo -eq 'FULL') {'Green'} else {'Gray'}
    PrintGateRow $c.label '0.65' $w65 '' $col
    PrintGateRow $c.label '0.55' $w55 $promo $col
}

if ($promoted.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 47.I (mini v2) ===' -ForegroundColor Yellow
    Write-Host '  -> Nessun checkpoint qualifica sotto gate v2. HARD CAP: il tentativo era uno.' -ForegroundColor Yellow
    Write-Host '     (Max un aggiustamento di quota K128 a decisione utente, poi Phase 47 si' -ForegroundColor Yellow
    Write-Host '     chiude comunque: stack di stabilita'' validato, generatore non promosso.)' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}

# ============ FULL gate v2 ========================================================
$full_tasks = @()
foreach ($tp in $TEMPS) { foreach ($rng in $RNGS_ORIG) { $si=0
    foreach ($off in $SEEDS_STD) { $si++
        foreach ($c in $promoted) { $full_tasks += New-GenTask ($c.label) $c.wargs $tp $rng $off $si '_full' }
    } } }
Write-Host ''
Write-Host ('=== FULL gate v2: {0} promossi, {1} runs ===' -f $promoted.Count,$full_tasks.Count) -ForegroundColor Yellow
foreach ($c in $promoted) { foreach ($tp in $TEMPS) { $agg[$c.label][$tp] = @{ mets=@(); bpbs=@() } } }
Invoke-Throttled $full_tasks $THROTTLE
foreach ($t in $full_tasks) {
    $m = Get-AllMetrics $t.tf
    if ($m) { $agg[$t.label][$t.tp].mets += $m }
    $b = Get-SelfBPB $t.sf
    if ($null -ne $b) { $agg[$t.label][$t.tp].bpbs += $b }
}
Write-Host ''
Write-Host '======== FULL gate v2 (worst-case su 32, word + byte) ========' -ForegroundColor Cyan
Write-Host ('  {0,-8} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'config','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','gate')
$full_pass = @()
foreach ($c in $promoted) {
    $allfails = @()
    foreach ($tp in $TEMPS) {
        $w = WorstOf $agg[$c.label][$tp].mets $agg[$c.label][$tp].bpbs
        $tf2 = GateV2Fails $w $tp
        $allfails += $tf2
        $v = if (@($tf2).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) { 'PASS' } else { 'fail' }
        $col = if ($v -eq 'PASS') {'Green'} else {'Gray'}
        PrintGateRow $c.label $tp $w $v $col
        if (@($tf2).Count -gt 0) { Write-Host ('    fail: ' + ($tf2 -join ', ')) -ForegroundColor Red }
    }
    if (@($allfails).Count -eq 0 -and $c.valbpb -le $PROMO_BPB) { $full_pass += $c }
}

# ============ REPLICA PROTOCOL su ogni full-PASS ===================================
if ($full_pass.Count -eq 0) {
    Write-Host ''
    Write-Host '=== Verdetto 47.I ===' -ForegroundColor Yellow
    Write-Host '  -> Promossi al full ma nessun PASS pieno sotto gate v2. HARD CAP raggiunto:' -ForegroundColor Yellow
    Write-Host '     un tentativo, max un aggiustamento quota (utente), poi STEP 3 chiusura.' -ForegroundColor Yellow
    Write-Host ('  Files: ' + $RDIR)
    exit 0
}
$USED_WIN = @(
    @{ lo=[long][math]::Floor($fsz/5);    hi=[long]([math]::Floor($fsz/5)+$TRAIN_N+$margin) },
    @{ lo=[long][math]::Floor($fsz/2);    hi=[long]([math]::Floor($fsz/2)+$TRAIN_N+$margin) },
    @{ lo=[long][math]::Floor(0.65*$fsz); hi=[long]([math]::Floor(0.65*$fsz)+$TRAIN_N+$margin) },
    @{ lo=[long][math]::Floor(0.80*$fsz); hi=[long]([math]::Floor(0.80*$fsz)+$TRAIN_N+$margin) }
)
function New-HeldOutSeeds([double]$frac) {
    $hs = @()
    for ($i = 0; $i -lt $NSEEDS; $i++) {
        $off = [long][math]::Floor(($i + $frac) * ($fsz - $margin) / $NSEEDS)
        $moved = $true
        while ($moved) {
            $moved = $false
            foreach ($uw in $USED_WIN) {
                if ($off -ge ($uw.lo - $margin) -and $off -lt $uw.hi) { $off = $uw.hi + 1048576; $moved = $true }
            }
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
    Write-Host ('=== REPLICA PROTOCOL (gate v2) su {0}: 4 repliche x 32 x 2 temp ===' -f $c.label) -ForegroundColor Yellow
    $rep_tasks = @()
    foreach ($rep in $REPLICAS) {
        foreach ($tp in $TEMPS) { foreach ($rng in $rep.rngs) { $si=0
            foreach ($off in $rep.seeds) { $si++
                $rep_tasks += New-GenTask ($c.label+'_'+$rep.name) $c.wargs $tp $rng $off $si ''
            } } }
    }
    Invoke-Throttled $rep_tasks $THROTTLE
    $rep_agg = @{}
    foreach ($rep in $REPLICAS) { $rep_agg[$rep.name] = @{}; foreach ($tp in $TEMPS) { $rep_agg[$rep.name][$tp] = @{ mets=@(); bpbs=@() } } }
    foreach ($t in $rep_tasks) {
        $repname = ($t.label -split '_')[-1]
        $m = Get-AllMetrics $t.tf
        if ($m) { $rep_agg[$repname][$t.tp].mets += $m }
        $b = Get-SelfBPB $t.sf
        if ($null -ne $b) { $rep_agg[$repname][$t.tp].bpbs += $b }
    }
    $rep_pass = 0
    Write-Host ('  {0,-8} {1,5} {2,5} {3,5} {4,5} {5,6} {6,6} {7,6} {8,7} {9,5} {10,7} {11,8}' -f 'replica','temp','bi','al','run','name','self','wsRun','chRun','wsFr','nonPr','verdict')
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
    # lettura umana: 5 sample/temp dalla replica R1
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
        Write-Host ('  -> {0} = 4/4 sotto GATE V2. Candidato confermato dalla statistica; resta la' -f $c.label) -ForegroundColor Green
        Write-Host '     LETTURA UMANA (componente permanente del gate) e la decisione utente.' -ForegroundColor Green
        Write-Host '     Aspettativa a verbale: anche un pass pieno a ~2.25 BPB legge come word-salad' -ForegroundColor Green
        Write-Host '     strutturato — la promozione misura la stabilita'', la lingua e'' substrato.' -ForegroundColor Green
    } else {
        Write-Host ('  -> {0} non regge le repliche sotto gate v2.' -f $c.label) -ForegroundColor Yellow
    }
}
Write-Host ''
Write-Host 'HARD CAP: tentativo unico esaurito. STEP 3 (chiusura Phase 47, HANDOFF, commit) a ordine utente.' -ForegroundColor Yellow
Write-Host ('  Files: ' + $RDIR)
