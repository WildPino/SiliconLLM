# Phase 43 - Deterministic Word-Gate Re-check (generation only, NO retraining)
#
# The RNG fix (--rng-seed) changed what every PRE-FIX word-gate meant: old runs
# measured sampling noise. So the C2/C3 word-gate verdicts must be re-verified on
# the EXISTING weights, deterministically, before any structural decision.
#
# Most important: C2.B (26c eta3e-4) had the strongest BPB (2.2572) but a
# "disaster" word-gate that was measured PRE-FIX. It is the last candidate not
# definitively dead. This script re-scores it (and its neighbours) with fixed RNG.
#
# Configs (existing weights, no training):
#   SEE-V1S c20   phase43h_c20.bin       (BPB 2.2656)
#   SEE-V2 13c    phase43c_eta1e3.bin    (BPB 2.2617)
#   C2.A 26c e1e-3 phase43c2_C2A.bin
#   C2.B 26c e3e-4 phase43c2_C2B.bin     <- the candidate to settle
#   C2.C 26c e1e-4 phase43c2_C2C.bin
#   C3.B 26c e1e-3 b0.50 phase43c3_C3B.bin
#   C3.C 26c e3e-4 b0.25 phase43c3_C3C.bin
#
# Promotion gate (per 26c candidate): val BPB <= 2.2597 AND name-ish worst <= 20
#   AND word-run worst <= 5 AND top bigram <= 8 AND alt loop <= 2 AND self in [0.8,2.0]
#
# Decision: if C2.B passes -> promote 26c eta3e-4. If C2.B fails -> close Phase 43,
#   dual baseline SEE-V1S/SEE-V2, next phase = structural jump (hierarchical memory).
#
# Run:  .\benchmarks\phase38-42\phase43_wordgate_recheck.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43_recheck'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
$C2LOG   = $ROOT + '\results\phase43c2\phase43c2_train.txt'
$C3LOG   = $ROOT + '\results\phase43c3\phase43c3_train.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'
$RNG     = 12345         # explicit fixed sampling seed -> deterministic
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0
$SEE_V2_BPB = 2.2617

$GATE_NAMEISH_WST = 20.0
$GATE_WORDRUN_WST = 5.0
$GATE_TOPBI       = 8.0
$GATE_ALTLOOP     = 2.0
$GATE_BPB_IMPROVE = 0.002

$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# -------- Compile generator (ensure --rng-seed build) -------------------------
Write-Host ''
Write-Host '=== COMPILING generator ===' -ForegroundColor Cyan
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Resolve val BPB from training logs ----------------------------------
function Get-LogBPB([string]$log, [string]$sfx) {
    if (-not (Test-Path $log)) { return $null }
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $log | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}

# label, weights, isbase, fixed valbpb (or null -> parse log), log, sfx
$gen_configs = @(
    @{ label='c20';    weights=$WDIR+'\phase43h_c20.bin';    isbase=$true;  valbpb=2.2656; log=$null;  sfx=$null },
    @{ label='SEE-V2'; weights=$WDIR+'\phase43c_eta1e3.bin'; isbase=$true;  valbpb=$SEE_V2_BPB; log=$null; sfx=$null },
    @{ label='C2.A';   weights=$WDIR+'\phase43c2_C2A.bin';   isbase=$false; valbpb=$null; log=$C2LOG; sfx='_C2A.bin' },
    @{ label='C2.B';   weights=$WDIR+'\phase43c2_C2B.bin';   isbase=$false; valbpb=$null; log=$C2LOG; sfx='_C2B.bin' },
    @{ label='C2.C';   weights=$WDIR+'\phase43c2_C2C.bin';   isbase=$false; valbpb=$null; log=$C2LOG; sfx='_C2C.bin' },
    @{ label='C3.B';   weights=$WDIR+'\phase43c3_C3B.bin';   isbase=$false; valbpb=$null; log=$C3LOG; sfx='_C3B.bin' },
    @{ label='C3.C';   weights=$WDIR+'\phase43c3_C3C.bin';   isbase=$false; valbpb=$null; log=$C3LOG; sfx='_C3C.bin' }
)
foreach ($cfg in $gen_configs) {
    if ($null -eq $cfg.valbpb -and $null -ne $cfg.log) { $cfg.valbpb = Get-LogBPB $cfg.log $cfg.sfx }
}

# -------- Word-level metrics --------------------------------------------------
function Get-WordMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }
    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^a-zA-Z]', ' '
    $toks = @($text.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
    $nt = $toks.Count
    if ($nt -lt 4) { return $null }

    $wf = @{}
    foreach ($w in $toks) { $wf[$w] = ($wf[$w] -as [int]) + 1 }
    $top_word = [math]::Round(($wf.Values | Measure-Object -Maximum).Maximum / $nt * 100.0, 1)

    $name_ct = 0
    foreach ($w in $toks) { if ($NAME_SET.ContainsKey($w)) { $name_ct++ } }
    $nameish = [math]::Round($name_ct / $nt * 100.0, 1)

    $run = 1; $maxrun = 1
    for ($j = 1; $j -lt $nt; $j++) {
        if ($toks[$j] -eq $toks[$j-1]) { $run++ } else { $run = 1 }
        if ($run -gt $maxrun) { $maxrun = $run }
    }
    $bf = @{}
    for ($j = 0; $j -lt $nt-1; $j++) { $k = $toks[$j]+' '+$toks[$j+1]; $bf[$k] = ($bf[$k] -as [int]) + 1 }
    $top_bi = if ($bf.Count) { ($bf.Values | Measure-Object -Maximum).Maximum } else { 0 }
    $alt = 0; $altmax = 0
    for ($j = 0; $j -lt $nt-2; $j++) {
        if ($toks[$j] -eq $toks[$j+2]) { $alt++ } else { $alt = 0 }
        if ($alt -gt $altmax) { $altmax = $alt }
    }
    $top_words = $wf.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 6 |
        ForEach-Object { "$($_.Key)=$($_.Value)" }
    return [PSCustomObject]@{
        top_word=$top_word; nameish=$nameish; maxrun=$maxrun; top_bi=$top_bi; altloop=$altmax
        top_words=($top_words -join ' ')
    }
}

# -------- Generate + score (deterministic) ------------------------------------
Write-Host ''
Write-Host ('=== Deterministic word-gate: {0} seeds, T={1}, rng={2} ===' -f $NSEEDS,$T,$RNG) -ForegroundColor Yellow
Write-Host '  Generation only, existing weights, NO retraining.'
Write-Host ''

$fsz = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS = @()
for ($i = 0; $i -lt $NSEEDS; $i++) { $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS)) }

$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{ mets=@(); bpbs=@() } }

$si = 0
foreach ($off in $SEEDS) {
    $si++
    foreach ($cfg in $gen_configs) {
        if (-not (Test-Path $cfg.weights)) { Write-Host ('  {0} seed{1}: weights missing, skip' -f $cfg.label,$si) -ForegroundColor Red; continue }
        $lbl = $cfg.label + '_s' + $si
        $tf  = $RDIR + '\gen_' + $lbl + '.txt'
        $sf  = $RDIR + '\gen_' + $lbl + '_stats.txt'
        Write-Host ('  {0,-8} seed{1,-2} (off={2}) ...' -f $cfg.label, $si, $off) -NoNewline
        $cmdLine = "`"$BINDIR\phase43_generator.exe`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $T --seed-start $off --rng-seed $RNG > `"$tf`" 2>`"$sf`""
        cmd /c $cmdLine
        if ($LASTEXITCODE -ne 0) { Write-Host ' FAILED' -ForegroundColor Red; continue }
        Write-Host ' OK' -ForegroundColor Green
        $m = Get-WordMetrics $tf
        if ($m) { $agg[$cfg.label].mets += $m }
        $bl = Select-String 'self_BPB' $sf | Select-Object -Last 1
        if ($bl) { $agg[$cfg.label].bpbs += [double]($bl.Line.Trim() -replace 'self_BPB:\s*','') }
    }
}

function Avg([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }
function Mx ([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; ($a|Measure-Object -Maximum).Maximum }

# -------- Summary table -------------------------------------------------------
Write-Host ''
Write-Host '======================================================================' -ForegroundColor Cyan
Write-Host ('  Phase 43 Word-Gate Re-check ({0} seeds, deterministic)' -f $NSEEDS)
Write-Host '======================================================================'
Write-Host ('  gate(26c): valBPB<={0}  nameWst<={1}  runWst<={2}  topBi<={3}  altLp<={4}  self in [{5},{6}]' -f `
    ($SEE_V2_BPB-$GATE_BPB_IMPROVE),$GATE_NAMEISH_WST,$GATE_WORDRUN_WST,$GATE_TOPBI,$GATE_ALTLOOP,$BPB_LO,$BPB_HI)
Write-Host ''
Write-Host ('  {0,-8} {1,9} {2,8} {3,14} {4,8} {5,7} {6,7} {7,9} {8,6}' -f `
    'config','valBPB','dBPB','nameish(av/wst)','runWst','topBi','altLp','selfBPB','gate')
Write-Host ('  ' + '-'*86)

$rows = @{}
foreach ($cfg in $gen_configs) {
    $ms = @($agg[$cfg.label].mets); $bs = @($agg[$cfg.label].bpbs)
    if ($ms.Count -eq 0) { Write-Host ('  {0,-8} [no data]' -f $cfg.label); continue }

    $ni = $ms | ForEach-Object { [double]$_.nameish }
    $mr = $ms | ForEach-Object { [double]$_.maxrun }
    $bi = $ms | ForEach-Object { [double]$_.top_bi }
    $al = $ms | ForEach-Object { [double]$_.altloop }

    $vb = $cfg.valbpb
    $row = [PSCustomObject]@{
        valbpb=$vb; ni_avg=(Avg $ni); ni_wst=(Mx $ni); mr_wst=(Mx $mr)
        bi_wst=(Mx $bi); al_wst=(Mx $al); selfbpb=(Avg $bs); isbase=$cfg.isbase
    }
    $rows[$cfg.label] = $row
    $dbpb = if ($vb -ne $null) { $vb - $SEE_V2_BPB } else { $null }

    $gate = 'ref'
    if (-not $cfg.isbase) {
        $bpb_ok = ($vb -ne $null) -and ($vb -le ($SEE_V2_BPB - $GATE_BPB_IMPROVE))
        $ni_ok  = $row.ni_wst -le $GATE_NAMEISH_WST
        $mr_ok  = $row.mr_wst -le $GATE_WORDRUN_WST
        $bi_ok  = $row.bi_wst -le $GATE_TOPBI
        $al_ok  = $row.al_wst -le $GATE_ALTLOOP
        $sb_ok  = ($row.selfbpb -ge $BPB_LO) -and ($row.selfbpb -le $BPB_HI)
        $gate = if ($bpb_ok -and $ni_ok -and $mr_ok -and $bi_ok -and $al_ok -and $sb_ok) { 'PASS' } else { 'fail' }
    }

    $vb_str = if ($vb -ne $null) { '{0:F4}' -f $vb } else { 'N/A' }
    $db_str = if ($dbpb -ne $null) { '{0:+0.000;-0.000;0.000}' -f $dbpb } else { '-' }

    Write-Host ('  {0,-8} {1,9} {2,8} {3,14} {4,8} {5,7} {6,7} {7,9} {8,6}' -f `
        $cfg.label, $vb_str, $db_str,
        ('{0:F1}/{1:F1}' -f $row.ni_avg, $row.ni_wst),
        ('{0:F0}' -f $row.mr_wst),
        ('{0:F0}' -f $row.bi_wst),
        ('{0:F0}' -f $row.al_wst),
        ('{0:F2}' -f $row.selfbpb),
        $gate)
}

# -------- Top words per seed --------------------------------------------------
Write-Host ''
Write-Host '=== Top words - seeds 1..4 ===' -ForegroundColor Cyan
foreach ($cfg in $gen_configs) {
    Write-Host ('  -- {0}' -f $cfg.label) -ForegroundColor Magenta
    $s = 0
    foreach ($m in @($agg[$cfg.label].mets)) { $s++; if ($s -gt 4) { break }; Write-Host ('     s{0,-2} {1}' -f $s, $m.top_words) }
}

# -------- Verdict: BEST PASSING config (not best BPB) ------------------------
# A candidate is promotable only if it clears ALL gates. Among those, pick the
# lowest val BPB. (Earlier bug: spotlighting the best-BPB config alone wrongly
# concluded "close Phase 43" when a lower-BPB-but-failing config existed while a
# slightly-higher-BPB config passed everything.)
function Is-Pass($lbl) {
    if (-not $rows.ContainsKey($lbl)) { return $false }
    $r = $rows[$lbl]
    if ($null -eq $r.valbpb) { return $false }
    return ($r.valbpb -le ($SEE_V2_BPB - $GATE_BPB_IMPROVE)) -and `
           ($r.ni_wst -le $GATE_NAMEISH_WST) -and ($r.mr_wst -le $GATE_WORDRUN_WST) -and `
           ($r.bi_wst -le $GATE_TOPBI) -and ($r.al_wst -le $GATE_ALTLOOP) -and `
           ($r.selfbpb -ge $BPB_LO) -and ($r.selfbpb -le $BPB_HI)
}

Write-Host ''
Write-Host '=== Verdetto: best PASSING config (non best BPB) ===' -ForegroundColor Yellow

$win_lbl=''; $win_bpb=$null
foreach ($cfg in $gen_configs) {
    if ($cfg.isbase) { continue }
    if (Is-Pass $cfg.label) {
        $b = $rows[$cfg.label].valbpb
        if ($null -eq $win_bpb -or $b -lt $win_bpb) { $win_bpb=$b; $win_lbl=$cfg.label }
    }
}

# Show which candidates pass / why others fail
foreach ($cfg in $gen_configs) {
    if ($cfg.isbase) { continue }
    if (-not $rows.ContainsKey($cfg.label)) { continue }
    $r = $rows[$cfg.label]
    if (Is-Pass $cfg.label) {
        Write-Host ('  {0,-6} PASS  (valBPB {1:F4}, nameWst {2:F1}, topBi {3:F0})' -f $cfg.label,$r.valbpb,$r.ni_wst,$r.bi_wst) -ForegroundColor Green
    } else {
        $why=@()
        if ($null -eq $r.valbpb -or $r.valbpb -gt ($SEE_V2_BPB-$GATE_BPB_IMPROVE)) { $why+='BPB' }
        if ($r.ni_wst -gt $GATE_NAMEISH_WST) { $why+='nameWst' }
        if ($r.mr_wst -gt $GATE_WORDRUN_WST) { $why+='runWst' }
        if ($r.bi_wst -gt $GATE_TOPBI) { $why+='topBi' }
        if ($r.al_wst -gt $GATE_ALTLOOP) { $why+='altLp' }
        if ($r.selfbpb -lt $BPB_LO -or $r.selfbpb -gt $BPB_HI) { $why+='self' }
        Write-Host ('  {0,-6} fail  (fails: {1})' -f $cfg.label,($why -join ',')) -ForegroundColor DarkGray
    }
}

Write-Host ''
if ($null -ne $win_bpb) {
    Write-Host ('  -> WINNER: {0}  (valBPB {1:F4}, {2:+0.000;-0.000;0.000} vs SEE-V2)' -f `
        $win_lbl, $win_bpb, ($win_bpb-$SEE_V2_BPB)) -ForegroundColor Green
    Write-Host '     Primo 26-cell che passa BPB + word-gate. NON chiudere Phase 43.' -ForegroundColor Green
    Write-Host '     Prossimo: confirmation 32-seed (o 16x2 rng) su WINNER vs c20 vs SEE-V2.' -ForegroundColor Green
    Write-Host '     Se ripassa -> SEE-V3 = SEE-V1S c20 + 26c Oja eta=1e-3.' -ForegroundColor Green
} else {
    Write-Host '  -> Nessun candidato passa TUTTI i gate.' -ForegroundColor Red
    Write-Host '     CHIUDI Phase 43. Baseline duale SEE-V1S c20 / SEE-V2 13c.' -ForegroundColor Red
    Write-Host '     Phase 44 = salto strutturale: memoria gerarchica / boundary-gated.' -ForegroundColor Red
}

Write-Host ''
Write-Host ('  Files: ' + $RDIR)
