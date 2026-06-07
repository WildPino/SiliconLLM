# Phase 43.C - Word-Level Semantic Gate
#
# The byte-level collapse gates (longest char run, top1 byte %, printable) are
# BLIND to semantic loops: a generation can pass every byte gate while spinning
# in a "My / Max / Mom / Mia" name tunnel. This script adds the word-level gate
# that caught oja_1e4 and tempered oja_1e3:
#
#   - top repeated word %        (single most frequent word token / all tokens)
#   - name-ish word %            (share of tokens in the attractor-name lexicon)
#   - max same-word run          (longest run of the SAME word back-to-back)
#   - top bigram / trigram freq  (most repeated adjacent pair / triple)
#   - alt-loop run               (longest A B A B ... period-2 oscillation)
#
# Head-to-head, N seeds, T=0.65, on ONLY the two survivors:  base c20 vs oja 1e-3.
# oja_1e4 was scrapped on manual read (26.9% name-ish avg, max word run 30).
#
# Promotion rule: oja_1e3 is promoted DEFINITIVELY only if it does NOT regress
# vs base on the semantic gate (name-ish avg AND worst-seed, max-word-run worst)
# while keeping its BPB edge. Otherwise: useful BPB, hidden nominal attractor.
#
# NOTE: weights/binaries are produced elsewhere. This script (re)generates text
# from existing weights and scores it. It does NOT train.
#
# Run:  .\benchmarks\phase38-42\phase43c_wordgate.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43c\wordgate'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$GEN_EXE = $BINDIR + '\phase43_generator.exe'
if (-not (Test-Path $GEN_EXE)) { Write-Error "Generator not built: $GEN_EXE (compile via phase43c_oja.ps1)"; exit 1 }

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16          # set to 16 for the final word-gate verdict
$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'
$MINLEN  = 2          # min word length to count as a token (keeps 'my','max','mom')

# Attractor-name lexicon. EDIT FREELY - these are the TinyStories proper names /
# pronoun-attractors that form the observed "My/Max/Mom/Mia" tunnels. 'my' is in
# because it is part of the observed tunnel, not because it is a name.
$NAME_WORDS = @(
    'lily','max','mom','mommy','mum','mummy','mia','tim','tom','ben','sam','sue',
    'dad','daddy','anna','lucy','jack','sara','my','spot','bella','leo','amy'
)
$NAME_SET = @{}; foreach ($w in $NAME_WORDS) { $NAME_SET[$w] = $true }

# Two survivors only.
$gen_configs = @(
    @{ label='base_c20'; weights=$WDIR+'\phase43h_c20.bin' },
    @{ label='oja_1e3';  weights=$WDIR+'\phase43c_eta1e3.bin' }
)
foreach ($c in $gen_configs) {
    if (-not (Test-Path $c.weights)) { Write-Error "Missing weights: $($c.weights)"; exit 1 }
}

# Evenly spaced seed offsets across the corpus, leaving room for warmup+seed.
$fsz    = (Get-Item $TS_DATA).Length
$margin = $WARMUP + 1024
$SEEDS  = @()
for ($i = 0; $i -lt $NSEEDS; $i++) {
    $SEEDS += [int]([math]::Floor($i * ($fsz - $margin) / $NSEEDS))
}

# -------- Word-level metrics --------------------------------------------------
function Get-WordMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    if ($bytes.Length -eq 0) { return $null }

    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^a-zA-Z]', ' '
    $toks = @($text.ToLower() -split '\s+' | Where-Object { $_.Length -ge $MINLEN })
    $nt   = $toks.Count
    if ($nt -lt 4) { return $null }

    # word frequencies
    $wf = @{}
    foreach ($w in $toks) { $wf[$w] = ($wf[$w] -as [int]) + 1 }
    $topw      = ($wf.Values | Measure-Object -Maximum).Maximum
    $top_word  = [math]::Round($topw / $nt * 100.0, 1)

    # name-ish share
    $name_ct = 0
    foreach ($w in $toks) { if ($NAME_SET.ContainsKey($w)) { $name_ct++ } }
    $nameish = [math]::Round($name_ct / $nt * 100.0, 1)

    # max same-word run (A A A ...)
    $run = 1; $maxrun = 1
    for ($j = 1; $j -lt $nt; $j++) {
        if ($toks[$j] -eq $toks[$j-1]) { $run++ } else { $run = 1 }
        if ($run -gt $maxrun) { $maxrun = $run }
    }

    # bigram / trigram peak frequency
    $bf = @{}; $tf = @{}
    for ($j = 0; $j -lt $nt-1; $j++) { $k = $toks[$j]+' '+$toks[$j+1]; $bf[$k] = ($bf[$k] -as [int]) + 1 }
    for ($j = 0; $j -lt $nt-2; $j++) { $k = $toks[$j]+' '+$toks[$j+1]+' '+$toks[$j+2]; $tf[$k] = ($tf[$k] -as [int]) + 1 }
    $top_bi = if ($bf.Count) { ($bf.Values | Measure-Object -Maximum).Maximum } else { 0 }
    $top_tri = if ($tf.Count) { ($tf.Values | Measure-Object -Maximum).Maximum } else { 0 }

    # alt-loop run: longest period-2 oscillation (A B A B ...): toks[j]==toks[j+2]
    $alt = 0; $altmax = 0
    for ($j = 0; $j -lt $nt-2; $j++) {
        if ($toks[$j] -eq $toks[$j+2]) { $alt++ } else { $alt = 0 }
        if ($alt -gt $altmax) { $altmax = $alt }
    }

    # top words string for eyeballing the tunnel
    $top_words = $wf.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 6 |
        ForEach-Object { "$($_.Key)=$($_.Value)" }

    return [PSCustomObject]@{
        tokens=$nt; top_word=$top_word; nameish=$nameish; maxrun=$maxrun
        top_bi=$top_bi; top_tri=$top_tri; altloop=$altmax
        top_words=($top_words -join ' ')
    }
}

# -------- Generate + score ----------------------------------------------------
Write-Host ''
Write-Host "=== Phase 43.C Word-Gate - $NSEEDS seeds, T=$T ===" -ForegroundColor Yellow
Write-Host '  Head-to-head: base_c20 vs oja_1e3 (oja_1e4 already scrapped)'
Write-Host ''

$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{ mets=@(); bpbs=@() } }

$si = 0
foreach ($off in $SEEDS) {
    $si++
    foreach ($cfg in $gen_configs) {
        $lbl = $cfg.label + '_s' + $si
        $tf  = $RDIR + '\gen_' + $lbl + '.txt'
        $sf  = $RDIR + '\gen_' + $lbl + '_stats.txt'
        Write-Host ('  {0} seed{1} (off={2}) ...' -f $cfg.label, $si, $off) -NoNewline
        $cmdLine = "`"$GEN_EXE`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $T --seed-start $off > `"$tf`" 2>`"$sf`""
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

# -------- Summary -------------------------------------------------------------
Write-Host ''
Write-Host '======================================================================' -ForegroundColor Cyan
Write-Host "  Word-Level Semantic Gate (avg / worst over $NSEEDS seeds)"
Write-Host '======================================================================'
Write-Host ('  {0,-10} {1,7} {2,16} {3,14} {4,8} {5,8} {6,12}' -f `
    'config','topW%','nameish%(avg/wst)','wordRun(avg/wst)','topBi','altLp','self_BPB')
Write-Host ('  ' + '-'*78)

$rows = @{}
foreach ($cfg in $gen_configs) {
    $ms = @($agg[$cfg.label].mets); $bs = @($agg[$cfg.label].bpbs)
    if ($ms.Count -eq 0) { Write-Host ('  {0,-10} [no data]' -f $cfg.label); continue }

    $tw  = $ms | ForEach-Object { [double]$_.top_word }
    $ni  = $ms | ForEach-Object { [double]$_.nameish }
    $mr  = $ms | ForEach-Object { [double]$_.maxrun }
    $bi  = $ms | ForEach-Object { [double]$_.top_bi }
    $al  = $ms | ForEach-Object { [double]$_.altloop }

    $row = [PSCustomObject]@{
        ni_avg=(Avg $ni); ni_wst=(Mx $ni); mr_avg=(Avg $mr); mr_wst=(Mx $mr)
        tw_avg=(Avg $tw); bi_avg=(Avg $bi); al_avg=(Avg $al)
        bpb=(Avg $bs)
    }
    $rows[$cfg.label] = $row

    Write-Host ('  {0,-10} {1,7} {2,16} {3,14} {4,8} {5,8} {6,12}' -f `
        $cfg.label,
        ('{0:F1}' -f $row.tw_avg),
        ('{0:F1}/{1:F1}' -f $row.ni_avg, $row.ni_wst),
        ('{0:F0}/{1:F0}' -f $row.mr_avg, $row.mr_wst),
        ('{0:F0}' -f $row.bi_avg),
        ('{0:F0}' -f $row.al_avg),
        ('{0:F4}' -f $row.bpb))
}

# Top words per seed - eyeball the tunnel
Write-Host ''
Write-Host '=== Top words per seed (spot the name tunnel) ===' -ForegroundColor Cyan
foreach ($cfg in $gen_configs) {
    Write-Host ("  -- $($cfg.label)") -ForegroundColor Magenta
    $s = 0
    foreach ($m in @($agg[$cfg.label].mets)) { $s++; Write-Host ('     s{0,-2} {1}' -f $s, $m.top_words) }
}

# -------- Head-to-head verdict ------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto word-gate: oja_1e3 vs base_c20 ===' -ForegroundColor Yellow
if ($rows.ContainsKey('base_c20') -and $rows.ContainsKey('oja_1e3')) {
    $b = $rows['base_c20']; $o = $rows['oja_1e3']
    $tol = 0.5  # pct tolerance: oja must not be meaningfully worse

    # The promotion criterion is SEMANTIC: does Oja reduce the name tunnels.
    # self_BPB is NOT a beat-the-base comparison -- it is a closed-loop
    # auto-confidence signal that only needs to stay in a HEALTHY band. Too low
    # (< 0.8) = collapse/overconfidence; too high (> 2.0) = incoherent drift.
    # The real BPB win is the teacher-forced VAL BPB measured in phase43c_oja
    # (2.2617 vs 2.2656), not this self_BPB.
    $BPB_LO = 0.8; $BPB_HI = 2.0

    $ni_avg_ok = $o.ni_avg -le ($b.ni_avg + $tol)
    $ni_wst_ok = $o.ni_wst -le ($b.ni_wst + $tol)
    $mr_wst_ok = $o.mr_wst -le $b.mr_wst
    $bpb_ok    = ($o.bpb -ge $BPB_LO) -and ($o.bpb -le $BPB_HI)   # healthy band, not beat

    function YN($c) { if ($c) { 'PASS' } else { 'fail' } }
    Write-Host ('  name-ish avg   base {0,5:F1}  oja {1,5:F1}   [{2}]' -f $b.ni_avg,$o.ni_avg,(YN $ni_avg_ok))
    Write-Host ('  name-ish worst base {0,5:F1}  oja {1,5:F1}   [{2}]' -f $b.ni_wst,$o.ni_wst,(YN $ni_wst_ok))
    Write-Host ('  word-run worst base {0,5:F0}  oja {1,5:F0}   [{2}]' -f $b.mr_wst,$o.mr_wst,(YN $mr_wst_ok))
    Write-Host ('  self_BPB band  oja {0,5:F2}  in [{1},{2}]   [{3}]' -f $o.bpb,$BPB_LO,$BPB_HI,(YN $bpb_ok))
    Write-Host ('  (info) self_BPB base {0,5:F2}  -- not a gate, base may differ' -f $b.bpb)
    Write-Host ''
    # Semantic stabilization = oja clearly BELOW base on the tunnel metrics.
    $semantic_win = ($o.ni_avg -lt $b.ni_avg) -and ($o.ni_wst -lt $b.ni_wst) -and ($o.mr_wst -le $b.mr_wst)
    if ($ni_avg_ok -and $ni_wst_ok -and $mr_wst_ok -and $bpb_ok) {
        if ($semantic_win) {
            Write-Host '  -> SEE-V2 CONFERMATO: Oja STABILIZZA semanticamente (tunnel ridotti),' -ForegroundColor Green
            Write-Host '     self_BPB sano. Procedi a 43.C2 (scala plastic cells 10% -> 20%).' -ForegroundColor Green
        } else {
            Write-Host '  -> Oja non rompe e regge i gate, ma non riduce chiaramente i tunnel.' -ForegroundColor Yellow
            Write-Host '     Promozione difendibile su BPB; il guadagno semantico e marginale.' -ForegroundColor Yellow
        }
    } elseif (-not $bpb_ok) {
        Write-Host '  -> self_BPB fuori banda sana: collasso o drift. Rivedi eta/celle.' -ForegroundColor Red
    } else {
        Write-Host '  -> Oja regredisce sul gate semantico: attractor nominali peggiori.' -ForegroundColor Red
        Write-Host '     NON promuovere. Rivedi eta / # plastic cells.' -ForegroundColor Red
    }
} else {
    Write-Host '  [dati insufficienti per il confronto]'
}

Write-Host ''
Write-Host ('  Files: ' + $RDIR)
Write-Host '  Per il verdetto finale, imposta $NSEEDS = 16 in cima e rilancia.'
