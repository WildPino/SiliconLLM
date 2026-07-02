# Phase 43.C2 - Plastic Capacity Scaling Tribunal
#
# Question: can SEE-V2 use MORE local plasticity (13 -> 26 cells) without
# reopening semantic attractors? "Remember more without falling into your own
# attractors." This is the step toward an LLM - NOT routing/MLP/Transformer yet.
#
# base:    SEE-V2 = c20 + Oja 13 cells eta=1e-3   (weights/phase43c_eta1e3.bin)
# variants (self-describing magic 0x53454539, n_oja in header):
#   C2.0  13 cells eta=1e-3   control / reproduce SEE-V2
#   C2.A  26 cells eta=1e-3
#   C2.B  26 cells eta=3e-4
#   C2.C  26 cells eta=1e-4   safety low-eta
#
# Metrics: teacher-forced val BPB (trainer) + 16-seed gen word-gate + self_BPB band.
#
# Promotion gate (per 26-cell variant):
#   val BPB improves >= 0.002 vs SEE-V2 (2.2617)
#   name-ish worst <= 20%   word-run worst <= 5   top bigram <= 8   alt loop <= 2
#   0.8 <= self_BPB <= 2.0
#
# Run:  .\benchmarks\phase38-42\phase43c2_scaling.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43c2'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$BASE_W = $WDIR + '\phase43c_eta1e3.bin'   # SEE-V2
$C20_W  = $WDIR + '\phase43h_c20.bin'      # warm-start readout source
$WP     = $WDIR + '\phase43c2'             # output prefix -> phase43c2_C20.bin etc.
$SEE_V2_BPB = 2.2617

foreach ($w in @($BASE_W, $C20_W)) {
    if (-not (Test-Path $w)) { Write-Error "Missing weights: $w"; exit 1 }
}

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# -------- Config --------------------------------------------------------------
$NSEEDS  = 16
$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'
$MINLEN  = 2
$BPB_LO  = 0.8
$BPB_HI  = 2.0

# Promotion-gate thresholds (per user spec)
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

# -------- Compile -------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
Write-Host '  phase43c2_scaling.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43c2_scaling.c' @SRC_CORE -o "$BINDIR\phase43c2_scaling.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43c2_scaling'; exit 1 }
Write-Host '  phase43_generator.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }
Write-Host 'Compiled OK.' -ForegroundColor Green

# -------- Train ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Phase 43.C2 - Training 4 variants (C2.0/A/B/C) ===' -ForegroundColor Yellow
Write-Host '  Warm start readout from phase43h_c20.bin; clamp +/-2.0; magic 0x53454539'
Write-Host ''
$TRAIN_OUT = $RDIR + '\phase43c2_train.txt'
& "$BINDIR\phase43c2_scaling.exe" $TS_DATA $WP $C20_W 2>&1 | Tee-Object $TRAIN_OUT
if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }
Write-Host ''
Write-Host 'Training complete.' -ForegroundColor Green

# Parse teacher-forced val BPB per variant from "Saved <path><sfx>  BPB=..."
function Get-TrainBPB([string]$sfx) {
    $line = Select-String ('Saved .*' + [regex]::Escape($sfx) + '\s+BPB=') $TRAIN_OUT | Select-Object -Last 1
    if ($line) { return [double](($line.Line -replace '.*BPB=','').Trim()) }
    return $null
}

# -------- Generation configs --------------------------------------------------
$gen_configs = @(
    @{ label='SEE-V2_13'; weights=$BASE_W;        isbase=$true;  valbpb=$SEE_V2_BPB },
    @{ label='C2.0_13';   weights=$WP+'_C20.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C20.bin') },
    @{ label='C2.A_26';   weights=$WP+'_C2A.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C2A.bin') },
    @{ label='C2.B_26';   weights=$WP+'_C2B.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C2B.bin') },
    @{ label='C2.C_26';   weights=$WP+'_C2C.bin'; isbase=$false; valbpb=(Get-TrainBPB '_C2C.bin') }
)

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

# -------- Generate + score ----------------------------------------------------
Write-Host ''
Write-Host ('=== Generation word-gate: {0} seeds, T={1} ===' -f $NSEEDS, $T) -ForegroundColor Yellow
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
        Write-Host ('  {0,-10} seed{1,-2} (off={2}) ...' -f $cfg.label, $si, $off) -NoNewline
        $cmdLine = "`"$BINDIR\phase43_generator.exe`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $T --seed-start $off > `"$tf`" 2>`"$sf`""
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
Write-Host ('  Phase 43.C2 - Capacity Scaling ({0} seeds, T={1})' -f $NSEEDS, $T)
Write-Host '======================================================================'
Write-Host ('  gate(26c): valBPB<=SEE-V2-{0}  nameWst<={1}  runWst<={2}  topBi<={3}  altLp<={4}  self in [{5},{6}]' -f `
    $GATE_BPB_IMPROVE,$GATE_NAMEISH_WST,$GATE_WORDRUN_WST,$GATE_TOPBI,$GATE_ALTLOOP,$BPB_LO,$BPB_HI)
Write-Host ''
Write-Host ('  {0,-10} {1,9} {2,8} {3,14} {4,8} {5,7} {6,7} {7,9} {8,6}' -f `
    'config','valBPB','dBPB','nameish(av/wst)','runWst','topBi','altLp','selfBPB','gate')
Write-Host ('  ' + '-'*88)

$rows = @{}
foreach ($cfg in $gen_configs) {
    $ms = @($agg[$cfg.label].mets); $bs = @($agg[$cfg.label].bpbs)
    if ($ms.Count -eq 0) { Write-Host ('  {0,-10} [no data]' -f $cfg.label); continue }

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

    # Gate only applies to non-base variants
    $gate = '-'
    if (-not $cfg.isbase) {
        $bpb_ok = ($vb -ne $null) -and ($vb -le ($SEE_V2_BPB - $GATE_BPB_IMPROVE))
        $ni_ok  = $row.ni_wst -le $GATE_NAMEISH_WST
        $mr_ok  = $row.mr_wst -le $GATE_WORDRUN_WST
        $bi_ok  = $row.bi_wst -le $GATE_TOPBI
        $al_ok  = $row.al_wst -le $GATE_ALTLOOP
        $sb_ok  = ($row.selfbpb -ge $BPB_LO) -and ($row.selfbpb -le $BPB_HI)
        $gate = if ($bpb_ok -and $ni_ok -and $mr_ok -and $bi_ok -and $al_ok -and $sb_ok) { 'PASS' } else { 'fail' }
    }

    $vb_str  = if ($vb -ne $null) { '{0:F4}' -f $vb } else { 'N/A' }
    $db_str  = if ($dbpb -ne $null) { '{0:+0.000;-0.000;0.000}' -f $dbpb } else { '-' }

    Write-Host ('  {0,-10} {1,9} {2,8} {3,14} {4,8} {5,7} {6,7} {7,9} {8,6}' -f `
        $cfg.label, $vb_str, $db_str,
        ('{0:F1}/{1:F1}' -f $row.ni_avg, $row.ni_wst),
        ('{0:F0}' -f $row.mr_wst),
        ('{0:F0}' -f $row.bi_wst),
        ('{0:F0}' -f $row.al_wst),
        ('{0:F2}' -f $row.selfbpb),
        $gate)
}

# -------- Top words per seed (eyeball) ----------------------------------------
Write-Host ''
Write-Host '=== Top words - seeds 1..4 (spot the name tunnel) ===' -ForegroundColor Cyan
foreach ($cfg in $gen_configs) {
    Write-Host ('  -- {0}' -f $cfg.label) -ForegroundColor Magenta
    $s = 0
    foreach ($m in @($agg[$cfg.label].mets)) {
        $s++; if ($s -gt 4) { break }
        Write-Host ('     s{0,-2} {1}' -f $s, $m.top_words)
    }
}

# -------- Interpretation ------------------------------------------------------
Write-Host ''
Write-Host '=== Verdetto 43.C2 ===' -ForegroundColor Yellow
if ($rows.ContainsKey('C2.A_26')) {
    # best 26-cell by valBPB
    $best=$null; $bestlbl=''
    foreach ($l in @('C2.A_26','C2.B_26','C2.C_26')) {
        if ($rows.ContainsKey($l) -and $rows[$l].valbpb -ne $null) {
            if ($best -eq $null -or $rows[$l].valbpb -lt $best) { $best=$rows[$l].valbpb; $bestlbl=$l }
        }
    }
    if ($best -ne $null) {
        $bpb_improves = $best -le ($SEE_V2_BPB - $GATE_BPB_IMPROVE)
        $r = $rows[$bestlbl]
        $wordgate_ok = ($r.ni_wst -le $GATE_NAMEISH_WST) -and ($r.mr_wst -le $GATE_WORDRUN_WST) -and `
                       ($r.bi_wst -le $GATE_TOPBI) -and ($r.al_wst -le $GATE_ALTLOOP)
        Write-Host ('  Best 26-cell: {0}  valBPB={1:F4} (d={2:+0.000;-0.000;0.000})  wordgate_ok={3}' -f `
            $bestlbl, $best, ($best-$SEE_V2_BPB), $wordgate_ok)
        Write-Host ''
        if ($bpb_improves -and $wordgate_ok) {
            Write-Host '  -> 26 cells migliora BPB E passa word-gate: il substrate accetta piu' -ForegroundColor Green
            Write-Host '     plasticita. PROMUOVI SEE-V2.1.' -ForegroundColor Green
        } elseif ($bpb_improves -and -not $wordgate_ok) {
            Write-Host '  -> 26 cells migliora BPB MA peggiora word-gate: piu capacita crea' -ForegroundColor Yellow
            Write-Host '     attractor. Serve regolarizzazione / omeostasi plastica.' -ForegroundColor Yellow
        } elseif (-not $bpb_improves -and $best -ge $SEE_V2_BPB) {
            Write-Host '  -> 26 cells NON migliora BPB: 13 celle sono gia lo sweet spot.' -ForegroundColor Yellow
            Write-Host '     Passa a 43.D byte-to-lane routing.' -ForegroundColor Yellow
        } else {
            Write-Host '  -> 26 cells migliora < 0.002 (sotto soglia): 13 probabile sweet spot.' -ForegroundColor Yellow
            Write-Host '     Plasticita locale utile come micro-correzione, non come scala primaria.' -ForegroundColor Yellow
        }
    }
    # Control sanity
    if ($rows.ContainsKey('C2.0_13') -and $rows['C2.0_13'].valbpb -ne $null) {
        $d0 = [math]::Abs($rows['C2.0_13'].valbpb - $SEE_V2_BPB)
        Write-Host ''
        Write-Host ('  Control C2.0 vs SEE-V2: |d|={0:F4} (atteso ~0 = pipeline riproducibile)' -f $d0)
    }
}
Write-Host ''
Write-Host '  Guida: un LLM non e solo predire meglio il prossimo byte;'
Write-Host '         e ricordare di piu senza cadere nei propri attractor.'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
