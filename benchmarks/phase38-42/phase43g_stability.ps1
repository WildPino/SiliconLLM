# Phase 43.G — Closed-Loop Stability Tribunal
#
# Domanda: quando collassa ms_f05? E anti-attractor sampling aiuta?
#
# Metriche per ogni run:
#   distinct_bytes, top1_share, top5_share, longest_run,
#   lily_count, mommy_count, sentence_count, printable_ratio
#   + BPB curve per 100 byte (dal generator stderr)
#
# Run:  .\benchmarks\phase38-42\phase43g_stability.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\stability'
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'
$GEN_EXE = $BINDIR + '\phase43_generator.exe'

Set-Location $ROOT

if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

# ---- Compile ----------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING phase43_generator ===' -ForegroundColor Cyan
$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE `
    -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed'; exit 1 }
Write-Host 'OK' -ForegroundColor Green

# ---- Weights -----------------------------------------------------------------
$W_LEGACY = $WDIR + '\phase42a_sum.bin'     # Phase 42 baseline, 2.3197 BPB
$W_MS05   = $WDIR + '\phase43a2_f05.bin'    # SEE-V1 ms_f0.5, 2.2757 BPB

foreach ($w in @($W_LEGACY, $W_MS05)) {
    if (-not (Test-Path $w)) { Write-Error "Weights not found: $w"; exit 1 }
}

# ---- Run matrix: T=0.65 (where ms_f05 showed attractor at T=0.65) -----------
# Baseline: T=0.55 (last stable point for ms_f05)
# Anti-attractor: rep-penalty, entropy-boost, combo
$GEN_LEN = 2000
$WARMUP  = 5000
$T = '0.65'

$runs = @(
    # --- Baselines ---
    @{ label='legacy_t065_plain';    weights=$W_LEGACY; extra='' },
    @{ label='ms_f05_t065_plain';    weights=$W_MS05;   extra='' },
    @{ label='ms_f05_t055_plain';    weights=$W_MS05;   extra='--temp 0.55' },

    # --- Rep penalty only ---
    @{ label='ms_f05_t065_rp05';     weights=$W_MS05;   extra='--rep-penalty 0.5  --rep-window 30' },
    @{ label='ms_f05_t065_rp10';     weights=$W_MS05;   extra='--rep-penalty 1.0  --rep-window 30' },
    @{ label='ms_f05_t065_rp05_w50'; weights=$W_MS05;   extra='--rep-penalty 0.5  --rep-window 50' },

    # --- Entropy boost only ---
    @{ label='ms_f05_t065_eb15';     weights=$W_MS05;   extra='--entropy-boost 1.5 --entropy-thresh 3.0' },
    @{ label='ms_f05_t065_eb20';     weights=$W_MS05;   extra='--entropy-boost 2.0 --entropy-thresh 3.5' },

    # --- Combo: rep + entropy ---
    @{ label='ms_f05_t065_rp05_eb15';weights=$W_MS05;   extra='--rep-penalty 0.5  --rep-window 30 --entropy-boost 1.5 --entropy-thresh 3.0' },

    # --- top-k as reference ---
    @{ label='ms_f05_t065_k20';      weights=$W_MS05;   extra='--top-k 20' }
)

Write-Host ''
Write-Host '=== Phase 43.G — Closed-Loop Stability Tribunal ===' -ForegroundColor Yellow
Write-Host "  Base temp: T=$T  |  Warmup: $WARMUP bytes  |  Gen: $GEN_LEN bytes"
Write-Host '  Anti-attractor: rep-penalty, entropy-boost, combo'
Write-Host ''

# Helper: compute static collapse metrics on a binary file
function Get-CollapseMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $n = $bytes.Length
    if ($n -eq 0) { return $null }

    # distinct bytes
    $freq = @{}
    foreach ($b in $bytes) { $freq[$b] = ($freq[$b] -as [int]) + 1 }
    $distinct = $freq.Count

    # top-1, top-5 share
    $sorted_freq = $freq.Values | Sort-Object -Descending
    $top1  = [math]::Round($sorted_freq[0] / $n * 100, 1)
    $top5f = ($sorted_freq | Select-Object -First 5 | Measure-Object -Sum).Sum
    $top5  = [math]::Round($top5f / $n * 100, 1)

    # longest run of same byte
    $longest = 1; $cur = 1
    for ($i = 1; $i -lt $n; $i++) {
        if ($bytes[$i] -eq $bytes[$i-1]) { $cur++ } else { $cur = 1 }
        if ($cur -gt $longest) { $longest = $cur }
    }

    # ASCII printable ratio (32-126)
    $printable = ($bytes | Where-Object { $_ -ge 32 -and $_ -le 126 }).Count
    $print_pct = [math]::Round($printable / $n * 100, 1)

    # Text-level: decode as ASCII (replace non-printable) for string searches
    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^\x20-\x7E\r\n]', '?'

    # Name counts (case-insensitive)
    $lily  = ([regex]::Matches($text, '(?i)lily')).Count
    $mommy = ([regex]::Matches($text, '(?i)mommy')).Count

    # Sentence boundary count (approx)
    $sentences = ([regex]::Matches($text, '[.!?]')).Count

    return [PSCustomObject]@{
        distinct   = $distinct
        top1       = $top1
        top5       = $top5
        longest_run= $longest
        printable  = $print_pct
        lily       = $lily
        mommy      = $mommy
        sentences  = $sentences
    }
}

# Run all configs
$results = @{}
$metrics = @{}

foreach ($r in $runs) {
    $label      = $r.label
    $text_file  = $RDIR + '\' + $label + '.txt'
    $stats_file = $RDIR + '\' + $label + '_stats.txt'

    Write-Host ("  $label ...") -NoNewline

    $tempFlag = if ($r.extra -notmatch '--temp') { "--temp $T" } else { '' }
    $cmdLine = "`"$GEN_EXE`" `"$TS_DATA`" `"$($r.weights)`" --gen-len $GEN_LEN --warmup $WARMUP $tempFlag $($r.extra) > `"$text_file`" 2>`"$stats_file`""
    cmd /c $cmdLine

    if ($LASTEXITCODE -ne 0) {
        Write-Host ' FAILED' -ForegroundColor Red
        $results[$label] = 'FAIL'
    } else {
        Write-Host ' OK' -ForegroundColor Green
        $statLine = Select-String 'self_BPB' $stats_file | Select-Object -First 1
        $results[$label] = if ($statLine) { $statLine.Line.Trim() } else { 'N/A' }
        $metrics[$label] = Get-CollapseMetrics $text_file
    }
}

# ---- BPB curve for key configs -----------------------------------------------
Write-Host ''
Write-Host '=== BPB Curve (per 100 bytes) ===' -ForegroundColor Cyan
$curve_labels = @('legacy_t065_plain','ms_f05_t065_plain','ms_f05_t065_rp05','ms_f05_t065_eb15','ms_f05_t065_rp05_eb15')
foreach ($lbl in $curve_labels) {
    $sf = $RDIR + '\' + $lbl + '_stats.txt'
    if (-not (Test-Path $sf)) { continue }
    $curve = Select-String 'bpb_at_' $sf | ForEach-Object { $_.Line.Trim() } | Select-Object -Last 10
    if ($curve) {
        Write-Host ("  -- $lbl") -ForegroundColor Magenta
        $curve | ForEach-Object { Write-Host "     $_" }
    }
}

# ---- Summary table -----------------------------------------------------------
Write-Host ''
Write-Host '============================================================================' -ForegroundColor Cyan
Write-Host '  Stability Tribunal — Collapse Metrics'
Write-Host '============================================================================'
Write-Host ('  {0,-30} {1,5} {2,5} {3,5} {4,7} {5,5} {6,5} {7,5} {8,6}' -f `
    'config','dist','top1%','top5%','longest','print%','lily','mommy','sentenc')
Write-Host ('  ' + '-'*80)

foreach ($r in $runs) {
    $lbl = $r.label
    $m   = $metrics[$lbl]
    if ($m) {
        Write-Host ('  {0,-30} {1,5} {2,5} {3,5} {4,7} {5,5} {6,5} {7,5} {8,6}' -f `
            $lbl, $m.distinct, $m.top1, $m.top5, $m.longest_run, $m.printable, $m.lily, $m.mommy, $m.sentences)
    } else {
        Write-Host ('  {0,-30} [no data]' -f $lbl)
    }
}

# ---- self_BPB summary --------------------------------------------------------
Write-Host ''
Write-Host '=== self_BPB summary ===' -ForegroundColor Cyan
foreach ($r in $runs) {
    Write-Host ('  {0,-35} {1}' -f $r.label, $results[$r.label])
}

# ---- Text previews: top candidates ------------------------------------------
Write-Host ''
Write-Host '=== Text preview — best candidates (first 600 chars) ===' -ForegroundColor Yellow
$preview_lbls = @('legacy_t065_plain','ms_f05_t065_plain','ms_f05_t065_rp05','ms_f05_t065_eb15','ms_f05_t065_rp05_eb15')
foreach ($lbl in $preview_lbls) {
    $f = $RDIR + '\' + $lbl + '.txt'
    Write-Host ''
    Write-Host ("--- $lbl ---") -ForegroundColor Magenta
    if (Test-Path $f) {
        $bytes = [System.IO.File]::ReadAllBytes($f)
        $n = [Math]::Min(600, $bytes.Length)
        Write-Host ([System.Text.Encoding]::ASCII.GetString($bytes[0..($n-1)]))
    } else { Write-Host '  [not found]' -ForegroundColor Red }
}

Write-Host ''
Write-Host '=== Decisione ===' -ForegroundColor Yellow
Write-Host '  Anti-attractor aiuta (distinct > legacy, longest_run < baseline):'
Write-Host '    -> decoding fix valido; substrate SEE-V1 ok; vai 43.C Oja'
Write-Host '  Anti-attractor non basta (longest_run >= baseline, lily/mommy ancora dominanti):'
Write-Host '    -> problema e'' nello stato; Oja prima di fix decoding non misurerebbe bene'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
