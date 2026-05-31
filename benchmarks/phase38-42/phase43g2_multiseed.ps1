# Phase 43.G2 — Closed-Loop Stability Tribunal (Multi-Seed)
#
# Domanda: state attractor vs readout attractor?
#   Se reset/clamp riduce Lily/Mommy -> state attractor
#   Se non cambia                    -> readout/distribuzione
#
# Configs:
#   legacy T=0.55/0.65      (baseline)
#   ms_f05 T=0.55/0.65      (SEE-V1 plain)
#   ms_f05 T=0.65 + reset-interval 256
#   ms_f05 T=0.65 + reset-interval 512
#   ms_f05 T=0.65 + feat-clamp 2.5
#
# Seeds: 8 offsets evenly spaced in TinyStories 64MB
#
# Run:  .\benchmarks\phase38-42\phase43g2_multiseed.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\g2_multiseed'
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

# ---- Weights ----------------------------------------------------------------
$W_LEGACY = $WDIR + '\phase42a_sum.bin'
$W_MS05   = $WDIR + '\phase43a2_f05.bin'
foreach ($w in @($W_LEGACY, $W_MS05)) {
    if (-not (Test-Path $w)) { Write-Error "Weights not found: $w"; exit 1 }
}

# ---- Seeds: 8 evenly spaced offsets in 64MB file ----------------------------
# Each seed_start needs room for warmup(5000) + seed(512)
# Max safe: ~67M - 6000 ~= 67102864
$SEED_OFFSETS = @(0, 8388608, 16777216, 25165824, 33554432, 41943040, 50331648, 58720256)

# ---- Configs ----------------------------------------------------------------
$GEN_LEN = 2000
$WARMUP  = 5000

$configs = @(
    @{ name='legacy_t055';      weights=$W_LEGACY; temp='0.55'; extra='' },
    @{ name='legacy_t065';      weights=$W_LEGACY; temp='0.65'; extra='' },
    @{ name='ms_f05_t055';      weights=$W_MS05;   temp='0.55'; extra='' },
    @{ name='ms_f05_t065';      weights=$W_MS05;   temp='0.65'; extra='' },
    @{ name='ms_f05_t065_rst256'; weights=$W_MS05; temp='0.65'; extra='--reset-interval 256' },
    @{ name='ms_f05_t065_rst512'; weights=$W_MS05; temp='0.65'; extra='--reset-interval 512' },
    @{ name='ms_f05_t065_fc25';   weights=$W_MS05; temp='0.65'; extra='--feat-clamp 2.5' }
)

# ---- Metrics helpers --------------------------------------------------------
function Get-StaticMetrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $n = $bytes.Length; if ($n -eq 0) { return $null }

    # byte frequency
    $freq = @{}
    foreach ($b in $bytes) { $freq[$b] = ($freq[$b] -as [int]) + 1 }
    $distinct = $freq.Count
    $sorted_v = $freq.Values | Sort-Object -Descending
    $top1 = [math]::Round($sorted_v[0] / $n * 100.0, 1)
    $top5sum = ($sorted_v | Select-Object -First 5 | Measure-Object -Sum).Sum
    $top5 = [math]::Round($top5sum / $n * 100.0, 1)

    # longest same-byte run
    $longest = 1; $cur = 1
    for ($j = 1; $j -lt $n; $j++) {
        if ($bytes[$j] -eq $bytes[$j-1]) { $cur++ } else { $cur = 1 }
        if ($cur -gt $longest) { $longest = $cur }
    }

    # printable ratio
    $pr = ($bytes | Where-Object { $_ -ge 32 -and $_ -le 126 }).Count
    $print_pct = [math]::Round($pr / $n * 100.0, 1)

    # text-level: decode as ASCII
    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^\x20-\x7E\r\n]', ' '

    # sentence count
    $sentences = ([regex]::Matches($text, '[.!?]')).Count

    # word frequency (words >= 3 chars)
    $words = ($text -split '[^a-zA-Z]+') | Where-Object { $_.Length -ge 3 }
    $wfreq = @{}
    foreach ($w in $words) {
        $wl = $w.ToLower()
        $wfreq[$wl] = ($wfreq[$wl] -as [int]) + 1
    }
    $top_words = $wfreq.GetEnumerator() |
        Sort-Object Value -Descending |
        Select-Object -First 10 |
        ForEach-Object { "$($_.Key)=$($_.Value)" }

    return [PSCustomObject]@{
        distinct   = $distinct
        top1       = $top1
        top5       = $top5
        longest    = $longest
        printable  = $print_pct
        sentences  = $sentences
        top_words  = ($top_words -join ' ')
    }
}

function Get-FinalBPB([string]$stats_path) {
    if (-not (Test-Path $stats_path)) { return 'N/A' }
    $line = Select-String 'self_BPB' $stats_path | Select-Object -Last 1
    if ($line) { return $line.Line.Trim() } else { return 'N/A' }
}

function Get-BPBCurve([string]$stats_path) {
    if (-not (Test-Path $stats_path)) { return @() }
    return (Select-String 'bpb_at_' $stats_path | ForEach-Object { $_.Line.Trim() })
}

# ---- Run all (config x seed) ------------------------------------------------
Write-Host ''
Write-Host '=== Phase 43.G2 — Multi-Seed Stability Tribunal ===' -ForegroundColor Yellow
Write-Host ("  Configs: " + $configs.Count + "  x  Seeds: " + $SEED_OFFSETS.Count + "  =  " + ($configs.Count * $SEED_OFFSETS.Count) + " runs")
Write-Host "  Gen: $GEN_LEN bytes | Warmup: $WARMUP bytes"
Write-Host ''

# Store per-config aggregate data
$agg = @{}
foreach ($cfg in $configs) { $agg[$cfg.name] = @{ metrics=@(); bpbs=@(); bpb_curves=@{} } }

$seed_idx = 0
foreach ($offset in $SEED_OFFSETS) {
    $seed_idx++
    Write-Host ("  Seed $seed_idx/$($SEED_OFFSETS.Count) (offset=$offset) ...") -ForegroundColor Cyan

    foreach ($cfg in $configs) {
        $lbl  = $cfg.name + '_s' + $seed_idx
        $tf   = $RDIR + '\' + $lbl + '.txt'
        $sf   = $RDIR + '\' + $lbl + '_stats.txt'

        Write-Host ("    $($cfg.name) ...") -NoNewline

        $cmdLine = "`"$GEN_EXE`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $($cfg.temp) --seed-start $offset $($cfg.extra) > `"$tf`" 2>`"$sf`""
        cmd /c $cmdLine

        if ($LASTEXITCODE -ne 0) {
            Write-Host ' FAILED' -ForegroundColor Red
        } else {
            Write-Host ' OK' -ForegroundColor Green
            $m = Get-StaticMetrics $tf
            if ($m) { $agg[$cfg.name].metrics += $m }
            $bpb_line = Get-FinalBPB $sf
            if ($bpb_line -ne 'N/A') {
                $val = [double]($bpb_line -replace 'self_BPB:\s*', '')
                $agg[$cfg.name].bpbs += $val
            }
            # Save BPB curve for seed 1 (representative)
            if ($seed_idx -eq 1) {
                $agg[$cfg.name].bpb_curves['s1'] = Get-BPBCurve $sf
            }
        }
    }
}

# ---- Aggregate metrics -------------------------------------------------------
function Avg([double[]]$arr) {
    if ($arr.Count -eq 0) { return 0.0 }
    return [math]::Round(($arr | Measure-Object -Average).Average, 2)
}
function Std([double[]]$arr) {
    if ($arr.Count -lt 2) { return 0.0 }
    $m = ($arr | Measure-Object -Average).Average
    $v = ($arr | ForEach-Object { ($_ - $m)*($_ - $m) } | Measure-Object -Sum).Sum / ($arr.Count-1)
    return [math]::Round([math]::Sqrt($v), 2)
}

# ---- Summary: collapse metrics table ----------------------------------------
Write-Host ''
Write-Host '=============================================================================' -ForegroundColor Cyan
Write-Host '  43.G2 — Stability Metrics (mean over 8 seeds)'
Write-Host '============================================================================='
Write-Host ('  {0,-28} {1,6} {2,6} {3,6} {4,8} {5,7} {6,7} {7,8}' -f `
    'config','dist','top1%','top5%','longest','print%','sents','BPB')
Write-Host ('  ' + '-'*80)

foreach ($cfg in $configs) {
    $ms = $agg[$cfg.name].metrics
    $bs = $agg[$cfg.name].bpbs
    if ($ms.Count -eq 0) {
        Write-Host ('  {0,-28} [no data]' -f $cfg.name)
        continue
    }
    $dist_arr    = $ms | ForEach-Object { [double]$_.distinct }
    $top1_arr    = $ms | ForEach-Object { [double]$_.top1 }
    $top5_arr    = $ms | ForEach-Object { [double]$_.top5 }
    $longest_arr = $ms | ForEach-Object { [double]$_.longest }
    $print_arr   = $ms | ForEach-Object { [double]$_.printable }
    $sent_arr    = $ms | ForEach-Object { [double]$_.sentences }

    $bpb_str = if ($bs.Count -gt 0) { '{0:F4}±{1:F4}' -f (Avg $bs), (Std $bs) } else { 'N/A' }

    Write-Host ('  {0,-28} {1,6} {2,6} {3,6} {4,8} {5,7} {6,7} {7,8}' -f `
        $cfg.name,
        ('{0:F0}±{1:F0}' -f (Avg $dist_arr),(Std $dist_arr)),
        ('{0:F1}' -f (Avg $top1_arr)),
        ('{0:F1}' -f (Avg $top5_arr)),
        ('{0:F0}±{1:F0}' -f (Avg $longest_arr),(Std $longest_arr)),
        ('{0:F1}' -f (Avg $print_arr)),
        ('{0:F0}' -f (Avg $sent_arr)),
        $bpb_str)
}

# ---- BPB curves (seed 1 representative) -------------------------------------
Write-Host ''
Write-Host '=== BPB Curve — Seed 1 (shows when collapse starts) ===' -ForegroundColor Cyan
$curve_cfgs = @('legacy_t065','ms_f05_t065','ms_f05_t065_rst256','ms_f05_t065_rst512','ms_f05_t065_fc25')
foreach ($cname in $curve_cfgs) {
    $curve = $agg[$cname].bpb_curves['s1']
    if ($curve -and $curve.Count -gt 0) {
        Write-Host ("  -- $cname") -ForegroundColor Magenta
        $curve | ForEach-Object { Write-Host "     $_" }
    }
}

# ---- Top words per config (seed 1) ------------------------------------------
Write-Host ''
Write-Host '=== Top Words — Seed 1 ===' -ForegroundColor Cyan
foreach ($cfg in $configs) {
    $tf = $RDIR + '\' + $cfg.name + '_s1.txt'
    if (Test-Path $tf) {
        $m = Get-StaticMetrics $tf
        if ($m) {
            Write-Host ('  {0,-28} {1}' -f $cfg.name, $m.top_words)
        }
    }
}

# ---- Text preview (seed 1, first 500 chars) ---------------------------------
Write-Host ''
Write-Host '=== Text Preview — Seed 1 (500 chars) ===' -ForegroundColor Yellow
$preview_cfgs = @('legacy_t065','ms_f05_t065','ms_f05_t065_rst256','ms_f05_t065_fc25')
foreach ($cname in $preview_cfgs) {
    $tf = $RDIR + '\' + $cname + '_s1.txt'
    Write-Host ''
    Write-Host ("--- $cname ---") -ForegroundColor Magenta
    if (Test-Path $tf) {
        $bytes = [System.IO.File]::ReadAllBytes($tf)
        $n = [Math]::Min(500, $bytes.Length)
        Write-Host ([System.Text.Encoding]::ASCII.GetString($bytes[0..($n-1)]))
    } else { Write-Host '  [not found]' -ForegroundColor Red }
}

# ---- Diagnosis ---------------------------------------------------------------
Write-Host ''
Write-Host '=== 43.G2 Diagnosis ===' -ForegroundColor Yellow
Write-Host ''
Write-Host '  Confronta ms_f05_t065_plain vs ms_f05_t065_rst256/rst512:'
Write-Host '    distinct SALE con reset  -> stato accumula attractor (state problem)'
Write-Host '    distinct UGUALE          -> il problema e'' nel readout (W/B/trigram)'
Write-Host ''
Write-Host '  Confronta ms_f05_t065_plain vs ms_f05_t065_fc25:'
Write-Host '    top1% SCENDE con clamp   -> features escono dai range di training (drift)'
Write-Host '    top1% UGUALE             -> le features sono ok, il readout pesa male'
Write-Host ''
Write-Host '  Dopo questo audit: 43.C Oja con collapse metrics come gate, non solo BPB'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
