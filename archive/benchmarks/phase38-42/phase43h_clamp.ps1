# Phase 43.H — Feature Clamp Tribunal
#
# Testa feat_clamp = {2.0, 2.5, 3.0} applicato dopo normalizzazione
# sia in training che in generation.
# Warm start da SEE-V1 (phase43a2_f05.bin).
# Gate: val BPB <= 2.28 + generation collapse metrics
#
# Run:  .\benchmarks\phase38-42\phase43h_clamp.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43h'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT

if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$BASE_W  = $WDIR + '\phase43a2_f05.bin'   # SEE-V1 warm start
$WP43H   = $WDIR + '\phase43h'            # prefix for output weights

if (-not (Test-Path $BASE_W)) { Write-Error "Base weights not found: $BASE_W"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# ---- Compile -----------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan

Write-Host '  phase43h_clamp.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43h_clamp.c' @SRC_CORE `
    -o "$BINDIR\phase43h_clamp.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43h_clamp'; exit 1 }

Write-Host '  phase43_generator.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE `
    -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }

Write-Host 'Compiled OK.' -ForegroundColor Green

# ---- Phase 43.H Training: clamp grid ----------------------------------------
Write-Host ''
Write-Host '=== Phase 43.H — Training: feat_clamp {2.0, 2.5, 3.0} ===' -ForegroundColor Yellow
Write-Host '  Warm start from SEE-V1 (phase43a2_f05.bin)'
Write-Host '  Clamp applied after normalize, in both train and val features'
Write-Host '  3 LR stages x 3 epochs (warm start -> fewer epochs needed)'
Write-Host ''

$TRAIN_OUT = $RDIR + '\phase43h_train.txt'
& "$BINDIR\phase43h_clamp.exe" $TS_DATA $WP43H $BASE_W 2>&1 |
    Tee-Object $TRAIN_OUT

if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }

Write-Host ''
Write-Host 'Training complete.' -ForegroundColor Green

# Extract BPB results from training output
$bpb_lines = Select-String 'PASS\|FAIL' $TRAIN_OUT | ForEach-Object { '  ' + $_.Line.Trim() }
if ($bpb_lines) {
    Write-Host ''
    Write-Host '  BPB Gate results:' -ForegroundColor Cyan
    $bpb_lines | ForEach-Object { Write-Host $_ }
}

# ---- Generation audit: multi-seed collapse metrics --------------------------
Write-Host ''
Write-Host '=== Generation Audit: collapse metrics per clamp value ===' -ForegroundColor Yellow
Write-Host '  T=0.65 (stress test — where plain ms_f05 showed attractor)'
Write-Host '  4 seeds (fast audit, not full 8-seed)'
Write-Host ''

$GEN_EXE = $BINDIR + '\phase43_generator.exe'
$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'

# 4 evenly spaced seeds
$SEEDS = @(0, 16777216, 33554432, 50331648)

# Configs to test: baseline (no new clamp, uses fc25 from generation flag) + each 43H weight
$gen_configs = @(
    @{ label='baseline_t065';  weights=$BASE_W;             extra='' },
    @{ label='h_c20_t065';     weights=$WDIR+'\phase43h_c20.bin'; extra='' },  # feat_clamp auto from header
    @{ label='h_c25_t065';     weights=$WDIR+'\phase43h_c25.bin'; extra='' },
    @{ label='h_c30_t065';     weights=$WDIR+'\phase43h_c30.bin'; extra='' }
)

function Get-Metrics([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $n = $bytes.Length; if ($n -eq 0) { return $null }

    $freq = @{}
    foreach ($b in $bytes) { $freq[$b] = ($freq[$b] -as [int]) + 1 }
    $distinct = $freq.Count
    $sv = $freq.Values | Sort-Object -Descending
    $top1 = [math]::Round($sv[0] / $n * 100.0, 1)
    $top5 = [math]::Round((($sv | Select-Object -First 5 | Measure-Object -Sum).Sum) / $n * 100.0, 1)

    $longest = 1; $cur = 1
    for ($j = 1; $j -lt $n; $j++) {
        if ($bytes[$j] -eq $bytes[$j-1]) { $cur++ } else { $cur = 1 }
        if ($cur -gt $longest) { $longest = $cur }
    }

    $pr = ($bytes | Where-Object { $_ -ge 32 -and $_ -le 126 }).Count
    $print_pct = [math]::Round($pr / $n * 100.0, 1)

    $text = [System.Text.Encoding]::ASCII.GetString($bytes) -replace '[^a-zA-Z ]', ' '
    $wfreq = @{}
    ($text -split '\s+') | Where-Object { $_.Length -ge 3 } | ForEach-Object {
        $wl = $_.ToLower(); $wfreq[$wl] = ($wfreq[$wl] -as [int]) + 1
    }
    $top_words = $wfreq.GetEnumerator() |
        Sort-Object Value -Descending | Select-Object -First 8 |
        ForEach-Object { "$($_.Key)=$($_.Value)" }

    $sentences = ([regex]::Matches($text, '[.!?]')).Count

    return [PSCustomObject]@{
        distinct=$distinct; top1=$top1; top5=$top5; longest=$longest
        printable=$print_pct; sentences=$sentences; top_words=($top_words -join ' ')
    }
}

# Aggregate across seeds
$agg = @{}
foreach ($cfg in $gen_configs) { $agg[$cfg.label] = @{ mets=@(); bpbs=@() } }

$si = 0
foreach ($off in $SEEDS) {
    $si++
    foreach ($cfg in $gen_configs) {
        if (-not (Test-Path $cfg.weights)) { continue }
        $lbl = $cfg.label + '_s' + $si
        $tf  = $RDIR + '\gen_' + $lbl + '.txt'
        $sf  = $RDIR + '\gen_' + $lbl + '_stats.txt'

        Write-Host ("  $($cfg.label) seed$si ...") -NoNewline
        $cmdLine = "`"$GEN_EXE`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --warmup $WARMUP --temp $T --seed-start $off $($cfg.extra) > `"$tf`" 2>`"$sf`""
        cmd /c $cmdLine

        if ($LASTEXITCODE -ne 0) {
            Write-Host ' FAILED' -ForegroundColor Red
        } else {
            Write-Host ' OK' -ForegroundColor Green
            $m = Get-Metrics $tf
            if ($m) { $agg[$cfg.label].mets += $m }
            $bl = Select-String 'self_BPB' $sf | Select-Object -Last 1
            if ($bl) {
                $v = [double]($bl.Line.Trim() -replace 'self_BPB:\s*','')
                $agg[$cfg.label].bpbs += $v
            }
        }
    }
}

function Avg([double[]]$a) { if ($a.Count -eq 0) { return 0.0 }; [math]::Round(($a|Measure-Object -Average).Average,2) }
function Std([double[]]$a) {
    if ($a.Count -lt 2) { return 0.0 }
    $m=($a|Measure-Object -Average).Average
    $v=($a|ForEach-Object{($_-$m)*($_-$m)}|Measure-Object -Sum).Sum/($a.Count-1)
    [math]::Round([math]::Sqrt($v),2)
}

# ---- Summary -----------------------------------------------------------------
Write-Host ''
Write-Host '======================================================================' -ForegroundColor Cyan
Write-Host '  Phase 43.H — Collapse Metrics (mean over 4 seeds, T=0.65)'
Write-Host '======================================================================'
Write-Host '  Gate: longest <= 10, top1% <= 26%, printable >= 96%, self_BPB > 0.8'
Write-Host ''
Write-Host ('  {0,-22} {1,6} {2,6} {3,6} {4,8} {5,7} {6,7} {7,8}' -f `
    'config','dist','top1%','top5%','longest','print%','sents','self_BPB')
Write-Host ('  ' + '-'*72)

foreach ($cfg in $gen_configs) {
    $ms = $agg[$cfg.label].mets
    $bs = $agg[$cfg.label].bpbs
    if ($ms.Count -eq 0) { Write-Host ('  {0,-22} [no data]' -f $cfg.label); continue }

    $d_a  = $ms | ForEach-Object { [double]$_.distinct }
    $t1_a = $ms | ForEach-Object { [double]$_.top1 }
    $t5_a = $ms | ForEach-Object { [double]$_.top5 }
    $lr_a = $ms | ForEach-Object { [double]$_.longest }
    $pr_a = $ms | ForEach-Object { [double]$_.printable }
    $se_a = $ms | ForEach-Object { [double]$_.sentences }

    $bpb_str = if ($bs.Count -gt 0) { '{0:F4}±{1:F4}' -f (Avg $bs),(Std $bs) } else { 'N/A' }
    $lr_str  = '{0:F0}±{1:F0}' -f (Avg $lr_a),(Std $lr_a)

    $longest_ok = (Avg $lr_a) -le 10
    $top1_ok    = (Avg $t1_a) -le 26
    $print_ok   = (Avg $pr_a) -ge 96
    $bpb_ok     = ($bs.Count -gt 0) -and ((Avg $bs) -gt 0.8)
    $gate_str   = if ($longest_ok -and $top1_ok -and $print_ok -and $bpb_ok) { 'PASS' } else { 'fail' }

    Write-Host ('  {0,-22} {1,6} {2,6} {3,6} {4,8} {5,7} {6,7} {7,8}  [{8}]' -f `
        $cfg.label,
        ('{0:F0}' -f (Avg $d_a)),
        ('{0:F1}' -f (Avg $t1_a)),
        ('{0:F1}' -f (Avg $t5_a)),
        $lr_str,
        ('{0:F1}' -f (Avg $pr_a)),
        ('{0:F0}' -f (Avg $se_a)),
        $bpb_str,
        $gate_str)
}

# Top words
Write-Host ''
Write-Host '=== Top Words — Seed 1 ===' -ForegroundColor Cyan
foreach ($cfg in $gen_configs) {
    $tf = $RDIR + '\gen_' + $cfg.label + '_s1.txt'
    $m  = Get-Metrics $tf
    if ($m) { Write-Host ('  {0,-22} {1}' -f $cfg.label, $m.top_words) }
}

# BPB curve (seed 1)
Write-Host ''
Write-Host '=== BPB Curve Seed 1 ===' -ForegroundColor Cyan
foreach ($cfg in $gen_configs) {
    $sf = $RDIR + '\gen_' + $cfg.label + '_s1_stats.txt'
    if (-not (Test-Path $sf)) { continue }
    $curve = Select-String 'bpb_at_' $sf | ForEach-Object { $_.Line.Trim() }
    if ($curve) {
        Write-Host ("  -- $($cfg.label)") -ForegroundColor Magenta
        $curve | ForEach-Object { Write-Host "     $_" }
    }
}

# Text preview (seed 1)
Write-Host ''
Write-Host '=== Text Preview Seed 1 (500 chars) ===' -ForegroundColor Yellow
foreach ($cfg in $gen_configs) {
    $tf = $RDIR + '\gen_' + $cfg.label + '_s1.txt'
    Write-Host ''
    Write-Host ("--- $($cfg.label) ---") -ForegroundColor Magenta
    if (Test-Path $tf) {
        $bytes = [System.IO.File]::ReadAllBytes($tf)
        $n = [Math]::Min(500, $bytes.Length)
        Write-Host ([System.Text.Encoding]::ASCII.GetString($bytes[0..($n-1)]))
    }
}

Write-Host ''
Write-Host '=== Verdetto 43.H ===' -ForegroundColor Yellow
Write-Host '  Se h_c25/h_c30 PASS tutti i gate:'
Write-Host '    -> SEE-V1S = SEE-V1 + feat_clamp (promuovi, poi 43.C Oja)'
Write-Host '  Se solo BPB regge ma generazione ancora instabile:'
Write-Host '    -> clamp stabilizza manifold ma non abbastanza; indaga readout'
Write-Host '  Se BPB peggiora oltre 2.28:'
Write-Host '    -> clamp troppo aggressivo, prova clamp piu largo (3.5, 4.0)'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
