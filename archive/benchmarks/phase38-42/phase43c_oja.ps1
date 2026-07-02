# Phase 43.C — Oja Plasticity Tribunal
#
# Substrate: SEE-V1S (multiscale f0.5/0.9/0.99 + feat_clamp 2.0), warm-started
# from phase43h_c20.bin. Adds Oja plastic cells to the first SEE_N_OJA=13 cells
# of the L1 fast band (unsupervised local PCA, readout W/B stays frozen-then-
# fine-tuned on the Oja-shaped features).
#
# Protocol (phase43c_oja.exe):
#   Pass 1: stream train with eta_oja active -> W_oja converges (unsupervised)
#   Freeze W_oja (eta=0, projection still applied), re-extract train+val
#   Normalize + clamp +/-2.0, warm-start W/B from c20, fine-tune (3 LR x 3 ep)
#   Grid: eta_oja in {1e-3, 1e-4} -> phase43c_eta1e3.bin / phase43c_eta1e4.bin
#
# Generation freezes W_oja (eta=0) so gen features match Pass-2 re-extraction.
#
# Gate (vs SEE-V1S c20 baseline = 2.2656 BPB):
#   val BPB < 2.26  AND  longest <= 10  AND  top1% <= 26%  AND  printable >= 96%
#
# Run:  .\benchmarks\phase38-42\phase43c_oja.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\phase43c'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'

Set-Location $ROOT

if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

$BASE_W  = $WDIR + '\phase43h_c20.bin'   # SEE-V1S c20 warm start
$WP43C   = $WDIR + '\phase43c'           # prefix -> phase43c_eta1e3.bin etc.

if (-not (Test-Path $BASE_W)) { Write-Error "Base weights not found: $BASE_W"; exit 1 }

$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')

# ---- Compile -----------------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan

Write-Host '  phase43c_oja.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43c_oja.c' @SRC_CORE `
    -o "$BINDIR\phase43c_oja.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43c_oja'; exit 1 }

Write-Host '  phase43_generator.exe ...'
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE `
    -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed: phase43_generator'; exit 1 }

Write-Host 'Compiled OK.' -ForegroundColor Green

# ---- Phase 43.C Training: Oja eta grid --------------------------------------
Write-Host ''
Write-Host '=== Phase 43.C — Training: Oja eta {1e-3, 1e-4} ===' -ForegroundColor Yellow
Write-Host '  Substrate SEE-V1S (f0.5, clamp 2.0); warm start phase43h_c20.bin'
Write-Host '  Pass1 Oja adapt -> freeze W_oja -> re-extract -> fine-tune W/B'
Write-Host ''

$TRAIN_OUT = $RDIR + '\phase43c_train.txt'
& "$BINDIR\phase43c_oja.exe" $TS_DATA $WP43C $BASE_W 2>&1 |
    Tee-Object $TRAIN_OUT

if ($LASTEXITCODE -ne 0) { Write-Error 'Training failed'; exit 1 }

Write-Host ''
Write-Host 'Training complete.' -ForegroundColor Green

# Echo the trainer's own tribunal table + signal line
$verdict_lines = Select-String 'Val BPB|eta|SIGNAL|Best' $TRAIN_OUT |
    ForEach-Object { '  ' + $_.Line.Trim() }
if ($verdict_lines) {
    Write-Host ''
    Write-Host '  Trainer BPB tribunal:' -ForegroundColor Cyan
    $verdict_lines | ForEach-Object { Write-Host $_ }
}

# ---- Generation audit: multi-seed collapse metrics --------------------------
Write-Host ''
Write-Host '=== Generation Audit: collapse metrics ===' -ForegroundColor Yellow
Write-Host '  T=0.65 (stress test), 4 seeds, W_oja frozen during generation'
Write-Host ''

$GEN_EXE = $BINDIR + '\phase43_generator.exe'
$GEN_LEN = 2000
$WARMUP  = 5000
$T       = '0.65'

# 4 evenly spaced seeds
$SEEDS = @(0, 16777216, 33554432, 50331648)

# Configs: SEE-V1S c20 baseline + the two Oja variants (feat_clamp auto from header)
$gen_configs = @(
    @{ label='base_c20_t065';  weights=$BASE_W;                    extra='' },
    @{ label='oja_1e3_t065';   weights=$WDIR+'\phase43c_eta1e3.bin'; extra='' },
    @{ label='oja_1e4_t065';   weights=$WDIR+'\phase43c_eta1e4.bin'; extra='' }
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
Write-Host '  Phase 43.C — Collapse Metrics (mean over 4 seeds, T=0.65)'
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

# Top words (seed 1)
Write-Host ''
Write-Host '=== Top Words — Seed 1 ===' -ForegroundColor Cyan
foreach ($cfg in $gen_configs) {
    $tf = $RDIR + '\gen_' + $cfg.label + '_s1.txt'
    $m  = Get-Metrics $tf
    if ($m) { Write-Host ('  {0,-22} {1}' -f $cfg.label, $m.top_words) }
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
Write-Host '=== Verdetto 43.C ===' -ForegroundColor Yellow
Write-Host '  Se un oja_* batte c20 su val BPB (< 2.2656) E passa i gate collasso:'
Write-Host '    -> SEE-V1S+Oja promosso. Il silicio ha imparato direzioni utili.'
Write-Host '  Se BPB neutro/peggiore ma generazione stabile:'
Write-Host '    -> PCA non supervisionata non aiuta il readout in questo regime.'
Write-Host '       Ferma a SEE-V1S; valuta 43.D (byte-to-lane routing).'
Write-Host '  Se generazione collassa (longest/top1 oltre soglia):'
Write-Host '    -> W_oja sposta le feature fuori manifold; eta troppo alto.'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
