# Phase 43 Sampling Audit
# Domanda: ms_f05 a T basse / top-k batte legacy come campione narrativo?
#
# Matrix:
#   legacy (phase42a_sum.bin):  T=0.55, T=0.65  (no top-k, e' il baseline)
#   ms_f05 (phase43a2_f05.bin): T=0.45, T=0.55, T=0.65  (plain + top-k=20 + top-k=40)
#
# Run:  .\benchmarks\phase38-42\phase43_sampling_audit.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\sampling_audit'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
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
$W_LEGACY = $WDIR + '\phase42a_sum.bin'     # 0x53454531, Phase 42 baseline, 2.3197 BPB
$W_MS05   = $WDIR + '\phase43a2_f05.bin'    # 0x53454535, multiscale f0.5,   2.2757 BPB

foreach ($w in @($W_LEGACY, $W_MS05)) {
    if (-not (Test-Path $w)) { Write-Error "Weights not found: $w"; exit 1 }
}

# ---- Run matrix --------------------------------------------------------------
$GEN_LEN = 2000
$WARMUP  = 5000

# Each entry: label, weights-path, temp, top-k (0=off)
$runs = @(
    @{ label='legacy_t055';         weights=$W_LEGACY; temp='0.55'; topk=0 },
    @{ label='legacy_t065';         weights=$W_LEGACY; temp='0.65'; topk=0 },
    @{ label='ms_f05_t045';         weights=$W_MS05;   temp='0.45'; topk=0 },
    @{ label='ms_f05_t055';         weights=$W_MS05;   temp='0.55'; topk=0 },
    @{ label='ms_f05_t065';         weights=$W_MS05;   temp='0.65'; topk=0 },
    @{ label='ms_f05_t055_k20';     weights=$W_MS05;   temp='0.55'; topk=20 },
    @{ label='ms_f05_t055_k40';     weights=$W_MS05;   temp='0.55'; topk=40 },
    @{ label='ms_f05_t065_k20';     weights=$W_MS05;   temp='0.65'; topk=20 },
    @{ label='ms_f05_t065_k40';     weights=$W_MS05;   temp='0.65'; topk=40 }
)

Write-Host ''
Write-Host '=== Sampling Audit: legacy vs ms_f05 at lower T + top-k ===' -ForegroundColor Yellow
Write-Host "  legacy: T=0.55, 0.65 (no top-k)"
Write-Host "  ms_f05: T=0.45/0.55/0.65 plain + T=0.55/0.65 with top-k=20/40"
Write-Host "  Warmup: $WARMUP bytes | Gen: $GEN_LEN bytes each"
Write-Host ''

$self_bpb = @{}

foreach ($r in $runs) {
    $label      = $r.label
    $text_file  = $RDIR + '\' + $label + '.txt'
    $stats_file = $RDIR + '\' + $label + '_stats.txt'

    Write-Host ("  $label ...") -NoNewline

    $topkFlag = if ($r.topk -gt 0) { "--top-k $($r.topk)" } else { '' }
    $cmdLine = "`"$GEN_EXE`" `"$TS_DATA`" `"$($r.weights)`" --gen-len $GEN_LEN --temp $($r.temp) --warmup $WARMUP $topkFlag > `"$text_file`" 2>`"$stats_file`""
    cmd /c $cmdLine

    if ($LASTEXITCODE -ne 0) {
        Write-Host ' FAILED' -ForegroundColor Red
        $self_bpb[$label] = 'FAIL'
    } else {
        Write-Host ' OK' -ForegroundColor Green
        $statLine = Select-String 'self_BPB' $stats_file | Select-Object -First 1
        $self_bpb[$label] = if ($statLine) { $statLine.Line.Trim() } else { 'N/A' }
    }
}

# ---- Summary table -----------------------------------------------------------
Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  Sampling Audit — self_BPB summary'
Write-Host '============================================================'
Write-Host ('  {0,-25} {1,-6} {2,-6} {3}' -f 'config','temp','top-k','self_BPB')
Write-Host ('  ' + '-'*55)
foreach ($r in $runs) {
    $topkStr = if ($r.topk -gt 0) { $r.topk } else { 'off' }
    Write-Host ('  {0,-25} {1,-6} {2,-6} {3}' -f $r.label, $r.temp, $topkStr, $self_bpb[$r.label])
}

# ---- Text previews T=0.55 (chiave per confronto) ----------------------------
Write-Host ''
Write-Host '=== Text preview T=0.55 (first 600 chars) ===' -ForegroundColor Yellow
$preview_labels = @('legacy_t055','ms_f05_t055','ms_f05_t055_k20','ms_f05_t055_k40')
foreach ($lbl in $preview_labels) {
    $out_file = $RDIR + '\' + $lbl + '.txt'
    Write-Host ''
    Write-Host ("--- $lbl ---") -ForegroundColor Magenta
    if (Test-Path $out_file) {
        $bytes = [System.IO.File]::ReadAllBytes($out_file)
        $n = [Math]::Min(600, $bytes.Length)
        Write-Host ([System.Text.Encoding]::ASCII.GetString($bytes[0..($n-1)]))
    } else {
        Write-Host '  [not found]' -ForegroundColor Red
    }
}

# ---- Text previews T=0.65 ---------------------------------------------------
Write-Host ''
Write-Host '=== Text preview T=0.65 (first 600 chars) ===' -ForegroundColor Yellow
$preview_labels65 = @('legacy_t065','ms_f05_t065','ms_f05_t065_k20','ms_f05_t065_k40')
foreach ($lbl in $preview_labels65) {
    $out_file = $RDIR + '\' + $lbl + '.txt'
    Write-Host ''
    Write-Host ("--- $lbl ---") -ForegroundColor Magenta
    if (Test-Path $out_file) {
        $bytes = [System.IO.File]::ReadAllBytes($out_file)
        $n = [Math]::Min(600, $bytes.Length)
        Write-Host ([System.Text.Encoding]::ASCII.GetString($bytes[0..($n-1)]))
    } else {
        Write-Host '  [not found]' -ForegroundColor Red
    }
}

Write-Host ''
Write-Host '=== Checklist ===' -ForegroundColor Yellow
Write-Host '  [ ] ms_f05 T=0.55 vs legacy T=0.55: piu narrativo o piu attractor?'
Write-Host '  [ ] top-k=20/40 riduce fixation su Lily/Tommy?'
Write-Host '  [ ] self_BPB scende con top-k (atteso) senza collasso a argmax?'
Write-Host '  [ ] ms_f05 T=0.45: troppo deterministico o narrativamente valido?'
Write-Host ''
Write-Host '=== Decisione attesa ==='
Write-Host '  ms_f05 batte visivamente -> SEE-V1 promosso su BPB + generation -> 43.C Oja'
Write-Host '  ms_f05 ha attractor -> SEE-V1 promosso su BPB, audit generativo segnala limite'
Write-Host ''
Write-Host ('  Files: ' + $RDIR)
