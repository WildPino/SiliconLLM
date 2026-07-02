# Phase 43 Generation Audit
# Confronta legacy (Phase 42.0) vs SEE-V1 (ms_f0.5) a T=0.5, 0.7, 0.9
# Stesso seed fisso per tutti i run -> confronto diretto.
#
# Domanda: il guadagno BPB -0.044 si trasforma in linguaggio migliore?
#
# Run:  .\benchmarks\phase38-42\phase43_gen_audit.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\gen_audit'
$TS_DATA = $ROOT + '\data\corpora\tinystories_64mb.txt'
$GEN_EXE = $BINDIR + '\phase43_generator.exe'

Set-Location $ROOT

# Crea output dir
if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }

if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found: $TS_DATA"; exit 1 }

# ---- Compile generator -------------------------------------------------------
Write-Host ''
Write-Host '=== COMPILING phase43_generator ===' -ForegroundColor Cyan
$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE `
    -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed'; exit 1 }
Write-Host 'OK' -ForegroundColor Green

# ---- Weights ----------------------------------------------------------------
$W_LEGACY = $WDIR + '\phase42a_sum.bin'      # format 0x53454531, Phase 42 baseline, BPB 2.3197
$W_MS05   = $WDIR + '\phase43a2_f05.bin'     # format 0x53454535, multiscale f0.5, BPB 2.2757

foreach ($w in @($W_LEGACY, $W_MS05)) {
    if (-not (Test-Path $w)) { Write-Error "Weights not found: $w"; exit 1 }
}

# ---- Run matrix: 2 models x 3 temps -----------------------------------------
# Temperatures as strings with '.' decimal to avoid Italian locale '0,5' issue
# (PS5.1 formats floats with locale separator; cmd passes literal string to atof).
$TEMPS     = @('0.5', '0.7', '0.9')
$GEN_LEN   = 2000
$WARMUP    = 5000   # bytes for online mean/std computation before seed

$configs = @(
    @{ label='legacy'; weights=$W_LEGACY; flags='' },   # 0x53454531: auto-detects legacy
    @{ label='ms_f05';  weights=$W_MS05;   flags='' }
)

Write-Host ''
Write-Host '=== Generation Audit: 2 models x 3 temperatures ===' -ForegroundColor Yellow
Write-Host "  Warmup: $WARMUP bytes (online stats) + 512 bytes seed context"
Write-Host '  Gen:  2000 bytes each'
Write-Host '  Models: legacy/phase42a_sum (2.3197 BPB) vs ms_f0.5 (2.2757 BPB, -0.044)'
Write-Host ''

$self_bpb = @{}

foreach ($cfg in $configs) {
    foreach ($t in $TEMPS) {
        $tlabel   = $t -replace '\.', ''        # '0.7' -> '07'
        $label    = $cfg.label + '_t' + $tlabel
        $text_file  = $RDIR + '\' + $label + '.txt'
        $stats_file = $RDIR + '\' + $label + '_stats.txt'

        Write-Host ("  Running $($cfg.label) T=$t ...") -NoNewline

        # cmd /c: clean binary stdout + text stderr, no PS5.1 ErrorRecord wrapping.
        # Decimal '.' is safe inside cmd /c string — C atof uses C locale, not Italian.
        $flagStr = $cfg.flags
        $cmdLine = "`"$GEN_EXE`" `"$TS_DATA`" `"$($cfg.weights)`" --gen-len $GEN_LEN --temp $t --warmup $WARMUP $flagStr > `"$text_file`" 2>`"$stats_file`""
        cmd /c $cmdLine

        if ($LASTEXITCODE -ne 0) {
            Write-Host ' FAILED' -ForegroundColor Red
        } else {
            Write-Host ' OK' -ForegroundColor Green
            $statLine = Select-String 'self_BPB' $stats_file | Select-Object -First 1
            $self_bpb[$label] = if ($statLine) { $statLine.Line.Trim() } else { 'N/A' }
        }
    }
}

# ---- Print summary table ----------------------------------------------------
Write-Host ''
Write-Host '======================================================' -ForegroundColor Cyan
Write-Host '  Generation Audit — self_BPB summary'
Write-Host '======================================================'
Write-Host '  (low self_BPB = model is internally consistent with its own output)'
Write-Host ''
Write-Host ('  {0,-15} {1,-6} {2}' -f 'config','temp','self_BPB')
Write-Host ('  ' + '-'*45)
foreach ($cfg in $configs) {
    foreach ($t in $TEMPS) {
        $tlabel = $t -replace '\.', ''
        $label  = $cfg.label + '_t' + $tlabel
        Write-Host ('  {0,-15} {1,-6} {2}' -f $label, $t, $self_bpb[$label])
    }
}

Write-Host ''
Write-Host '======================================================' -ForegroundColor Cyan
Write-Host '  Generated text files in: ' + $RDIR
Write-Host ''

# ---- Print first 600 chars of each T=0.7 run --------------------------------
Write-Host '=== Text preview T=0.7 ===' -ForegroundColor Yellow
foreach ($cfg in $configs) {
    $label = $cfg.label + '_t07'    # matches $t='0.7' -> tlabel='07'
    $out_file = $RDIR + '\' + $label + '.txt'
    Write-Host ''
    Write-Host ("--- $($cfg.label) T=0.7 ---") -ForegroundColor Magenta
    if (Test-Path $out_file) {
        $content = Get-Content $out_file -Raw
        if ($content.Length -gt 600) { $content = $content.Substring(0,600) + '...' }
        Write-Host $content
    }
}

Write-Host ''
Write-Host '=== Checklist ===' -ForegroundColor Yellow
Write-Host '  [ ] parole intere vs frammenti'
Write-Host '  [ ] stabilita inglese (niente garbled bytes)'
Write-Host '  [ ] ripetizioni (the the, to to)'
Write-Host '  [ ] punteggiatura corretta'
Write-Host '  [ ] chiusura frasi brevi'
Write-Host '  [ ] ms_f05 visibilmente migliore di legacy?'
Write-Host ''
Write-Host '  Files completi: ' $RDIR
