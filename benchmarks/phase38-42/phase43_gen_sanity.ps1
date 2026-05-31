# Phase 43 Generator Sanity Check
# Verifica che phase43_generator produca output paragonabile a phase41_generator
# sullo stesso file di weights (phase42a_sum.bin, formato 0x53454531).
#
# Se phase41_generator fa pseudo-inglese e phase43_generator fa garbage -> bug nel nuovo gen.
# Se entrambi fanno pseudo-inglese -> nuovo generator e' affidabile per legacy.
#
# Run:  .\benchmarks\phase38-42\phase43_gen_sanity.ps1

$ROOT    = Split-Path (Split-Path $PSScriptRoot)
$BINDIR  = $ROOT + '\bin'
$WDIR    = $ROOT + '\weights'
$RDIR    = $ROOT + '\results\gen_sanity'
$TS_DATA = $ROOT + '\experiments\phase41a\corpora\tinystories_64mb.txt'

Set-Location $ROOT

if (-not (Test-Path $RDIR)) { New-Item -ItemType Directory -Path $RDIR | Out-Null }
if (-not (Test-Path $TS_DATA)) { Write-Error "TinyStories not found"; exit 1 }

# ---- Compile phase43_generator -----------------------------------------------
Write-Host ''
Write-Host '=== COMPILING ===' -ForegroundColor Cyan
$GCC_FLAGS = @('-O3','-march=native','-mavx2','-mfma','-lm','-I','.')
$SRC_CORE  = @('src/silicon_entropy.c','src/silicon_v0.c')
& gcc @GCC_FLAGS 'benchmarks/phase38-42/phase43_generator.c' @SRC_CORE `
    -o "$BINDIR\phase43_generator.exe"
if ($LASTEXITCODE -ne 0) { Write-Error 'Compile failed phase43_generator'; exit 1 }
Write-Host 'OK' -ForegroundColor Green

# ---- Weights ----------------------------------------------------------------
$W42A = $WDIR + '\phase42a_sum.bin'   # format 0x53454531, legacy, pool=sum
if (-not (Test-Path $W42A)) { Write-Error "phase42a_sum.bin not found: $W42A"; exit 1 }

$GEN41  = $BINDIR + '\phase38-42\phase41_generator.exe'
$GEN43  = $BINDIR + '\phase43_generator.exe'

if (-not (Test-Path $GEN41)) { Write-Warning "phase41_generator not found: $GEN41 -- skipping old generator comparison" }

# ---- Run comparison: T=0.7, 2000 bytes, same seed ---------------------------
$TEMP    = '0.7'
$GEN_LEN = 2000
$WARMUP  = 5000

Write-Host ''
Write-Host '=== Sanity Check: phase41 vs phase43 generator on phase42a_sum.bin ===' -ForegroundColor Yellow
Write-Host "  Weights: phase42a_sum.bin (0x53454531, legacy pool=sum, BPB~2.32)"
Write-Host "  Temp: $TEMP | Gen: $GEN_LEN bytes | Warmup (gen43 only): $WARMUP"
Write-Host ''

# Run phase41_generator (if present)
if (Test-Path $GEN41) {
    Write-Host '  phase41_generator + phase42a_sum.bin ...' -NoNewline
    $out41   = $RDIR + '\gen41_phase42a_t07.txt'
    $stats41 = $RDIR + '\gen41_phase42a_t07_stats.txt'
    cmd /c "`"$GEN41`" `"$TS_DATA`" `"$W42A`" --seed-len 512 --gen-len $GEN_LEN --temp $TEMP > `"$out41`" 2>`"$stats41`""
    if ($LASTEXITCODE -ne 0) { Write-Host ' FAILED' -ForegroundColor Red }
    else { Write-Host ' OK' -ForegroundColor Green }
} else {
    Write-Host '  [phase41_generator not found -- skipping reference run]' -ForegroundColor Yellow
}

# Run phase43_generator with same weights (auto-detects 0x53454531, sets legacy=1)
Write-Host '  phase43_generator + phase42a_sum.bin ...' -NoNewline
$out43   = $RDIR + '\gen43_phase42a_t07.txt'
$stats43 = $RDIR + '\gen43_phase42a_t07_stats.txt'
cmd /c "`"$GEN43`" `"$TS_DATA`" `"$W42A`" --gen-len $GEN_LEN --temp $TEMP --warmup $WARMUP > `"$out43`" 2>`"$stats43`""
if ($LASTEXITCODE -ne 0) { Write-Host ' FAILED' -ForegroundColor Red }
else { Write-Host ' OK' -ForegroundColor Green }

# Also run phase43_generator on phase43a_legacy.bin (format 0x53454535, --legacy flag)
$W43L = $WDIR + '\phase43a_legacy.bin'
if (Test-Path $W43L) {
    Write-Host '  phase43_generator + phase43a_legacy.bin (--legacy) ...' -NoNewline
    $out43L   = $RDIR + '\gen43_legacy_t07.txt'
    $stats43L = $RDIR + '\gen43_legacy_t07_stats.txt'
    cmd /c "`"$GEN43`" `"$TS_DATA`" `"$W43L`" --gen-len $GEN_LEN --temp $TEMP --warmup $WARMUP --legacy > `"$out43L`" 2>`"$stats43L`""
    if ($LASTEXITCODE -ne 0) { Write-Host ' FAILED' -ForegroundColor Red }
    else { Write-Host ' OK' -ForegroundColor Green }
}

# ---- Show stats ----------------------------------------------------------------
Write-Host ''
Write-Host '=== Stats ===' -ForegroundColor Cyan
foreach ($pair in @(
    @{ label='phase41 + phase42a'; stats=$RDIR+'\gen41_phase42a_t07_stats.txt' },
    @{ label='phase43 + phase42a'; stats=$RDIR+'\gen43_phase42a_t07_stats.txt' },
    @{ label='phase43 + p43legacy'; stats=$RDIR+'\gen43_legacy_t07_stats.txt' }
)) {
    if (Test-Path $pair.stats) {
        $bpb = Select-String 'self_BPB' $pair.stats | Select-Object -First 1
        $bpbStr = if ($bpb) { $bpb.Line.Trim() } else { 'N/A' }
        Write-Host ('  {0,-25} {1}' -f $pair.label, $bpbStr)
    }
}

# ---- Text preview ---------------------------------------------------------------
Write-Host ''
Write-Host '=== Text preview T=0.7 (first 500 chars) ===' -ForegroundColor Yellow

$previews = @(
    @{ label='phase41_generator + phase42a_sum'; file=$RDIR+'\gen41_phase42a_t07.txt' },
    @{ label='phase43_generator + phase42a_sum'; file=$RDIR+'\gen43_phase42a_t07.txt' },
    @{ label='phase43_generator + p43a_legacy '; file=$RDIR+'\gen43_legacy_t07.txt' }
)
foreach ($p in $previews) {
    Write-Host ''
    Write-Host ("--- $($p.label) ---") -ForegroundColor Magenta
    if (Test-Path $p.file) {
        $bytes = [System.IO.File]::ReadAllBytes($p.file)
        $n = [Math]::Min(500, $bytes.Length)
        Write-Host ([System.Text.Encoding]::ASCII.GetString($bytes[0..($n-1)]))
    } else {
        Write-Host '  [file not found]' -ForegroundColor Red
    }
}

Write-Host ''
Write-Host '=== Interpretation ===' -ForegroundColor Yellow
Write-Host '  phase41 + phase42a = reference (known to produce pseudo-English in Phase 42)'
Write-Host '  phase43 + phase42a = sanity check for new generator legacy path'
Write-Host '  phase43 + p43a_legacy = same weights, different format (0x53454535 vs 0x53454531)'
Write-Host '  Expected: all three produce similar quality (broken but English-like)'
Write-Host '  If phase43 produces garbage where phase41 makes English -> generator bug'
