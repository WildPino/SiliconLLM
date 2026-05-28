$ErrorActionPreference = "Stop"

Write-Host "=== Phase 16: Robustness & Serialization Audit ==="
Write-Host "Compiling Engine..."
gcc -O3 -march=native -mavx2 -mfma benchmarks/train_entropy_readout.c src/silicon_entropy.c src/silicon_v0.c -o train_readout.exe
gcc -O3 -march=native -mavx2 -mfma benchmarks/eval_entropy_stream.c src/silicon_entropy.c src/silicon_v0.c -o eval_stream.exe
Write-Host "Compilation Successful.`n"

# Test 1: Determinism
Write-Host "--- Test 1: Determinism ---"
Write-Host "Training on c_code.c (0-50%)..."
.\train_readout.exe data/c_code.c entropy_weights.bin --train-start 0 --train-len 50 --val-start 50 --val-len 25 > $null
Write-Host "Run 1: Eval on c_code.c (75-100%)..."
$out1 = .\eval_stream.exe data/c_code.c entropy_weights.bin --eval-start 75 --eval-len 25
$bpb1 = ($out1 | Select-String "Model BPB:").ToString().Split()[-1]
Write-Host "Run 2: Eval on c_code.c (75-100%)..."
$out2 = .\eval_stream.exe data/c_code.c entropy_weights.bin --eval-start 75 --eval-len 25
$bpb2 = ($out2 | Select-String "Model BPB:").ToString().Split()[-1]

if ($bpb1 -eq $bpb2) {
    Write-Host "[PASS] Determinism: BPB exactly matches ($bpb1 == $bpb2)" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Determinism: BPB mismatch ($bpb1 != $bpb2)" -ForegroundColor Red
    exit 1
}

# Test 2: Cross-File Decoupling
Write-Host "`n--- Test 2: Cross-File Decoupling ---"
Write-Host "Training on promessi_sposi.txt (100%)..."
# We use 100% train, 0% val to simplify. The script supports 0 len.
.\train_readout.exe data/promessi_sposi.txt entropy_weights.bin --train-start 0 --train-len 99 --val-start 99 --val-len 1 > $null
Write-Host "Evaluating on markdown_docs.md (100%)..."
$outCross = .\eval_stream.exe data/markdown_docs.md entropy_weights.bin --eval-start 0 --eval-len 100
$bpbCross = ($outCross | Select-String "Model BPB:").ToString().Split()[-1]

if ([double]$bpbCross -gt 0 -and [double]$bpbCross -lt 10) {
    Write-Host "[PASS] Cross-file: Pipeline runs and produces finite BPB ($bpbCross)" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Cross-file: Invalid BPB ($bpbCross)" -ForegroundColor Red
    exit 1
}

# Test 3: Shuffled Control
Write-Host "`n--- Test 3: Shuffled Control ---"
Write-Host "Training on c_code.c (50%)..."
.\train_readout.exe data/c_code.c entropy_weights.bin --train-start 0 --train-len 50 --val-start 50 --val-len 25 > $null
Write-Host "Evaluating on shuffled c_code.c (25%)..."
$outShuffle = .\eval_stream.exe data/c_code.c entropy_weights.bin --eval-start 75 --eval-len 25 --shuffled
$bpbShuffleModel = [double]($outShuffle | Select-String "Model BPB:").ToString().Split()[-1]
$bpbShuffleUni = [double]($outShuffle | Select-String "Unigram BPB:").ToString().Split()[-1]

if ($bpbShuffleModel -ge ($bpbShuffleUni - 0.01)) {
    Write-Host "[PASS] Shuffled Control: Model collapses as expected (Model: $bpbShuffleModel >= Unigram: $bpbShuffleUni)" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Shuffled Control: Model mysteriously gained compression! (Model: $bpbShuffleModel < Unigram: $bpbShuffleUni)" -ForegroundColor Red
    exit 1
}

# Test 4: Header Mismatch
Write-Host "`n--- Test 4: Header Mismatch ---"
Write-Host "Simulating user requesting wrong chunk-size (--chunk-size 8)..."
$errorMatched = $false
try {
    .\eval_stream.exe data/c_code.c entropy_weights.bin --eval-start 75 --eval-len 25 --chunk-size 8 2>&1 | Out-Null
} catch {
    $errorMatched = $true
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[PASS] Header Mismatch: Engine aborted gracefully on wrong configuration." -ForegroundColor Green
} else {
    Write-Host "[FAIL] Header Mismatch: Engine ran despite configuration mismatch!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== ALL TESTS PASSED ===" -ForegroundColor Cyan
