# Phase 53.B - B-SFDPE feasibility probe (local plasticity overlay on the FROZEN substrate).
# TF decodability only. NOT a generator. Substrate read-only. No commit. magic 0x5345455E.
# The .exe runs the whole mandated run-order internally and prints RAW numbers (no auto PASS/FAIL).
# Run-order inside the exe: GATE(1) A2-scalar flat -> GATE(2) A5-scramble collapses -> CONTENT vs
#   FROZEN/PARITY/BLIND/POS -> ablations A1/A3/A4b/A4c/A6 -> kernel-check. Human reads the verdict.
#
# Usage:
#   pwsh benchmarks/phase52/phase53b.ps1            # build + tiny smoke (deterministic sanity)
#   pwsh benchmarks/phase52/phase53b.ps1 -Real      # PRINT the real-run command (the user launches it)
param([switch]$Real, [switch]$SkipBuild)

$ErrorActionPreference = "Stop"
$root = "D:\_THINGS\Progetti\SiliconLLM"
Set-Location $root
$src  = "benchmarks/phase52/phase53b_bsfdpe.c"
$exe  = "bin/phase53b_bsfdpe.exe"
$data = "data/corpora/tinystories_64mb.txt"
$d1   = "weights/phase44f_F0.bin"
$bpe  = "weights/bpe1024.bin"
New-Item -ItemType Directory -Force "results/phase53" | Out-Null

if (-not $SkipBuild) {
    Write-Host "== build ==" -ForegroundColor Cyan
    & gcc -O3 -march=native -mavx2 -mfma $src src/silicon_entropy.c src/silicon_v0.c -o $exe -lm -I .
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
    Write-Host "build OK" -ForegroundColor Green
}

if ($Real) {
    Write-Host ""
    Write-Host "== REAL RUN (the user/architect launches this; long) ==" -ForegroundColor Yellow
    Write-Host "$exe $data $d1 $bpe results\phase53\P53B --len 150000 --epochs 6 > results\phase53\P53B_run.out 2> results\phase53\P53B_run.err"
    Write-Host ""
    Write-Host "After it finishes, read results\phase53\P53B_run.out. Decision rule (HUMAN reads, no auto-verdict):"
    Write-Host "  GATE(1): A2-scalar[x|h] ~= FROZEN-POINT (flat). else harness/probe broken -> STOP."
    Write-Host "  GATE(2): A5-scramble headroom ~0 AND A5_h ~0 (substrate load-bearing). else measuring text, not substrate -> STOP."
    Write-Host "  SUCCESS iff: CONTENT headroom(=CONTENT-FROZEN) >> PARITY-headroom, >> POS-headroom, > 0, and CONTENT > BLIND;"
    Write-Host "              and ablations behave (A1->parity, A3->POS, A5->collapse); and K_CONTENT distinct from K_PARITY."
    Write-Host "  Caveat: this is TF decodability = 'signal extractable', NOT a generator. No promotion, no commit."
    return
}

Write-Host ""
Write-Host "== TINY SMOKE (deterministic sanity; mechanics only, NOT the verdict) ==" -ForegroundColor Cyan
& $exe $data $d1 $bpe "results/phase53/SMOKE53B" --smoke --max-bytes 4000000 2>$null | Tee-Object "results/phase53/smoke53b.out" | Select-String -Pattern "GATE|CONTENT=|headroom|h-only|A5scramble_h|A2-scalar\[|dU_F|actTr"
Write-Host ""
Write-Host "SMOKE sanity checks (read by eye):" -ForegroundColor Green
Write-Host "  - end-to-end ran, deterministic (re-run -> identical output)."
Write-Host "  - GATE(1): A2-scalar ~= FROZEN-POINT (flat) -> bank is sane."
Write-Host "  - on 24k/3ep everything sits at the noise floor (often negative) -> EXPECTED, decide only on the real run."
Write-Host ""
Write-Host "Next: pwsh benchmarks/phase52/phase53b.ps1 -Real   (prints the real-run command for the user to launch)"
