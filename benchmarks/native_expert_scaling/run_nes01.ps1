$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$resultDir = Join-Path $repoRoot 'results\native_expert_scaling'
$recordDir = Join-Path $repoRoot 'docs\research\NATIVE_EXPERT_SCALING_20260925'
$pidPath = Join-Path $resultDir 'nes01_launcher.pid'
$statusPath = Join-Path $recordDir 'nes01_status.json'
$stdoutPath = Join-Path $recordDir 'nes01_e128_train.stdout.log'
$stderrPath = Join-Path $recordDir 'nes01_e128_train.stderr.log'

New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
$PID | Set-Content -LiteralPath $pidPath
$startedAt = (Get-Date).ToUniversalTime().ToString('o')
$exitCode = -1
try {
    Push-Location -LiteralPath $repoRoot
    try {
        & $pythonPath -u benchmarks\phase57\phase59_moe.py `
            --arm moe-gran --experts 128 --sparse-moe --steps 4000 `
            --seq 512 --batch 4 --accum 4 --bf16 `
            --eval-tok 200000 --measure-batches 24 --device cuda:0 `
            --checkpoint-every 200 --max-hours 10 `
            --resume-state results\native_expert_scaling\nes01_e128_resume.pt `
            --save results\native_expert_scaling\nes01_e128_final.pt `
            1> $stdoutPath 2> $stderrPath
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrPath
    $exitCode = 1
} finally {
    [ordered]@{
        started_at_utc = $startedAt
        ended_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        launcher_pid = $PID
        exit_code = $exitCode
        final_checkpoint_exists = Test-Path -LiteralPath (Join-Path $resultDir 'nes01_e128_final.pt')
        resume_state_exists = Test-Path -LiteralPath (Join-Path $resultDir 'nes01_e128_resume.pt')
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath
}
exit $exitCode
