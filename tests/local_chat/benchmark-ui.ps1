param(
    [Parameter(Mandatory=$true)][string]$BaselineExecutable,
    [Parameter(Mandatory=$true)][string]$CandidateExecutable,
    [Parameter(Mandatory=$true)][string]$QtPath,
    [Parameter(Mandatory=$true)][string]$CompilerPath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [int]$Iterations = 7,
    [int]$DurationMs = 3000
)
$ErrorActionPreference = 'Stop'
if ($Iterations -lt 7 -or $DurationMs -lt 1000) { throw 'Use at least seven iterations and a one-second measurement window.' }
$baseline = (Resolve-Path -LiteralPath $BaselineExecutable).Path
$candidate = (Resolve-Path -LiteralPath $CandidateExecutable).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
if ($output.Contains('"')) { throw 'Output path cannot contain a quote.' }
New-Item -ItemType Directory -Path $output -Force | Out-Null
$configurations = @(
    @{ name='baseline-light'; executable=$baseline; theme='light' },
    @{ name='candidate-light'; executable=$candidate; theme='light' },
    @{ name='candidate-dark'; executable=$candidate; theme='dark' }
)
$records = [System.Collections.Generic.List[object]]::new()
$previousPath = $env:PATH
$previousPlatform = $env:QT_QPA_PLATFORM
try {
    $env:PATH = "$CompilerPath;$(Join-Path $QtPath 'bin');$previousPath"
    $env:QT_QPA_PLATFORM = 'windows'
    foreach ($scenario in @('idle','stream')) {
        for ($iteration = 1; $iteration -le $Iterations; $iteration++) {
            $ordered = if ($iteration % 2 -eq 0) { @($configurations[2],$configurations[1],$configurations[0]) } else { $configurations }
            foreach ($configuration in $ordered) {
                $jsonPath = Join-Path $output "$($configuration.name)-$scenario-$iteration.json"
                $arguments = @('--scenario', $scenario, '--theme', $configuration.theme, '--duration', $DurationMs, '--output', ('"' + $jsonPath + '"'))
                $process = Start-Process -FilePath $configuration.executable -ArgumentList $arguments -WindowStyle Hidden -PassThru -Wait
                if ($process.ExitCode -ne 0) { throw "Rendering probe failed: $($configuration.name) / $scenario / $iteration (exit $($process.ExitCode))." }
                $record = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
                if ($record.synthetic_ui_only -ne $true -or $record.native_platform -ne 'windows' -or $record.visible_window -ne $true) { throw 'Probe did not use the expected native synthetic UI scenario.' }
                $records.Add([pscustomobject]@{configuration=$configuration.name;iteration=$iteration;measurement=$record})
            }
            Write-Output "$scenario : iteration $iteration/$Iterations complete"
        }
    }
    $report = [ordered]@{
        schema='silicon-ui-comparison-v1'; synthetic_ui_only=$true;
        baseline_executable_sha256=(Get-FileHash -LiteralPath $baseline -Algorithm SHA256).Hash.ToLowerInvariant();
        candidate_executable_sha256=(Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant();
        iterations=$Iterations; duration_ms=$DurationMs; viewport_width=1180; viewport_height=780;
        note='UI process resource measurements only. No LLM inference or decode-rate claim.';
        samples=$records
    }
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'comparison.json') -Encoding utf8
} finally {
    $env:PATH = $previousPath
    if ($null -eq $previousPlatform) { Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue } else { $env:QT_QPA_PLATFORM = $previousPlatform }
}
