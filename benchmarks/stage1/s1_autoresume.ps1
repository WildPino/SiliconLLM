# Keeps the stage 1 (a) run going across a shutdown or a reboot. It starts at logon from the Startup folder, and once by
# hand for the run already going.
# - Run complete: it does nothing.
# - Runner not running: it relaunches s1_run.py, which resumes after the last COMPLETE panel (the panel that was cut is
#   redone from scratch: the tools open their outputs with "wb").
# - Runner running: every 5 min it copies the report to a backup if the report parses, and keeps Windows awake.
# It never relaunches over a VOID or STOPPED panel, after a launch that made no progress, or more than $MaxLaunches times.
# The parameters other than $MaxLaunches exist only to test the branches on copies.
param([int]$MaxLaunches = 4,
      [string]$Work = 'E:\SiliconLLM_Artifacts\stage1\s1a',
      [string]$Report = 'D:\_THINGS\Progetti\SiliconLLM\benchmarks\stage1\results\s1a_run.json',
      [string]$RunnerPattern = '*s1_run.py*',
      [string]$MutexName = 'Local\s1a_autoresume')
$ErrorActionPreference = 'Continue'
$repo = 'D:\_THINGS\Progetti\SiliconLLM'
$work = $Work
$report = $Report
$bak = Join-Path $work 's1a_run.json.bak'
$log = Join-Path $work 'autoresume.log'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
$py = (Get-Command python).Source

function Say([string]$m) { Add-Content -Path $log -Value ('{0} {1}' -f (Get-Date -Format s), $m) -Encoding utf8 }
function State([string]$p) { (& $py -I (Join-Path $repo 'benchmarks\stage1\s1_run_state.py') $p | Select-Object -First 1) }
function Done([string]$s) { if ($s -match '^\w+ (\d+)/') { [int]$Matches[1] } else { -1 } }
function Runner {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like $RunnerPattern } | Select-Object -First 1
}

$mutex = New-Object System.Threading.Mutex($false, $MutexName)
if (-not $mutex.WaitOne(0)) { exit 0 }
Add-Type -Namespace S1 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
Say "start pid=$PID"
$launches = 0
$doneAtLaunch = -1
while ($true) {
    $r = Runner
    if ($r) {
        [S1.Power]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if ((State $report) -ne 'CORRUPT') { Copy-Item $report $bak -Force }
        Start-Sleep -Seconds 300
        continue
    }
    [S1.Power]::SetThreadExecutionState([uint32]'0x80000000') | Out-Null
    $s = State $report
    if ($s -eq 'CORRUPT') {
        $kept = Join-Path $work ('s1a_run.json.corrupt-' + (Get-Date -Format 'yyyyMMddTHHmmss'))
        Copy-Item $report $kept -Force
        if ((Test-Path $bak) -and ((State $bak) -ne 'CORRUPT')) {
            Copy-Item $bak $report -Force
            Say "report did not parse: kept as $kept, restored from the backup"
            continue
        }
        Say "report did not parse and there is no readable backup: kept as $kept, stopping"
        break
    }
    Say "runner not running, state: $s"
    if ($s -eq 'DONE' -or $s -like 'HOLD*') { break }
    if ($launches -ge $MaxLaunches) { Say "launch limit $MaxLaunches reached, stopping"; break }
    if ($launches -gt 0 -and (Done $s) -le $doneAtLaunch) { Say 'the last launch made no progress, stopping'; break }
    $ts = Get-Date -Format 'yyyyMMddTHHmmss'
    $p = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -ArgumentList '-I', 'benchmarks/stage1/s1_run.py', '--work', $work, '--out', $report `
        -RedirectStandardOutput (Join-Path $work "run_$ts.out.log") -RedirectStandardError (Join-Path $work "run_$ts.err.log")
    $launches++
    $doneAtLaunch = Done $s
    Say "launched runner pid=$($p.Id) (launch $launches, logs run_$ts.*.log)"
    Start-Sleep -Seconds 60
}
$mutex.ReleaseMutex()
Say 'exit'
