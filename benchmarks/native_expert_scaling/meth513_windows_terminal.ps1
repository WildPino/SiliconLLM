$ErrorActionPreference = 'Stop'
$taskDoc = Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskStages = @()
foreach ($taskKind in @('main', 'audit')) {
    $taskRawPath = Join-Path $taskDoc ('meth513_' + $taskKind + '_result.json')
    $taskRaw = Get-Content -Raw -LiteralPath $taskRawPath | ConvertFrom-Json
    $taskStart = [DateTimeOffset]::Parse($taskRaw.started_utc).UtcDateTime.AddSeconds(-1).ToString('o')
    $taskEnd = [DateTime]::UtcNow.AddSeconds(1).ToString('o')
    $taskXPath = "*[System[(EventID=1000) and TimeCreated[@SystemTime >= '$taskStart' and @SystemTime <= '$taskEnd']]]"
    $taskAvailable = $true
    $taskError = $null
    $taskEvents = @()
    try { $taskEvents = @(Get-WinEvent -LogName Application -FilterXPath $taskXPath -ErrorAction Stop) }
    catch {
        if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') { $taskAvailable = $false; $taskError = $_.ToString() }
    }
    $taskMatches = @()
    foreach ($taskEvent in $taskEvents) {
        [xml]$taskXml = $taskEvent.ToXml()
        $taskData = @{}
        foreach ($taskField in $taskXml.Event.EventData.Data) { $taskData[$taskField.Name] = $taskField.'#text' }
        $taskEventPid = if ($taskData.ProcessId -like '0x*') { [Convert]::ToInt64($taskData.ProcessId.Substring(2), 16) } else { [long]$taskData.ProcessId }
        $taskCreate = $null
        if ($taskData.ProcessCreationTime -like '0x*') {
            $taskFileTime = [Convert]::ToInt64($taskData.ProcessCreationTime.Substring(2), 16)
            $taskCreate = ([DateTimeOffset][DateTime]::FromFileTimeUtc($taskFileTime)).ToUnixTimeMilliseconds() / 1000.0
        }
        if ($taskEventPid -eq $taskRaw.process_instance.pid -and $null -ne $taskCreate -and [Math]::Abs($taskCreate - $taskRaw.process_instance.create_time_unix) -lt .002) {
            $taskMatches += @{RecordId=$taskEvent.RecordId; XML=$taskEvent.ToXml()}
        }
    }
    $taskStages += @{kind=$taskKind; raw_sha256=(Get-FileHash -LiteralPath $taskRawPath -Algorithm SHA256).Hash.ToLower(); process_instance=$taskRaw.process_instance; query_available=$taskAvailable; query_error=$taskError; xpath=$taskXPath; relevant_events=$taskMatches}
}
$taskControls = @(Get-WinEvent -LogName Application -FilterXPath '*[System[(EventID=1000) and (EventRecordID=179810 or EventRecordID=179791)]]' -ErrorAction Stop)
if (@($taskControls).Count -ne 2) { throw 'positive control unavailable' }
$taskResult = @{stages=$taskStages; positive_control_ids=@($taskControls | ForEach-Object { $_.RecordId }); positive_control_XML=@($taskControls | ForEach-Object { $_.ToXml() }); new_numerical_model_C_calls=0}
$taskDestination = Join-Path $taskDoc 'meth513_windows_terminal.json'
if (Test-Path -LiteralPath $taskDestination) { throw 'terminal metadata already exists' }
$taskResult | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $taskDestination -Encoding utf8
if (@($taskStages | Where-Object { -not $_.query_available -or @($_.relevant_events).Count -ne 0 }).Count -gt 0) { throw 'terminal event gate failed; receipt retained' }
Write-Output ('Typed UTC main/audit queries passed; positive controls present; relevant events0.')
