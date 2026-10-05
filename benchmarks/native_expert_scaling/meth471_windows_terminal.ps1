$taskRawPath = [IO.Path]::GetFullPath('docs/research/NATIVE_EXPERT_SCALING_20260925/meth471_switch_development_capture_result.json')
$taskRawText = [IO.File]::ReadAllText($taskRawPath, [Text.Encoding]::UTF8)
$taskRaw = $taskRawText | ConvertFrom-Json
$taskIsoStart = [regex]::Match($taskRawText, '"start_utc"\s*:\s*"([^"]+)"').Groups[1].Value
$taskStart = ([DateTimeOffset]::Parse($taskIsoStart, [Globalization.CultureInfo]::InvariantCulture)).UtcDateTime.AddSeconds(-2)
$taskEnd = [DateTime]::UtcNow
$taskMainPid = [int]$taskRaw.main_process_instance.pid
$taskNativePids = @($taskRaw.commands | ForEach-Object { [int]$_.pid } | Sort-Object -Unique)
$taskAllPids = @($taskMainPid) + $taskNativePids
$taskEvents = @()
$taskQueryOK = $false
$taskQueryError = $null
try {
 $taskEvents = @(Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=$taskStart.ToLocalTime(); EndTime=$taskEnd.ToLocalTime()} -ErrorAction Stop)
 $taskQueryOK = $true
} catch {
 if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $taskQueryOK = $true } else { $taskQueryError = $_.Exception.Message }
}
$taskMatching = @()
$taskOther = @()
foreach ($taskEvent in $taskEvents) {
 $taskXml = [xml]$taskEvent.ToXml()
 $taskPidField = @($taskXml.Event.EventData.Data | Where-Object { $_.Name -eq 'ProcessId' })
 $taskNameField = @($taskXml.Event.EventData.Data | Where-Object { $_.Name -eq 'AppName' })
 $taskEventPid = -1
 if ($taskPidField.Count -eq 1) {
  $taskPidText = $taskPidField[0].'#text'
  if ($taskPidText.StartsWith('0x')) { $taskEventPid = [Convert]::ToInt32($taskPidText.Substring(2),16) } else { $taskEventPid = [int]$taskPidText }
 }
 $taskEventName = if ($taskNameField.Count -eq 1) { $taskNameField[0].'#text' } else { '' }
 $taskSummary = @{record_id=$taskEvent.RecordId; time_utc=$taskEvent.TimeCreated.ToUniversalTime().ToString('o'); pid=$taskEventPid; app_name=$taskEventName}
 if ($taskAllPids -contains $taskEventPid -and $taskEventName -in @('python.exe','meth393_switch_router_audit.exe')) {
  $taskSummary.xml = $taskEvent.ToXml()
  $taskMatching += $taskSummary
 } else { $taskOther += $taskSummary }
}
$taskState = [ordered]@{experiment='METH471 development capture actual Windows terminal metadata'; main_pid=$taskMainPid; native_pids=$taskNativePids; query_start_utc=$taskStart.ToString('o'); query_end_utc=$taskEnd.ToString('o'); query_available=$taskQueryOK; query_error=$taskQueryError; event_id=1000; returned_event_count=$taskEvents.Count; matching_main_or_native_events=$taskMatching; other_events=$taskOther}
$taskEventPath = [IO.Path]::GetFullPath('results/native_expert_scaling/meth471_windows_terminal.json')
$taskBytes = [Text.UTF8Encoding]::new($false).GetBytes(($taskState | ConvertTo-Json -Depth 8) + "`r`n")
$taskFile = [IO.File]::Open($taskEventPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try { $taskFile.Write($taskBytes, 0, $taskBytes.Length) } finally { $taskFile.Dispose() }
[ordered]@{query_available=$taskQueryOK; main_pid=$taskMainPid; distinct_native_pids=$taskNativePids.Count; events=$taskEvents.Count; matching_events=$taskMatching.Count; query_start_utc=$taskStart.ToString('o'); query_end_utc=$taskEnd.ToString('o')} | ConvertTo-Json
