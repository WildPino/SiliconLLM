$taskRaw = Get-Content -LiteralPath 'docs/research/NATIVE_EXPERT_SCALING_20260925/meth469_switch_native_domain_capture_result.json' -Raw | ConvertFrom-Json
$taskProgress = Get-Content -LiteralPath 'results/native_expert_scaling/meth469_switch_native_domain_capture/progress.jsonl' -TotalCount 1 | ConvertFrom-Json
$taskRawText = [IO.File]::ReadAllText([IO.Path]::GetFullPath('docs/research/NATIVE_EXPERT_SCALING_20260925/meth469_switch_native_domain_capture_result.json'))
$taskIsoStart = [regex]::Match($taskRawText, '"start_utc"\s*:\s*"([^"]+)"').Groups[1].Value
$taskStart = ([DateTimeOffset]::Parse($taskIsoStart, [Globalization.CultureInfo]::InvariantCulture)).UtcDateTime.AddSeconds(-2)
$taskEnd = [DateTime]::UtcNow
$taskNativePids = @($taskRaw.commands | ForEach-Object { [int]$_.pid } | Sort-Object -Unique)
$taskAllPids = @([int]$taskProgress.pid) + $taskNativePids
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
$taskState = [ordered]@{experiment='METH469 completed native capture Windows terminal metadata'; main_pid=[int]$taskProgress.pid; native_pids=$taskNativePids; query_start_utc=$taskStart.ToString('o'); query_end_utc=$taskEnd.ToString('o'); query_available=$taskQueryOK; query_error=$taskQueryError; event_id=1000; returned_event_count=$taskEvents.Count; matching_main_or_native_events=$taskMatching; other_events=$taskOther}
$taskEventPath = [IO.Path]::GetFullPath('results/native_expert_scaling/meth469_windows_terminal_v2.json')
if (Test-Path -LiteralPath $taskEventPath) { throw 'terminal record already exists' }
[IO.File]::WriteAllText($taskEventPath, ($taskState | ConvertTo-Json -Depth 8) + "`r`n", [Text.UTF8Encoding]::new($false))
[ordered]@{query_available=$taskQueryOK; main_pid=[int]$taskProgress.pid; distinct_native_pids=$taskNativePids.Count; events=$taskEvents.Count; matching_events=$taskMatching.Count; query_start_utc=$taskStart.ToString('o'); query_end_utc=$taskEnd.ToString('o')} | ConvertTo-Json
