$ErrorActionPreference = 'Stop'
$rawPath = Join-Path (Get-Location) 'docs/research/NATIVE_EXPERT_SCALING_20260925/meth475_switch_coarse_certificate_result.json'
$destination = Join-Path (Get-Location) 'results/native_expert_scaling/meth475_windows_terminal.json'
if (Test-Path -LiteralPath $destination) { throw 'Exclusive event output already exists' }
$raw = [IO.File]::ReadAllText($rawPath, [Text.Encoding]::UTF8)
$parsed = $raw | ConvertFrom-Json
$literal = [regex]::Match($raw, '"start_utc"\s*:\s*"([^"]+)"').Groups[1].Value
if (-not $literal) { throw 'Missing literal ISO start_utc' }
$start = [DateTimeOffset]::Parse($literal, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime().AddSeconds(-2)
$end = [DateTimeOffset]::UtcNow
$mainPid = [int]$parsed.main_process_instance.pid
$available = $false; $queryError = $null; $events = @()
try {
    $events = @(Get-WinEvent -FilterHashtable @{ LogName='Application'; Id=1000; StartTime=$start.UtcDateTime; EndTime=$end.UtcDateTime } -ErrorAction Stop)
    $available = $true
} catch {
    if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $available = $true }
    else { $queryError = $_.Exception.Message }
}
$matching = @(); $other = @()
foreach ($event in $events) {
    [xml]$xml = $event.ToXml(); $values = @{}
    foreach ($entry in $xml.Event.EventData.Data) { $values[$entry.Name] = $entry.'#text' }
    $eventPid = 0; $value = [string]$values.ProcessId
    if ($value -match '^0x') { $eventPid = [Convert]::ToInt32($value.Substring(2), 16) }
    elseif ($value -match '^\d+$') { $eventPid = [int]$value }
    $row = @{ RecordId=$event.RecordId; TimeUtc=$event.TimeCreated.ToUniversalTime().ToString('o'); Data=$values; XML=$event.ToXml(); ProcessId=$eventPid }
    if ($eventPid -eq $mainPid) { $matching += $row } else { $other += $row }
}
$result = @{ experiment='METH475 actual main Windows terminal metadata'; main_pid=$mainPid; query_start_utc=$start.ToString('o'); query_end_utc=$end.ToString('o'); query_available=$available; query_error=$queryError; event_id=1000; returned_event_count=$events.Count; matching_main_events=$matching; other_events=$other }
[IO.File]::WriteAllText($destination, ($result | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$result | Select-Object main_pid,query_start_utc,query_end_utc,query_available,returned_event_count,query_error | ConvertTo-Json
