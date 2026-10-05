$ErrorActionPreference = 'Stop'
$rawPath = Join-Path (Get-Location) 'docs/research/NATIVE_EXPERT_SCALING_20260925/meth478_grouped_router_result.json'
$destination = Join-Path (Get-Location) 'results/native_expert_scaling/meth478_windows_terminal.json'
if (Test-Path -LiteralPath $destination) { throw 'Exclusive event output already exists' }
if (-not (Test-Path -LiteralPath $rawPath)) { $rawPath = $rawPath.Replace('.json', '.failure.json') }
$raw = [IO.File]::ReadAllText($rawPath, [Text.Encoding]::UTF8)
$parsed = $raw | ConvertFrom-Json
$literal = [regex]::Match($raw, '"start_utc"\s*:\s*"([^"]+)"').Groups[1].Value
if (-not $literal) { throw 'Missing literal ISO start_utc' }
$start = [DateTimeOffset]::Parse($literal, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime().AddSeconds(-2)
$end = [DateTimeOffset]::UtcNow
$instances = @(@{pid=[int]$parsed.process_instance.pid;create_time_unix=$parsed.process_instance.create_time_unix;label='controller';start_utc=$parsed.start_utc;end_utc=$parsed.end_utc})
foreach ($command in $parsed.commands) {
    $instances += @{pid=[int]$command.process_instance.pid;create_time_unix=$command.process_instance.create_time_unix;label=$command.label;start_utc=$command.start_utc;end_utc=$command.end_utc}
    foreach ($child in $command.descendant_process_peaks) { $instances += @{pid=[int]$child.pid;create_time_unix=$child.create_time_unix;label=($command.label+'.descendant');start_utc=$command.start_utc;end_utc=$command.end_utc} }
}
$available = $false; $queryError = $null; $events = @()
try {
    $events = @(Get-WinEvent -FilterHashtable @{ LogName='Application'; Id=1000; StartTime=$start.UtcDateTime; EndTime=$end.UtcDateTime } -ErrorAction Stop)
    $available = $true
} catch {
    if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $available = $true } else { $queryError = $_.Exception.Message }
}
$matching = @(); $other = @()
foreach ($event in $events) {
    [xml]$xml = $event.ToXml(); $values = @{}
    foreach ($entry in $xml.Event.EventData.Data) { $values[$entry.Name] = $entry.'#text' }
    $eventPid = 0; $value = [string]$values.ProcessId
    if ($value -match '^0x') { $eventPid = [Convert]::ToInt32($value.Substring(2), 16) } elseif ($value -match '^\d+$') { $eventPid = [int]$value }
    $time = [DateTimeOffset]$event.TimeCreated
    $hit = @($instances | Where-Object { $_.pid -eq $eventPid -and $time -ge ([DateTimeOffset]::Parse($_.start_utc).AddSeconds(-2)) -and $time -le ([DateTimeOffset]::Parse($_.end_utc).AddSeconds(2)) })
    $row = @{RecordId=$event.RecordId;TimeUtc=$event.TimeCreated.ToUniversalTime().ToString('o');Data=$values;XML=$event.ToXml();ProcessId=$eventPid;MatchedInstances=$hit}
    if ($hit.Count -gt 0) { $matching += $row } else { $other += $row }
}
$result = @{experiment='METH478 actual controller/compiler/native Windows terminal metadata';query_start_utc=$start.ToString('o');query_end_utc=$end.ToString('o');query_available=$available;query_error=$queryError;event_id=1000;returned_event_count=$events.Count;instances=$instances;matching_scientific_events=$matching;other_events=$other}
[IO.File]::WriteAllText($destination, ($result | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$result | Select-Object query_start_utc,query_end_utc,query_available,returned_event_count,query_error | ConvertTo-Json
