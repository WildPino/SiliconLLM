$ErrorActionPreference = 'Stop'
$taskDoc = Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925'
function Utc($v) {
    if ($v -is [DateTimeOffset]) { return $v.ToUniversalTime() }
    if ($v -is [DateTime]) { return [DateTimeOffset]($v.ToUniversalTime()) }
    return [DateTimeOffset]::Parse([string]$v,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
$epochCheck = [DateTimeOffset]([DateTime]::FromFileTimeUtc(116444736000000000))
if (($epochCheck - [DateTimeOffset]'1970-01-01T00:00:00Z').TotalSeconds -ne 0) { throw 'UTC epoch fault' }
$taskStages = @()
foreach ($taskKind in @('binding', 'main', 'audit')) {
    $taskRawPath = if ($taskKind -eq 'binding') { Join-Path $taskDoc 'meth516_binding.json' } else { Join-Path $taskDoc ('meth516_' + $taskKind + '_result.json') }
    $taskRaw = Get-Content -Raw -LiteralPath $taskRawPath | ConvertFrom-Json
    $taskStart = (Utc $taskRaw.started_utc).UtcDateTime.AddSeconds(-1).ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $taskEnd = [DateTime]::UtcNow.AddSeconds(1).ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
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
        $taskNs = [System.Xml.XmlNamespaceManager]::new($taskXml.NameTable)
        $taskNs.AddNamespace('e', 'http://schemas.microsoft.com/win/2004/08/events/event')
        foreach ($taskField in $taskXml.SelectNodes('//e:EventData/e:Data', $taskNs)) {
            $taskName = $taskField.GetAttribute('Name')
            if ($taskName) { $taskData[$taskName] = $taskField.InnerText }
        }
        $taskEventTime = Utc $taskXml.Event.System.TimeCreated.SystemTime
        if ($taskEventTime -lt (Utc $taskStart) -or $taskEventTime -gt (Utc $taskEnd)) { throw 'event outside typed UTC bounds' }
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
$taskDestination = Join-Path $taskDoc 'meth516_windows_terminal.json'
if (Test-Path -LiteralPath $taskDestination) { throw 'terminal metadata already exists' }
$taskResult | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $taskDestination -Encoding utf8
if (@($taskStages | Where-Object { -not $_.query_available -or @($_.relevant_events).Count -ne 0 }).Count -gt 0) { throw 'terminal event gate failed; receipt retained' }
Write-Output ('Typed UTC binding/main/audit queries passed; positive controls present; relevant events0.')
