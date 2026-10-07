$ErrorActionPreference = 'Stop'
$taskDoc = Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925'
function Utc($value) {
    if ($value -is [DateTimeOffset]) { return $value.ToUniversalTime() }
    if ($value -is [DateTime]) { return [DateTimeOffset]($value.ToUniversalTime()) }
    return [DateTimeOffset]::Parse([string]$value,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
$taskSpecs = @(
    @{kind='qwen'; file='chatbot_qwen_source_contract_20261007.json'; sha='258313ec5e4800615d30e44ed61fd9e90086a295205fb3322054117a3c3fc225'; pid=21408; created=1791369616.4518359},
    @{kind='gigachat'; file='chatbot_gigachat_source_contract_20261007.json'; sha='88f25eb6055c0b4e7b06c5c755d1c3aded3ab1ac05574febac29eee6d2aabc81'; pid=29664; created=1791369627.4306371}
)
$taskStages = @()
foreach ($taskSpec in $taskSpecs) {
    $taskPath = Join-Path $taskDoc $taskSpec.file
    if ((Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLower() -ne $taskSpec.sha) { throw 'wrong preflight report hash' }
    $taskRaw = Get-Content -Raw -LiteralPath $taskPath | ConvertFrom-Json
    if ($taskRaw.process_instance.pid -ne $taskSpec.pid -or [Math]::Abs($taskRaw.process_instance.create_time_unix - $taskSpec.created) -ge .002) { throw 'wrong preflight process identity' }
    if ($taskRaw.pipeline_complete -or $taskRaw.new_model_native_calls -ne 0 -or $taskRaw.new_tensor_value_bytes_read -ne 0) { throw 'wrong metadata-only scope' }
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
        $taskNs = [Xml.XmlNamespaceManager]::new($taskXml.NameTable)
        $taskNs.AddNamespace('e','http://schemas.microsoft.com/win/2004/08/events/event')
        $taskData = @{}
        foreach ($taskField in $taskXml.SelectNodes('//e:EventData/e:Data',$taskNs)) { $taskData[$taskField.GetAttribute('Name')] = $taskField.InnerText }
        $taskTime = Utc $taskXml.Event.System.TimeCreated.SystemTime
        if ($taskTime -lt (Utc $taskStart) -or $taskTime -gt (Utc $taskEnd)) { throw 'event outside UTC query' }
        $taskPid = if ($taskData.ProcessId -like '0x*') { [Convert]::ToInt64($taskData.ProcessId.Substring(2),16) } else { [long]$taskData.ProcessId }
        $taskCreated = $null
        if ($taskData.ProcessCreationTime -like '0x*') { $taskCreated = ([DateTimeOffset][DateTime]::FromFileTimeUtc([Convert]::ToInt64($taskData.ProcessCreationTime.Substring(2),16))).ToUnixTimeMilliseconds()/1000.0 }
        if ($taskPid -eq $taskSpec.pid -and $null -ne $taskCreated -and [Math]::Abs($taskCreated - $taskSpec.created) -lt .002) { $taskMatches += @{record_id=$taskEvent.RecordId; xml=$taskEvent.ToXml()} }
    }
    $taskProc = Get-Process -Id $taskSpec.pid -ErrorAction SilentlyContinue
    if ($taskProc) {
        $taskCurrentCreated = ([DateTimeOffset]$taskProc.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()/1000.0
        if ([Math]::Abs($taskCurrentCreated - $taskSpec.created) -lt .002) { throw 'owned preflight process still live' }
    }
    $taskStages += @{kind=$taskSpec.kind;raw_sha256=$taskSpec.sha;process_instance=$taskRaw.process_instance;closed=$true;query_available=$taskAvailable;query_error=$taskError;xpath=$taskXPath;relevant_events=$taskMatches}
}
$taskControls = @(Get-WinEvent -LogName Application -FilterXPath '*[System[(EventID=1000) and (EventRecordID=179810 or EventRecordID=179791)]]' -ErrorAction Stop)
if (@($taskControls).Count -ne 2) { throw 'positive event controls missing' }
foreach ($taskControl in $taskControls) { [xml]$taskXml = $taskControl.ToXml(); [void](Utc $taskXml.Event.System.TimeCreated.SystemTime) }
$taskResult = @{stages=$taskStages;positive_control_ids=@($taskControls | ForEach-Object {$_.RecordId});positive_control_xml=@($taskControls | ForEach-Object {$_.ToXml()});new_numerical_model_C_calls=0}
$taskDestination = Join-Path $taskDoc 'chatbot_preflight_windows_terminal_20261007.json'
if (Test-Path -LiteralPath $taskDestination) { throw 'metadata terminal already exists' }
$taskResult | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $taskDestination -Encoding utf8
if (@($taskStages | Where-Object {-not $_.query_available -or @($_.relevant_events).Count -ne 0}).Count -gt 0) { throw 'event gate fault retained' }
Write-Output 'Both metadata inspections closed; typed UTC queries available, positive controls present, zero relevant Event1000.'
