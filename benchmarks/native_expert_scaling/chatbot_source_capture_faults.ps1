# Retain actual Arrow Event1000 faults, matching PID AND exact creation FILETIME.
$ErrorActionPreference='Stop'
$taskDoc=(Resolve-Path (Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925')).Path
$taskOut=Join-Path $taskDoc 'chatbot_source_capture_windows_faults_20261007.json'
if(Test-Path -LiteralPath $taskOut){throw 'Fault receipt exists'}
function Utc($value) {
    if($value -is [DateTimeOffset]){return $value.ToUniversalTime()}
    if($value -is [DateTime]){return [DateTimeOffset]($value.ToUniversalTime())}
    return [DateTimeOffset]::Parse([string]$value,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
$taskControls=@(Get-WinEvent -LogName Application -FilterXPath '*[System[(EventID=1000) and (EventRecordID=179810 or EventRecordID=179791)]]' -ErrorAction Stop)
if($taskControls.Count -ne 2){throw 'Positive controls absent'}
$taskSpecs=@(@('chatbot_original_capture_20261007.launcher_failure.json',180220),@('chatbot_original_capture_repair2_20261007.launcher_failure.json',180239))
$taskResults=@();$taskInstances=@()
foreach($taskSpec in $taskSpecs) {
    $taskReport=Join-Path $taskDoc $taskSpec[0]
    $taskRaw=Get-Content -LiteralPath $taskReport -Raw | ConvertFrom-Json -AsHashtable
    $taskEvent=@(Get-WinEvent -LogName Application -FilterXPath ("*[System[(EventID=1000) and (EventRecordID="+$taskSpec[1]+")]]") -ErrorAction Stop)
    if($taskEvent.Count -ne 1){throw 'Fault record cardinality'}
    [xml]$taskXML=$taskEvent[0].ToXml();$taskData=@{}
    $taskNS=[Xml.XmlNamespaceManager]::new($taskXML.NameTable);$taskNS.AddNamespace('e','http://schemas.microsoft.com/win/2004/08/events/event')
    foreach($taskField in $taskXML.SelectNodes('//e:EventData/e:Data',$taskNS)){$taskData[$taskField.GetAttribute('Name')]=$taskField.InnerText}
    $taskPIDValue=[Convert]::ToInt64($taskData.ProcessId.Substring(2),16)
    $taskTicks=[Convert]::ToInt64($taskData.ProcessCreationTime.Substring(2),16)
    if($taskPIDValue -ne $taskRaw.worker_instance.pid -or $taskTicks -ne $taskRaw.worker_instance.creation_FILETIME){throw 'PID/creation mismatched'}
    if($taskData.ModuleName -ne 'arrow.dll' -or $taskData.ExceptionCode -ne 'c0000005'){throw 'Unexpected actual fault'}
    $taskFaultTime=Utc $taskXML.Event.System.TimeCreated.SystemTime
    $taskStart=Utc $taskRaw.started_utc
    $taskEnd=Utc $taskRaw.ended_utc
    if($taskFaultTime -lt $taskStart -or $taskFaultTime -gt $taskEnd){throw 'Typed UTC fault interval'}
    $taskResults+=@{report=$taskSpec[0];report_sha256=(Get-FileHash -LiteralPath $taskReport).Hash.ToLowerInvariant();record_id=$taskSpec[1];
        pid=$taskPIDValue;creation_FILETIME=$taskTicks;typed_UTC_interval_matches=$true;XML=$taskEvent[0].ToXml()}
    foreach($taskRole in @('launcher_instance','worker_instance')) {
        $taskInstance=$taskRaw[$taskRole];$taskInstances+=@{role=$taskRole;pid=$taskInstance.pid;create_time=$taskInstance.create_time_unix}
    }
    foreach($taskChild in $taskRaw.unexpected_descendants){$taskInstances+=@{role='known native descendant';pid=$taskChild.pid;create_time=$taskChild.create_time}}
}
foreach($taskInstance in $taskInstances) {
    $taskLive=Get-Process -Id $taskInstance.pid -ErrorAction SilentlyContinue
    if($taskLive) {
        $taskCreate=([DateTimeOffset]$taskLive.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()/1000.0
        if([Math]::Abs($taskCreate-$taskInstance.create_time) -lt .002){throw 'Owned instance still live'}
        $taskInstance.reused_PID_with_other_creation=$true
    }
    $taskInstance.owned_instance_closed=$true
}
$taskResult=@{faults=$taskResults;known_instances=$taskInstances;positive_control_ids=@($taskControls|ForEach-Object {$_.RecordId});
    positive_control_XML=@($taskControls|ForEach-Object {$_.ToXml()});new_source_forwards=0;first_unknown_descendant_identity_and_peak='UNKNOWN; no retrospective family resource admission';
    decision='BOTH_ORIGINAL_WORKERS_NATIVE_ARROW_IMPORT_FAULT; FAMILIES_FAILED'}
$taskResult.metadata_date_repair='c459ee first administrative UTC cast failure retained; use typed DateTime/DateTimeOffset conversion, not string-cast UTC dates'
[IO.File]::WriteAllText($taskOut,($taskResult|ConvertTo-Json -Depth 12)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
Write-Output 'Two Arrow c0000005 OS faults match exact owned worker PID/creation and typed UTC; six known instances closed; original families remain failed.'
