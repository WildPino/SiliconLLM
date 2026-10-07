# Administrative closure of known NEW census instances; no operator replay.
$ErrorActionPreference='Stop'
$taskDoc=Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskDestination=Join-Path $taskDoc 'chatbot_operator_windows_terminal_20261007.json'
if(Test-Path -LiteralPath $taskDestination){throw 'Terminal namespace exists'}
function Utc($value) {
    if($value -is [DateTimeOffset]){return $value.ToUniversalTime()}
    if($value -is [DateTime]){return [DateTimeOffset]($value.ToUniversalTime())}
    return [DateTimeOffset]::Parse([string]$value,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
$taskSpecs=@(
    @('chatbot_qwen_operator_census_20261007.json','a4293870cb38706fc60d94562d9579653ddff1a1387c366c307a56b4b262bbc2'),
    @('chatbot_gigachat_operator_census_repair1_20261007.json','94e8a43db9b64d691ff9531b3cfd8eeb40b8062e1eb24f4198a11c0b5ba5236a')
)
$taskQueries=@()
foreach($taskSpec in $taskSpecs) {
    $taskPath=Join-Path $taskDoc $taskSpec[0]
    if((Get-FileHash -LiteralPath $taskPath).Hash.ToLowerInvariant() -ne $taskSpec[1]){throw 'Report SHA'}
    $taskRaw=Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json
    $taskInstance=$taskRaw.process_instance
    $taskLive=Get-Process -Id $taskInstance.pid -ErrorAction SilentlyContinue
    if($taskLive) {
        $taskLiveCreate=([DateTimeOffset]$taskLive.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()/1000.0
        if([Math]::Abs($taskLiveCreate-$taskInstance.create_time_unix) -lt .002){throw 'Owned instance still live'}
    }
    $taskStart=(Utc $taskRaw.started_utc).UtcDateTime.AddSeconds(-1).ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $taskEnd=[DateTime]::UtcNow.AddSeconds(1).ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $taskXPath="*[System[(EventID=1000) and TimeCreated[@SystemTime >= '$taskStart' and @SystemTime <= '$taskEnd']]]"
    $taskAvailable=$true; $taskError=$null; $taskEvents=@(); $taskMatches=@()
    try {$taskEvents=@(Get-WinEvent -LogName Application -FilterXPath $taskXPath -ErrorAction Stop)}
    catch {if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){$taskAvailable=$false;$taskError=$_.ToString()}}
    foreach($taskEvent in $taskEvents) {
        [xml]$taskXml=$taskEvent.ToXml(); $taskData=@{}
        $taskNs=[Xml.XmlNamespaceManager]::new($taskXml.NameTable)
        $taskNs.AddNamespace('e','http://schemas.microsoft.com/win/2004/08/events/event')
        foreach($taskField in $taskXml.SelectNodes('//e:EventData/e:Data',$taskNs)){
            $taskName=$taskField.GetAttribute('Name'); if($taskName){$taskData[$taskName]=$taskField.InnerText}
        }
        $taskTime=Utc $taskXml.Event.System.TimeCreated.SystemTime
        if($taskTime -lt (Utc $taskStart) -or $taskTime -gt (Utc $taskEnd)){throw 'UTC query interval'}
        $taskPid=if($taskData.ProcessId -like '0x*'){[Convert]::ToInt64($taskData.ProcessId.Substring(2),16)}else{[long]$taskData.ProcessId}
        $taskCreate=$null
        if($taskData.ProcessCreationTime -like '0x*'){
            $taskCreate=([DateTimeOffset][DateTime]::FromFileTimeUtc([Convert]::ToInt64($taskData.ProcessCreationTime.Substring(2),16))).ToUnixTimeMilliseconds()/1000.0
        }
        if($taskPid -eq $taskInstance.pid -and $null -ne $taskCreate -and [Math]::Abs($taskCreate-$taskInstance.create_time_unix) -lt .002){
            $taskMatches+=@{record_id=$taskEvent.RecordId;XML=$taskEvent.ToXml()}
        }
    }
    $taskQueries+=@{report=$taskSpec[0];report_sha256=$taskSpec[1];process_instance=$taskInstance;owned_instance_closed=$true;
                    query_available=$taskAvailable;query_error=$taskError;xpath=$taskXPath;relevant_events=$taskMatches}
}
$taskControls=@(Get-WinEvent -LogName Application -FilterXPath '*[System[(EventID=1000) and (EventRecordID=179810 or EventRecordID=179791)]]' -ErrorAction Stop)
if($taskControls.Count -ne 2){throw 'Positive controls absent'}
$taskResult=@{queries=$taskQueries;positive_control_ids=@($taskControls | ForEach-Object {$_.RecordId});positive_control_XML=@($taskControls | ForEach-Object {$_.ToXml()});
    first_Giga_fault_instance='UNKNOWN: first serializer discarded actor/resource report; no retrospective instance closure or peak claim';
    resource_scope='Raw OS peaks sampled before final JSON serialization. Whole-job final OS peak was not retained. Executor wall spans bound duration but do not supply OS peak.';
    new_scientific_calls=0; new_tensor_values=0}
[IO.File]::WriteAllText((Resolve-Path $taskDoc).Path+'/chatbot_operator_windows_terminal_20261007.json',($taskResult | ConvertTo-Json -Depth 10)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
if(@($taskQueries | Where-Object {-not $_.query_available -or @($_.relevant_events).Count -ne 0}).Count){throw 'Terminal gate failure; retained'}
Write-Output 'Both known census instances closed; typed UTC Event1000 queries available and zero relevant faults; two positive controls.'
