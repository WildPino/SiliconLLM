# Administrative OS-instance closure only; never render/tokenize or invoke a model.
$ErrorActionPreference='Stop'
$taskDoc=(Resolve-Path (Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925')).Path
$taskOutput=Join-Path $taskDoc 'chatbot_interaction_windows_terminal_20261007.json'
if(Test-Path -LiteralPath $taskOutput){throw 'Terminal namespace exists'}
function Utc($value) {
    if($value -is [DateTimeOffset]){return $value.ToUniversalTime()}
    if($value -is [DateTime]){return [DateTimeOffset]($value.ToUniversalTime())}
    return [DateTimeOffset]::Parse([string]$value,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
$taskNames=@('chatbot_qwen_interaction_20261007.launcher_failure.json',
    'chatbot_qwen_interaction_repair1_20261007.launcher_failure.json',
    'chatbot_qwen_interaction_repair2_20261007.launcher_failure.json',
    'chatbot_qwen_interaction_repair3_20261007.terminal.json')
$taskQueries=@()
foreach($taskName in $taskNames) {
    $taskPath=Join-Path $taskDoc $taskName
    $taskSHA=(Get-FileHash -LiteralPath $taskPath).Hash.ToLowerInvariant()
    $taskRaw=Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json
    foreach($taskRole in @('launcher_instance','worker_instance')) {
        $taskInstance=$taskRaw.$taskRole
        if($null -eq $taskInstance){continue}
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
            foreach($taskField in $taskXml.SelectNodes('//e:EventData/e:Data',$taskNs)) {
                $taskFieldName=$taskField.GetAttribute('Name'); if($taskFieldName){$taskData[$taskFieldName]=$taskField.InnerText}
            }
            $taskTime=Utc $taskXml.Event.System.TimeCreated.SystemTime
            if($taskTime -lt (Utc $taskStart) -or $taskTime -gt (Utc $taskEnd)){throw 'UTC query interval'}
            $taskPid=if($taskData.ProcessId -like '0x*'){[Convert]::ToInt64($taskData.ProcessId.Substring(2),16)}else{[long]$taskData.ProcessId}
            $taskCreate=$null
            if($taskData.ProcessCreationTime -like '0x*') {
                $taskCreate=([DateTimeOffset][DateTime]::FromFileTimeUtc([Convert]::ToInt64($taskData.ProcessCreationTime.Substring(2),16))).ToUnixTimeMilliseconds()/1000.0
            }
            if($taskPid -eq $taskInstance.pid -and $null -ne $taskCreate -and [Math]::Abs($taskCreate-$taskInstance.create_time_unix) -lt .002) {
                $taskMatches+=@{record_id=$taskEvent.RecordId;XML=$taskEvent.ToXml()}
            }
        }
        $taskQueries+=@{report=$taskName;report_sha256=$taskSHA;role=$taskRole;process_instance=$taskInstance;
            owned_instance_closed=$true;query_available=$taskAvailable;query_error=$taskError;xpath=$taskXPath;relevant_events=$taskMatches}
    }
}
$taskControls=@(Get-WinEvent -LogName Application -FilterXPath '*[System[(EventID=1000) and (EventRecordID=179810 or EventRecordID=179791)]]' -ErrorAction Stop)
if($taskControls.Count -ne 2){throw 'Positive controls absent'}
$taskResult=@{queries=$taskQueries;positive_control_ids=@($taskControls | ForEach-Object {$_.RecordId});
    positive_control_XML=@($taskControls | ForEach-Object {$_.ToXml()});new_scientific_calls=0;new_tensor_values=0;
    scope='Known seven interaction launcher/worker OS instances only. Failed Python assertions are retained separately; Event1000 absence does not convert them into passes.'}
[IO.File]::WriteAllText($taskOutput,($taskResult | ConvertTo-Json -Depth 10)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
if($taskQueries.Count -ne 7 -or @($taskQueries | Where-Object {-not $_.query_available -or @($_.relevant_events).Count -ne 0}).Count){throw 'Terminal gate failure; retained'}
Write-Output 'Seven known interaction OS instances closed; typed UTC Event1000 queries available and zero relevant faults; two positive controls.'
