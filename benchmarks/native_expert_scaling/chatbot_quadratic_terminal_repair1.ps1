# Administrative closure of source-null covariance or its new compiler phases.
param([ValidateSet('kernel','fit')][string]$Stage='kernel')
$ErrorActionPreference='Stop'
$taskDoc=(Resolve-Path (Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925')).Path
$taskOut=Join-Path $taskDoc ('chatbot_quadratic_'+$Stage+'_repair1_terminal_20261008.json')
if(Test-Path -LiteralPath $taskOut){throw 'Terminal receipt exists'}
function Utc($value) {
    if($value -is [DateTimeOffset]){return $value.ToUniversalTime()}
    if($value -is [DateTime]){return [DateTimeOffset]($value.ToUniversalTime())}
    return [DateTimeOffset]::Parse([string]$value,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
$taskControls=@(Get-WinEvent -LogName Application -FilterXPath '*[System[(EventID=1000) and (EventRecordID=179810 or EventRecordID=179791)]]' -ErrorAction Stop)
if($taskControls.Count -ne 2){throw 'Positive controls absent'}
$taskNames=if($Stage -eq 'kernel'){@('chatbot_quadratic_kernel_20261008.launcher_failure.json','chatbot_quadratic_kernel_repair1_20261008.terminal.json','chatbot_quadratic_kernel_repair1_audit_20261008.terminal.json')}else{@('chatbot_quadratic_fit_20261008.terminal.json','chatbot_quadratic_fit_audit_20261008.terminal.json')}
$taskResults=@()
foreach($taskName in $taskNames) {
    $taskPath=Join-Path $taskDoc $taskName
    $taskRaw=Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json -AsHashtable
    $taskStart=Utc $taskRaw.started_utc; $taskEnd=Utc $taskRaw.ended_utc
    $taskEvents=@()
    try {$taskEvents=@(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1000;StartTime=$taskStart.UtcDateTime;EndTime=$taskEnd.UtcDateTime} -ErrorAction Stop)}
    catch {if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){throw}}
    foreach($taskRole in @('launcher_instance','worker_instance')) {
        if(-not $taskRaw.ContainsKey($taskRole)){continue}
        $taskInstance=$taskRaw[$taskRole];$taskPIDValue=[int]$taskInstance.pid
        $taskUnix=[double]$taskInstance.create_time_unix
        $taskCreation=if($taskInstance.ContainsKey('creation_FILETIME')){[long]$taskInstance.creation_FILETIME}else{[long][Math]::Round($taskUnix*10000000+116444736000000000)}
        $taskTolerance=if($taskRole -eq 'worker_instance'){0}else{16}
        $taskMatches=@()
        foreach($taskEvent in $taskEvents) {
            [xml]$taskXML=$taskEvent.ToXml();$taskNS=[Xml.XmlNamespaceManager]::new($taskXML.NameTable)
            $taskNS.AddNamespace('e','http://schemas.microsoft.com/win/2004/08/events/event');$taskData=@{}
            foreach($taskField in $taskXML.SelectNodes('//e:EventData/e:Data',$taskNS)){$taskData[$taskField.GetAttribute('Name')]=$taskField.InnerText}
            if(-not $taskData.ContainsKey('ProcessId') -or -not $taskData.ContainsKey('ProcessCreationTime')){continue}
            $taskEventPID=[Convert]::ToInt64($taskData.ProcessId.Substring(2),16)
            $taskTicks=[Convert]::ToInt64($taskData.ProcessCreationTime.Substring(2),16)
            # Workers have exact held-handle FILETIME. Launcher psutil float loses
            # submicrosecond ticks, so explicitly allow <=16ticks only there.
            $taskTolerance=if($taskRole -eq 'worker_instance'){0}else{16}
            if($taskEventPID -eq $taskPIDValue -and [Math]::Abs($taskTicks-$taskCreation) -le $taskTolerance) {
                $taskFaultTime=Utc $taskXML.Event.System.TimeCreated.SystemTime
                if($taskFaultTime -lt $taskStart -or $taskFaultTime -gt $taskEnd){throw 'Typed UTC interval mismatch'}
                $taskMatches+=@{record_id=$taskEvent.RecordId;XML=$taskEvent.ToXml();typed_UTC_matches=$true}
            }
        }
        $taskLive=Get-Process -Id $taskPIDValue -ErrorAction SilentlyContinue
        $taskReused=$false
        if($taskLive) {
            $taskLiveUnix=([DateTimeOffset]$taskLive.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()/1000.0
            if([Math]::Abs($taskLiveUnix-$taskUnix) -lt .002){throw 'Owned instance still live'}
            $taskReused=$true
        }
        $taskResults+=@{report=$taskName;report_sha256=(Get-FileHash -LiteralPath $taskPath).Hash.ToLowerInvariant();
            role=$taskRole;pid=$taskPIDValue;create_time_unix=$taskUnix;creation_FILETIME=$taskCreation;
            FILETIME_tolerance_ticks=$taskTolerance;owned_instance_closed=$true;PID_reused=$taskReused;faults=$taskMatches}
    }
}
$taskResult=@{scope='ADMINISTRATIVE_TERMINAL_NO_SCIENTIFIC_REPLAY';instances=$taskResults;
    positive_control_ids=@($taskControls|ForEach-Object {$_.RecordId});positive_control_XML=@($taskControls|ForEach-Object {$_.ToXml()});
    new_source_J_acquisitions=0;new_original_BF16_full_forwards=0;
    metadata_UTC='Typed DateTime/DateTimeOffset conversion; worker exact FILETIME, launcher <=16ticks float tolerance'}
[IO.File]::WriteAllText($taskOut,($taskResult|ConvertTo-Json -Depth 15)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
Write-Output ('Quadratic transfer terminal closed '+$taskResults.Count+' known instances; matched OS faults '+@($taskResults|ForEach-Object {$_.faults}).Count+'. No derivative or source replay.')
