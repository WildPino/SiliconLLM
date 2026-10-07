param([Parameter(Mandatory=$true)][string]$ProofSha)
$ErrorActionPreference='Stop'
$taskDoc=Join-Path $PSScriptRoot '../../docs/research/NATIVE_EXPERT_SCALING_20260925'
function Utc($v) {
    if($v -is [DateTimeOffset]){return $v.ToUniversalTime()}
    if($v -is [DateTime]){return [DateTimeOffset]($v.ToUniversalTime())}
    return [DateTimeOffset]::Parse([string]$v,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
$taskSpecs=@(
    @('prefix_bound_result.failure','d33e082af7ca64f9ef94411a5de6cd1afcc5349df1d06cd5c35ae090add14d00'),
    @('prefix_bound_repair1_result.failure','57864c460164a817fae14811a12d62c9edfac444bfc3d96678fb2ccf39fbae68'),
    @('prefix_bound_repair2_result.failure','df05d59ab32e2d5553ed38a6c056d1675b4c15bcdb535104401ba687d659af62'),
    @('prefix_bound_repair3_result.failure','8ec7be567481dd06930d16587af68a31b3f4defb3cd82d84423e51e7b99d270e'),
    @('prefix_compiler_diag_result','d6592c623c07178068cd7400f09981395498c16204cbcffc0ca7cd54b350f955'),
    @('prefix_bound_repair4_result','81d5454195ce1ac9dae9bf791b31bd3edd13dc5fa4c77d6e1679e7d645a06f03'),
    @('prefix_energy_audit_repair4_result','8872d2df4c0d2b9d28804f586cd8857723cac5cc1cdf74166c0bf6642cb2ed6b'),
    @('rare_gate_result.failure','cebedb62ba73a54c945b29eb6b5064dd2a03d0a86c6311ec47e28156a1874124'),
    @('rare_gate_repair1_result.failure','bd17b8410ba1e68738eb9a1bba9aeace777f6c5a70d1e95085a8a7e7562b1d63'),
    @('rare_metadata_proof_result',$ProofSha)
)
$taskQueries=@()
foreach($taskSpec in $taskSpecs) {
    $taskRawPath=Join-Path $taskDoc ('meth528_'+$taskSpec[0]+'.json')
    if((Get-FileHash -LiteralPath $taskRawPath).Hash.ToLower() -ne $taskSpec[1]){throw 'raw SHA'}
    $taskRaw=Get-Content -LiteralPath $taskRawPath -Raw | ConvertFrom-Json
    $taskInstances=@($taskRaw.process_instance)+@($taskRaw.compiler.observed_instances | Where-Object {$null -ne $_})
    if($taskSpec[0] -eq 'prefix_compiler_diag_result') {$taskInstances+=@($taskRaw.root)+@($taskRaw.observed_descendants)}
    $taskStart=(Utc $taskRaw.started_utc).UtcDateTime.AddSeconds(-1).ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $taskEnd=[DateTime]::UtcNow.AddSeconds(1).ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $taskXPath="*[System[(EventID=1000) and TimeCreated[@SystemTime >= '$taskStart' and @SystemTime <= '$taskEnd']]]"
    $taskAvailable=$true;$taskError=$null;$taskEvents=@()
    try {$taskEvents=@(Get-WinEvent -LogName Application -FilterXPath $taskXPath -ErrorAction Stop)}
    catch {if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){$taskAvailable=$false;$taskError=$_.ToString()}}
    foreach($taskInstance in $taskInstances) {
        $taskLive=Get-Process -Id $taskInstance.pid -ErrorAction SilentlyContinue
        if($taskLive) {
            $taskLiveCreate=([DateTimeOffset]$taskLive.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()/1000.0
            if([Math]::Abs($taskLiveCreate-$taskInstance.create_time_unix)<.002){throw 'owned process still live'}
        }
        $taskMatches=@()
        foreach($taskEvent in $taskEvents) {
            [xml]$taskXml=$taskEvent.ToXml();$taskData=@{}
            $taskNs=[System.Xml.XmlNamespaceManager]::new($taskXml.NameTable);$taskNs.AddNamespace('e','http://schemas.microsoft.com/win/2004/08/events/event')
            foreach($taskField in $taskXml.SelectNodes('//e:EventData/e:Data',$taskNs)){$taskName=$taskField.GetAttribute('Name');if($taskName){$taskData[$taskName]=$taskField.InnerText}}
            $taskTime=Utc $taskXml.Event.System.TimeCreated.SystemTime
            if($taskTime -lt (Utc $taskStart) -or $taskTime -gt (Utc $taskEnd)){throw 'UTC interval'}
            $taskPid=if($taskData.ProcessId -like '0x*'){[Convert]::ToInt64($taskData.ProcessId.Substring(2),16)}else{[long]$taskData.ProcessId}
            $taskCreate=$null
            if($taskData.ProcessCreationTime -like '0x*'){$taskCreate=([DateTimeOffset][DateTime]::FromFileTimeUtc([Convert]::ToInt64($taskData.ProcessCreationTime.Substring(2),16))).ToUnixTimeMilliseconds()/1000.0}
            if($taskPid -eq $taskInstance.pid -and $null -ne $taskCreate -and [Math]::Abs($taskCreate-$taskInstance.create_time_unix)<.002){$taskMatches+=@{RecordId=$taskEvent.RecordId;XML=$taskEvent.ToXml()}}
        }
        $taskQueries+=@{stage=$taskSpec[0];raw_sha256=$taskSpec[1];process_instance=$taskInstance;query_available=$taskAvailable;query_error=$taskError;xpath=$taskXPath;relevant_events=$taskMatches;owned_instance_closed=$true}
    }
}
$taskControls=@(Get-WinEvent -LogName Application -FilterXPath '*[System[(EventID=1000) and (EventRecordID=179810 or EventRecordID=179791)]]' -ErrorAction Stop)
if($taskControls.Count -ne 2){throw 'positive controls'}
$taskResult=@{queries=$taskQueries;positive_control_ids=@($taskControls | ForEach-Object {$_.RecordId});positive_control_XML=@($taskControls | ForEach-Object {$_.ToXml()});first_two_fault_compiler_descendants='UNKNOWN identities/peaks/exits; initial family resource qualification remains FALSE';new_scientific_calls=0}
$taskDestination=Join-Path $taskDoc 'meth528_certificate_windows_terminal_20261007.json'
if(Test-Path -LiteralPath $taskDestination){throw 'terminal exists'}
$taskResult | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $taskDestination -Encoding utf8
if(@($taskQueries | Where-Object {-not $_.query_available -or @($_.relevant_events).Count -ne 0}).Count){throw 'terminal gate failure; retained'}
Write-Output ('All known '+$taskQueries.Count+' owned instances closed; typed UTC queries available; relevant Event1000=0; two positive controls present.')
