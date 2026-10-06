$ErrorActionPreference='Stop'
$epochCheck=[DateTimeOffset]([DateTime]::FromFileTimeUtc(116444736000000000))
if (($epochCheck-[DateTimeOffset]'1970-01-01T00:00:00Z').TotalSeconds -ne 0) {throw 'UTC FileTime epoch conversion fault'}
$root=(Get-Location).Path
$doc=Join-Path $root 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$destination=Join-Path $doc 'meth510_r1_windows_terminal.json'
if (Test-Path -LiteralPath $destination) { throw 'Exclusive510 UTC event receipt exists' }
function Utc($v) { return [DateTimeOffset]::Parse([string]$v,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime() }
function Query([DateTimeOffset]$start,[DateTimeOffset]$end) {
    $s=$start.UtcDateTime.ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ');$e=$end.UtcDateTime.ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $xpath="*[System[(EventID=1000) and TimeCreated[@SystemTime >= '$s' and @SystemTime <= '$e']]]"
    $available=$false;$errorText=$null;$events=@()
    try { $events=@(Get-WinEvent -LogName Application -FilterXPath $xpath -ErrorAction Stop);$available=$true }
    catch { if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {$available=$true} else {$errorText=$_.Exception.Message} }
    $rows=@()
    foreach ($event in $events) {
        [xml]$xml=$event.ToXml();$time=Utc $xml.Event.System.TimeCreated.SystemTime
        if ($time -lt $start -or $time -gt $end) { throw 'UTC XPath returned outside interval' }
        $values=@{};foreach ($v in $xml.Event.EventData.Data) {$values[$v.Name]=$v.'#text'}
        $eventPid=0;$literal=[string]$values.ProcessId
        if ($literal -match '^0x') {$eventPid=[Convert]::ToInt32($literal.Substring(2),16)} elseif ($literal -match '^\d+$') {$eventPid=[int]$literal}
        $creation=$null;$literal=[string]$values.ProcessCreationTime
        if ($literal -match '^0x') {$creation=([DateTimeOffset]([DateTime]::FromFileTimeUtc([Convert]::ToInt64($literal.Substring(2),16)))-[DateTimeOffset]'1970-01-01T00:00:00Z').TotalSeconds}
        $rows+=@{RecordId=$event.RecordId;time_utc=$time.ToString('o');pid=$eventPid;create_time_unix=$creation;data=$values;XML=$event.ToXml()}
    }
    return @{start_utc=$start.ToString('o');end_utc=$end.ToString('o');xpath=$xpath;query_available=$available;query_error=$errorText;events=$rows}
}
$legacyPath=Join-Path $root 'results/native_expert_scaling/meth506_r4_2_main_windows_terminal.json'
$legacy=Get-Content -LiteralPath $legacyPath -Raw -Encoding utf8 | ConvertFrom-Json
$known=@($legacy.other_events);if ($known.Count -ne 2) {throw 'Expected earlier positive pair'}
$times=@($known | ForEach-Object {Utc $_.TimeUtc} | Sort-Object)
$positive=Query $times[0].AddSeconds(-1) $times[-1].AddSeconds(1)
if (-not $positive.query_available) {throw 'Positive UTC query unavailable'}
foreach ($v in $known) {if ($v.RecordId -notin @($positive.events | ForEach-Object {$_.RecordId})) {throw 'Known Event1000 missing'}}
$stages=@()
foreach ($kind in @('main','audit')) {
    $path=Join-Path $doc ('meth510_r1_'+$kind+'_result.json');$raw=Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
    $start=(Utc $raw.started_utc).AddSeconds(-2);$end=[DateTimeOffset]::UtcNow
    $instances=@(@{label=$kind;pid=$raw.process_instance.pid;create_time_unix=$raw.process_instance.create_time_unix;start_utc=$start.ToString('o');end_utc=$end.ToString('o')})
    if ($kind -eq 'main') {
        $first=Get-Content -LiteralPath (Join-Path $doc 'meth510_main_result.failure.json') -Raw -Encoding utf8 | ConvertFrom-Json
        $instances+=@{label='original first fault';pid=$first.process_instance.pid;create_time_unix=$first.process_instance.create_time_unix;start_utc=$first.started_utc;end_utc=$end.ToString('o')}
        $start=(Utc $first.started_utc).AddSeconds(-2)
    }
    foreach ($cmd in @($raw.commands)) {
        $instances+=@{label=$cmd.label;pid=$cmd.process_instance.pid;create_time_unix=$cmd.process_instance.create_time_unix;start_utc=(Utc $cmd.start_utc).AddSeconds(-2).ToString('o');end_utc=(Utc $cmd.end_utc).AddSeconds(2).ToString('o')}
        foreach ($child in @($cmd.descendants)) {$instances+=@{label=$cmd.label+' descendant';pid=$child.pid;create_time_unix=$child.create_time_unix;start_utc=(Utc $cmd.start_utc).AddSeconds(-2).ToString('o');end_utc=(Utc $cmd.end_utc).AddSeconds(2).ToString('o')}}
    }
    $q=Query $start $end;$matches=@()
    foreach ($event in $q.events) {foreach ($instance in $instances) {
        $time=Utc $event.time_utc
        if ($event.pid -eq $instance.pid -and $time -ge (Utc $instance.start_utc) -and $time -le (Utc $instance.end_utc)) {
            if ($null -ne $event.create_time_unix -and [Math]::Abs($event.create_time_unix-[double]$instance.create_time_unix) -gt .002) {continue}
            $matches+=@{event=$event;instance=$instance;creation_time_available=($null -ne $event.create_time_unix)}
        }
    }}
    $stages+=@{name=$kind;raw_path=$path;raw_sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower();instances=$instances;query=$q;matching_events=$matches}
}
$result=@{experiment='METH510 terminal metadata XML UTC Event1000';positive_control=$positive;known_record_ids=@($known | ForEach-Object {$_.RecordId});stages=$stages;new_numerical_calls=0}
[IO.File]::WriteAllText($destination,($result | ConvertTo-Json -Depth 18),[Text.UTF8Encoding]::new($false))
$stages | ForEach-Object {@{stage=$_.name;available=$_.query.query_available;matching=@($_.matching_events).Count}} | ConvertTo-Json
