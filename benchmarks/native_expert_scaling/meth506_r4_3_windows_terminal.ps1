$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$destination=Join-Path $root 'results/native_expert_scaling/meth506_r4_3_windows_terminal.json'
if (Test-Path -LiteralPath $destination) { throw 'Exclusive UTC event receipt already exists' }
function Utc($v) {
    if ($v -is [DateTimeOffset]) { return $v.ToUniversalTime() }
    if ($v -is [DateTime]) { return [DateTimeOffset]$v.ToUniversalTime() }
    return [DateTimeOffset]::Parse([string]$v,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}
function Query([DateTimeOffset]$start,[DateTimeOffset]$end) {
    $s=$start.UtcDateTime.ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $e=$end.UtcDateTime.ToString('yyyy-MM-ddTHH:mm:ss.fffffffZ')
    $xpath="*[System[(EventID=1000) and TimeCreated[@SystemTime >= '$s' and @SystemTime <= '$e']]]"
    $available=$false;$errorText=$null;$events=@()
    try { $events=@(Get-WinEvent -LogName Application -FilterXPath $xpath -ErrorAction Stop);$available=$true }
    catch { if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $available=$true } else { $errorText=$_.Exception.Message } }
    $rows=@()
    foreach ($event in $events) {
        [xml]$xml=$event.ToXml();$time=Utc $xml.Event.System.TimeCreated.SystemTime
        if ($time -lt $start -or $time -gt $end) { throw 'UTC XPath returned outside requested interval' }
        $values=@{};foreach ($v in $xml.Event.EventData.Data) { $values[$v.Name]=$v.'#text' }
        $eventPid=0;$literal=[string]$values.ProcessId
        if ($literal -match '^0x') { $eventPid=[Convert]::ToInt32($literal.Substring(2),16) }
        elseif ($literal -match '^\d+$') { $eventPid=[int]$literal }
        $creation=$null;$literal=[string]$values.ProcessCreationTime
        if ($literal -match '^0x') { $creation=[DateTimeOffset][DateTime]::FromFileTimeUtc([Convert]::ToInt64($literal.Substring(2),16)) }
        $rows+=@{RecordId=$event.RecordId;time_utc=$time.ToString('o');pid=$eventPid;creation_utc=if ($creation) {$creation.ToString('o')} else {$null};creation_unix=if ($creation) {($creation-[DateTimeOffset]'1970-01-01T00:00:00Z').TotalSeconds} else {$null};data=$values;XML=$event.ToXml()}
    }
    return @{start_utc=$start.ToString('o');end_utc=$end.ToString('o');xpath=$xpath;query_available=$available;query_error=$errorText;events=$rows}
}
$legacyPath=Join-Path $root 'results/native_expert_scaling/meth506_r4_2_main_windows_terminal.json'
$legacy=Get-Content -LiteralPath $legacyPath -Raw -Encoding utf8 | ConvertFrom-Json
$known=@($legacy.other_events)
if ($known.Count -ne 2) { throw 'Expected retained positive-control event pair' }
$times=@($known | ForEach-Object { Utc $_.TimeUtc } | Sort-Object)
$control=Query $times[0].AddSeconds(-1) $times[-1].AddSeconds(1)
if (-not $control.query_available) { throw 'Positive UTC query unavailable' }
$ids=@($control.events | ForEach-Object { $_.RecordId })
foreach ($v in $known) { if ($v.RecordId -notin $ids) { throw 'Known actual Event1000 missing from UTC positive control' } }
$specs=@(
    @{name='prepareR3';path='docs/research/NATIVE_EXPERT_SCALING_20260925/meth506_r3_preparation_result.json'},
    @{name='originalMainAndAll578Commands';path='docs/research/NATIVE_EXPERT_SCALING_20260925/meth506_whole_result.failure.json'},
    @{name='qualityR4p2';path='docs/research/NATIVE_EXPERT_SCALING_20260925/meth506_r4_2_result.json'},
    @{name='auditR4p2';path='docs/research/NATIVE_EXPERT_SCALING_20260925/RETENTION_506_R4_4_20261006.json'})
$stages=@()
foreach ($spec in $specs) {
    $path=Join-Path $root $spec.path;$raw=Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
    $start=(Utc $raw.start_utc).AddSeconds(-2);$end=(Utc $raw.end_utc).AddSeconds(2)
    $instances=@(@{pid=[int]$raw.process_instance.pid;create_time_unix=$raw.process_instance.create_time_unix;creation_interval_utc=$raw.process_instance.creation_interval_utc;label=$spec.name;start_utc=$start.ToString('o');end_utc=$end.ToString('o')})
    # Recovery raw inherits old commands; actual original command scope is separate.
    if ($spec.name -eq 'originalMainAndAll578Commands') {
        foreach ($cmd in @($raw.commands)) {
            $instances+=@{pid=[int]$cmd.process_instance.pid;create_time_unix=[double]$cmd.process_instance.create_time_unix;label=$cmd.label;start_utc=(Utc $cmd.start_utc).AddSeconds(-2).ToString('o');end_utc=(Utc $cmd.end_utc).AddSeconds(2).ToString('o')}
        }
    }
    $q=Query $start $end;$matches=@()
    foreach ($event in $q.events) {
        $time=Utc $event.time_utc
        foreach ($instance in $instances) {
            if ($event.pid -eq $instance.pid -and $time -ge (Utc $instance.start_utc) -and $time -le (Utc $instance.end_utc)) {
                if ($null -ne $event.creation_unix -and $null -ne $instance.create_time_unix -and [Math]::Abs($event.creation_unix-[double]$instance.create_time_unix) -gt .002) { continue }
                if ($event.creation_utc -and $instance.creation_interval_utc) {
                    $born=Utc $event.creation_utc
                    if ($born -lt (Utc $instance.creation_interval_utc[0]) -or $born -gt (Utc $instance.creation_interval_utc[1])) { continue }
                }
                $matches+=@{event=$event;instance=$instance;creation_time_available=($null -ne $event.creation_unix)}
            }
        }
    }
    $stages+=@{name=$spec.name;raw_path=$path;raw_sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower();instances=$instances;query=$q;matching_scientific_events=$matches}
}
$result=@{experiment='METH506 R4.3 metadata-only actual UTC Windows requery';positive_control=$control;known_control_record_ids=@($known | ForEach-Object { $_.RecordId });stages=$stages;legacy_query_path=$legacyPath;legacy_query_sha256=(Get-FileHash -LiteralPath $legacyPath -Algorithm SHA256).Hash.ToLower();scope='Explicit Event XML UTC XPath with actual positive-control events; every returned timestamp checked. Retains original time-filter defect; no model/native/scientific replay or new quality/economic observation.'}
[IO.File]::WriteAllText($destination,($result | ConvertTo-Json -Depth 16),[Text.UTF8Encoding]::new($false))
$stages | ForEach-Object { @{stage=$_.name;available=$_.query.query_available;events=@($_.query.events).Count;matching=@($_.matching_scientific_events).Count} } | ConvertTo-Json
