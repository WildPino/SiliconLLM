param(
    [Parameter(Mandatory=$true)][string]$Step,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference='Stop'
$taskTimer=[Diagnostics.Stopwatch]::StartNew()
if(-not [BitConverter]::IsLittleEndian){throw 'Little-endian metadata required'}
if(Test-Path -LiteralPath $Out){throw 'Do not replace an existing observation'}
function Get-Extent([string]$TaskPath){
    $taskFile=Get-Item -LiteralPath $TaskPath
    return [ordered]@{path=$taskFile.FullName;bytes=$taskFile.Length;sha256=(Get-FileHash -LiteralPath $TaskPath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Check-Array($TaskItem){
    $taskGot=Get-Extent $TaskItem.path
    if($taskGot.bytes -ne $TaskItem.bytes -or $taskGot.sha256 -cne $TaskItem.sha256){throw 'Producer metadata array extent mismatch'}
    return $taskGot
}
$taskInputs=[Collections.Generic.List[object]]::new()
$taskInputs.Add((Get-Extent $PSCommandPath));$taskInputs.Add((Get-Extent $Step))
$taskRecord=Get-Content -LiteralPath $Step -Raw | ConvertFrom-Json
if($taskRecord.ordinal -ne 1 -or -not $taskRecord.history.reused){throw 'This observation is restricted to the first reused qualified gradient'}
$taskRoute=$taskRecord.history.artifacts.forward_route_ids
if($taskRoute.dtype -ne '<i4' -or $taskRoute.shape[1] -ne 6 -or $taskRoute.shape[2] -ne 8){throw 'Unexpected route metadata shape'}
$taskInputs.Add((Check-Array $taskRoute));$taskRouteBytes=[IO.File]::ReadAllBytes($taskRoute.path)
$taskSets=@(0..5 | ForEach-Object { [Collections.Generic.HashSet[int]]::new() })
for($taskPosition=0;$taskPosition -lt $taskRoute.shape[0];$taskPosition++){
    for($taskSite=0;$taskSite -lt 6;$taskSite++){
        for($taskRank=0;$taskRank -lt 8;$taskRank++){
            $taskId=[BitConverter]::ToInt32($taskRouteBytes,4*(($taskPosition*6+$taskSite)*8+$taskRank))
            if($taskId -lt 0 -or $taskId -ge 1152){throw 'Route ID outside declared bank'}
            [void]$taskSets[$taskSite].Add($taskId)
        }
    }
}
$taskRows=[Collections.Generic.List[object]]::new()
foreach($taskTrial in $taskRecord.trials){
    for($taskSite=0;$taskSite -lt 6;$taskSite++){
        foreach($taskOrgan in @('weight','bias')){
            $taskName="layers.$taskSite.bank.router.$taskOrgan"
            $taskArray=$taskTrial.group_stats.$taskName
            if($taskArray.dtype -ne '<f8' -or $taskArray.shape[0] -ne 1152 -or $taskArray.shape[1] -ne 6){throw 'Unexpected group-stat metadata shape'}
            $taskInputs.Add((Check-Array $taskArray));$taskData=[IO.File]::ReadAllBytes($taskArray.path)
            $taskExcludedNonzero=0;$taskExcludedMoved=0;$taskExcludedMovedCoefficients=0.0;$taskMin=[double]::PositiveInfinity;$taskMax=0.0;$taskRadiusMax=0.0
            for($taskId=0;$taskId -lt 1152;$taskId++){
                if($taskSets[$taskSite].Contains($taskId)){continue}
                $taskGradient=[BitConverter]::ToDouble($taskData,$taskId*48+8)
                $taskRadius=[BitConverter]::ToDouble($taskData,$taskId*48+16)
                $taskDisplacement=[BitConverter]::ToDouble($taskData,$taskId*48+24)
                $taskMoved=[BitConverter]::ToDouble($taskData,$taskId*48+40)
                if(-not [double]::IsFinite($taskGradient) -or $taskGradient -lt 0 -or $taskRadius -le 0 -or -not [double]::IsFinite($taskDisplacement) -or $taskMoved -lt 0){throw 'Invalid group-stat value'}
                if($taskGradient -gt 0){$taskExcludedNonzero++;$taskMin=[Math]::Min($taskMin,$taskGradient);$taskMax=[Math]::Max($taskMax,$taskGradient)}
                if($taskMoved -gt 0){$taskExcludedMoved++;$taskExcludedMovedCoefficients+=$taskMoved}
                $taskRadiusMax=[Math]::Max($taskRadiusMax,$taskDisplacement/$taskRadius)
            }
            $taskRows.Add([ordered]@{alpha=$taskTrial.alpha;site=$taskSite;organ=$taskOrgan;selected_union=$taskSets[$taskSite].Count;excluded_groups=1152-$taskSets[$taskSite].Count;
                excluded_nonzero_gradient_norms=$taskExcludedNonzero;excluded_moved_groups=$taskExcludedMoved;excluded_moved_coefficients=$taskExcludedMovedCoefficients;
                excluded_positive_gradient_norm_min=$(if($taskExcludedNonzero){$taskMin}else{$null});excluded_gradient_norm_max=$taskMax;excluded_relative_displacement_max=$taskRadiusMax})
            if($taskTimer.Elapsed.TotalSeconds -gt 10){throw 'Metadata observation exceeded10s cap'}
        }
    }
}
foreach($taskItem in $taskInputs){$taskCheck=Get-Extent $taskItem.path;if($taskCheck.sha256 -cne $taskItem.sha256 -or $taskCheck.bytes -ne $taskItem.bytes){throw 'Metadata changed while observing'}}
$taskResult=[ordered]@{schema='CATEGORICAL_ROUTER_SUPPORT_METADATA_OBSERVATION_V1';case=$taskRecord.id;inputs=$taskInputs;rows=$taskRows;
    seconds=$taskTimer.Elapsed.TotalSeconds;OS_peak=[Diagnostics.Process]::GetCurrentProcess().PeakWorkingSet64;model_loads=0;history_forwards=0;backwards=0;native_calls=0;GPU_calls=0;
    all_input_extents_exact_before_after=$true;gradient_and_displacement_bits_independently_read=$false;
    scope='Small metadata inspection during held producer monitoring. Counts derive from completed producer group statistics and route IDs, not a new tensor/numerical benchmark. Full campaign audit must independently verify actual gradients and reconstructed displacement bits. No change to consumed criteria.'}
$taskRaw=$taskResult | ConvertTo-Json -Depth 8
$taskStream=[IO.File]::Open([IO.Path]::GetFullPath($Out),[IO.FileMode]::CreateNew)
try{$taskWriter=[IO.StreamWriter]::new($taskStream,[Text.UTF8Encoding]::new($false));$taskWriter.WriteLine($taskRaw);$taskWriter.Flush();$taskStream.Flush($true)}finally{if($taskWriter){$taskWriter.Dispose()}else{$taskStream.Dispose()}}
$taskRows | Select-Object alpha,site,organ,selected_union,excluded_groups,excluded_nonzero_gradient_norms,excluded_moved_groups,excluded_relative_displacement_max | ConvertTo-Json
