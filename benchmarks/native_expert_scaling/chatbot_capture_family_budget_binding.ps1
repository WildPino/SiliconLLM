# Prospective monitor budget correction BEFORE any source observation.
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskOld=Join-Path $taskDoc 'chatbot_original_capture_binding_20261007.json'
$taskOut=Join-Path $taskDoc 'chatbot_original_capture_binding_repair1_20261007.json'
if(Test-Path -LiteralPath $taskOut){throw 'Binding namespace exists'}
if((Get-FileHash -LiteralPath $taskOld).Hash.ToLowerInvariant() -ne 'f9cdadde1fb6d293a7c3d674033fde9ce7124114b5b9cba274d9d94017bfeeae'){throw 'Original binding SHA'}
$taskBinding=Get-Content -LiteralPath $taskOld -Raw | ConvertFrom-Json -AsHashtable
function Entry($path) {
    $taskStream=[IO.File]::OpenRead($path);$taskLength=$taskStream.Length;$taskStream.Dispose()
    $taskInfo=Get-Item -LiteralPath $path
    $taskResolved=if($taskInfo.LinkType -eq 'SymbolicLink'){[IO.Path]::GetFullPath($taskInfo.Target)}else{$taskInfo.FullName}
    return @{path=$taskInfo.FullName;resolved_path=$taskResolved;bytes=$taskLength;sha256=(Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant()}
}
$taskChanges=@()
foreach($taskRel in @('benchmarks/native_expert_scaling/chatbot_capture_launch.py','docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_SOURCE_CAPTURE_PROTOCOL_20261007.md')) {
    $taskPath=Join-Path $taskRoot $taskRel
    $taskPrevious=@($taskBinding.inputs | Where-Object {$_.path -eq $taskPath})
    if($taskPrevious.Count -ne 1){throw 'Bound path cardinality'}
    $taskNew=Entry $taskPath
    $taskChanges+=@{path=$taskPath;old_sha=$taskPrevious[0].sha256;new_sha=$taskNew.sha256}
    $taskBinding.inputs=@($taskBinding.inputs | ForEach-Object {if($_.path -eq $taskPath){$taskNew}else{$_}})
}
$taskBinding.inputs+=@(Entry $taskOld;Entry $PSCommandPath)
$taskBinding.prospective_budget_change=@{stage='BEFORE_FIRST_SOURCE_VALUES_OR_MODEL_CALLS';worker_seconds=300;old_family_seconds=360;new_family_seconds=1200;
    reason='Administrative whole-runtime tree contains15994 files/4432800404B; price both before/after seals separately from worker';changes=$taskChanges}
[IO.File]::WriteAllText($taskOut,($taskBinding | ConvertTo-Json -Depth 20)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
Get-FileHash -LiteralPath $taskOut | Format-List Path,Hash
