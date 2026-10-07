# Correct metadata for existing symlink inputs; no fixture or worker execution.
$ErrorActionPreference='Stop'
$taskRoot=(Get-Location).Path
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskOriginal=Join-Path $taskDoc 'chatbot_interaction_binding_20261007.json'
$taskOutput=Join-Path $taskDoc 'chatbot_interaction_binding_repair1_20261007.json'
if(Test-Path -LiteralPath $taskOutput){throw 'Repair binding exists'}
if((Get-FileHash -LiteralPath $taskOriginal).Hash.ToLowerInvariant() -ne '257d944dac6431eb8abfe96cc221ae537b9d298f6aac2b066ec7a7eb626014f7'){throw 'Original binding SHA'}
function Entry([string]$path){
    $taskItem=Get-Item -LiteralPath $path;$taskStream=[IO.File]::OpenRead($taskItem.FullName)
    try {$taskBytes=$taskStream.Length} finally {$taskStream.Dispose()}
    return @{path=$taskItem.FullName;bytes=$taskBytes;sha256=(Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant()}
}
$taskBinding=Get-Content -LiteralPath $taskOriginal -Raw | ConvertFrom-Json
$taskChanges=@()
foreach($taskInput in $taskBinding.inputs){
    $taskNow=Entry $taskInput.path
    if([IO.Path]::GetFileName($taskInput.path) -ne 'chatbot_interaction_setup.ps1' -and $taskNow.sha256 -ne $taskInput.sha256){throw ('Input content drift '+$taskInput.path)}
    if($taskNow.bytes -ne $taskInput.bytes -or $taskNow.sha256 -ne $taskInput.sha256){
        $taskChanges+=@{path=$taskInput.path;old_bytes=$taskInput.bytes;new_bytes=$taskNow.bytes;old_sha256=$taskInput.sha256;new_sha256=$taskNow.sha256}
    }
    $taskInput.bytes=$taskNow.bytes;$taskInput.sha256=$taskNow.sha256
}
$taskBinding.inputs=@($taskBinding.inputs)
foreach($taskPath in @($taskOriginal,(Join-Path $taskDoc 'chatbot_qwen_interaction_20261007.launcher_failure.json'),
    (Join-Path $taskDoc 'CHATBOT_INTERACTION_REPAIR1_20261007.md'),(Join-Path $taskRoot 'benchmarks/native_expert_scaling/chatbot_interaction_binding_repair1.ps1'))){$taskBinding.inputs+=Entry $taskPath}
$taskBinding | Add-Member -NotePropertyName metadata_repair -NotePropertyValue @{scope='Physical symlink-target byte lengths and corrected administrative setup source only';changes=$taskChanges;fixture_or_worker_replay=$false}
[IO.File]::WriteAllText($taskOutput,($taskBinding | ConvertTo-Json -Depth 20)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256 | Format-List Path,Hash
