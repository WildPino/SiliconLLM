# Administrative package view and input sealing, BEFORE any new chat token observations.
$ErrorActionPreference='Stop'
$taskRoot=(Get-Location).Path
$taskRuntime=Join-Path $taskRoot 'results/native_expert_scaling/chatbot_interaction_runtime'
$taskSite=Join-Path $taskRuntime 'site'
$taskSourceSite=Join-Path $taskRoot '.venv/Lib/site-packages'
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskOutput=Join-Path $taskDoc 'chatbot_interaction_binding_20261007.json'
if((Test-Path -LiteralPath $taskRuntime) -or (Test-Path -LiteralPath $taskOutput)){throw 'Fresh namespace required'}
[void](New-Item -ItemType Directory -Path $taskSite)
function Entry([string]$path) {
    $taskItem=Get-Item -LiteralPath $path
    $taskStream=[IO.File]::OpenRead($taskItem.FullName)
    try {$taskBytes=$taskStream.Length} finally {$taskStream.Dispose()}
    return @{path=$taskItem.FullName;bytes=$taskBytes;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Tree([string]$path) {
    $taskItem=Get-Item -LiteralPath $path
    if(-not $taskItem.PSIsContainer){return @{tree_sha256=(Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant();files=1;bytes=$taskItem.Length}}
    $taskFiles=@(Get-ChildItem -LiteralPath $path -Recurse -File | Where-Object {
        $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and ($_.Extension -cin @('.py','.pyd','.dll','.json','.pem') -or $_.Name -cin @('METADATA','WHEEL','RECORD'))
    })
    $taskMap=@{}; $taskRelative=[Collections.Generic.List[string]]::new()
    foreach($taskFile in $taskFiles){$taskRel=$taskFile.FullName.Substring($taskItem.FullName.Length+1).Replace('\','/');$taskMap[$taskRel]=$taskFile;$taskRelative.Add($taskRel)}
    $taskRelative.Sort([StringComparer]::Ordinal)
    $taskDigest=[Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    $taskTotal=0L
    foreach($taskRel in $taskRelative){
        $taskFile=$taskMap[$taskRel];$taskHash=(Get-FileHash -LiteralPath $taskFile.FullName).Hash.ToLowerInvariant()
        $taskLine=$taskRel+[char]0+[string]$taskFile.Length+[char]0+$taskHash+[char]10
        $taskDigest.AppendData([Text.Encoding]::UTF8.GetBytes($taskLine));$taskTotal+=$taskFile.Length
    }
    $taskHash=[BitConverter]::ToString($taskDigest.GetHashAndReset()).Replace('-','').ToLowerInvariant();$taskDigest.Dispose()
    return @{tree_sha256=$taskHash;files=$taskFiles.Count;bytes=$taskTotal}
}
$taskModules=@('transformers','tokenizers','jinja2','markupsafe','packaging','numpy','numpy.libs','regex','safetensors','huggingface_hub',
    'tqdm','filelock','fsspec','yaml','_yaml','httpx','httpcore','anyio','certifi','idna','h11','click','typer','rich','markdown_it','mdurl',
    'shellingham','colorama','annotated_doc','psutil','pygments','typing_extensions.py')
$taskPackages=@('transformers','tokenizers','jinja2','markupsafe','packaging','numpy','regex','safetensors','huggingface_hub',
    'tqdm','filelock','fsspec','pyyaml','httpx','httpcore','anyio','certifi','idna','h11','click','typer','rich','markdown_it_py','mdurl',
    'shellingham','colorama','annotated_doc','psutil','pygments','typing_extensions')
$taskViews=@()
foreach($taskName in $taskModules){
    $taskPath=Join-Path $taskSourceSite $taskName
    if(-not (Test-Path -LiteralPath $taskPath)){throw ('Missing selected dependency '+$taskName)}
    $taskViews+=@{name=$taskName;path=$taskPath}
}
$taskVersions=@{}
foreach($taskPackage in $taskPackages){
    $taskMetadata=@(Get-ChildItem -LiteralPath $taskSourceSite -Directory | Where-Object {$_.Name -like ($taskPackage+'-*.dist-info')})
    if($taskMetadata.Count -ne 1){throw ('Dist-info cardinality '+$taskPackage)}
    $taskVersion=(Get-Content -LiteralPath (Join-Path $taskMetadata[0].FullName 'METADATA') | Where-Object {$_ -match '^Version: '}) -replace '^Version: ',''
    $taskVersions[$taskPackage.Replace('_','-')]=$taskVersion
    $taskViews+=@{name=$taskMetadata[0].Name;path=$taskMetadata[0].FullName}
}
$taskRoots=@()
foreach($taskView in $taskViews){
    $taskInfo=Get-Item -LiteralPath $taskView.path
    $taskLink=Join-Path $taskSite $taskView.name
    if($taskInfo.PSIsContainer){[void](New-Item -ItemType Junction -Path $taskLink -Target $taskInfo.FullName)}
    else{[void](New-Item -ItemType HardLink -Path $taskLink -Target $taskInfo.FullName)}
    $taskTree=Tree $taskInfo.FullName
    $taskTree.path=$taskInfo.FullName;$taskTree.view_name=$taskView.name;$taskRoots+=$taskTree
}
$taskPreflightPath=Join-Path $taskDoc 'chatbot_qwen_source_contract_20261007.json'
if((Get-FileHash -LiteralPath $taskPreflightPath).Hash.ToLowerInvariant() -ne '258313ec5e4800615d30e44ed61fd9e90086a295205fb3322054117a3c3fc225'){throw 'Preflight SHA'}
$taskPreflight=Get-Content -LiteralPath $taskPreflightPath -Raw | ConvertFrom-Json
$taskDonor=Split-Path -Parent (@($taskPreflight.inputs | Where-Object {[IO.Path]::GetFileName($_.path) -eq 'tokenizer.json'})[0].path)
$taskInputs=@(Entry $taskPreflightPath)
foreach($taskName in @('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json','chat_template.jinja')){
    $taskPath=Join-Path $taskDonor $taskName
    if(Test-Path -LiteralPath $taskPath){
        $taskEntry=Entry $taskPath
        $taskOld=@($taskPreflight.inputs | Where-Object {[IO.Path]::GetFileName($_.path) -eq $taskName})
        if($taskOld.Count -and $taskOld[0].sha256 -ne $taskEntry.sha256){throw ('Donor drift '+$taskName)}
        $taskInputs+=$taskEntry
    }
}
foreach($taskRel in @('benchmarks/native_expert_scaling/chatbot_interaction.py','benchmarks/native_expert_scaling/chatbot_interaction_launch.py',
    'benchmarks/native_expert_scaling/chatbot_interaction_goldens.json','benchmarks/native_expert_scaling/chatbot_interaction_setup.ps1',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_INTERACTION_PROTOCOL_20261007.md',
    'results/native_expert_scaling/meth511_runtime/venv/Lib/site-packages/psutil/__init__.py',
    'results/native_expert_scaling/meth511_runtime/venv/Lib/site-packages/psutil/_psutil_windows.pyd')){$taskInputs+=Entry (Join-Path $taskRoot $taskRel)}
foreach($taskPath in @('C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe','C:/Users/giosa/AppData/Local/Programs/Python/Python312/python312.dll')){$taskInputs+=Entry $taskPath}
$taskBinding=@{schema='QWEN_INTERACTION_BINDING_V1';created_utc=[DateTime]::UtcNow.ToString('o');
    python='C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe';donor_source=$taskDonor;versions=$taskVersions;
    runtime_roots=$taskRoots;inputs=$taskInputs;tree_scope='Sorted relative UTF8 path NUL size NUL SHA LF; py,pyd,dll,json,pem,METADATA,WHEEL,RECORD; no cached pyc';
    no_Torch_or_model_packages_in_runtime_view=$true;new_tensor_weights_or_model_calls=0}
[IO.File]::WriteAllText($taskOutput,($taskBinding | ConvertTo-Json -Depth 20)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256 | Format-List Path,Hash
