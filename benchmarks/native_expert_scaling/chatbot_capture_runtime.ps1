# New import view, not package modification/installation. Exclude unrelated Arrow.
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskView=Join-Path $taskRoot 'results/native_expert_scaling/chatbot_source_runtime/site'
if(Test-Path -LiteralPath $taskView){throw 'Runtime namespace exists'}
$taskOriginal=Join-Path $taskDoc 'chatbot_interaction_binding_repair3_20261007.json'
if((Get-FileHash -LiteralPath $taskOriginal).Hash.ToLowerInvariant() -ne '1b57abd45f37670b7f2e9293fda5d3ed299930dbb406003b59132a1f90b5fb97'){throw 'Interaction binding SHA'}
$taskB=Get-Content -LiteralPath $taskOriginal -Raw | ConvertFrom-Json -AsHashtable
$taskMappings=@($taskB.runtime_roots | ForEach-Object {@{name=$_.view_name;path=$_.path}})
$taskSite=Join-Path $taskRoot '.venv/Lib/site-packages'
foreach($taskName in @('torch','torchgen','functorch','sympy','mpmath','networkx')) {
    $taskPath=Join-Path $taskSite $taskName
    if(-not (Test-Path -LiteralPath $taskPath)){throw ('Required Torch module absent: '+$taskName)}
    $taskMappings+=@{name=$taskName;path=$taskPath}
}
foreach($taskPackage in @('torch','sympy','mpmath','networkx')) {
    $taskMatches=@(Get-ChildItem -LiteralPath $taskSite -Directory | Where-Object {$_.Name -like ($taskPackage+'-*.dist-info')})
    if($taskMatches.Count -ne 1){throw 'Torch dependency dist-info cardinality'}
    $taskMappings+=@{name=$taskMatches[0].Name;path=$taskMatches[0].FullName}
}
[void](New-Item -ItemType Directory -Path $taskView)
foreach($taskMapping in $taskMappings) {
    $taskPath=Join-Path $taskView $taskMapping.name
    if((Get-Item -LiteralPath $taskMapping.path).PSIsContainer){[void](New-Item -ItemType Junction -Path $taskPath -Target $taskMapping.path)}
    else{[void](New-Item -ItemType HardLink -Path $taskPath -Target $taskMapping.path)}
}
$taskReceipt=Join-Path $taskDoc 'chatbot_source_runtime_view_20261007.json'
if(Test-Path -LiteralPath $taskReceipt){throw 'View receipt exists'}
[IO.File]::WriteAllText($taskReceipt,(@{view=$taskView;mappings=$taskMappings;packages_modified=0;packages_installed=0;model_calls=0}|ConvertTo-Json -Depth 10)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
Write-Output ('Created restricted source runtime with '+$taskMappings.Count+' module/dist-info links; no packages modified')
