$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskBase=Get-Content (Join-Path $taskDoc 'meth528_prefix_manifest_repair4_20261007.json') -Raw | ConvertFrom-Json
$taskBinding=Get-Content (Join-Path $taskDoc 'meth528_binding.json') -Raw | ConvertFrom-Json
$taskFiles=@($taskBase.files)
$taskMain=Get-Content (Join-Path $taskDoc 'meth528_prefix_bound_repair4_result.json') -Raw | ConvertFrom-Json
$taskAudit=Get-Content (Join-Path $taskDoc 'meth528_prefix_energy_audit_repair4_result.json') -Raw | ConvertFrom-Json
foreach($v in @($taskMain.output_inventory)+@($taskAudit.output_inventory)) {if((Split-Path $v.path -Leaf) -in @('source_U640.npy','first_prefix_scalar.dll','independent_integer.dll','first_C_verification.json','independent_proof.json')) {$taskFiles+=$v}}
$taskNames=@('benchmarks/native_expert_scaling/meth528_atom_math.py','benchmarks/native_expert_scaling/meth528_rare_gate.py','benchmarks/native_expert_scaling/meth528_rare_manifest.ps1','docs/research/NATIVE_EXPERT_SCALING_20260925/METH_528_RARE_FIRST_PROTOCOL_20261007.md','docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_prefix_bound_repair4_result.json','docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_prefix_energy_audit_repair4_result.json')
foreach($rel in $taskNames){$p=Join-Path $taskRoot $rel;$taskFiles+=@{path=$p;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p).Hash.ToLower()}}
foreach($e in @(15,36,69,115,124)) {
    $p=Join-Path $taskRoot ('results/native_expert_scaling/meth528_atoms/e{0:000}_a0.bin' -f $e)
    $taskSeal=Get-Content (Join-Path $taskDoc 'RETENTION_528_FIRST_FAULT_20261007.json') -Raw | ConvertFrom-Json
    $v=$taskSeal.original_partial_output_inventory | Where-Object {$_.path -eq $p}
    if(@($v).Count -ne 1){throw 'bank missing'};$taskFiles+=$v
}
$taskExtra=@($taskBinding.source_extents | Where-Object {$_.name -match '\.expert_([0-9]+)\.' -and [int]$Matches[1] -in @(15,36,69,115,124)})
if($taskExtra.Count -ne 20){throw '20 extra extents expected'}
$taskResult=@{files=$taskFiles;source_extents=@($taskBase.source_extents)+$taskExtra;data_keys=$taskBase.data_keys;empty_cache=$taskBase.empty_cache;original_binding_sha256=$taskBase.original_binding_sha256}
$p=Join-Path $taskDoc 'meth528_rare_manifest_20261007.json';if(Test-Path -LiteralPath $p){throw 'manifest exists'}
$taskResult | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $p -Encoding utf8
Write-Output ('manifest_sha='+ (Get-FileHash -LiteralPath $p).Hash.ToLower())
