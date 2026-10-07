# Input selection only, no energy/response/model calculation. Run before freeze.
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskBindingPath=Join-Path $taskDoc 'meth528_binding.json'
$taskBindingSha='9de128cb77373c0e5c35cb24d398b1269d68b79f54fc2e943c2a845236ec49c2'
if ((Get-FileHash -LiteralPath $taskBindingPath).Hash.ToLower() -ne $taskBindingSha) {throw 'binding SHA'}
$taskB=Get-Content -LiteralPath $taskBindingPath -Raw | ConvertFrom-Json
$taskSealPath=Join-Path $taskDoc 'RETENTION_528_FIRST_FAULT_20261007.json'
if ((Get-FileHash -LiteralPath $taskSealPath).Hash.ToLower() -ne '411dd6a3cd875b94d8c5fefa61028f755643c4d223439e10ab4716382a282fd7') {throw 'seal SHA'}
$taskSeal=Get-Content -LiteralPath $taskSealPath -Raw | ConvertFrom-Json
$taskMap=@{}
function AddFile($v) {
    if ($taskMap.ContainsKey($v.path) -and $taskMap[$v.path].sha256 -ne $v.sha256) {throw 'conflicting SHA'}
    $taskMap[$v.path]=@{path=$v.path;bytes=[long]$v.bytes;sha256=$v.sha256}
}
foreach ($v in @($taskB.runtime_files)+@($taskB.compiler.files)+@($taskB.catalog | Where-Object {$_.path -like 'C:\Windows\System32\*'})) {AddFile $v}
$taskKeys=@('uid','occurrences','inputs','unweighted','targets','old_signed_dots','old_hinges','old_signs','old_anchors')
foreach ($k in $taskKeys) {AddFile $taskB.data.$k}
$taskOldNames=@('controls.json','development_mask_freeze.json','physical.npy','integer_WO.npy','hidden_alpha.npy')
foreach ($e in 1..12) {$taskOldNames+=('e{0:000}_a0.bin' -f $e);$taskOldNames+=('e{0:000}_a0_hidden_codes.npy' -f $e)}
foreach ($v in $taskSeal.original_partial_output_inventory) {if ((Split-Path $v.path -Leaf) -in $taskOldNames) {AddFile $v}}
$taskFound=@($taskSeal.original_partial_output_inventory | Where-Object {(Split-Path $_.path -Leaf) -in $taskOldNames}).Count
if ($taskFound -ne $taskOldNames.Count) {throw 'prefix partial catalog incomplete'}
$taskSourcePaths=@(
    'benchmarks/native_expert_scaling/meth528_operations.py',
    'benchmarks/native_expert_scaling/meth528_contract.py',
    'benchmarks/native_expert_scaling/meth528_verifier.c',
    'benchmarks/native_expert_scaling/meth528_prefix_io_repair2.py',
    'benchmarks/native_expert_scaling/meth528_prefix_bound_repair2.py',
    'benchmarks/native_expert_scaling/meth528_prefix_energy.c',
    'benchmarks/native_expert_scaling/meth528_prefix_energy_audit_repair2.py',
    'benchmarks/native_expert_scaling/meth528_prefix_manifest_repair2.ps1',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/METH_528_PREFIX_NECESSARY_FAILURE_PROTOCOL_20261007.md',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/METH_528_PREFIX_REPAIR1_PROTOCOL_20261007.md',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/METH_528_PREFIX_REPAIR2_PROTOCOL_20261007.md',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_prefix_bound_repair1_result.failure.json',
    'results/native_expert_scaling/meth528_prefix_bound_repair1/first_prefix_scalar_frontend_plan.log',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_prefix_compiler_diag_result.json',
    'results/native_expert_scaling/meth528_prefix_compiler_diag/driver_plan.log',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/RETENTION_528_PREFIX_FIRST_FAULT_20261007.json',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_prefix_bound_result.failure.json',
    'results/native_expert_scaling/meth528_prefix_bound/first_prefix_scalar_frontend.log',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_binding.json',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_main_result.failure.json',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/RETENTION_528_FIRST_FAULT_20261007.json'
)
foreach ($rel in $taskSourcePaths) {
    $p=Join-Path $taskRoot $rel
    AddFile @{path=$p;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p).Hash.ToLower()}
}
$taskExtents=@($taskB.source_extents | Where-Object {$_.name -match '\.expert_([0-9]+)\.' -and [int]$Matches[1] -ge 1 -and [int]$Matches[1] -le 12})
if ($taskExtents.Count -ne 48) {throw 'expected48 source extents'}
$taskFiles=@($taskMap.Values | Sort-Object path)
$taskResult=@{
    original_binding_sha256=$taskBindingSha
    original_fault_sha256='ffd1a55a824aa319b2a3ec63fad5a92dc7bd4425ed707a85e8c8a0634c9e3e42'
    original_retention_sha256='411dd6a3cd875b94d8c5fefa61028f755643c4d223439e10ab4716382a282fd7'
    files=$taskFiles;data_keys=$taskKeys;source_extents=$taskExtents
    fresh_file_hash_bytes=($taskFiles | Measure-Object bytes -Sum).Sum
    fresh_source_extent_hash_bytes=($taskExtents | Measure-Object bytes -Sum).Sum
    empty_cache=(Join-Path $taskRoot 'results/native_expert_scaling/meth511_runtime/empty_cache')
    scope='Prospective necessary528 prefix inquiry; selected used inputs only. Omitted full source bank extents13..127 and unused continuous/M/rotation/old fold outputs are not read or requalified.'
}
$taskDestination=Join-Path $taskDoc 'meth528_prefix_manifest_repair2_20261007.json'
if (Test-Path -LiteralPath $taskDestination) {throw 'manifest already exists'}
$taskResult | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $taskDestination -Encoding utf8
Write-Output ('files={0}; fresh_file_hash_bytes={1}; source_extent_bytes={2}; manifest_sha={3}' -f $taskFiles.Count,$taskResult.fresh_file_hash_bytes,$taskResult.fresh_source_extent_hash_bytes,(Get-FileHash -LiteralPath $taskDestination).Hash.ToLower())
