$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$b=Get-Content (Join-Path $taskDoc 'meth528_binding.json') -Raw | ConvertFrom-Json
$files=@($b.runtime_files)+@($b.catalog | Where-Object {$_.path -like 'C:\Windows\System32\*'})
foreach($k in @('uid','occurrences','unweighted','targets')){$files+=$b.data.$k}
$rels=@('benchmarks/native_expert_scaling/meth528_operations.py','benchmarks/native_expert_scaling/meth528_contract.py','benchmarks/native_expert_scaling/meth528_prefix_io_repair4.py','benchmarks/native_expert_scaling/meth528_prefix_bound_repair4.py','benchmarks/native_expert_scaling/meth528_rare_gate_repair1.py','benchmarks/native_expert_scaling/meth528_rare_metadata_proof.py','benchmarks/native_expert_scaling/meth528_rare_metadata_manifest.ps1','docs/research/NATIVE_EXPERT_SCALING_20260925/METH_528_RARE_METADATA_PROOF_PROTOCOL_20261007.md','docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_rare_gate_repair1_result.failure.json','docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_prefix_bound_repair4_result.json','docs/research/NATIVE_EXPERT_SCALING_20260925/meth528_prefix_energy_audit_repair4_result.json','results/native_expert_scaling/meth528_prefix_bound_repair4/source_U640.npy','results/native_expert_scaling/meth528_prefix_energy_audit_repair4/independent_integer.dll')
foreach($name in @('physical','integer','codes','alpha')){foreach($type in @('main','C')){$rels+=('results/native_expert_scaling/meth528_rare_gate_repair1/e015_'+$type+'_'+$name+'.npy')}}
foreach($rel in $rels){$p=Join-Path $taskRoot $rel;$files+=@{path=$p;bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p).Hash.ToLower()}}
$extents=@($b.source_extents | Where-Object {$_.name -match '\.expert_15\.'});if($extents.Count -ne 4){throw '4 extents'}
$result=@{files=$files;source_extents=$extents;data_keys=@('uid','occurrences','unweighted','targets');empty_cache=(Join-Path $taskRoot 'results/native_expert_scaling/meth511_runtime/empty_cache');original_binding_sha256='9de128cb77373c0e5c35cb24d398b1269d68b79f54fc2e943c2a845236ec49c2'}
$p=Join-Path $taskDoc 'meth528_rare_metadata_manifest_20261007.json';if(Test-Path $p){throw 'manifest exists'};$result | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $p -Encoding utf8
Write-Output ('manifest_sha='+(Get-FileHash -LiteralPath $p).Hash.ToLower())
