$ErrorActionPreference='Stop'
$taskClock=[Diagnostics.Stopwatch]::StartNew()
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskDoc=Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskOldSeal=Get-Content (Join-Path $taskDoc 'RETENTION_528_FIRST_FAULT_20261007.json') -Raw | ConvertFrom-Json
foreach($v in $taskOldSeal.original_partial_output_inventory){if((Get-Item -LiteralPath $v.path).Length -ne $v.bytes -or (Get-FileHash -LiteralPath $v.path).Hash.ToLower() -ne $v.sha256){throw 'original partial changed'}}
$taskNew=@()
$taskFolders=@('meth528_prefix_bound','meth528_prefix_bound_repair1','meth528_prefix_bound_repair2','meth528_prefix_bound_repair3','meth528_prefix_bound_repair4','meth528_prefix_energy_audit_repair4','meth528_prefix_compiler_diag','meth528_rare_gate','meth528_rare_gate_repair1','meth528_rare_metadata_proof')
foreach($name in $taskFolders){foreach($p in Get-ChildItem -LiteralPath (Join-Path $taskRoot ('results/native_expert_scaling/'+$name)) -File){$taskNew+=@{path=$p.FullName;bytes=$p.Length;sha256=(Get-FileHash -LiteralPath $p.FullName).Hash.ToLower()}}}
$taskMain=Get-Content (Join-Path $taskDoc 'meth528_prefix_bound_repair4_result.json') -Raw | ConvertFrom-Json
$taskAudit=Get-Content (Join-Path $taskDoc 'meth528_prefix_energy_audit_repair4_result.json') -Raw | ConvertFrom-Json
$taskProof=Get-Content (Join-Path $taskDoc 'meth528_rare_metadata_proof_result.json') -Raw | ConvertFrom-Json
foreach($v in @($taskMain.output_inventory)+@($taskAudit.output_inventory)+@($taskProof.output_inventory)){if((Get-FileHash -LiteralPath $v.path).Hash.ToLower() -ne $v.sha256){throw 'completed output changed'}}
$taskForeign=@{}
$b=Get-Content (Join-Path $taskDoc 'meth528_binding.json') -Raw | ConvertFrom-Json
foreach($v in $b.preserved.PSObject.Properties){$taskActual=(Get-FileHash -LiteralPath (Join-Path $taskRoot $v.Name)).Hash.ToLower();if($taskActual -ne $v.Value){throw 'foreign SHA'};$taskForeign[$v.Name]=$taskActual}
$taskCache=Join-Path $taskRoot 'results/native_expert_scaling/meth511_runtime/empty_cache';if(@(Get-ChildItem -LiteralPath $taskCache -Force).Count){throw 'cache changed'}
$taskOldBytes=($taskOldSeal.original_partial_output_inventory | Measure-Object bytes -Sum).Sum
$taskNewBytes=($taskNew | Measure-Object bytes -Sum).Sum
$taskDocBytes=(Get-ChildItem -LiteralPath $taskDoc -File | Where-Object {$_.Name -match '^(meth528|RETENTION_528|ADMISSION_528)' } | Measure-Object Length -Sum).Sum
if($taskNewBytes -gt (465MB) -or $taskOldBytes+$taskNewBytes+$taskDocBytes -gt 2GB){throw 'cumulative output budget'}
$taskReceipt=[ordered]@{decision='NECESSARY_FIXED_DOMAIN_FIDELITY_FAILURE_CERTIFIED';original528_resource_gate=$false;original_all288_partials_freshly_BYTE_preserved=$true;original_partial_bytes=$taskOldBytes;new_output_inventory=$taskNew;new_output_bytes=$taskNewBytes;current_original_plus_new_plus_numeric_metadata_bytes=$taskOldBytes+$taskNewBytes+$taskDocBytes;foreign_preserved=$taskForeign;empty_cache=$true;terminal_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $taskDoc 'meth528_certificate_windows_terminal_20261007.json')).Hash.ToLower();full528_or_chatbot_admission=$false;new_scientific_evaluations=0;metadata_elapsed_seconds=$taskClock.Elapsed.TotalSeconds;scope='Metadata-only SHA retention; reuse all completed numerical observations, no candidate/model/source/control/energy replay. First two unknown compiler-descendant resource qualifications stay FALSE.'}
$taskDestination=Join-Path $taskDoc 'RETENTION_528_NECESSARY_GATE_20261007.json';if(Test-Path -LiteralPath $taskDestination){throw 'retention exists'}
$taskReceipt | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $taskDestination -Encoding utf8
Write-Output ('original288 preserved; new_bytes='+$taskNewBytes+'; cumulative_bytes='+$taskReceipt.current_original_plus_new_plus_numeric_metadata_bytes+'; metadata_seconds='+$taskClock.Elapsed.TotalSeconds+'; receipt_sha='+(Get-FileHash -LiteralPath $taskDestination).Hash.ToLower())
