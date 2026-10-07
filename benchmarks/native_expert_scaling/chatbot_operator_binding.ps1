# Administrative input sealing only: no tensor descriptors or operator counts.
$ErrorActionPreference = 'Stop'
$taskRoot = (Get-Location).Path
$taskDoc = Join-Path $taskRoot 'docs/research/NATIVE_EXPERT_SCALING_20260925'
$taskOutput = Join-Path $taskDoc 'chatbot_operator_binding_20261007.json'
if (Test-Path -LiteralPath $taskOutput) { throw 'Binding namespace exists' }
function Entry([string]$path, [long]$count = -1) {
    $taskItem = Get-Item -LiteralPath $path
    if ($count -lt 0) { $count = $taskItem.Length }
    if ($count -gt 67108864 -or $count -gt $taskItem.Length) { throw 'Extent cap' }
    $taskStream = [IO.File]::OpenRead($taskItem.FullName)
    try {
        $taskBytes = New-Object byte[] $count
        $taskRead = 0
        while ($taskRead -lt $count) {
            $taskGot = $taskStream.Read($taskBytes, $taskRead, $count-$taskRead)
            if ($taskGot -eq 0) { throw 'Short read' }
            $taskRead += $taskGot
        }
    } finally { $taskStream.Dispose() }
    $taskSha = [Security.Cryptography.SHA256]::Create()
    try { $taskHash = [BitConverter]::ToString($taskSha.ComputeHash($taskBytes)).Replace('-','').ToLowerInvariant() }
    finally { $taskSha.Dispose() }
    return @{path=$taskItem.FullName; bytes=$count; file_bytes=$taskItem.Length; sha256=$taskHash; whole_file_SHA_verified=($count -eq $taskItem.Length)}
}
$taskCommon = @(
    'benchmarks/native_expert_scaling/chatbot_operator_census.py',
    'benchmarks/native_expert_scaling/chatbot_operator_binding.ps1',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_OPERATOR_CENSUS_PROTOCOL_20261007.md',
    '.venv/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py',
    '.venv/Lib/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py',
    'benchmarks/native_expert_scaling/meth284_source_operator.h',
    'benchmarks/native_expert_scaling/meth182_group64_ffn_cpu.c',
    'benchmarks/native_expert_scaling/meth285_model_operator.h',
    'benchmarks/native_expert_scaling/meth294_window_forward.h',
    'benchmarks/native_expert_scaling/meth295_native_primary_cpu.c',
    'benchmarks/native_expert_scaling/meth296_native_generation_cpu.c',
    'benchmarks/phase60/engine.c',
    'docs/research/NATIVE_EXPERT_SCALING_20260925/meth284_native_archive_qualification_result.json',
    'results/native_expert_scaling/meth511_runtime/venv/Scripts/python.exe',
    'results/native_expert_scaling/meth511_runtime/venv/Lib/site-packages/psutil/__init__.py',
    'results/native_expert_scaling/meth511_runtime/venv/Lib/site-packages/psutil/_psutil_windows.pyd'
) | ForEach-Object { Entry (Join-Path $taskRoot $_) }
$taskDonors = @{}
foreach ($taskDonor in @('qwen','gigachat')) {
    $taskPreflightPath = Join-Path $taskDoc ('chatbot_'+$taskDonor+'_source_contract_20261007.json')
    $taskPreflight = Get-Content -LiteralPath $taskPreflightPath -Raw | ConvertFrom-Json
    $taskHeaders = @($taskPreflight.inputs | Where-Object { $_.path.EndsWith('.safetensors') })
    $taskConfig = @($taskPreflight.inputs | Where-Object { [IO.Path]::GetFileName($_.path) -eq 'config.json' })
    if ($taskConfig.Count -ne 1) { throw 'Config binding cardinality' }
    $taskIndex = @($taskPreflight.inputs | Where-Object { $_.path.EndsWith('model.safetensors.index.json') })
    $taskDonors[$taskDonor] = @{preflight=(Entry $taskPreflightPath); config=$taskConfig[0]; headers=$taskHeaders; index=$null}
    if ($taskIndex.Count -eq 1) { $taskDonors[$taskDonor].index = $taskIndex[0] }
}
$taskBinding = @{schema='CHATBOT_OPERATOR_BINDING_V1'; created_utc=[DateTime]::UtcNow.ToString('o'); common=$taskCommon; donors=$taskDonors;
    qwen_archive=@{header=(Entry (Join-Path $taskRoot 'results/native_expert_scaling/meth276_qwen05b_i16_private128_diagnostic_repair1.safetensors') 225368);
                   catalog=(Entry (Join-Path $taskRoot 'benchmarks/native_expert_scaling/meth284_archive_catalog.h'))}}
# Actual newline, never a literal backslash sequence.
$taskJson = ($taskBinding | ConvertTo-Json -Depth 30)+[Environment]::NewLine
[IO.File]::WriteAllText($taskOutput,$taskJson,[Text.UTF8Encoding]::new($false))
Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256 | Format-List Path,Hash
