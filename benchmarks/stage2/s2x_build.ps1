# Builds stage 2 test 3b's panel tools against the pinned llama.cpp static libraries (5b335f4, AVX2), as s2t_build.ps1.
$ErrorActionPreference = 'Stop'
$T = 'E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4'
$gxx = (Get-ChildItem "$env:LOCALAPPDATA/Microsoft/WinGet/Packages/BrechtSanders.WinLibs*/mingw64/bin/g++.exe").FullName
$env:PATH = (Split-Path $gxx) + ';' + $env:PATH
$here = $PSScriptRoot
$s1 = Join-Path $here '../stage1'
New-Item -ItemType Directory -Force (Join-Path $here 'bin') | Out-Null
& $gxx --version | Select-Object -First 1
foreach ($tool in 's2x_llama_score', 's2x_llama_generate') {
    & $gxx -O2 -std=c++17 "-I$T/include" "-I$T/ggml/include" "-I$here" "-I$s1" (Join-Path $here "$tool.cpp") `
        "$T/build-avx2/src/libllama.a" "$T/build-avx2/ggml/src/ggml.a" "$T/build-avx2/ggml/src/ggml-cpu.a" `
        "$T/build-avx2/ggml/src/ggml-base.a" -fopenmp -static -o (Join-Path $here "bin/${tool}_avx2.exe")
    if ($LASTEXITCODE) { throw "build failed: $tool" }
    (Get-FileHash (Join-Path $here "bin/${tool}_avx2.exe") -Algorithm SHA256).Hash.ToLower()
}
