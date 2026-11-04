# Builds the routing recorder against the pinned llama.cpp static libraries (5b335f4, AVX2), as s2x_build.ps1.
$ErrorActionPreference = 'Stop'
$T = 'E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4'
$gxx = (Get-ChildItem "$env:LOCALAPPDATA/Microsoft/WinGet/Packages/BrechtSanders.WinLibs*/mingw64/bin/g++.exe").FullName
$env:PATH = (Split-Path $gxx) + ';' + $env:PATH
$here = $PSScriptRoot
$s1 = Join-Path $here '../stage1'
& $gxx -O2 -std=c++17 "-I$T/include" "-I$T/ggml/include" "-I$here" "-I$s1" (Join-Path $here 's2r_llama_score.cpp') `
    "$T/build-avx2/src/libllama.a" "$T/build-avx2/ggml/src/ggml.a" "$T/build-avx2/ggml/src/ggml-cpu.a" `
    "$T/build-avx2/ggml/src/ggml-base.a" -fopenmp -static -o (Join-Path $here 'bin/s2r_llama_score_avx2.exe')
if ($LASTEXITCODE) { throw 'build failed' }
(Get-FileHash (Join-Path $here 'bin/s2r_llama_score_avx2.exe') -Algorithm SHA256).Hash.ToLower()
