# The panel tools again, with ggml-cpu built with the assembler's unaligned vector moves (S2K, amendment 3): on this compiler ggml_gemv_q4_0_8x8_q8_0 stores to an unaligned
# stack slot, and only a Q4_0 model reaches it.  Same sources and flags as build-avx2; every other library is build-avx2's.  Output: bin/s2x_llama_{score,generate}_ua_avx2.exe.
$ErrorActionPreference = 'Stop'
$T = 'E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4'
$gxx = (Get-ChildItem "$env:LOCALAPPDATA/Microsoft/WinGet/Packages/BrechtSanders.WinLibs*/mingw64/bin/g++.exe").FullName
$bin = Split-Path $gxx
$env:PATH = "$bin;C:\Program Files\Git\cmd;" + $env:PATH
$here = $PSScriptRoot
$s1 = Join-Path $here '../stage1'
$fl = '-D_WIN32_WINNT=0x0A00 -Wa,-muse-unaligned-vector-move'
$B = "$T/build-avx2-ua"
if (-not (Test-Path "$B/ggml/src/ggml-cpu.a")) {
    & cmake -S $T -B $B -G Ninja "-DCMAKE_MAKE_PROGRAM=$bin/ninja.exe" "-DCMAKE_C_COMPILER=$bin/gcc.exe" "-DCMAKE_CXX_COMPILER=$gxx" -DCMAKE_BUILD_TYPE=Release `
        "-DCMAKE_C_FLAGS=$fl" "-DCMAKE_CXX_FLAGS=$fl" -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_F16C=ON -DGGML_FMA=ON `
        -DGGML_SSE42=ON -DGGML_OPENMP=ON -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
    if ($LASTEXITCODE) { throw 'configure failed' }
    & cmake --build $B --target ggml-cpu -j 10
    if ($LASTEXITCODE) { throw 'build failed' }
}
foreach ($tool in 's2x_llama_score', 's2x_llama_generate') {
    & $gxx -O2 -std=c++17 "-I$T/include" "-I$T/ggml/include" "-I$here" "-I$s1" (Join-Path $here "$tool.cpp") `
        "$T/build-avx2/src/libllama.a" "$T/build-avx2/ggml/src/ggml.a" "$B/ggml/src/ggml-cpu.a" `
        "$T/build-avx2/ggml/src/ggml-base.a" -fopenmp -static -o (Join-Path $here "bin/${tool}_ua_avx2.exe")
    if ($LASTEXITCODE) { throw "build failed: $tool" }
    (Get-FileHash (Join-Path $here "bin/${tool}_ua_avx2.exe") -Algorithm SHA256).Hash.ToLower()
}
