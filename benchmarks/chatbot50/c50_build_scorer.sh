#!/bin/sh
# Builds a C50 llama.cpp tool against the pinned llama.cpp static libraries (5b335f4), once per
# build flavour (avx2, generic), with the compiler that built those libraries.
# Usage: c50_build_scorer.sh [tool]   tool = c50_llama_score (default) or c50_llama_generate
set -eu
TOOL=${1:-c50_llama_score}
T=${LLAMA_PIN:-/e/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4}
CXX=${CXX:-/c/Users/giosa/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/g++.exe}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/bin"
for flavour in avx2 generic; do
  B=$T/build-$flavour
  "$CXX" -O2 -std=c++17 -I"$T/include" -I"$T/ggml/include" "$HERE/$TOOL.cpp" \
    "$B/src/libllama.a" "$B/ggml/src/ggml.a" "$B/ggml/src/ggml-cpu.a" "$B/ggml/src/ggml-base.a" \
    -fopenmp -static -o "$HERE/bin/${TOOL}_$flavour.exe"
  sha256sum "$HERE/bin/${TOOL}_$flavour.exe"
done
