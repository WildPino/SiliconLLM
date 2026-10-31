#!/bin/sh
# Builds a stage 1 llama.cpp tool against the pinned llama.cpp static libraries (5b335f4, the AVX2
# flavour of the C50 panel), as benchmarks/chatbot50/c50_build_scorer.sh does.
# Usage: s1_build.sh tool   tool = s1_llama_score or s1_llama_generate
# CXX defaults to the WinLibs POSIX MSVCRT g++ on PATH.  The libraries were built with g++ 15.2; the
# tool prints the compiler it was built with, and the parity checks of the stage 1 protocol compare
# its outputs bit for bit with the C50 tools'.
set -eu
TOOL=$1
T=${LLAMA_PIN:-/e/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4}
CXX=${CXX:-g++}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/bin"
B=$T/build-avx2
"$CXX" --version | head -n 1
"$CXX" -O2 -std=c++17 -I"$T/include" -I"$T/ggml/include" -I"$HERE" "$HERE/$TOOL.cpp" \
  "$B/src/libllama.a" "$B/ggml/src/ggml.a" "$B/ggml/src/ggml-cpu.a" "$B/ggml/src/ggml-base.a" \
  -fopenmp -static -o "$HERE/bin/${TOOL}_avx2.exe"
sha256sum "$HERE/bin/${TOOL}_avx2.exe"
