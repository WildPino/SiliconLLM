#!/bin/sh
# Builds c50_s3_teacher against the official llama.cpp release of the pinned commit (b11040 = 5b335f4),
# CUDA 12.4 x64, by linking the release's DLLs (C API only).  The executable goes next to those DLLs,
# because the release loads its backends (ggml-cuda, ggml-cpu-*) from the executable's directory.
# The release zips were checked against the published digests (ROADMAP §12, 19 October 2026).
set -eu
T=${LLAMA_PIN:-/e/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4}
R=${LLAMA_REL:-/e/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama-b11040-win-cuda-12.4-x64/bin}
CXX=${CXX:-/c/Users/giosa/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/g++.exe}
HERE=$(cd "$(dirname "$0")" && pwd)
"$CXX" -O2 -std=c++17 -I"$T/include" -I"$T/ggml/include" "$HERE/c50_s3_teacher.cpp" \
  "$R/llama.dll" "$R/ggml.dll" "$R/ggml-base.dll" -static-libstdc++ -static-libgcc -o "$R/c50_s3_teacher_cuda.exe"
sha256sum "$R/c50_s3_teacher_cuda.exe"
