#!/usr/bin/env bash
# Stage 1 (c): builds s1k_bench against the four ggml builds it measures, into $OUT (default C:/SiliconLLM_fast/s1k).
#   main:   llama.cpp 5b335f4, build-avx2 (the build of stage 1 (a) and (b))
#   ik:     ik_llama.cpp d4c1f75, build-native (stage 0's reference build)
#   bitnet: bitnet.cpp 0b341e5 with its llama.cpp 390c307, gcc, BITNET_X86_TL2=OFF (I2_S)
#   tl2:    the same with BITNET_X86_TL2=ON, kernels from utils/codegen_tl2.py for Ling-flash's shapes
#           (--model ling-flash --BM 256,256,128 --BK 96,96,96 --bm 32,32,32), plus s1k_tl2_glue.cpp
# The BitNet clone was prepared once, in its own directory:
#   git clone --recursive https://github.com/microsoft/BitNet.git   (0b341e5, llama.cpp submodule 390c307)
#   utils/codegen_tl2.py: ModelShapeDict gets "ling-flash": [[1024, 4096], [4096, 1024], [4096, 4096]]
#   python -I utils/codegen_tl2.py --model ling-flash --BM 256,256,128 --BK 96,96,96 --bm 32,32,32
#   include/bitnet-lut-kernels.h: drop " && tensor->backend == GGML_BACKEND_TYPE_CPU" (a field this ggml no longer has)
#   cmake -B build-gcc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
#     -DBITNET_X86_TL2=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
#     -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF && ninja -C build-gcc ggml
#   the same into build-tl2 with -DBITNET_X86_TL2=ON
#   python -I benchmarks/stage1/s1k_tl2_weights.py --bitnet <clone> --out C:/SiliconLLM_fast/s1k/tl2
# Usage: bash benchmarks/stage1/s1k_build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-C:/SiliconLLM_fast/s1k}"
M=E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4
I=E:/SiliconLLM_Artifacts/wallmap/tools/ik_llama.cpp
B=E:/SiliconLLM_Artifacts/stage1/s1k/src/BitNet
G=$B/3rdparty/llama.cpp/ggml
SRC="$HERE/s1k_bench.c"
mkdir -p "$OUT"
gcc -O2 -Wall -DMAINLINE -fopenmp "$SRC" -I$M/ggml/include $M/build-avx2/ggml/src/ggml.a \
    $M/build-avx2/ggml/src/ggml-cpu.a $M/build-avx2/ggml/src/ggml-base.a -lstdc++ -o "$OUT/s1k_bench_main.exe"
gcc -O2 -Wall -fopenmp "$SRC" -I$I/ggml/include $I/build-native/ggml/src/libggml.a -lstdc++ -o "$OUT/s1k_bench_ik.exe"
gcc -O2 -Wall -DMAINLINE -DBITNET -fopenmp "$SRC" -I$G/include $B/build-gcc/3rdparty/llama.cpp/ggml/src/ggml.a \
    $B/build-gcc/3rdparty/llama.cpp/ggml/src/ggml-cpu.a $B/build-gcc/3rdparty/llama.cpp/ggml/src/ggml-base.a -lstdc++ \
    -o "$OUT/s1k_bench_bitnet.exe"
g++ -O2 -march=native -DGGML_BITNET_X86_TL2 -c "$HERE/s1k_tl2_glue.cpp" -I$B/include -I$G/include -I$G/src \
    -I$G/src/ggml-cpu -o "$OUT/s1k_tl2_glue.o"
gcc -O2 -Wall -DMAINLINE -DBITNET -DBITNET_TL2 -DGGML_BITNET_X86_TL2 -fopenmp "$SRC" -I$G/include -I$B/include \
    "$OUT/s1k_tl2_glue.o" $B/build-tl2/3rdparty/llama.cpp/ggml/src/ggml.a $B/build-tl2/3rdparty/llama.cpp/ggml/src/ggml-cpu.a \
    $B/build-tl2/3rdparty/llama.cpp/ggml/src/ggml-base.a -lstdc++ -o "$OUT/s1k_bench_tl2.exe"
sha256sum "$OUT"/s1k_bench_*.exe
