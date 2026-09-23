#!/usr/bin/env python3
"""Build the pinned active-x86 oracle for Q4_K/Q8_K reduction parity."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from benchmarks.donor_adaptation.engine.build_strat01_rung2a_projection_diagnostic import (
    BuildError,
    PINNED_LLAMA,
    verify_pinned_llama,
)

SOURCE = Path(__file__).resolve().with_name("strat01_q4k_q8k_avx2_oracle.cpp")
TARGET = "strat01_q4k_q8k_avx2_oracle"


def cmake_project(source: Path, llama_source: Path) -> str:
    return f'''cmake_minimum_required(VERSION 3.20)
project({TARGET} LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_AVX ON CACHE BOOL "" FORCE)
set(GGML_AVX2 ON CACHE BOOL "" FORCE)
set(GGML_FMA ON CACHE BOOL "" FORCE)
set(GGML_F16C ON CACHE BOOL "" FORCE)
set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable({TARGET} "{source.as_posix()}")
target_compile_features({TARGET} PRIVATE cxx_std_17)
target_compile_options({TARGET} PRIVATE -mavx2 -mfma)
target_include_directories({TARGET} PRIVATE
    "{(llama_source / 'ggml' / 'include').as_posix()}"
    "{(llama_source / 'ggml' / 'src').as_posix()}"
    "{(llama_source / 'ggml' / 'src' / 'ggml-cpu').as_posix()}")
target_link_libraries({TARGET} PRIVATE llama)
'''


def build(build_dir: Path, configuration: str = "Release") -> Path:
    verify_pinned_llama()
    build_dir.mkdir(parents=True, exist_ok=True)
    toolchain = build_dir / "x86_64-toolchain.cmake"
    toolchain.write_text(
        'set(CMAKE_SYSTEM_NAME Windows)\nset(CMAKE_SYSTEM_PROCESSOR x86_64)\n',
        encoding="utf-8", newline="\n",
    )
    (build_dir / "CMakeLists.txt").write_text(
        cmake_project(SOURCE, PINNED_LLAMA), encoding="utf-8", newline="\n"
    )
    configured = subprocess.run(
        ["cmake", "-S", str(build_dir), "-B", str(build_dir / "cmake-build"),
         f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"],
        text=True, capture_output=True, check=False,
    )
    if configured.returncode:
        raise BuildError("CMake configure failed:\n" + "\n".join((configured.stdout + configured.stderr).splitlines()[-30:]))
    commands = json.loads((build_dir / "cmake-build" / "compile_commands.json").read_text(encoding="utf-8"))
    x86_quants = [
        entry for entry in commands
        if entry["file"].replace("\\", "/").endswith("/ggml-cpu/arch/x86/quants.c")
    ]
    if len(x86_quants) != 1:
        raise BuildError("oracle did not select exactly one pinned x86 quants.c")
    command = x86_quants[0]["command"]
    if "-mavx2" not in command or "-mfma" not in command or "-DGGML_CPU_GENERIC" in command:
        raise BuildError("pinned x86 oracle lacks the required AVX2/FMA compile contract")
    compiled = subprocess.run(
        ["cmake", "--build", str(build_dir / "cmake-build"), "--target", TARGET, "--config", configuration],
        text=True, capture_output=True, check=False,
    )
    if compiled.returncode:
        raise BuildError("CMake build failed:\n" + "\n".join((compiled.stdout + compiled.stderr).splitlines()[-40:]))
    names = (f"{TARGET}.exe", TARGET)
    candidates = [candidate for name in names for candidate in (build_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise BuildError("build did not produce exactly one AVX2 oracle")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    print(build(args.build_dir, args.config))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        raise SystemExit(f"build_strat01_q4k_q8k_avx2_oracle: {error}")
