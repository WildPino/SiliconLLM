#!/usr/bin/env python3
"""Build and verify the pinned baseline GGML generic Q4_K/Q8_K oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from benchmarks.donor_adaptation.engine.build_strat01_rung2a_projection_diagnostic import (
    BuildError,
    PINNED_LLAMA,
    verify_pinned_llama,
)

SOURCE = Path(__file__).resolve().with_name("strat01_q4k_q8k_reference_generic_oracle.cpp")
GENERIC_SOURCE = PINNED_LLAMA / "ggml/src/ggml-cpu/quants.c"
GENERIC_SOURCE_SHA = "459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad"
TARGET = "strat01_q4k_q8k_reference_generic_oracle"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmake_project(source: Path, llama_source: Path) -> str:
    return f'''cmake_minimum_required(VERSION 3.20)
project({TARGET} LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable({TARGET} "{source.as_posix()}")
target_compile_features({TARGET} PRIVATE cxx_std_17)
target_include_directories({TARGET} PRIVATE
    "{(llama_source / 'ggml' / 'include').as_posix()}"
    "{(llama_source / 'ggml' / 'src').as_posix()}"
    "{(llama_source / 'ggml' / 'src' / 'ggml-cpu').as_posix()}")
target_link_libraries({TARGET} PRIVATE llama)
'''


def build(build_dir: Path, configuration: str = "Release") -> Path:
    verify_pinned_llama()
    if sha256_file(GENERIC_SOURCE) != GENERIC_SOURCE_SHA:
        raise BuildError("pinned generic quants.c hash mismatch")
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeLists.txt").write_text(cmake_project(SOURCE, PINNED_LLAMA), encoding="utf-8", newline="\n")
    configured = subprocess.run(
        ["cmake", "-S", str(build_dir), "-B", str(build_dir / "cmake-build")],
        text=True, capture_output=True, check=False,
    )
    if configured.returncode:
        raise BuildError("CMake configure failed:\n" + "\n".join((configured.stdout + configured.stderr).splitlines()[-30:]))
    commands = json.loads((build_dir / "cmake-build/compile_commands.json").read_text(encoding="utf-8"))
    generic = [entry for entry in commands if entry["file"].replace("\\", "/").endswith("/ggml-cpu/quants.c")]
    x86 = [entry for entry in commands if entry["file"].replace("\\", "/").endswith("/ggml-cpu/arch/x86/quants.c")]
    if len(generic) != 1 or x86:
        raise BuildError("oracle did not select exactly the baseline generic quants.c")
    command = generic[0]["command"]
    if "-DGGML_CPU_GENERIC" not in command or "-mavx" in command or "-mfma" in command:
        raise BuildError("oracle compile command is not baseline GGML_CPU_GENERIC")
    compiled = subprocess.run(
        ["cmake", "--build", str(build_dir / "cmake-build"), "--target", TARGET, "--config", configuration],
        text=True, capture_output=True, check=False,
    )
    if compiled.returncode:
        raise BuildError("CMake build failed:\n" + "\n".join((compiled.stdout + compiled.stderr).splitlines()[-40:]))
    candidates = [candidate for name in (f"{TARGET}.exe", TARGET) for candidate in (build_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise BuildError("build did not produce exactly one reference-generic oracle")
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
        raise SystemExit(f"build_strat01_q4k_q8k_reference_generic_oracle: {error}")
