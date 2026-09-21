#!/usr/bin/env python3
"""Build the pinned-GGML attention-stage helper."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine.build_strat01_rung2a_projection_diagnostic import (
    BuildError,
    PINNED_LLAMA,
    verify_pinned_llama,
)

SOURCE = Path(__file__).resolve().with_name("strat01_attention_stage_diagnostic.cpp")


def cmake_project(source: Path, llama_source: Path) -> str:
    return f'''cmake_minimum_required(VERSION 3.20)
project(strat01_attention_stage_diagnostic LANGUAGES C CXX)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable(strat01_attention_stage_diagnostic "{source.as_posix()}")
target_compile_features(strat01_attention_stage_diagnostic PRIVATE cxx_std_17)
target_compile_options(strat01_attention_stage_diagnostic PRIVATE -mavx2 -mfma)
target_include_directories(strat01_attention_stage_diagnostic PRIVATE "{(llama_source / 'ggml' / 'include').as_posix()}")
target_link_libraries(strat01_attention_stage_diagnostic PRIVATE llama)
'''


def build(build_dir: Path, configuration: str = "Release") -> Path:
    verify_pinned_llama()
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeLists.txt").write_text(cmake_project(SOURCE, PINNED_LLAMA), encoding="utf-8", newline="\n")
    configured = subprocess.run(["cmake", "-S", str(build_dir), "-B", str(build_dir / "cmake-build")], text=True, capture_output=True, check=False)
    if configured.returncode:
        raise BuildError("CMake configure failed:\n" + "\n".join((configured.stdout + configured.stderr).splitlines()[-30:]))
    compiled = subprocess.run(["cmake", "--build", str(build_dir / "cmake-build"), "--target", "strat01_attention_stage_diagnostic", "--config", configuration], text=True, capture_output=True, check=False)
    if compiled.returncode:
        raise BuildError("CMake build failed:\n" + "\n".join((compiled.stdout + compiled.stderr).splitlines()[-40:]))
    candidates = [candidate for name in ("strat01_attention_stage_diagnostic.exe", "strat01_attention_stage_diagnostic") for candidate in (build_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise BuildError("build did not produce exactly one attention-stage helper")
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
        raise SystemExit(f"build_strat01_attention_stage_diagnostic: {error}")
