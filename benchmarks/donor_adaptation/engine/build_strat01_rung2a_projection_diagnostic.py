#!/usr/bin/env python3
"""Build the pinned-GGML helper for the offline Rung-2A projection diagnostic."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "strat01_rung2a_projection_diagnostic.cpp"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4")
PINNED_HEAD = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"


class BuildError(RuntimeError):
    pass


def validate_pinned_state(head: str, status: str) -> None:
    if head.strip() != PINNED_HEAD or status.strip():
        raise BuildError("llama.cpp source is not the clean pinned commit")


def verify_pinned_llama(source: Path = PINNED_LLAMA) -> None:
    if not (source / "CMakeLists.txt").is_file():
        raise BuildError("pinned llama.cpp source is unavailable")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=source, text=True, capture_output=True, check=False)
    if head.returncode or status.returncode:
        raise BuildError("cannot inspect pinned llama.cpp source")
    validate_pinned_state(head.stdout, status.stdout)


def cmake_project(source: Path, llama_source: Path) -> str:
    return f'''cmake_minimum_required(VERSION 3.20)
project(strat01_rung2a_projection_diagnostic LANGUAGES C CXX)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable(strat01_rung2a_projection_diagnostic "{source.as_posix()}")
target_compile_features(strat01_rung2a_projection_diagnostic PRIVATE cxx_std_17)
target_compile_options(strat01_rung2a_projection_diagnostic PRIVATE -mavx2 -mfma)
target_include_directories(strat01_rung2a_projection_diagnostic PRIVATE
    "{(llama_source / 'ggml' / 'include').as_posix()}"
    "{(llama_source / 'ggml' / 'src').as_posix()}"
    "{(llama_source / 'ggml' / 'src' / 'ggml-cpu').as_posix()}")
target_link_libraries(strat01_rung2a_projection_diagnostic PRIVATE llama)
'''


def build(build_dir: Path, configuration: str = "Release") -> Path:
    verify_pinned_llama()
    if not SOURCE.is_file():
        raise BuildError("projection diagnostic source is unavailable")
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeLists.txt").write_text(cmake_project(SOURCE, PINNED_LLAMA), encoding="utf-8", newline="\n")
    configured = subprocess.run(
        ["cmake", "-S", str(build_dir), "-B", str(build_dir / "cmake-build")],
        text=True,
        capture_output=True,
        check=False,
    )
    if configured.returncode:
        raise BuildError("CMake configure failed:\n" + "\n".join((configured.stdout + configured.stderr).splitlines()[-30:]))
    compiled = subprocess.run(
        ["cmake", "--build", str(build_dir / "cmake-build"), "--target", "strat01_rung2a_projection_diagnostic", "--config", configuration],
        text=True,
        capture_output=True,
        check=False,
    )
    if compiled.returncode:
        raise BuildError("CMake build failed:\n" + "\n".join((compiled.stdout + compiled.stderr).splitlines()[-40:]))
    names = ("strat01_rung2a_projection_diagnostic.exe", "strat01_rung2a_projection_diagnostic")
    candidates = [candidate for name in names for candidate in (build_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise BuildError("build did not produce exactly one diagnostic binary")
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
        raise SystemExit(f"build_strat01_rung2a_projection_diagnostic: {error}")
