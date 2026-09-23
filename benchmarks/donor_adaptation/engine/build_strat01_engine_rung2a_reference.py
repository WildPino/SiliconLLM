#!/usr/bin/env python3
"""Build the model-free-testable pinned llama.cpp Rung-2A trace apparatus."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4")
PINNED_HEAD = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"


class BuildError(RuntimeError):
    pass


def verify_pinned_llama(source: Path = PINNED_LLAMA) -> None:
    if not (source / "CMakeLists.txt").is_file():
        raise BuildError("pinned llama.cpp source is unavailable")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=source, text=True, capture_output=True, check=False)
    if head.returncode or head.stdout.strip() != PINNED_HEAD or status.returncode or status.stdout.strip():
        raise BuildError("llama.cpp source is not the clean pinned commit")


def cmake_project(source: Path, llama_source: Path, *, rung2b: bool = False, rung2c: bool = False) -> str:
    if rung2b and rung2c:
        raise BuildError("reference build variants are mutually exclusive")
    variant = "rung2c" if rung2c else "rung2b" if rung2b else "rung2a"
    target = f"strat01_engine_{variant}_reference"
    definition = f"target_compile_definitions({target} PRIVATE STRAT01_{variant.upper()}=1)\n" if variant != "rung2a" else ""
    return f'''cmake_minimum_required(VERSION 3.20)
project({target} LANGUAGES C CXX)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("{llama_source.as_posix()}" llama-cpp)
add_executable({target} "{source.as_posix()}")
target_compile_features({target} PRIVATE cxx_std_17)
{definition}target_link_libraries({target} PRIVATE llama)
'''


def build(build_dir: Path, configuration: str = "Release", *, rung2b: bool = False, rung2c: bool = False) -> Path:
    verify_pinned_llama()
    if not SOURCE.is_file():
        raise BuildError("reference trace source is unavailable")
    if rung2b and rung2c:
        raise BuildError("reference build variants are mutually exclusive")
    variant = "rung2c" if rung2c else "rung2b" if rung2b else "rung2a"
    target = f"strat01_engine_{variant}_reference"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeLists.txt").write_text(cmake_project(SOURCE, PINNED_LLAMA, rung2b=rung2b, rung2c=rung2c), encoding="utf-8", newline="\n")
    configured = subprocess.run(["cmake", "-S", str(build_dir), "-B", str(build_dir / "cmake-build")], text=True, capture_output=True, check=False)
    if configured.returncode:
        raise BuildError("CMake configure failed:\n" + "\n".join((configured.stdout + configured.stderr).splitlines()[-20:]))
    compiled = subprocess.run(["cmake", "--build", str(build_dir / "cmake-build"), "--target", target, "--config", configuration], text=True, capture_output=True, check=False)
    if compiled.returncode:
        raise BuildError("CMake build failed:\n" + "\n".join((compiled.stdout + compiled.stderr).splitlines()[-30:]))
    candidates = [candidate for name in (target + ".exe", target) for candidate in (build_dir / "cmake-build").rglob(name)]
    if len(candidates) != 1:
        raise BuildError("build did not produce exactly one reference trace binary")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    if args.build_dir:
        print(build(args.build_dir, args.config))
        return 0
    with tempfile.TemporaryDirectory(prefix="strat01-rung2a-reference-") as temporary:
        print(build(Path(temporary), args.config))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        raise SystemExit(f"build_strat01_engine_rung2a_reference: {error}")
