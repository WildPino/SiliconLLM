"""Build and adjudicate STRAT-01 C-engine GGUF parity rung 0.

This runner performs one bounded-memory inspection of the frozen accepted Q4
artifact.  It is an identity/layout gate, not a throughput benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENGINE_SOURCE = ROOT / "benchmarks" / "phase60" / "engine.c"
INSPECTOR_SOURCE = ROOT / "benchmarks" / "phase60" / "strat01_gguf_inspect.h"
DEFAULT_MODEL = (
    ROOT
    / "benchmarks"
    / "donor_adaptation"
    / "density"
    / "results"
    / "strat01_gigachat_q4_97045b2"
    / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks"
    / "donor_adaptation"
    / "engine"
    / "results"
    / "strat01_gigachat_engine_rung0_20260921"
)
EXPECTED_SIZE = 6_474_702_976
EXPECTED_SHA256 = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
EXPECTED_TYPES = {"F32": 129, "Q5_0": 26, "Q4_K": 233, "Q6_K": 26}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = args.model.resolve()
    output = args.output_dir.resolve()
    inventory_path = output / "inventory.json"
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    clang = shutil.which("clang")
    if not clang:
        raise SystemExit("clang not found")
    if not model.is_file():
        raise SystemExit(f"accepted artifact not found: {model}")
    if model.stat().st_size != EXPECTED_SIZE:
        raise SystemExit(f"accepted artifact size mismatch: {model.stat().st_size}")

    binary = output / "engine_rung0.exe"
    compile_command = [
        clang,
        "-std=c11",
        "-O3",
        "-mavx2",
        "-mfma",
        str(ENGINE_SOURCE),
        "-o",
        str(binary),
        "-lm",
    ]
    started = datetime.now(timezone.utc).isoformat()
    compile_started = time.perf_counter()
    compiled = capture(compile_command)
    compile_seconds = time.perf_counter() - compile_started
    (output / "compile.stdout.log").write_text(compiled.stdout, encoding="utf-8")
    (output / "compile.stderr.log").write_text(compiled.stderr, encoding="utf-8")

    run_command = [
        str(binary),
        "--strat01-gguf-inspect",
        str(model),
        "--json",
        str(inventory_path),
    ]
    run_returncode: int | None = None
    run_seconds: float | None = None
    run_stdout = ""
    run_stderr = ""
    if compiled.returncode == 0:
        run_started = time.perf_counter()
        run = capture(run_command)
        run_seconds = time.perf_counter() - run_started
        run_returncode = run.returncode
        run_stdout, run_stderr = run.stdout, run.stderr
    (output / "run.stdout.log").write_text(run_stdout, encoding="utf-8")
    (output / "run.stderr.log").write_text(run_stderr, encoding="utf-8")

    errors: list[str] = []
    inventory: dict[str, object] | None = None
    if compiled.returncode != 0:
        errors.append(f"compile return code {compiled.returncode}")
    elif run_returncode != 0:
        errors.append(f"inspect return code {run_returncode}")
    if inventory_path.is_file():
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid inventory JSON: {exc}")
    else:
        errors.append("inventory JSON missing")

    if inventory is not None:
        checks = {
            "gate_status": inventory.get("gate_status") == "PASS_ENGINE_RUNG0",
            "byte_size": inventory.get("byte_size") == EXPECTED_SIZE,
            "sha256": inventory.get("sha256") == EXPECTED_SHA256,
            "metadata_count": inventory.get("metadata_count") == 47,
            "tensor_count": inventory.get("tensor_count") == 414,
            "type_census": inventory.get("type_census") == EXPECTED_TYPES,
            "mtp_excluded": inventory.get("mtp_excluded") is True,
            "descriptor_count": len(inventory.get("tensors", [])) == 414,
            "gguf_contract": inventory.get("gguf")
            == {"magic": "GGUF", "version": 3, "alignment": 32, "data_offset": 6_102_912},
        }
        errors.extend(name for name, passed in checks.items() if not passed)
    else:
        checks = {}

    head = capture(["git", "rev-parse", "HEAD"]).stdout.strip()
    compiler_version = capture([clang, "--version"]).stdout.splitlines()
    manifest = {
        "schema": "strat01_gigachat_engine_rung0_v1",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ENGINE_RUNG0" if not errors else "FAIL_ENGINE_RUNG0",
        "scope": "accepted Q4 GGUF identity and complete descriptor inventory through phase60/engine.c",
        "non_claims": [
            "quantized matvec parity",
            "operator or tokenizer parity",
            "generation quality",
            "accepted-token throughput",
        ],
        "git_head": head,
        "engine_source": {"path": str(ENGINE_SOURCE), "sha256": sha256_file(ENGINE_SOURCE)},
        "inspector_source": {"path": str(INSPECTOR_SOURCE), "sha256": sha256_file(INSPECTOR_SOURCE)},
        "binary": {"path": str(binary), "sha256": sha256_file(binary) if binary.is_file() else None},
        "model": {"path": str(model), "bytes": model.stat().st_size, "expected_sha256": EXPECTED_SHA256},
        "compile": {"command": compile_command, "returncode": compiled.returncode, "seconds": compile_seconds},
        "run": {"command": run_command, "returncode": run_returncode, "seconds": run_seconds},
        "checks": checks,
        "errors": errors,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "compiler": compiler_version,
            "cwd": os.getcwd(),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(output), "errors": errors}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
