"""Freeze and, only on explicit request, execute the synthetic Q4_K layout gate.

This does not measure latency or validate donor weights. No command runs by default.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FILES = {
    "fixture": HERE / "granite_child256_layout.c",
    "runner": Path(__file__).resolve(),
    "q4_header": ROOT / "benchmarks/phase60/strat01_q4k_q8k.h",
    "protocol": ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/GRANITE_CHILD256_LAYOUT_PROTOCOL_20261015.md",
}
FLAGS = ["-O2", "-std=c11", "-march=x86-64-v3", "-fno-fast-math", "-ffp-contract=off"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compiler():
    found = shutil.which("clang")
    if not found:
        raise RuntimeError("clang_not_found")
    return Path(found).resolve()


def snapshot():
    cc = compiler()
    return {
        "schema": "granite_child256_layout_freeze_v1",
        "files_sha256": {key: digest(value) for key, value in FILES.items()},
        "compiler": str(cc),
        "compiler_sha256": digest(cc),
        "flags": FLAGS,
        "synthetic_only": True,
    }


def write_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute(args):
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    if binding != snapshot():
        raise RuntimeError("freeze_binding_mismatch")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / ("granite_child256_layout.exe" if os.name == "nt" else "granite_child256_layout")
    record = {
        "schema": "granite_child256_layout_result_v1",
        "binding_sha256": digest(args.binding),
        "synthetic_only": True,
        "cost_or_rate_admission": False,
        "compile": None,
        "fixture": None,
    }
    try:
        cmd = [binding["compiler"], *FLAGS, str(FILES["fixture"]), "-lm", "-o", str(binary)]
        t0 = time.monotonic()
        cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
        record["compile"] = {"argv": cmd, "exit_code": cp.returncode,
                             "elapsed_s": time.monotonic() - t0,
                             "stdout": cp.stdout[-8192:], "stderr": cp.stderr[-8192:]}
        if cp.returncode:
            raise RuntimeError("compile_failed")
        t0 = time.monotonic()
        run = subprocess.run([str(binary)], cwd=ROOT, capture_output=True, text=True,
                             timeout=60, check=False)
        receipt = {"exit_code": run.returncode, "elapsed_s": time.monotonic() - t0,
                   "stdout": run.stdout[-8192:], "stderr": run.stderr[-8192:]}
        record["fixture"] = receipt
        if run.returncode:
            raise RuntimeError("fixture_failed")
        parsed = json.loads(run.stdout)
        expected = {"status": "layout_pass", "synthetic_only": True,
                    "q4_block_bytes": 144, "q8_block_bytes": 292,
                    "copied_blocks": 6144,
                    "gate_up_parent_bytes": 589824, "gate_up_child_bytes": 589824,
                    "down_parent_bytes": 294912, "down_child_bytes": 294912}
        if any(parsed.get(k) != value for k, value in expected.items()):
            raise RuntimeError("layout_receipt_mismatch")
        for key in ("gate_up_relative_l2", "down_relative_l2",
                    "down_child_reconstruction_relative_l2", "swiglu_relative_l2"):
            if not 0 <= parsed[key] <= 1e-5:
                raise RuntimeError("numeric_gate_failed:" + key)
        record["parsed"] = parsed
        record["decision"] = "layout_pass_only"
    except Exception as exc:
        record["decision"] = "layout_fail"
        record["error"] = f"{type(exc).__name__}:{exc}"
    write_exclusive(out / "result.json", record)
    print(json.dumps({"result": str(out / "result.json"), "decision": record["decision"]}))
    return 0 if record["decision"] == "layout_pass_only" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path, metavar="JSON")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--binding", type=Path, help="Required with --run")
    parser.add_argument("--output", type=Path, help="New exclusive directory for --run")
    args = parser.parse_args()
    if args.freeze:
        if args.binding or args.output:
            parser.error("--freeze accepts no --binding or --output")
        write_exclusive(args.freeze, snapshot())
        print(str(args.freeze.resolve()))
        return 0
    if not args.binding or not args.output:
        parser.error("--run requires --binding and --output")
    return execute(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"layout_controller_failure:{type(exc).__name__}:{exc}", file=sys.stderr)
        sys.exit(2)
