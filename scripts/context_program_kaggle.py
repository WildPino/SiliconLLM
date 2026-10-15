#!/usr/bin/env python3
"""Freeze, launch and retrieve the bounded private context-program pilot.

No local model imports, weight downloads or numerical work. Credentials remain
in the existing external account store and are handled by kaggle_ops.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import kaggle_ops

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/context_program_pilot"
CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/context_program_20261015")
MAX_SECONDS = 14400


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def invoke(account: str, *args: str) -> subprocess.CompletedProcess:
    return kaggle_ops.kaggle(account, *args, check=False)


def bundle_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,45}", run_id):
        raise ValueError("invalid run id")
    result = CAMPAIGN / run_id
    if result.parent.resolve() != CAMPAIGN.resolve():
        raise ValueError("campaign boundary violation")
    return result


def build(args: argparse.Namespace) -> None:
    target = bundle_dir(args.run_id)
    if target.exists():
        raise RuntimeError("run directory already exists; never overwrite frozen runs")
    sources = {}
    for name in ("pilot.py", "followup.py", "test_pilot.py", "config.json"):
        data = (SOURCE / name).read_bytes()
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=name)
        else:
            cfg = json.loads(data)
            if not 0 < cfg["wall_seconds"] <= 12600:
                raise ValueError("pilot exceeds internal wall budget")
        sources[name] = data
    sources["PROTOCOL.md"] = (ROOT / "docs/research/CONTEXT_PROGRAM_PILOT_20261015/PROTOCOL.md").read_bytes()
    # Source-only archive: never enumerate the workspace or attach old artifacts.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    file_pins = {n: sha(d) for n, d in sources.items()}
    code = BOOTSTRAP.replace("PAYLOAD_BASE64", encoded).replace("PAYLOAD_SHA", sha(payload))
    code = code.replace("FILE_PINS_JSON", repr(json.dumps(file_pins, sort_keys=True)))
    ast.parse(code, filename="context_program_entry.py")
    owner = kaggle_ops.server_identity(args.account)
    if not owner or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise RuntimeError("server identity not established")
    title = "context-program-" + args.run_id
    metadata = {
        "id": owner + "/" + title,
        "title": title,
        "code_file": "context_program_entry.py",
        "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }
    target.mkdir(parents=True)
    (target / "context_program_entry.py").write_text(code, encoding="utf-8", newline="\n")
    save(target / "kernel-metadata.json", metadata)
    save(target / "bundle_receipt.json", {
        "schema": "CONTEXT_PROGRAM_BUNDLE_V1", "created_wall_time": time.time(),
        "account": args.account, "owner": owner, "kernel": metadata["id"],
        "max_seconds": MAX_SECONDS, "conservative_t4_hours_cap": 8,
        "source_files": file_pins, "payload_sha256": sha(payload),
        "entry_sha256": sha((target / "context_program_entry.py").read_bytes()),
        "metadata_sha256": sha((target / "kernel-metadata.json").read_bytes()),
        "protocol_sha256": sha((ROOT / "docs/research/CONTEXT_PROGRAM_PILOT_20261015/PROTOCOL.md").read_bytes()),
    })
    print(json.dumps({"bundle": str(target), "kernel": metadata["id"]}))


def load_run(args: argparse.Namespace) -> tuple[Path, dict]:
    path = bundle_dir(args.run_id)
    receipt = json.loads((path / "bundle_receipt.json").read_text(encoding="utf-8"))
    for name, key in (("context_program_entry.py", "entry_sha256"), ("kernel-metadata.json", "metadata_sha256")):
        if sha((path / name).read_bytes()) != receipt[key]:
            raise RuntimeError("frozen bundle changed: " + name)
    return path, receipt


def launch(args: argparse.Namespace) -> None:
    path, receipt = load_run(args)
    if (path / "launch_attempt.json").exists():
        raise RuntimeError("launch already attempted; inspect remote status, never blindly retry")
    attempts = list(CAMPAIGN.glob("*/launch_attempt.json"))
    if attempts:
        raise RuntimeError("first-stage campaign already has a launch; qualification required before any successor")
    account, kernel = receipt["account"], receipt["kernel"]
    if kaggle_ops.server_identity(account) != receipt["owner"]:
        raise RuntimeError("account identity drift")
    inventory = invoke(account, "kernels", "list", "--mine", "--csv", "--page-size", "200")
    if inventory.returncode:
        raise RuntimeError("cannot establish fresh kernel inventory")
    rows = list(csv.DictReader(io.StringIO(inventory.stdout)))
    if len(rows) >= 200 or any(row.get("ref") == kernel for row in rows):
        raise RuntimeError("kernel is not demonstrably fresh")
    if not rows or "ref" not in rows[0]:
        raise RuntimeError("unexpected kernel inventory")
    quota = invoke(account, "quota", "--csv")
    if quota.returncode:
        raise RuntimeError("quota read failed")
    qrows = list(csv.DictReader(io.StringIO(quota.stdout)))
    # The raw non-secret quota response is retained for review and never guessed.
    gpu = [r for r in qrows if any(str(v).upper() == "GPU" for v in r.values())]
    if len(gpu) != 1:
        raise RuntimeError("unrecognized GPU quota response")
    remaining = next((v for k, v in gpu[0].items() if "remaining" in k.lower()), None)
    if remaining is None or float(remaining.removesuffix("h")) < 8:
        raise RuntimeError("require at least eight remaining quota hours conservatively")
    save(path / "preflight.json", {"wall_time": time.time(), "quota_csv": quota.stdout,
         "inventory_refs": [r.get("ref") for r in rows], "kernel": kernel})
    save(path / "launch_attempt.json", {"wall_time": time.time(), "account": account,
         "kernel": kernel, "max_seconds": MAX_SECONDS, "bundle_sha256": sha((path / "bundle_receipt.json").read_bytes())})
    result = invoke(account, "kernels", "push", "--path", str(path),
                    "--timeout", str(MAX_SECONDS), "--accelerator", "NvidiaTeslaT4")
    save(path / "push_result.json", {"wall_time": time.time(), "returncode": result.returncode,
         "stdout": result.stdout, "stderr": result.stderr})
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


def inspect(args: argparse.Namespace) -> None:
    path, receipt = load_run(args)
    if args.command == "status":
        result = invoke(receipt["account"], "kernels", "status", receipt["kernel"])
    else:
        out = path / ("output-" + str(int(time.time())))
        out.mkdir()
        result = invoke(receipt["account"], "kernels", "output", receipt["kernel"], "--path", str(out))
    save(path / (args.command + "-" + str(time.time_ns()) + ".json"),
         {"wall_time": time.time(), "returncode": result.returncode,
          "stdout": result.stdout, "stderr": result.stderr})
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


BOOTSTRAP = r'''# Frozen source-only Kaggle entrypoint.
import base64, hashlib, io, json, os, pathlib, subprocess, sys, time, traceback, zipfile
started = time.monotonic()
out = pathlib.Path('/kaggle/working/context_program_output')
out.mkdir(parents=True, exist_ok=True)
root = pathlib.Path('/kaggle/working/context_program_source')
root.mkdir(parents=True, exist_ok=True)
terminal = {'schema': 'CONTEXT_PROGRAM_BOOTSTRAP_V1', 'started_wall_time': time.time()}
def run(command):
    remaining = 14040 - (time.monotonic() - started)
    if remaining < 30:
        raise TimeoutError('total cloud bootstrap budget exhausted')
    return subprocess.run(command, check=True, timeout=remaining)
try:
    payload = base64.b64decode('PAYLOAD_BASE64')
    assert hashlib.sha256(payload).hexdigest() == 'PAYLOAD_SHA'
    pins = json.loads(FILE_PINS_JSON)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert sorted(z.namelist()) == sorted(pins)
        for name in pins:
            data = z.read(name)
            assert hashlib.sha256(data).hexdigest() == pins[name]
            (root / name).write_bytes(data)
    (out / 'source_manifest.json').write_text(json.dumps(pins, indent=2))
    run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
         'transformers==4.51.3', 'datasets==3.5.0'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    run([sys.executable, str(root / 'pilot.py'), '--self-test'])
    run([sys.executable, str(root / 'followup.py'), '--self-test'])
    effective = json.loads((root / 'config.json').read_text())
    effective['wall_seconds'] = min(effective['wall_seconds'], int(13860 - (time.monotonic() - started)))
    if effective['wall_seconds'] < effective['reserve_seconds'] + 600:
        raise TimeoutError('dependency setup left insufficient safe pilot budget')
    (root / 'effective_config.json').write_text(json.dumps(effective, indent=2))
    run([sys.executable, str(root / 'pilot.py'), '--config', str(root / 'effective_config.json'), '--output', str(out)])
    terminal['status'] = 'PROCESS_COMPLETED_INSPECT_SCIENTIFIC_REPORT'
except BaseException as exc:
    terminal.update(status='FAILED', exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
    raise
finally:
    terminal['elapsed_seconds'] = time.monotonic() - started
    terminal['finished_wall_time'] = time.time()
    (out / 'bootstrap_terminal.json').write_text(json.dumps(terminal, indent=2))
'''


def main() -> None:
    # The child CLI prints Unicode progress/log text on Windows. Its locale
    # encoding otherwise fails after downloads have already succeeded.
    os.environ["PYTHONUTF8"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "launch", "status", "output"):
        p = sub.add_parser(command)
        p.add_argument("--run-id", required=True)
        if command == "build":
            p.add_argument("--account", choices=kaggle_ops.ACCOUNTS, default="acct1")
    args = parser.parse_args()
    {"build": build, "launch": launch, "status": inspect, "output": inspect}[args.command](args)


if __name__ == "__main__":
    main()
