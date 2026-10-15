#!/usr/bin/env python3
"""Launch three private, bounded mathematical diagnostics on existing accounts.

No local model acquisition, tensor calculations, or numerical training.
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
import sys
import time
import zipfile

import kaggle_ops

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/predictive_geometry_pilot"
CONTROL = Path("E:/SiliconLLM_Artifacts/campaigns/predictive_geometry_20261015")
PROTOCOL = ROOT / "docs/research/PREDICTIVE_GEOMETRY_20261015/PROTOCOL.md"
ARMS = {"geometry": "acct1", "temporal": "acct2", "reuse": "acct3"}
MAX_SECONDS = 7200


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def cli(account, *args):
    return kaggle_ops.kaggle(account, *args, check=False)


def build(mode):
    out = CONTROL / mode
    if out.exists():
        raise RuntimeError("refusing to overwrite a frozen arm")
    files = {name: (SOURCE / name).read_bytes() for name in
             ("support.py", "geometry.py", "temporal.py", "reuse.py", "exclusions.json")}
    files["common.py"] = (ROOT / "benchmarks/context_program_pilot/pilot.py").read_bytes()
    files["config.json"] = (SOURCE / (mode + "_config.json")).read_bytes()
    files["PROTOCOL.md"] = PROTOCOL.read_bytes()
    for name, data in files.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=name)
    config = json.loads(files["config.json"])
    if config["mode"] != mode or config["wall_seconds"] > 6000:
        raise RuntimeError("wrong arm or excessive internal cap")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    payload = buffer.getvalue()
    pins = {name: sha(data) for name, data in files.items()}
    code = BOOTSTRAP.replace("EMBEDDED_PAYLOAD", base64.b64encode(payload).decode())
    code = code.replace("EMBEDDED_SHA", sha(payload)).replace("EMBEDDED_PINS", repr(json.dumps(pins)))
    ast.parse(code)
    account = ARMS[mode]
    owner = kaggle_ops.server_identity(account)
    if not owner:
        raise RuntimeError("server identity is unproven")
    title = "predictive-" + mode + "-20261015"
    metadata = {"id": owner + "/" + title, "title": title, "code_file": "entry.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [], "kernel_sources": [], "competition_sources": []}
    out.mkdir(parents=True)
    (out / "entry.py").write_text(code, encoding="utf-8", newline="\n")
    save(out / "kernel-metadata.json", metadata)
    receipt = {"schema": "PREDICTIVE_GEOMETRY_BUNDLE_V1", "mode": mode, "account": account,
               "owner": owner, "kernel": metadata["id"], "created_wall_time": time.time(),
               "source_files": pins, "payload_sha256": sha(payload),
               "entry_sha256": sha((out / "entry.py").read_bytes()),
               "metadata_sha256": sha((out / "kernel-metadata.json").read_bytes()),
               "max_seconds": MAX_SECONDS, "conservative_t4_hours_cap": 4}
    save(out / "bundle_receipt.json", receipt)
    print(json.dumps({"mode": mode, "kernel": metadata["id"], "bundle": str(out)}))


def load(mode):
    out = CONTROL / mode
    receipt = json.loads((out / "bundle_receipt.json").read_text())
    for name, key in (("entry.py", "entry_sha256"), ("kernel-metadata.json", "metadata_sha256")):
        if sha((out / name).read_bytes()) != receipt[key]:
            raise RuntimeError("frozen bundle changed")
    return out, receipt


def launch(mode):
    out, receipt = load(mode)
    if (out / "launch_attempt.json").exists():
        raise RuntimeError("attempt exists; inspect the remote state, do not blindly retry")
    attempts = list(CONTROL.glob("*/launch_attempt.json"))
    if len(attempts) >= 3:
        raise RuntimeError("three-arm allocation limit reached")
    account, kernel = receipt["account"], receipt["kernel"]
    if kaggle_ops.server_identity(account) != receipt["owner"]:
        raise RuntimeError("identity mismatch")
    inventory = cli(account, "kernels", "list", "--mine", "--csv", "--page-size", "200")
    if inventory.returncode:
        raise RuntimeError("cannot establish remote freshness")
    rows = list(csv.DictReader(io.StringIO(inventory.stdout)))
    if not rows or "ref" not in rows[0] or len(rows) >= 200:
        raise RuntimeError("ambiguous remote inventory")
    if any(row["ref"] == kernel for row in rows):
        raise RuntimeError("remote kernel already exists")
    quota = cli(account, "quota", "--csv")
    if quota.returncode:
        raise RuntimeError("quota request failed")
    gpu = [r for r in csv.DictReader(io.StringIO(quota.stdout)) if r.get("resource") == "GPU"]
    if len(gpu) != 1 or float(gpu[0]["remaining"].removesuffix("h")) < 4:
        raise RuntimeError("insufficient verified GPU quota")
    save(out / "preflight.json", {"wall_time": time.time(), "quota_csv": quota.stdout,
         "inventory_refs": [r["ref"] for r in rows], "kernel": kernel})
    save(out / "launch_attempt.json", {"mode": mode, "kernel": kernel, "account": account,
         "wall_time": time.time(), "max_seconds": MAX_SECONDS,
         "conservative_t4_hours_cap": 4,
         "bundle_receipt_sha256": sha((out / "bundle_receipt.json").read_bytes())})
    result = cli(account, "kernels", "push", "--path", str(out), "--timeout", str(MAX_SECONDS),
                 "--accelerator", "NvidiaTeslaT4")
    save(out / "push_result.json", {"returncode": result.returncode, "wall_time": time.time(),
         "stdout": result.stdout, "stderr": result.stderr})
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


def inspect(mode, command):
    out, receipt = load(mode)
    if command == "status":
        result = cli(receipt["account"], "kernels", "status", receipt["kernel"])
    elif command == "readback":
        target = out / ("readback-" + str(time.time_ns()))
        result = cli(receipt["account"], "kernels", "pull", receipt["kernel"], "--metadata", "--path", str(target))
    else:
        target = out / ("output-" + str(time.time_ns()))
        result = cli(receipt["account"], "kernels", "output", receipt["kernel"], "--path", str(target), "--page-size", "200")
    save(out / (command + "-" + str(time.time_ns()) + ".json"),
         {"returncode": result.returncode, "wall_time": time.time(), "stdout": result.stdout, "stderr": result.stderr})
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


BOOTSTRAP = r'''import base64, hashlib, io, json, os, pathlib, subprocess, sys, time, traceback, zipfile
started=time.monotonic()
root=pathlib.Path('/kaggle/working/predictive_source'); root.mkdir(exist_ok=True)
out=pathlib.Path('/kaggle/working/predictive_output'); out.mkdir(exist_ok=True)
terminal={'schema':'PREDICTIVE_BOOTSTRAP_V1','started_wall_time':time.time()}
def run(command):
    left=6900-(time.monotonic()-started)
    if left<30: raise TimeoutError('bounded bootstrap budget exhausted')
    subprocess.run(command,check=True,timeout=left)
try:
    data=base64.b64decode('EMBEDDED_PAYLOAD')
    assert hashlib.sha256(data).hexdigest()=='EMBEDDED_SHA'
    pins=json.loads(EMBEDDED_PINS)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert sorted(z.namelist())==sorted(pins)
        for name,expected in pins.items():
            content=z.read(name); assert hashlib.sha256(content).hexdigest()==expected
            (root/name).write_bytes(content)
    (out/'source_manifest.json').write_text(json.dumps(pins,indent=2))
    run([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','transformers==4.51.3','datasets==3.5.0'])
    os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ['PYTHONDONTWRITEBYTECODE']='1'
    cfg=json.loads((root/'config.json').read_text())
    cfg['wall_seconds']=min(cfg['wall_seconds'],int(6720-(time.monotonic()-started)))
    if cfg['wall_seconds']<1200: raise TimeoutError('insufficient budget after setup')
    (root/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    run([sys.executable,'-B',str(root/'support.py'),'--config',str(root/'effective_config.json'),'--output',str(out)])
    terminal['status']='PROCESS_COMPLETE_INSPECT_SCIENTIFIC_REPORT'
except BaseException as exc:
    terminal.update(status='FAILED',exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc())
    raise
finally:
    terminal.update(elapsed_seconds=time.monotonic()-started,finished_wall_time=time.time())
    manifest=out/'artifact_manifest.json'
    if manifest.exists(): terminal['artifact_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    (out/'bootstrap_terminal.json').write_text(json.dumps(terminal,indent=2))
'''


def main():
    os.environ["PYTHONUTF8"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "launch", "status", "readback", "output"))
    parser.add_argument("--mode", choices=(*ARMS, "all"), required=True)
    args = parser.parse_args()
    modes = list(ARMS) if args.mode == "all" else [args.mode]
    for mode in modes:
        if args.command == "build": build(mode)
        elif args.command == "launch": launch(mode)
        else: inspect(mode, args.command)


if __name__ == "__main__":
    main()
