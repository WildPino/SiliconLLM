#!/usr/bin/env python3
"""Verified Kaggle dispatcher for H1 session 3 and H2I Phase B.

This is operational apparatus only.  It does not choose hyperparameters or interpret GPU
progress.  It validates the frozen local bundle, stages it without modifying the bundle,
uploads it through scripts/kaggle_ops.py, waits for the dataset to become attachable, and
pushes a private T4 kernel whose first action is to rehash the mounted payload.

Commands:
    check     {h1,h2i,all}
    preflight {h1,h2i} ACCOUNT
    upload    {h1,h2i} ACCOUNT
    push      {h1,h2i} ACCOUNT
    launch    {h1,h2i} ACCOUNT
    status    {h1,h2i} ACCOUNT
    output    {h1,h2i} ACCOUNT [--path DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import kaggle_ops  # noqa: E402

MACHINE = "NvidiaTeslaT4"
SESSION_SECONDS = 12 * 3600
READY_TIMEOUT_S = 20 * 60


TARGETS = {
    "h2i": {
        "bundle": HERE / "_h2i_bundle",
        "manifest_sha256": "a37d65fdc3e7833bb2b539abd43a07d958a98bc9dd65d091240716a5606abf20",
        "dataset_slug": "h2i-one-byte-phase-b-bundle",
        "dataset_title": "H2I one-byte Phase B bundle",
        "kernel_slug": "h2i-one-byte-phase-b",
        "kernel_title": "H2I one-byte Phase B",
        "copy_to_working": True,
        "manifest_test": {"stage": "H2I_PHASE_B", "status": "READY_NOT_RUN"},
        "command_test": {
            "steps": 4000, "bs": 2, "accum": 8, "lr": 0.0002,
            "router_lr": 0.0003, "aux": 0.01, "every": 250,
            "seed": 4242, "max_hours": 11.0, "resume": None,
        },
        "command_key": "command",
        "command": [
            "python3", "h2i_qat.py",
            "--manifest", "MANIFEST.json",
            "--factors", "h0_trained3.npz",
            "--labels", "labels_E256.npz",
            "--stats", "h1_actstats.npz",
            "--routers", "e37_routers_E256.npz",
            "--phase-a", "h2i_applied_8L.json",
            "--train", "h1_train.npz",
            "--heldout", "h1_heldout.npz",
            "--calib", "h1_calib.npz",
            "--out", "/kaggle/working/h2i_trained_s1.npz",
            "--steps", "4000", "--bs", "2", "--accum", "8",
            "--lr", "2e-4", "--router-lr", "3e-4", "--aux", "0.01",
            "--every", "250", "--seed", "4242", "--max-hours", "11.0",
        ],
        "outputs": ["h2i_trained_s1.npz", "h2i_trained_s1.json"],
    },
    "h1": {
        "bundle": HERE / "_h1_bundle",
        "manifest_sha256": "bae2701ac8091b79a0f690298d0aa3ddfd3bad255f714dbe5aa05b678bbea0ee",
        "dataset_slug": "h1-qat-bundle",
        "dataset_title": "H1 carve-trained bundle",
        "kernel_slug": "h1-qat-run-session-3",
        "kernel_title": "H1 QAT run session 3",
        "copy_to_working": False,
        "manifest_test": {"sessions": 3},
        "command_test": {
            "resume": "h1_trained_s2.npz", "factors": "h0_trained3.npz",
            "out": "h1_trained_s3.npz", "steps": 4000, "bs": 2, "accum": 8,
            "lr": "2e-4", "router_lr": "0.0003", "aux": "0.01", "every": 250,
            "seed": 3141, "max_hours": 11.0,
            "expected_steps_at_51.7_s_per_step": 765,
        },
        "command_key": "command_session3",
        "command": [
            "python3", "h1_qat.py",
            "--factors", "h0_trained3.npz",
            "--resume", "h1_trained_s2.npz",
            "--labels", "labels_E256.npz",
            "--stats", "h1_actstats.npz",
            "--train", "h1_train.npz",
            "--heldout", "h1_heldout.npz",
            "--calib", "h1_calib.npz",
            "--out", "/kaggle/working/h1_trained_s3.npz",
            "--router-lr", "0.0003", "--aux", "0.01",
            "--steps", "4000", "--bs", "2", "--accum", "8", "--lr", "2e-4",
            "--every", "250", "--seed", "3141", "--max-hours", "11.0",
        ],
        "outputs": ["h1_trained_s3.npz", "h1_trained_s3.json"],
    },
}


def log(*items) -> None:
    print(*items, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target(name: str) -> dict:
    return TARGETS[name]


def verify_bundle(name: str) -> dict:
    spec = target(name)
    bundle = spec["bundle"]
    manifest_path = bundle / "MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{name}: missing manifest {manifest_path}")
    actual_manifest_sha = sha256(manifest_path)
    if actual_manifest_sha != spec["manifest_sha256"]:
        raise SystemExit(
            f"{name}: manifest sha mismatch {actual_manifest_sha} != {spec['manifest_sha256']}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in spec["manifest_test"].items():
        if manifest.get(key) != expected:
            raise SystemExit(f"{name}: manifest {key}={manifest.get(key)!r}, expected {expected!r}")
    actual_command = manifest.get(spec["command_key"])
    if actual_command != spec["command_test"]:
        raise SystemExit(
            f"{name}: frozen command metadata differs\n"
            f"actual={json.dumps(actual_command, sort_keys=True)}\n"
            f"expected={json.dumps(spec['command_test'], sort_keys=True)}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"{name}: empty files map")
    total = 0
    for filename, metadata in files.items():
        path = bundle / filename
        if not path.is_file():
            raise SystemExit(f"{name}: missing payload {filename}")
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(metadata["bytes"]) or digest != metadata["sha256"]:
            raise SystemExit(
                f"{name}: payload mismatch {filename}: {size} {digest} != "
                f"{metadata['bytes']} {metadata['sha256']}"
            )
        total += size
    log(
        f"{name}: LOCAL BUNDLE PASS -- {len(files)} manifest payloads, "
        f"{total:,} bytes, manifest {actual_manifest_sha}"
    )
    return manifest


def runtime_shim(name: str) -> str:
    spec = target(name)
    # The mounted dataset name/version is deliberately not assumed.  The sha identifies the
    # sole valid manifest even if Kaggle changes the mount directory between versions.
    script = f'''import glob, hashlib, json, os, shutil, subprocess, sys
TARGET = {name!r}
MANIFEST_SHA256 = {spec["manifest_sha256"]!r}
COMMAND = {spec["command"]!r}
EXPECTED_OUTPUTS = {spec["outputs"]!r}
COPY_TO_WORKING = {spec["copy_to_working"]!r}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

hits = [p for p in glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)
        if sha256(p) == MANIFEST_SHA256]
assert len(hits) == 1, ("expected exactly one frozen manifest", hits)
manifest_path = hits[0]
root = os.path.dirname(manifest_path)
with open(manifest_path, encoding="utf-8") as fh:
    manifest = json.load(fh)

def verify_root(folder):
    assert sha256(os.path.join(folder, "MANIFEST.json")) == MANIFEST_SHA256
    for filename, metadata in manifest["files"].items():
        path = os.path.join(folder, filename)
        assert os.path.isfile(path), ("missing payload", filename)
        assert os.path.getsize(path) == int(metadata["bytes"]), ("size mismatch", filename)
        assert sha256(path) == metadata["sha256"], ("sha mismatch", filename)

verify_root(root)
print("KAGGLE_MOUNT_GATE PASS", TARGET, len(manifest["files"]), MANIFEST_SHA256, flush=True)
if COPY_TO_WORKING:
    writable_root = os.path.join("/kaggle/working", TARGET + "_bundle")
    assert not os.path.exists(writable_root), ("working copy already exists", writable_root)
    shutil.copytree(root, writable_root)
    verify_root(writable_root)
    root = writable_root
    print("KAGGLE_WORKING_COPY_GATE PASS", TARGET, root, flush=True)
cmd = list(COMMAND)
cmd[0] = sys.executable
print("EXEC", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=root, check=True)
missing = [name for name in EXPECTED_OUTPUTS
           if not os.path.isfile(os.path.join("/kaggle/working", name))]
assert not missing, ("trainer returned without final outputs", missing)
print("KAGGLE_OUTPUT_GATE PASS", TARGET, EXPECTED_OUTPUTS, flush=True)
'''
    compile(script, f"<{name}_kaggle_shim>", "exec")
    return script


def staging_root() -> Path:
    explicit = os.environ.get("SILICONLLM_KAGGLE_STAGING")
    if explicit:
        root = Path(explicit)
    elif Path(r"D:\_ktmp").is_dir():
        root = Path(r"D:\_ktmp") / "siliconllm_kaggle_staging"
    else:
        root = REPO / ".kaggle_staging"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_dataset(name: str, account: str, destination: Path) -> None:
    spec = target(name)
    manifest = verify_bundle(name)
    for filename in list(manifest["files"]) + ["MANIFEST.json"]:
        link_or_copy(spec["bundle"] / filename, destination / filename)
    metadata = {
        "title": spec["dataset_title"],
        "id": f"{kaggle_ops.username(account)}/{spec['dataset_slug']}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (destination / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def remote_ref(name: str, account: str, kind: str) -> str:
    spec = target(name)
    slug = spec["dataset_slug"] if kind == "dataset" else spec["kernel_slug"]
    return f"{kaggle_ops.username(account)}/{slug}"


def print_result(result: subprocess.CompletedProcess) -> None:
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")


def dataset_status(name: str, account: str) -> subprocess.CompletedProcess:
    return kaggle_ops.kaggle(
        account, "datasets", "status", remote_ref(name, account, "dataset"), check=False
    )


def dataset_ready(name: str, account: str) -> bool:
    result = dataset_status(name, account)
    return result.returncode == 0 and "ready" in (result.stdout or "").lower()


def wait_dataset(name: str, account: str) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    ref = remote_ref(name, account, "dataset")
    while time.monotonic() < deadline:
        if dataset_ready(name, account):
            log(f"{name}: DATASET READY -- {ref}")
            return
        log(f"{name}: dataset processing; next check in 30 s -- {ref}")
        time.sleep(30)
    raise SystemExit(f"{name}: dataset did not become READY within {READY_TIMEOUT_S}s: {ref}")


def check_identity(account: str) -> None:
    file_user = kaggle_ops.username(account)
    server_user = kaggle_ops.server_identity(account)
    if server_user != file_user:
        raise SystemExit(
            f"{account}: credential identity mismatch: file={file_user!r}, server={server_user!r}"
        )
    log(f"{account}: IDENTITY PASS -- {server_user}")


def quota(account: str) -> str:
    result = kaggle_ops.kaggle(account, "quota", "-v", check=False)
    if result.returncode != 0:
        print_result(result)
        raise SystemExit(f"{account}: quota query failed")
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    gpu = next((line for line in lines if line.startswith("GPU,")), None)
    if gpu is None:
        raise SystemExit(f"{account}: GPU quota row missing")
    remaining = float(gpu.split(",")[2].rstrip("h"))
    if remaining < 11.0:
        raise SystemExit(f"{account}: only {remaining:.2f} GPU-h remain; 11.0 required")
    log(f"{account}: QUOTA PASS -- {gpu}")
    return gpu


def cmd_check(args: argparse.Namespace) -> int:
    names = list(TARGETS) if args.target == "all" else [args.target]
    for name in names:
        verify_bundle(name)
        runtime_shim(name)
        log(f"{name}: KERNEL SHIM COMPILES")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    verify_bundle(args.target)
    runtime_shim(args.target)
    check_identity(args.account)
    quota(args.account)
    ref = remote_ref(args.target, args.account, "kernel")
    status = kaggle_ops.kaggle(args.account, "kernels", "status", ref, check=False)
    text = ((status.stdout or "") + (status.stderr or "")).strip()
    if status.returncode == 0 and any(word in text.lower() for word in ("running", "queued", "pending")):
        raise SystemExit(f"{args.target}: refusing duplicate active kernel: {text}")
    log(f"{args.target}: REMOTE PREFLIGHT PASS -- {ref} is not active")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    cmd_preflight(args)
    spec = target(args.target)
    root = staging_root()
    with tempfile.TemporaryDirectory(prefix=f"{args.target}_dataset_", dir=str(root)) as raw:
        staged = Path(raw)
        stage_dataset(args.target, args.account, staged)
        remote = dataset_status(args.target, args.account)
        exists = remote.returncode == 0
        if exists and "ready" not in (remote.stdout or "").lower():
            wait_dataset(args.target, args.account)
        action = "version" if exists else "create"
        log(
            f"{args.target}: DATASET {action.upper()} -- "
            f"{remote_ref(args.target, args.account, 'dataset')}"
        )
        saved = {key: os.environ.get(key) for key in ("TMP", "TEMP", "TMPDIR")}
        try:
            for key in saved:
                os.environ[key] = str(root)
            command = ["datasets", action, "-p", str(staged), "--dir-mode", "zip"]
            if action == "version":
                command[2:2] = ["-m", f"{args.target} frozen training bundle"]
            result = kaggle_ops.kaggle(args.account, *command, check=False)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        print_result(result)
        if result.returncode != 0:
            raise SystemExit(f"{args.target}: Kaggle dataset {action} failed")
    wait_dataset(args.target, args.account)
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    cmd_preflight(args)
    if not dataset_ready(args.target, args.account):
        wait_dataset(args.target, args.account)
    spec = target(args.target)
    user = kaggle_ops.username(args.account)
    root = staging_root()
    with tempfile.TemporaryDirectory(prefix=f"{args.target}_kernel_", dir=str(root)) as raw:
        staged = Path(raw)
        code_name = f"{args.target}_kaggle_entry.py"
        (staged / code_name).write_text(runtime_shim(args.target), encoding="utf-8")
        metadata = {
            "id": f"{user}/{spec['kernel_slug']}",
            "title": spec["kernel_title"],
            "code_file": code_name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": MACHINE,
            "dataset_sources": [f"{user}/{spec['dataset_slug']}"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (staged / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        log(
            f"{args.target}: KERNEL PUSH -- {user}/{spec['kernel_slug']} "
            f"(machine={MACHINE}, cap={SESSION_SECONDS}s)"
        )
        result = kaggle_ops.kaggle(
            args.account, "kernels", "push", "-p", str(staged),
            "-t", str(SESSION_SECONDS), check=False,
        )
        print_result(result)
        blob = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            raise SystemExit(f"{args.target}: kernel push failed")
        if "not valid dataset sources" in blob or "could not be added to the kernel" in blob:
            raise SystemExit(
                f"{args.target}: Kaggle silently rejected the dataset source; kernel is invalid"
            )
    time.sleep(5)
    return cmd_status(args)


def cmd_launch(args: argparse.Namespace) -> int:
    cmd_upload(args)
    return cmd_push(args)


def cmd_status(args: argparse.Namespace) -> int:
    result = kaggle_ops.kaggle(
        args.account, "kernels", "status", remote_ref(args.target, args.account, "kernel"),
        check=False,
    )
    print_result(result)
    return result.returncode


def cmd_output(args: argparse.Namespace) -> int:
    spec = target(args.target)
    destination = Path(args.path) if args.path else (
        HERE / "results" / f"{args.target}_kaggle_return"
    )
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"refusing to mix/overwrite existing output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    result = kaggle_ops.kaggle(
        args.account, "kernels", "output", remote_ref(args.target, args.account, "kernel"),
        "-p", str(destination), check=False,
    )
    print_result(result)
    if result.returncode != 0:
        raise SystemExit(f"{args.target}: output download failed")
    missing = [filename for filename in spec["outputs"] if not (destination / filename).is_file()]
    if missing:
        raise SystemExit(f"{args.target}: completed output missing {missing}")
    log(f"{args.target}: OUTPUT GATE PASS -- {destination}")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command_name", required=True)
    check = sub.add_parser("check")
    check.add_argument("target", choices=["h1", "h2i", "all"])
    check.set_defaults(func=cmd_check)
    for command_name, function in (
        ("preflight", cmd_preflight), ("upload", cmd_upload), ("push", cmd_push),
        ("launch", cmd_launch), ("status", cmd_status), ("output", cmd_output),
    ):
        command = sub.add_parser(command_name)
        command.add_argument("target", choices=["h1", "h2i"])
        command.add_argument("account", choices=kaggle_ops.ACCOUNTS)
        if command_name == "output":
            command.add_argument("--path")
        command.set_defaults(func=function)
    return top


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
