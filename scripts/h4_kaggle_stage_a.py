#!/usr/bin/env python3
"""Prepare, preflight, and (only when explicitly confirmed) launch H4 Stage A.

This is operational glue, not H4 science. It never changes the frozen bundle.
The only state-changing subcommands are ``upload`` (one new private dataset)
and ``launch`` (one new kernel). Neither retries or polls.

Examples (these commands are *not* run by this script unless selected):

  # Local, read-only integrity check; no Kaggle API call.
  python scripts/h4_kaggle_stage_a.py validate

  # Remote, read-only checks.  Supply actual owner/slug values at decision time.
  python scripts/h4_kaggle_stage_a.py preflight --account acct1 \
      --dataset OWNER/DATASET --kernel OWNER/NEW-KERNEL-SLUG

  # Create exactly one new private dataset after main-agent review.
  python scripts/h4_kaggle_stage_a.py upload --account acct1 \
      --dataset OWNER/NEW-DATASET-SLUG

  # The one permitted push.  It has no retry and requires a fresh kernel slug.
  python scripts/h4_kaggle_stage_a.py launch --account acct1 \
      --dataset OWNER/DATASET --kernel OWNER/NEW-KERNEL-SLUG \
      --confirm-launch

The remote dataset is intentionally an argument, not a constant.  No account,
dataset, or kernel identity is guessed from prior experiments.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "benchmarks" / "donor_adaptation" / "s1" / "_h4_bundle_v2"
MANIFEST = BUNDLE / "MANIFEST.json"
ACCOUNTS = ("acct1", "acct2", "acct3")
MACHINE_SHAPE = "NvidiaTeslaT4"
KERNEL_CODE_FILE = "h4_stage_a_notebook.py"
KERNEL_LOG = "h4_stage_a.log"
MIN_GPU_HOURS = 3.0
DATASET_PAGE_SIZE = 20
MAX_DATASET_PAGES = 100
KERNEL_PAGE_SIZE = 200

# The CLI's owner inventories are the only accepted freshness authority.  A
# per-resource ``status`` response can be an authorization failure (not an
# absence), so it is intentionally not used for create/push freshness gates.
KERNEL_LIST_HEADER = ("ref", "title", "author", "lastruntime", "totalvotes")
DATASET_REQUIRED_LIST_FIELDS = frozenset(("ref", "title", "size", "lastupdated"))

# This is read from the frozen local bundle at authoring time, and is also
# recomputed by validate().  A dataset that merely calls itself H4 is not enough.
EXPECTED_MANIFEST_SHA256 = "a97e444359684b2bd769932248493873fe7a7f4b4e63f3f811cf3caf17f2c331"

REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}/[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise SystemExit(f"missing frozen H4 manifest: {MANIFEST}")
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid frozen H4 manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit("frozen H4 manifest is not a JSON object")
    return manifest


def exact_cli(manifest: dict[str, Any]) -> list[str]:
    cli = manifest.get("cli")
    payload = manifest.get("payload")
    expected = {
        "steps": 4000,
        "bs": 2,
        "accum": 8,
        "lr": 0.0002,
        "every": 250,
        "max_hours": 2.8,
        "seed": 1717,
    }
    if manifest.get("stage") != "H4" or manifest.get("rank") != 48:
        raise SystemExit("manifest is not the frozen H4 rank-48 bundle")
    if cli != expected:
        raise SystemExit(f"unexpected H4 CLI pins: {cli!r}")
    if not isinstance(payload, dict) or payload != {
        "trainer": "h4_qat.py",
        "factors": "h4_factors.npz",
        "train": "h4_train.npz",
        "probe": "h4_probe.json",
        "output": "h4_trained.npz",
    }:
        raise SystemExit("unexpected H4 payload declaration")
    return [
        "python", "h4_qat.py", "--factors", "h4_factors.npz", "--train", "h4_train.npz",
        "--probe", "h4_probe.json", "--out", "h4_trained.npz", "--steps", "4000",
        "--bs", "2", "--accum", "8", "--lr", "2e-4", "--every", "250",
        "--max-hours", "2.8", "--seed", "1717",
    ]


def verify_local_bundle() -> tuple[dict[str, Any], str]:
    """Hash every frozen, transported input without importing model libraries."""
    manifest = load_manifest()
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise SystemExit(
            "frozen H4 MANIFEST.json SHA-256 differs from this operational launcher; "
            "do not launch until the main audit explicitly reconciles the mismatch"
        )
    exact_cli(manifest)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("frozen H4 manifest has no file inventory")
    output = manifest["payload"]["output"]
    for name, record in files.items():
        if name == output:
            raise SystemExit("terminal output must not be shipped in a Stage A input bundle")
        if not isinstance(record, dict):
            raise SystemExit(f"invalid file record for {name!r}")
        path = BUNDLE / name
        if not path.is_file():
            raise SystemExit(f"frozen H4 bundle is missing {name}")
        if path.stat().st_size != record.get("bytes"):
            raise SystemExit(f"frozen H4 byte mismatch for {name}")
        if sha256_file(path) != record.get("sha256"):
            raise SystemExit(f"frozen H4 SHA-256 mismatch for {name}")
    if (BUNDLE / output).exists() or (BUNDLE / Path(output).with_suffix(".json")).exists():
        raise SystemExit("frozen H4 bundle wrongly contains terminal Stage A output")
    return manifest, manifest_sha


def validate_ref(value: str, label: str) -> str:
    if not REF_RE.fullmatch(value):
        raise SystemExit(f"{label} must be an explicit OWNER/SLUG reference")
    return value


def remote_file_inventory(text: str) -> dict[str, int]:
    """Parse exact Kaggle CSV inventory; reject zip-only or ambiguous listings."""
    rows = csv.DictReader(text.splitlines())
    if not rows.fieldnames or not {"name", "size"}.issubset({x.lower() for x in rows.fieldnames}):
        raise SystemExit("remote dataset inventory lacks name/size CSV fields")
    inventory: dict[str, int] = {}
    for row in rows:
        lowered = {key.lower(): value for key, value in row.items() if key}
        name = lowered.get("name", "")
        size = lowered.get("size", "")
        if not name or not size or name in inventory:
            raise SystemExit("remote dataset inventory has missing or duplicate file entries")
        try:
            nbytes = int(size)
        except ValueError as exc:
            raise SystemExit("remote dataset inventory has an unreadable byte size") from exc
        if nbytes < 0:
            raise SystemExit("remote dataset inventory has a negative byte size")
        inventory[name] = nbytes
    return inventory


def expected_inventory(manifest: dict[str, Any]) -> dict[str, int]:
    return {**{name: int(record["bytes"]) for name, record in manifest["files"].items()},
            "MANIFEST.json": MANIFEST.stat().st_size}


@contextmanager
def _isolated_kaggle_credentials(account: str):
    """Use kaggle_ops' per-account file while hiding global OAuth credentials."""
    import kaggle_ops

    neutral_home = kaggle_ops._NEUTRAL_HOME
    neutral_home.mkdir(parents=True, exist_ok=True)
    replacements = {
        "KAGGLE_CONFIG_DIR": str(kaggle_ops.config_dir(account)),
        "HOME": str(neutral_home),
        "USERPROFILE": str(neutral_home),
    }
    cleared = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
    prior = {key: os.environ.get(key) for key in (*replacements, *cleared)}
    try:
        os.environ.update(replacements)
        for key in cleared:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _whole_second_duration_compatible(original: Any):
    """Accept protobuf Duration values such as ``0s`` without altering kagglesdk."""
    def decode(value: Any):
        if isinstance(value, str) and value.endswith("s") and "." not in value:
            try:
                return datetime.timedelta(seconds=int(value[:-1]))
            except ValueError:
                pass
        return original(value)
    return decode


def _quota_view(account: str) -> Any:
    """Read the quota endpoint once with an in-memory kagglesdk compatibility shim."""
    try:
        with _isolated_kaggle_credentials(account):
            from kaggle import KaggleApi
            from kagglesdk.kaggle_object import TimeDeltaSerializer

            original = TimeDeltaSerializer._from_dict_value
            TimeDeltaSerializer._from_dict_value = staticmethod(_whole_second_duration_compatible(original))
            try:
                api = KaggleApi()
                api.authenticate()
                return api.quota_view()
            finally:
                TimeDeltaSerializer._from_dict_value = staticmethod(original)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit("read-only direct Kaggle GPU quota query failed") from exc


def check_quota(account: str) -> dict[str, Any]:
    response = _quota_view(account)
    quota = getattr(response, "gpu_quota", None)
    try:
        used = float(quota.time_used.total_seconds()) / 3600
        total = float(quota.total_time_allowed.total_seconds()) / 3600
    except (AttributeError, TypeError, ValueError) as exc:
        raise SystemExit("Kaggle direct GPU quota response is incomplete") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (used, total)) or used > total:
        raise SystemExit("Kaggle direct GPU quota response is invalid")
    remaining = total - used
    observed = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if remaining < MIN_GPU_HOURS:
        raise SystemExit(f"Kaggle GPU quota has {remaining:.2f} h remaining; {MIN_GPU_HOURS:.1f} h required")
    return {
        "gpu_quota_source": "direct_api_in_memory_duration_shim",
        "gpu_quota_used_hours": used,
        "gpu_quota_total_hours": total,
        "gpu_quota_remaining_hours": remaining,
        "gpu_quota_observed_at_utc": observed,
        "gpu_quota_min_hours": MIN_GPU_HOURS,
    }


def _nonblank_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _has_continuation_marker(text: str) -> bool:
    return bool(re.search(r"(?:next\s+page\s+token|continuation(?:\s+token)?)", text, re.IGNORECASE))


def parse_owner_inventory_csv(text: str, resource: str, owner: str) -> list[str]:
    """Parse one owner-only Kaggle list page, rejecting ambiguous CSV."""
    lines = _nonblank_lines(text)
    if not lines:
        raise SystemExit(f"{resource} owner inventory is empty or not CSV")
    if _has_continuation_marker("\n".join(lines)):
        raise SystemExit(f"{resource} owner inventory has a continuation marker")
    try:
        rows = list(csv.reader(lines, strict=True))
    except csv.Error as exc:
        raise SystemExit(f"{resource} owner inventory is malformed CSV") from exc
    if not rows or not rows[0] or any(not field.strip() for field in rows[0]):
        raise SystemExit(f"{resource} owner inventory has an invalid CSV header")
    header = tuple(field.strip().lower() for field in rows[0])
    if len(set(header)) != len(header):
        raise SystemExit(f"{resource} owner inventory has duplicate CSV header fields")
    if resource == "kernels":
        if header != KERNEL_LIST_HEADER:
            raise SystemExit("kernels owner inventory has an unexpected CSV header")
    elif resource == "datasets":
        if not DATASET_REQUIRED_LIST_FIELDS.issubset(header):
            raise SystemExit("datasets owner inventory lacks required CSV fields")
    else:
        raise SystemExit(f"unsupported owner inventory resource {resource!r}")

    refs: list[str] = []
    seen: set[str] = set()
    width = len(header)
    for row in rows[1:]:
        if len(row) != width:
            raise SystemExit(f"{resource} owner inventory has a malformed CSV row")
        ref = row[0].strip()
        if not REF_RE.fullmatch(ref):
            raise SystemExit(f"{resource} owner inventory has an invalid reference")
        row_owner, _ = ref.split("/", 1)
        if row_owner != owner:
            raise SystemExit(f"{resource} owner inventory contains a wrong-owner reference")
        if ref in seen:
            raise SystemExit(f"{resource} owner inventory contains a duplicate reference")
        seen.add(ref)
        refs.append(ref)
    return refs


def _owner_inventory_result(account: str, resource: str, *args: str) -> Any:
    result = _kaggle(account, resource, "list", "--mine", "--csv", *args)
    if result.returncode != 0:
        raise SystemExit(f"{resource} owner inventory command failed")
    return result


def assert_remote_ref_absent(account: str, resource: str, reference: str) -> None:
    """Prove a fresh same-owner ref from a complete owner inventory only.

    Kernel listing has a single 200-row bounded page.  Dataset listing uses the
    CLI's fixed 20-row pages and keeps scanning until a short page proves the
    inventory complete.  A known exact ``No datasets found`` end marker is
    accepted only after a full preceding page.
    """
    reference = validate_ref(reference, resource[:-1])
    owner, _ = reference.split("/", 1)
    if resource == "kernels":
        result = _owner_inventory_result(account, resource, "--page-size", str(KERNEL_PAGE_SIZE))
        refs = parse_owner_inventory_csv(result.stdout or "", resource, owner)
        if len(refs) >= KERNEL_PAGE_SIZE:
            raise SystemExit("kernels owner inventory reached page capacity; completeness is unproven")
        if reference in refs:
            raise SystemExit("H4 kernel already exists; refusing an update/version")
        return
    if resource != "datasets":
        raise SystemExit(f"unsupported freshness resource {resource!r}")

    all_refs: set[str] = set()
    seen_pages: set[str] = set()
    prior_was_full = False
    for page in range(1, MAX_DATASET_PAGES + 1):
        result = _owner_inventory_result(account, resource, "--page", str(page))
        raw = result.stdout or ""
        normalized = "\n".join(_nonblank_lines(raw)).strip()
        if normalized == "No datasets found":
            if not prior_was_full:
                raise SystemExit("datasets owner inventory ended with an unexpected non-CSV response")
            if reference in all_refs:
                raise SystemExit("H4 dataset already exists; refusing an update/version")
            return
        if normalized in seen_pages:
            raise SystemExit("datasets owner inventory repeated a page; completeness is unproven")
        seen_pages.add(normalized)
        refs = parse_owner_inventory_csv(raw, resource, owner)
        if len(refs) > DATASET_PAGE_SIZE:
            raise SystemExit("datasets owner inventory exceeded fixed page capacity")
        overlap = all_refs.intersection(refs)
        if overlap:
            raise SystemExit("datasets owner inventory repeats a reference across pages")
        all_refs.update(refs)
        if len(refs) < DATASET_PAGE_SIZE:
            if reference in all_refs:
                raise SystemExit("H4 dataset already exists; refusing an update/version")
            return
        prior_was_full = True
    raise SystemExit("datasets owner inventory exceeded safe page bound; completeness is unproven")


def dataset_is_ready(result: Any) -> bool:
    if result.returncode != 0:
        return False
    status = (result.stdout or "").lower()
    return bool(re.search(r"\bready\b", status)) and not bool(re.search(r"\bnot\s+ready\b", status))


def check_identity(account: str, owner: str) -> tuple[str, str]:
    import kaggle_ops
    configured = kaggle_ops.username(account)
    identity = kaggle_ops.server_identity(account)
    if not identity or configured != identity:
        raise SystemExit("selected Kaggle credentials do not establish one matching server identity")
    if owner != identity:
        raise SystemExit(f"requested owner {owner!r} does not equal account server identity {identity!r}")
    return configured, identity


def _kaggle(account: str, *args: str):
    # Imported only for a remote subcommand.  validate/render stay standard-library only.
    import kaggle_ops

    return kaggle_ops.kaggle(account, *args, check=False)


def remote_preflight(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    """Read-only remote checks.  No push, download, polling, or GPU allocation."""
    dataset = validate_ref(args.dataset, "dataset")
    kernel = validate_ref(args.kernel, "kernel")
    if dataset == kernel:
        raise SystemExit("dataset and fresh kernel references cannot be identical")

    owner, _ = kernel.split("/", 1)
    configured, identity = check_identity(args.account, owner)
    if dataset.split("/", 1)[0] != identity:
        raise SystemExit("dataset owner does not equal selected account server identity")
    quota = check_quota(args.account)

    dataset_status = _kaggle(args.account, "datasets", "status", dataset)
    if not dataset_is_ready(dataset_status):
        raise SystemExit("dataset is not server-side READY; refusing a launch with a missing/stale mount")

    inventory = _kaggle(args.account, "datasets", "files", dataset, "--csv", "--page-size", "200")
    if inventory.returncode != 0:
        raise SystemExit("could not read the remote dataset inventory; refusing to infer its contents")
    remote_files = remote_file_inventory(inventory.stdout or "")
    expected = expected_inventory(manifest)
    if remote_files != expected:
        raise SystemExit("remote dataset inventory differs from the frozen flat H4 inventory/byte sizes; "
                         "a zip-only or nested listing is not sufficient")

    # A new slug avoids silently turning this into a version/retry.  Freshness
    # comes only from a complete owner ``--mine`` inventory: a ``status`` 403
    # is ambiguous and cannot be treated as an absent kernel.
    assert_remote_ref_absent(args.account, "kernels", kernel)
    return {
        "account_alias": args.account,
        "configured_username": configured,
        "server_identity": identity,
        "dataset": dataset,
        "remote_inventory_files": len(remote_files),
        "remote_inventory_match": True,
        "kernel": kernel,
        "fresh_kernel_gate": True,
        **quota,
    }


NOTEBOOK_TEMPLATE = r'''#!/usr/bin/env python3
# Generated by scripts/h4_kaggle_stage_a.py.  Operational wrapper only.
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = __H4_MANIFEST_SHA256__
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
BUNDLE_COPY = WORKING / "h4_bundle"
OUT_NPZ = WORKING / "h4_trained.npz"
OUT_JSON = WORKING / "h4_trained.json"
LOG = WORKING / "h4_stage_a.log"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message):
    raise SystemExit("H4 Stage A preflight FAILED: " + message)


def find_and_verify_mount():
    if not INPUT.is_dir():
        fail("/kaggle/input is absent")
    matches = []
    # Recursive discovery is intentional: Kaggle's mount nesting is not stable.
    for candidate in sorted(INPUT.glob("**/MANIFEST.json")):
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("stage") == "H4" and sha256_file(candidate) == EXPECTED_MANIFEST_SHA256:
            matches.append((candidate, record))
    if len(matches) != 1:
        fail("expected exactly one recursively discovered H4 manifest with the pinned SHA; got %d" % len(matches))
    manifest_path, manifest = matches[0]
    if manifest.get("rank") != 48 or manifest.get("seed") != 1717:
        fail("H4 manifest identity/rank/seed mismatch")
    expected_cli = {"steps": 4000, "bs": 2, "accum": 8, "lr": 0.0002, "every": 250, "max_hours": 2.8, "seed": 1717}
    if manifest.get("cli") != expected_cli:
        fail("frozen H4 CLI pins differ")
    payload = {"trainer": "h4_qat.py", "factors": "h4_factors.npz", "train": "h4_train.npz", "probe": "h4_probe.json", "output": "h4_trained.npz"}
    if manifest.get("payload") != payload:
        fail("frozen H4 payload declaration differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        fail("manifest has no file inventory")
    root = manifest_path.parent
    for name, expected in sorted(files.items()):
        path = root / name
        if not path.is_file():
            fail("mount lacks frozen input " + name)
        if path.stat().st_size != expected.get("bytes"):
            fail("mount byte mismatch for " + name)
        if sha256_file(path) != expected.get("sha256"):
            fail("mount SHA-256 mismatch for " + name)
    return root, manifest


def copy_and_reverify(mount_root, manifest):
    if BUNDLE_COPY.exists():
        fail("working H4 bundle path already exists; refusing overwrite")
    shutil.copytree(mount_root, BUNDLE_COPY)
    for name, expected in sorted(manifest["files"].items()):
        path = BUNDLE_COPY / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            fail("working-copy verification failed for " + name)


def append_log(line):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main():
    # Nothing importing torch/transformers, opening the donor, or asking CUDA for
    # a device occurs before this entire mount+manifest verification succeeds.
    if OUT_NPZ.exists() or OUT_JSON.exists() or LOG.exists():
        fail("terminal output/log path exists; one launch cannot overwrite prior artifacts")
    mount_root, manifest = find_and_verify_mount()
    copy_and_reverify(mount_root, manifest)
    receipt = {
        "stage": "H4_STAGE_A",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "mount_root": str(mount_root),
        "verified_files": len(manifest["files"]),
        "working_copy_verified": True,
        "donor_load_started": False,
        "gpu_checked": False,
    }
    (WORKING / "h4_stage_a_preflight.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    # Kaggle's T4 preset may expose one or two physical T4s. Inspect before
    # masking CUDA, then expose only index 0 to both wrapper and trainer.
    physical = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if physical.returncode != 0:
        fail("nvidia-smi could not enumerate physical GPUs")
    physical_names = [name.strip() for name in physical.stdout.splitlines() if name.strip()]
    if len(physical_names) not in (1, 2) or "T4" not in physical_names[0].upper():
        fail("Stage A requires physical cuda:0 to be a T4 on a one- or two-GPU preset")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("CUDA mask did not expose exactly one GPU to the trainer")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        fail("Stage A requires a T4, found " + gpu_name)
    os.environ["H4_DEV"] = "cuda:0"
    receipt["gpu_checked"] = True
    receipt["physical_gpu_count"] = len(physical_names)
    receipt["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    receipt["trainer_visible_gpu_count"] = torch.cuda.device_count()
    receipt["gpu_name"] = gpu_name
    receipt["trainer_device"] = os.environ["H4_DEV"]
    (WORKING / "h4_stage_a_preflight.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    # Do not let an incidental notebook environment turn this into an H4 smoke
    # run.  All scientific pins are supplied as explicit CLI values below.
    os.environ.pop("H4_SMOKE", None)
    cmd = [
        sys.executable, str(BUNDLE_COPY / "h4_qat.py"),
        "--factors", str(BUNDLE_COPY / "h4_factors.npz"),
        "--train", str(BUNDLE_COPY / "h4_train.npz"),
        "--probe", str(BUNDLE_COPY / "h4_probe.json"),
        "--out", str(OUT_NPZ),
        "--steps", "4000", "--bs", "2", "--accum", "8", "--lr", "2e-4",
        "--every", "250", "--max-hours", "2.8", "--seed", "1717",
    ]
    append_log("H4 Stage A exact pinned command: " + " ".join(cmd))
    with LOG.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd, cwd=str(WORKING), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise SystemExit("H4 trainer exited %d; no retry was attempted" % code)
    if not OUT_NPZ.is_file() or not OUT_JSON.is_file():
        fail("trainer exited zero without both terminal h4_trained artifacts")
    terminal = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if terminal.get("terminal_checkpoint") is not True or terminal.get("status") not in {"TIME_CAP", "STEPS_COMPLETE"}:
        fail("terminal metadata is not a valid H4 terminal checkpoint")
    append_log("H4 Stage A terminal artifacts present; CPU fp32 h4_eval.py remains the sole adjudicator.")


if __name__ == "__main__":
    main()
'''


def notebook_source(manifest_sha: str) -> str:
    return NOTEBOOK_TEMPLATE.replace("__H4_MANIFEST_SHA256__", repr(manifest_sha))


def kernel_metadata(kernel: str, dataset: str) -> dict[str, Any]:
    return {
        "id": kernel,
        "title": kernel.split("/", 1)[1],
        "code_file": KERNEL_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [dataset],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def stage(staging: Path, manifest_sha: str, kernel: str, dataset: str) -> None:
    staging.mkdir(parents=True, exist_ok=False)
    (staging / KERNEL_CODE_FILE).write_text(notebook_source(manifest_sha), encoding="utf-8", newline="\n")
    (staging / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(kernel, dataset), indent=2) + "\n", encoding="utf-8"
    )


def stage_dataset(staging: Path, manifest: dict[str, Any], manifest_sha: str, dataset: str) -> None:
    """Create a flat upload package without writing into the frozen source."""
    for name, record in manifest["files"].items():
        source = BUNDLE / name
        destination = staging / name
        shutil.copy2(source, destination)
        if destination.stat().st_size != record["bytes"] or sha256_file(destination) != record["sha256"]:
            raise SystemExit(f"H4 upload staging hash/size mismatch for {name}")
    shutil.copy2(MANIFEST, staging / "MANIFEST.json")
    if sha256_file(staging / "MANIFEST.json") != manifest_sha:
        raise SystemExit("H4 upload staging manifest hash mismatch")
    metadata = {
        "title": "H4 Stage A frozen rank 48 bundle",
        "id": dataset,
        "licenses": [{"name": "other"}],
        "description": ("Private research bundle. The rank-48 initialization derives from "
                        "Qwen/Qwen2.5-1.5B (Apache-2.0); the tokenized research stream "
                        "and other inputs are not relicensed as CC0. No public redistribution intended."),
        "isPrivate": True,
    }
    (staging / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def cmd_validate(_: argparse.Namespace) -> int:
    manifest, manifest_sha = verify_local_bundle()
    print("H4 local bundle: PASS")
    print(f"  manifest_sha256: {manifest_sha}")
    print(f"  frozen_inputs: {len(manifest['files'])}")
    print("  pinned_cli: " + " ".join(exact_cli(manifest)))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    _, manifest_sha = verify_local_bundle()
    validate_ref(args.dataset, "dataset")
    validate_ref(args.kernel, "kernel")
    target = Path(args.out).resolve()
    stage(target, manifest_sha, args.kernel, args.dataset)
    print(f"staged only: {target}")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    manifest, manifest_sha = verify_local_bundle()
    report = remote_preflight(args, manifest)
    report["bundle_manifest_sha256"] = manifest_sha
    report["mode"] = "READ_ONLY_PREFLIGHT"
    print(json.dumps(report, indent=2))
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    manifest, manifest_sha = verify_local_bundle()
    dataset = validate_ref(args.dataset, "dataset")
    owner, _ = dataset.split("/", 1)
    _, identity = check_identity(args.account, owner)
    # Do not ask status whether a new dataset is absent: Kaggle can return a
    # forbidden response for an owner-owned, as-yet absent slug.  The complete
    # owner inventory below is the only safe freshness proof.
    assert_remote_ref_absent(args.account, "datasets", dataset)
    with tempfile.TemporaryDirectory(prefix="siliconllm_h4_dataset_") as tmp:
        staging = Path(tmp)
        stage_dataset(staging, manifest, manifest_sha, dataset)
        # Flat root files: no directory mode and no alternate zip-only inventory.
        # Private is the CLI default; no --public flag is supplied.
        result = _kaggle(args.account, "datasets", "create", "-p", str(staging), "--dir-mode", "skip")
    if result.returncode != 0:
        raise SystemExit("H4 dataset create failed; no retry was attempted")
    after = _kaggle(args.account, "datasets", "status", dataset)
    state = (after.stdout or "").strip() if after.returncode == 0 else "STATUS_UNAVAILABLE"
    report: dict[str, Any] = {
        "status": "DATASET_CREATE_SUBMITTED_ONCE",
        "account_alias": args.account,
        "server_identity": identity,
        "dataset": dataset,
        "bundle_manifest_sha256": manifest_sha,
        "remote_state": state,
        "expected_flat_files": len(manifest["files"]) + 1,
        "private_requested": True,
        "no_retry": True,
    }
    if dataset_is_ready(after):
        listing = _kaggle(args.account, "datasets", "files", dataset, "--csv", "--page-size", "200")
        if listing.returncode == 0:
            try:
                report["remote_inventory_match"] = (
                    remote_file_inventory(listing.stdout or "") == expected_inventory(manifest)
                )
            except SystemExit:
                report["remote_inventory_match"] = False
        else:
            report["remote_inventory_match"] = False
        if not report["remote_inventory_match"]:
            report["next_action"] = "Remote inventory differs; do not launch. Inspect the dataset without retrying upload."
    else:
        report["next_action"] = "Dataset may still be processing; check status/inventory later before launch."
    print(json.dumps(report, indent=2))
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    if not args.confirm_launch:
        raise SystemExit("launch is armed only by the explicit --confirm-launch flag")
    manifest, manifest_sha = verify_local_bundle()
    report = remote_preflight(args, manifest)
    # A temporary staging directory is the only local write.  It disappears
    # after the single CLI push; no generated kernel metadata enters the repo.
    with tempfile.TemporaryDirectory(prefix="siliconllm_h4_stage_a_") as tmp:
        staging = Path(tmp) / "kernel"
        stage(staging, manifest_sha, args.kernel, args.dataset)
        result = _kaggle(args.account, "kernels", "push", "-p", str(staging))
    blob = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise SystemExit("Kaggle kernels push failed; no retry was attempted")
    if "not valid dataset sources" in blob.lower() or "could not be added to the kernel" in blob.lower():
        raise SystemExit("Kaggle accepted the push but rejected its dataset source; treat as failed, no retry attempted")
    print(json.dumps({
        "status": "PUSH_SUBMITTED_ONCE",
        "bundle_manifest_sha256": manifest_sha,
        **report,
        "trainer_cli": " ".join(exact_cli(manifest)),
        "no_retry": True,
        "no_best_checkpoint_selection": True,
    }, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="hash-check frozen local H4 bundle only").set_defaults(func=cmd_validate)

    def remote_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--account", choices=ACCOUNTS, required=True, help="credential alias; no default")
        p.add_argument("--dataset", required=True, help="existing frozen-bundle OWNER/SLUG")
        p.add_argument("--kernel", required=True, help="new H4 kernel OWNER/SLUG")

    render = sub.add_parser("render", help="write only a local temporary/selected kernel staging directory")
    render.add_argument("--dataset", required=True)
    render.add_argument("--kernel", required=True)
    render.add_argument("--out", required=True, help="new empty directory; never the repository bundle")
    render.set_defaults(func=cmd_render)

    upload = sub.add_parser("upload", help="create one new private flat-file H4 dataset")
    upload.add_argument("--account", choices=ACCOUNTS, required=True)
    upload.add_argument("--dataset", required=True, help="new H4 dataset OWNER/SLUG")
    upload.set_defaults(func=cmd_upload)

    preflight = sub.add_parser("preflight", help="read-only credential, dataset, and fresh-kernel checks")
    remote_arguments(preflight)
    preflight.set_defaults(func=cmd_preflight)

    launch = sub.add_parser("launch", help="one Kaggle push; requires an explicit confirmation flag")
    remote_arguments(launch)
    launch.add_argument("--confirm-launch", action="store_true")
    launch.set_defaults(func=cmd_launch)
    return ap


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
