#!/usr/bin/env python3
"""Orchestrates the phase-64 rung-1 stage-3 alpha-sweep across three Kaggle accounts.

Three arms differ ONLY by --alpha (0.0 / 0.25 / 0.5); every pin is identical, so the
sweep reads as a trend on one variable. One arm per account, one Kaggle account per arm:

    arm4 (alpha 0.00) -> acct1 (wildpino)
    arm5 (alpha 0.25) -> acct2 (giggio253)
    arm6 (alpha 0.50) -> acct3 (sirwildpino)

Auth isolation (the OAuth-token trap that made all three look like sirwildpino) is handled
in kaggle_ops.kaggle(): every child gets a neutral HOME and the token env stripped.

Subcommands
    upload  <acct> <which|all>   create-or-version a data dataset for an account
    push    <arm> [--relay]       push the arm's script kernel as a batch (Save & Run All)
    status  <arm|all>            print each kernel's run state
    output  <arm> [-p DIR]        download a finished/failed kernel's files + log
    plan                         print the arm/account/dataset wiring and exit

The batch kernel runs to the 12 h session limit and stops. To continue, `push --relay`:
it re-pushes the SAME kernel with its own previous OUTPUT attached as an input, which is
what the notebook's rehydrate() reads (resume_<arm>.pt) to pick the run back up. There is
no Kaggle "persistence" flag over the API -- this input-chaining is the resume mechanism.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kaggle_ops  # noqa: E402  (reuses the isolated-auth kaggle() wrapper)

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "kaggle_rung1"

# account -> real Kaggle username (the KEY's server identity, verified, NOT the file field)
USER = {"acct1": "wildpino", "acct2": "giggio253", "acct3": "sirwildpino"}

# the three data datasets every arm attaches. slug is per-account (namespaced by username).
# dir_mode=zip because the uploader does not recurse; Kaggle re-extracts the archive so the
# data/ and logits_s0/ subtrees the trainer needs are preserved on the mount.
DATASETS = {
    "stage3-data": {"folder": PACK / "stage3_data",  "slug": "phase64-stage3-data", "title": "phase64 stage3 data"},
    "code":        {"folder": PACK / "code_bundle",  "slug": "phase64-code",        "title": "phase64 code bundle"},
    "logits":      {"folder": PACK / "logits_bundle","slug": "phase64-logits",      "title": "phase64 teacher logits"},
}

# arm -> (account, notebook script, kernel slug). alpha lives inside the script; nothing here sets it.
# NOTE on slugs: Kaggle derives the kernel slug from the TITLE, not from the id suffix.
# "phase64 arm4" -> phase64-arm4, so an id of phase64-arm4-a000 silently became phase64-arm4
# (with a warning). The slug here MUST be the real one, because relay attaches the kernel's
# own output via kernel_sources=[user/slug] -- a wrong slug means resume can't find itself.
ARMS = {
    "arm4": {"acct": "acct1", "script": PACK / "NOTEBOOK_arm4_a000_V2048.py", "slug": "phase64-arm4"},
    "arm5": {"acct": "acct2", "script": PACK / "NOTEBOOK_arm5_a025_V2048.py", "slug": "phase64-arm5"},
    "arm6": {"acct": "acct3", "script": PACK / "NOTEBOOK_arm6_a050_V2048.py", "slug": "phase64-arm6"},
    # stage-4: alpha 0.0 + kd-resident 16 (new knob) + slice-sha now VERIFIED. New code bundle
    # (CODE_SHA 27b22582); data + logits unchanged from stage-3, so only phase64-code is re-uploaded.
    "arm7": {"acct": "acct1", "script": PACK / "NOTEBOOK_arm7_w16_V2048.py", "slug": "phase64-arm7"},
    # throwaway: validates the resume CHAIN (stop-incomplete -> push --relay -> rehydrate -> continue)
    # before the multi-session main run relies on it unattended. Tiny budget forces an incomplete stop.
    "relaytest": {"acct": "acct1", "script": PACK / "NOTEBOOK_relaytest_V2048.py", "slug": "phase64-relaytest"},
}

MACHINE = "NvidiaTeslaT4"        # the enum has no explicit 2xT4; the cell auto-adapts --accum to device_count
SESSION_SECONDS = 12 * 3600      # Kaggle's batch cap; we set it explicitly rather than hope for the default


def _run_cli(account: str, *args: str, check: bool = True):
    r = kaggle_ops.kaggle(account, *args, check=False)
    sys.stdout.write(r.stdout or "")
    if (r.stderr or "").strip():
        sys.stderr.write(r.stderr)
    if check and r.returncode != 0:
        sys.exit(f"[{account}] kaggle {' '.join(args)} -> exit {r.returncode}")
    return r


# ---- datasets ----------------------------------------------------------------------------------

def _write_dataset_metadata(folder: Path, user: str, slug: str, title: str) -> None:
    meta = {
        "title": title,
        "id": f"{user}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def cmd_upload(args: argparse.Namespace) -> int:
    # The uploader zips each subdir into a temp archive (18 GB for logits) via tempfile,
    # which lands on C: by default -- only ~68 GB free there, and three of these at once
    # would overrun it. Point every temp mechanism at D: (1.3 TB free) for this process.
    scratch = REPO / "kaggle_rung1" / "_ziptmp"
    scratch.mkdir(parents=True, exist_ok=True)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(scratch)

    which = list(DATASETS) if args.which == "all" else [args.which]
    user = USER[args.account]
    for name in which:
        d = DATASETS[name]
        folder, slug, title = d["folder"], d["slug"], d["title"]
        if not folder.is_dir():
            sys.exit(f"missing payload folder: {folder}")
        _write_dataset_metadata(folder, user, slug, title)
        # does this account already have the dataset? -> version; else -> create.
        r = kaggle_ops.kaggle(args.account, "datasets", "status", f"{user}/{slug}", check=False)
        exists = r.returncode == 0 and "ready" in r.stdout.lower()
        print(f"\n=== [{args.account}/{user}] {name} -> {slug}  ({'version' if exists else 'create'}) ===", flush=True)
        if exists:
            _run_cli(args.account, "datasets", "version", "-p", str(folder),
                     "-m", "re-version via kaggle_run", "--dir-mode", "zip")
        else:
            _run_cli(args.account, "datasets", "create", "-p", str(folder),
                     "--dir-mode", "zip")
    return 0


# ---- kernels -----------------------------------------------------------------------------------

def cmd_push(args: argparse.Namespace) -> int:
    arm = ARMS[args.arm]
    acct, script, slug = arm["acct"], arm["script"], arm["slug"]
    user = USER[acct]
    if not script.is_file():
        sys.exit(f"missing arm script: {script}")
    ds_sources = [f"{user}/{DATASETS[n]['slug']}" for n in ("stage3-data", "code", "logits")]
    kernel_sources = [f"{user}/{slug}"] if args.relay else []   # self-output = the resume input

    staging = Path(tempfile.mkdtemp(prefix=f"k_{args.arm}_"))
    try:
        code_name = script.name
        shutil.copy(script, staging / code_name)
        meta = {
            "id": f"{user}/{slug}",
            "title": f"phase64 {args.arm}",
            "code_file": code_name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "false",
            "machine_shape": MACHINE,
            "dataset_sources": ds_sources,
            "kernel_sources": kernel_sources,
            "competition_sources": [],
            "model_sources": [],
        }
        (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"=== push {args.arm} -> {user}/{slug}  (relay={args.relay}) ===")
        print("  datasets:", ", ".join(ds_sources))
        if kernel_sources:
            print("  resume-from:", ", ".join(kernel_sources))
        _run_cli(acct, "kernels", "push", "-p", str(staging), "-t", str(SESSION_SECONDS))
        print(f"\nwatch: python scripts/kaggle_run.py status {args.arm}")
        print(f"logs : python scripts/kaggle_run.py output {args.arm}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for a in arms:
        arm = ARMS[a]
        user = USER[arm["acct"]]
        r = kaggle_ops.kaggle(arm["acct"], "kernels", "status", f"{user}/{arm['slug']}", check=False)
        line = (r.stdout or r.stderr).strip().replace("\n", " ")
        print(f"{a:5} {user}/{arm['slug']:22} {line}")
    return 0


def cmd_output(args: argparse.Namespace) -> int:
    arm = ARMS[args.arm]
    user = USER[arm["acct"]]
    dest = args.path or str(REPO / "kaggle_rung1" / "results" / args.arm)
    Path(dest).mkdir(parents=True, exist_ok=True)
    _run_cli(arm["acct"], "kernels", "output", f"{user}/{arm['slug']}", "-p", dest)
    print(f"downloaded to {dest}")
    return 0


def cmd_plan(_args: argparse.Namespace) -> int:
    print("arm    account   user           alpha  kernel-slug            script")
    alphas = {"arm4": "0.00", "arm5": "0.25", "arm6": "0.50", "arm7": "0.00/w16"}
    for a, arm in ARMS.items():
        print(f"{a:6} {arm['acct']:9} {USER[arm['acct']]:14} {alphas[a]:6} {arm['slug']:22} {arm['script'].name}")
    print("\ndatasets attached to every arm (per account):")
    for n, d in DATASETS.items():
        print(f"  {n:12} {d['slug']:22} <- {d['folder']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("upload"); u.add_argument("account", choices=USER)
    u.add_argument("which", choices=[*DATASETS, "all"]); u.set_defaults(func=cmd_upload)

    pu = sub.add_parser("push"); pu.add_argument("arm", choices=ARMS)
    pu.add_argument("--relay", action="store_true"); pu.set_defaults(func=cmd_push)

    st = sub.add_parser("status"); st.add_argument("arm", choices=[*ARMS, "all"]); st.set_defaults(func=cmd_status)
    ou = sub.add_parser("output"); ou.add_argument("arm", choices=ARMS)
    ou.add_argument("-p", "--path"); ou.set_defaults(func=cmd_output)
    sub.add_parser("plan").set_defaults(func=cmd_plan)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
