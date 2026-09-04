#!/usr/bin/env python3
"""Package and push the S1 scale arm to Kaggle T4x2.  BRIEF_S1 Amendment 1 A1.5.

REUSES `scripts/kaggle_ops.py` for authentication -- that module carries the fix for the
recorded trap where a global OAuth access token overrides KAGGLE_CONFIG_DIR and every account
silently authenticates as one of them.  Nothing here re-implements auth.

It deliberately does NOT touch `scripts/kaggle_run.py`'s ARMS/DATASETS tables: that file drives
the phase-64 rung-1 arms and its main run is staged-but-unlaunched on acct2.  Adding an S1 row
to those tables risks disarming a pin on a run this probe has nothing to do with.  The push
here follows the same shape (code dataset + script kernel) without editing that apparatus.

WHAT SHIPS.  The bundle carries the CACHED EVAL SLICE (~101 KB), not the 140 MB corpus -- so
the GPU run scores the BYTE-IDENTICAL span the local CPU anchor scored, which is precisely
what the A1.2 platform control requires.  It also carries `anchor_cpu.json`, the completed
local CPU result, so the kernel can gate itself on the anchor without phoning home.

Subcommands
    build                 assemble the bundle locally and print an inventory
    whoami                verify the account's token is live (reuses kaggle_ops)
    upload  <acct>        create-or-version the code dataset
    push    <acct>        push the script kernel (Save & Run All)
    status  <acct>        print the kernel's run state
    output  <acct> [-p]   download the finished kernel's files + log
    estimate              print the GPU-hour estimate and its arithmetic
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DENSITY = REPO / "benchmarks" / "donor_adaptation" / "density"
sys.path.insert(0, str(REPO / "scripts"))
import kaggle_ops  # noqa: E402  (isolated-auth wrapper; the OAuth-trap fix lives here)

BUNDLE = HERE / "_kaggle_bundle"
DS_SLUG = "s1-sparsity-bpb-code"
DS_TITLE = "S1 sparsity BPB code bundle"
KERNEL_SLUG = "s1-sparsity-bpb-scale-arm"
KERNEL_TITLE = "S1 sparsity BPB scale arm"
# The accelerator string is a server-side enum that is NOT shipped in kagglesdk (see the
# comment in kaggle_api_extended.kernels_push).  "gpuT4x2" was a guess and is unverified;
# "NvidiaTeslaT4" is the value phase-64 pushed eight arms with, and every one of them came
# up with torch.cuda.device_count() == 2 (NOTEBOOK_main asserts --expect-gpus 2).  Use the
# value that has been observed to work rather than the one that reads like it should.
MACHINE = "NvidiaTeslaT4"        # ACHIEVED device count is asserted at run time, not here
SESSION_SECONDS = 12 * 3600

SLICE_NAME = "slice_heldout_24x512_s1234.pt"
CODE_FILES = ["s1_sparsity_bpb.py", "s1_kaggle_entry.py", "s1_tables.py", "s1_oracle.py"]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------- build
def cmd_build(args):
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "results").mkdir(parents=True)
    (BUNDLE / "corpus").mkdir(parents=True)

    for f in CODE_FILES:
        shutil.copy(HERE / f, BUNDLE / f)
    shutil.copy(DENSITY / "common.py", BUNDLE / "common.py")

    src_slice = DENSITY / "results" / SLICE_NAME
    if not src_slice.is_file():
        sys.exit(f"missing pinned eval slice: {src_slice}\n"
                 f"  The GPU run must score the SAME span as the CPU anchor; without this file "
                 f"it would redraw one and the A1.2 control would be meaningless.")
    shutil.copy(src_slice, BUNDLE / "results" / SLICE_NAME)
    shutil.copy(DENSITY / "corpus" / "manifest.json", BUNDLE / "corpus" / "manifest.json")

    # ---- the anchor: the completed local CPU result, or an explicit absence
    anchor_src = HERE / "results" / "s1_run_qwen2.5-1.5b.json"
    d = json.load(open(anchor_src)) if anchor_src.is_file() else None

    # COMPLETENESS GATE.  stage_run CHECKPOINTS after every p, so this file exists and parses
    # long before the run is finished -- and the earlier version of this function happily wrote
    # an "anchor" out of a partial one and printed "from the completed local run".  That is the
    # brief's "write no number from a stage still running", committed as an artefact: the A1.2
    # gate would then have compared 3 p-values instead of 10 while reporting a full comparison,
    # i.e. a WEAKER gate wearing the label of the strong one.
    # `elapsed_s` is written exactly once, by the final dump; the p-set is checked too, because
    # a resumed run could in principle carry a stale elapsed_s forward.
    missing = []
    if d is not None:
        sys.path.insert(0, str(HERE))
        import s1_sparsity_bpb as _S
        missing = [str(x) for x in _S.P_GRID if str(x) not in d.get("results", {})]
        if "elapsed_s" not in d or missing:
            log(f"  !! the local CPU run at results/s1_run_qwen2.5-1.5b.json is STILL RUNNING or "
                f"INCOMPLETE (elapsed_s={'present' if 'elapsed_s' in d else 'ABSENT'}, "
                f"{len(d.get('results', {}))}/{len(_S.P_GRID)} p-values, missing "
                f"{missing or 'none'}).")
            if not getattr(args, "allow_partial_anchor", False):
                log("     NO anchor is bundled.  The kernel will report the A1.2 check as "
                    "UNAVAILABLE and will NOT run the sizes above 1.5B.  Re-build when it "
                    "finishes.  (--allow-partial-anchor overrides, and marks the artefact.)")
                d = None
            else:
                log("     --allow-partial-anchor: bundling it ANYWAY, marked PARTIAL.")
    if d is not None:
        slim = {"baseline_bpb": d["baseline_bpb"], "donor": d["donor"], "slice": d["slice"],
                "env": d["env"], "elapsed_s": d.get("elapsed_s"),
                "results": {p: {a: {"bpb": r[a]["bpb"], "delta_bpb": r[a]["delta_bpb"],
                                    "achieved": r[a]["achieved"]}
                                for a in r if a in ("A", "B", "C", "D", "D2")}
                            for p, r in d["results"].items()},
                "controls": d.get("controls", {}),
                "PROVENANCE": "local CPU fp32 run on this machine; the A1.2 platform anchor",
                "COMPLETE": not missing,
                "p_values_missing_vs_P_GRID": missing}
        (BUNDLE / "anchor_cpu.json").write_text(json.dumps(slim, indent=2))
        n_p = len(slim["results"])
        log(f"  anchor_cpu.json written from the local CPU run "
            f"(baseline {slim['baseline_bpb']:.9f}, {n_p} p-values, "
            f"COMPLETE={not missing})")
    elif not anchor_src.is_file():
        log("  !! NO local anchor yet -- the kernel will report the A1.2 check as UNAVAILABLE "
            "and will NOT run the sizes above 1.5B.  Re-build once the local run finishes.")

    tot = 0
    log(f"\nbundle: {BUNDLE}")
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file():
            tot += p.stat().st_size
            log(f"  {p.relative_to(BUNDLE).as_posix():44} {p.stat().st_size:>10,} B")
    log(f"  {'TOTAL':44} {tot:>10,} B")

    meta = {"title": DS_TITLE, "id": f"REPLACED_AT_UPLOAD/{DS_SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    (BUNDLE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return 0


# ------------------------------------------------------------------------------- kaggle
def _user(acct):
    return kaggle_ops.username(acct)


def cmd_whoami(args):
    ns = argparse.Namespace(accounts=[args.account] if args.account else list(kaggle_ops.ACCOUNTS))
    return kaggle_ops.cmd_whoami(ns)


def cmd_upload(args):
    if not BUNDLE.is_dir():
        sys.exit("run `build` first")
    user = _user(args.account)
    meta = json.loads((BUNDLE / "dataset-metadata.json").read_text())
    meta["id"] = f"{user}/{DS_SLUG}"
    (BUNDLE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    r = kaggle_ops.kaggle(args.account, "datasets", "list", "--mine", "--search", DS_SLUG,
                          "--csv", check=False)
    exists = DS_SLUG in (r.stdout or "")
    if exists:
        r = kaggle_ops.kaggle(args.account, "datasets", "version", "-p", str(BUNDLE),
                              "-m", "s1 code bundle update", "--dir-mode", "zip", check=False)
    else:
        r = kaggle_ops.kaggle(args.account, "datasets", "create", "-p", str(BUNDLE),
                              "--dir-mode", "zip", check=False)
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    log(f"\ndataset: {user}/{DS_SLUG}  ({'new version' if exists else 'created'})")
    return r.returncode


def cmd_push(args):
    user = _user(args.account)
    staging = Path(tempfile.mkdtemp(prefix="s1k_"))
    try:
        # the kernel body is a 3-line shim: everything real lives in the attached dataset,
        # so the code that runs on the GPU is byte-identical to the code that ran on the CPU.
        shim = (
            "import glob, os, runpy, sys\n"
            "os.environ.setdefault('S1_RESULTS', '/kaggle/working/s1_results')\n"
            + (f"os.environ['S1_ONLY'] = {args.only!r}\n" if args.only else "")
            + (f"os.environ['S1_LADDER_GRID'] = '1'\n" if args.ladder_grid else "")
            + "hits = glob.glob('/kaggle/input/**/s1_kaggle_entry.py', recursive=True)\n"
              "assert len(hits) == 1, hits\n"
              "d = os.path.dirname(hits[0]); sys.path.insert(0, d)\n"
              "runpy.run_path(hits[0], run_name='__main__')\n"
        )
        (staging / "s1_kernel.py").write_text(shim)
        meta = {
            "id": f"{user}/{KERNEL_SLUG}",
            "title": KERNEL_TITLE,
            "code_file": "s1_kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            # internet ON: the donors are pulled from the HF hub at run time.  The eval slice
            # is NOT pulled -- it ships in the dataset, pinned.
            "enable_internet": "true",
            "machine_shape": MACHINE,
            "dataset_sources": [f"{user}/{DS_SLUG}"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        log(f"=== push {user}/{KERNEL_SLUG}  (machine={MACHINE}, only={args.only}) ===")
        # A freshly created dataset is not immediately ATTACHABLE: Kaggle processes the upload
        # first, and a push inside that window prints "not valid dataset sources" and then
        # SUCCEEDS -- creating a kernel version that runs with no code bundle on the mount.
        # kaggle_run.py already carries this guard (it was written after the failure happened
        # for real on 2026-08-01); reuse it rather than write a second one.
        sys.path.insert(0, str(REPO / "scripts"))
        from kaggle_run import _wait_datasets_ready
        late = _wait_datasets_ready(args.account, [f"{user}/{DS_SLUG}"])
        if late:
            sys.exit(f"push aborted: {', '.join(late)} never became READY. Pushing now would "
                     f"create a version that silently runs WITHOUT the code bundle.")
        r = kaggle_ops.kaggle(args.account, "kernels", "push", "-p", str(staging),
                              "-t", str(SESSION_SECONDS), check=False)
        blob = (r.stdout or "") + (r.stderr or "")
        sys.stdout.write(r.stdout or "")
        sys.stderr.write(r.stderr or "")
        # same backstop kaggle_run.py uses: a rejected dataset source still exits 0
        if "not valid dataset sources" in blob or "could not be added to the kernel" in blob:
            sys.exit("push FAILED SILENTLY: Kaggle rejected the dataset source. The kernel "
                     "exists but has no code bundle attached; delete or re-push it.")
        log(f"\nwatch: python {Path(__file__).name} status {args.account}")
        return r.returncode
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def cmd_status(args):
    r = kaggle_ops.kaggle(args.account, "kernels", "status",
                          f"{_user(args.account)}/{KERNEL_SLUG}", check=False)
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    return r.returncode


def cmd_output(args):
    dest = args.path or str(HERE / "results" / "kaggle")
    os.makedirs(dest, exist_ok=True)
    r = kaggle_ops.kaggle(args.account, "kernels", "output",
                          f"{_user(args.account)}/{KERNEL_SLUG}", "-p", dest, check=False)
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    log(f"\n-> {dest}")
    return r.returncode


# ------------------------------------------------------------------------------- estimate
ESTIMATE = {
    "MEASURED_on_this_machine": {
        "cpu_fp32_1.5B_forward_per_batch_4x512_s": 22.7,
        "cpu_fp32_1.5B_mask_per_layer_batch_s": 0.30,
        "evals_per_size_full_grid": "10 p x 5 arms + 1 baseline + 2 planted = 53",
        "note": "these are the only timing facts here that were measured; everything below is "
                "an ESTIMATE and is labelled as one",
    },
    "ESTIMATED_gpu": {
        "basis": "T4 fp16 realised throughput taken at ~20 TFLOP/s (peak 65, GEMM-realistic "
                 "fraction); one eval = 2*N_params*12288 tokens of FLOPs plus the LM head",
        "qwen2.5-0.5b": {"per_eval_s": 8, "evals": 53, "run_h": 0.12, "download_min": 2},
        "qwen2.5-1.5b": {"per_eval_s": 20, "evals": 53, "run_h": 0.29, "download_min": 5},
        "qwen2.5-7b": {"per_eval_s": 95, "evals": 53, "run_h": 1.40, "download_min": 20},
        "qwen2.5-14b": {"per_eval_s": 190, "evals": 53, "run_h": 2.80, "download_min": 35,
                        "fits": "TIGHT across 2x16 GB in fp16 (~29.6 GB weights); may OOM, in "
                                "which case it is SKIPPED, never re-run quantised"},
    },
    "TOTALS_estimated_gpu_hours": {
        "smoke_0.5B_only": 0.2,
        "0.5B + 1.5B (anchor gate)": 0.6,
        "0.5B + 1.5B + 7B": 2.4,
        "0.5B + 1.5B + 7B + 14B": 5.8,
        "weekly_budget_per_account": 30,
    },
    "CAVEAT": "no GPU is present on this machine, so no GPU number here was measured. The "
              "estimate exists to size the request, not to be quoted as a result.",
}


def cmd_estimate(_args):
    log(json.dumps(ESTIMATE, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--allow-partial-anchor", action="store_true",
                   help="bundle an anchor from a run that has NOT finished; the "
                        "artefact is marked COMPLETE=false")
    b.set_defaults(func=cmd_build)
    w = sub.add_parser("whoami"); w.add_argument("account", nargs="?")
    w.set_defaults(func=cmd_whoami)
    for name, fn in (("upload", cmd_upload), ("status", cmd_status)):
        s = sub.add_parser(name); s.add_argument("account", choices=kaggle_ops.ACCOUNTS)
        s.set_defaults(func=fn)
    pu = sub.add_parser("push"); pu.add_argument("account", choices=kaggle_ops.ACCOUNTS)
    pu.add_argument("--only", help="run a single size, e.g. qwen2.5-0.5b (the smoke)")
    pu.add_argument("--ladder-grid", action="store_true", help="use the reduced 6-point p grid")
    pu.set_defaults(func=cmd_push)
    ou = sub.add_parser("output"); ou.add_argument("account", choices=kaggle_ops.ACCOUNTS)
    ou.add_argument("-p", "--path"); ou.set_defaults(func=cmd_output)
    sub.add_parser("estimate").set_defaults(func=cmd_estimate)
    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
