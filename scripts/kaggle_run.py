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
import time
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
    "main-data":   {"folder": PACK / "main_run",     "slug": "phase64-main-data",   "title": "phase64 main run data"},
    # The main run's code bundle gets its OWN slug rather than re-versioning "phase64-code". Those are
    # two different generations -- relaytest pins 27b22582, the main run pins f86fdc6f -- and one slug
    # cannot hold both. Sharing it means every upload for one arm silently disarms the other's pin on
    # that account. The pin would catch it (loudly, at session start), but only after a session was spent.
    "code-main":   {"folder": PACK / "code_bundle",  "slug": "phase64-code-main",   "title": "phase64 main run code"},
    # The generation relaytest pins (27b22582), pulled back down from acct1 so the throwaway arm can be
    # given a second account WITHOUT re-versioning any dataset a closed arm was measured against.
    "code-rt":     {"folder": PACK / "code_bundle_rt", "slug": "phase64-code-rt",   "title": "phase64 relaytest code"},
}

# The dataset set the screening arms attach. Named because it is now one of two, not the only one.
SCREEN_DS = ("stage3-data", "code", "logits")

# arm -> (account, notebook script, kernel slug). alpha lives inside the script; nothing here sets it.
# NOTE on slugs: Kaggle derives the kernel slug from the TITLE, not from the id suffix.
# "phase64 arm4" -> phase64-arm4, so an id of phase64-arm4-a000 silently became phase64-arm4
# (with a warning). The slug here MUST be the real one, because relay attaches the kernel's
# own output via kernel_sources=[user/slug] -- a wrong slug means resume can't find itself.
#
# "datasets" is per-arm and NOT optional-by-accident. It used to be one hard-coded triple, which was
# harmless while every arm read the same corpus and became a live hazard the moment a second package
# existed: the cells locate their data by globbing /kaggle/input for PACKAGE_MANIFEST.json and taking
# the FIRST hit. Two packages on one mount makes that a coin flip. assert_package would refuse the wrong
# one (it pins the arm id), so the failure is loud rather than silent -- but it would be a session spent
# to learn a wiring fact, and which way the coin lands is not stable across pushes. Each arm attaches
# exactly the packages it reads.
ARMS = {
    "arm4": {"acct": "acct1", "script": PACK / "NOTEBOOK_arm4_a000_V2048.py", "slug": "phase64-arm4",
             "datasets": SCREEN_DS},
    "arm5": {"acct": "acct2", "script": PACK / "NOTEBOOK_arm5_a025_V2048.py", "slug": "phase64-arm5",
             "datasets": SCREEN_DS},
    "arm6": {"acct": "acct3", "script": PACK / "NOTEBOOK_arm6_a050_V2048.py", "slug": "phase64-arm6",
             "datasets": SCREEN_DS},
    # stage-4: alpha 0.0 + the kd-resident knob (new) + slice-sha now VERIFIED. New code bundle
    # (CODE_SHA 27b22582); data + logits unchanged from stage-3, so only phase64-code was re-uploaded.
    # The file is named w16 and the run that produced 1.1446 was w8: --kd-resident 16 was host-OOM
    # killed and the Builder's documented fallback edited the cell in place. The arm is closed and is
    # not relaunched, so the name is left as the record of the file, not of the number; the authority
    # for what ran is the RUN: line in results/arm7/. Flagged by the Builder 2026-08-01.
    "arm7": {"acct": "acct1", "script": PACK / "NOTEBOOK_arm7_w16_V2048.py", "slug": "phase64-arm7",
             "datasets": SCREEN_DS},
    # throwaway: validates the resume CHAIN (stop-incomplete -> push --relay -> rehydrate -> continue)
    # before the multi-session main run relies on it unattended. Tiny budget forces an incomplete stop.
    # Also the ROTATION rehearsal: it carries the same resume_*.pt / dataset-handoff machinery the main
    # run will, at a 4-minute session instead of an 11-hour one, so the cross-account path is proven on
    # something disposable first. Two accounts are enough to prove a handoff; the main run uses three.
    "relaytest": {"acct": "acct1", "script": PACK / "NOTEBOOK_relaytest_V2048.py", "slug": "phase64-relaytest",
                  "datasets": ("stage3-data", "code-rt", "logits"),
                  "accts": ("acct1", "acct3"), "resume_slug": "phase64-relaytest-resume"},
    # THE MAIN RUN. 1.5 B tokens, curriculum C->F, CE-primary (LOGITS_SHA empty), x_proj r=26, V=2048.
    # It attaches TWO datasets, not three: its own 4.3 GB package and the code bundle. The 18 GB logits
    # are deliberately absent -- a CE-primary run never opens them, and the stage-3 corpus is absent
    # because a second PACKAGE_MANIFEST.json on the mount is the coin flip described above.
    # Runs on acct2: idle since 2026-07-24, so the weekly GPU quota is whole at launch.
    # ROTATION. Kaggle bills 30 GPU-h per account per WEEK, and a session is ~11 h, so one account
    # carries under three sessions a week while the other two sit idle -- a 3x difference in how far a
    # multi-week run gets. "accts" is the rotation order; the run moves between them by carrying its
    # resume checkpoint across as a dataset (a private dataset cannot be read by another account, so
    # every account needs its own copy of the package, and the handoff has to physically move state).
    # The hardware pin is what makes this sound: every account runs the same T4x2, asserted by
    # --expect-gpus/--expect-gpu-name, so crossing accounts is numerically the same event as crossing
    # sessions on one. What it costs is bookkeeping -- the run's log lives in three kernels, not one.
    "mainrun": {"acct": "acct2", "script": PACK / "NOTEBOOK_main_V2048_r26_CE.py", "slug": "phase64-mainrun",
                "datasets": ("main-data", "code-main"),
                "accts": ("acct2", "acct3", "acct1"), "resume_slug": "phase64-mainrun-resume"},
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

def _wait_datasets_ready(account: str, refs, timeout_s: int = 1200):
    """Block until every dataset is READY, returning those that never got there.

    A freshly created dataset is not immediately attachable: Kaggle processes the upload first. Pushing
    in that window is the failure this exists for, and it is a quiet one -- `kernels push` prints
    "The following are not valid dataset sources and could not be added" and then SUCCEEDS, creating a
    version that runs without the dataset. For a rotation handoff that means the session starts from
    scratch instead of resuming, throws away every hour the run had banked, and reports success while
    doing it. Observed for real on the first cross-account handoff (2026-08-01).
    """
    pending = list(refs)
    deadline = time.time() + timeout_s
    while True:
        still = []
        for ref in pending:
            r = kaggle_ops.kaggle(account, "datasets", "status", ref, check=False)
            if not (r.returncode == 0 and "ready" in (r.stdout or "").lower()):
                still.append(ref)
        if not still or time.time() >= deadline:
            return still
        print(f"  waiting for {len(still)} dataset(s) to finish processing: {', '.join(still)}", flush=True)
        time.sleep(15)


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
    # Overridable, because the uploader keeps its resume-token cache under gettempdir()/.kaggle/uploads:
    # two uploads sharing one scratch share that cache, and a cache error there has already silently
    # dropped a dataset's final commit once (acct2, stage-3). Concurrent uploads get separate scratches.
    scratch = Path(os.environ.get("KAGGLE_ZIPTMP") or (REPO / "kaggle_rung1" / "_ziptmp"))
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
    acct = getattr(args, "acct", None) or arm["acct"]
    script, slug = arm["script"], arm["slug"]
    user = USER[acct]
    if not script.is_file():
        sys.exit(f"missing arm script: {script}")
    ds_sources = [f"{user}/{DATASETS[n]['slug']}" for n in arm["datasets"]]
    kernel_sources = [f"{user}/{slug}"] if args.relay else []   # self-output = the resume input

    # EXACTLY ONE SOURCE OF RESUME STATE, enforced rather than intended.
    # rehydrate() finds the checkpoint with glob('/kaggle/input/**/resume_<arm>.pt') and takes hit[0] --
    # an arbitrary one when there are several. Attaching both the kernel's own previous output AND a
    # handoff dataset puts two resume files of DIFFERENT ages on the mount, and the older one winning is
    # invisible: the run resumes, trains cleanly, and quietly redoes hours it had already done. The
    # trainer's own resume checks do not catch it either, because a stale checkpoint of the same arm is
    # internally consistent -- it is the right file, from the wrong time.
    if getattr(args, "resume_ds", False):
        if args.relay:
            sys.exit("push: --relay and --resume-ds are mutually exclusive. --relay attaches this "
                     "kernel's own previous output; --resume-ds attaches a carried-over checkpoint. "
                     "Both at once puts two resume files on the mount and rehydrate() picks arbitrarily.")
        ds_sources.append(f"{user}/{arm['resume_slug']}")

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

        late = _wait_datasets_ready(acct, ds_sources)
        if late:
            sys.exit(f"push aborted: dataset(s) not ready after the wait: {', '.join(late)}\n"
                     f"  Pushing now would create a version that silently runs WITHOUT them.")

        r = _run_cli(acct, "kernels", "push", "-p", str(staging), "-t", str(SESSION_SECONDS))
        # BACKSTOP. Kaggle reports a rejected dataset source on stdout and still exits 0, so a push that
        # dropped an input looks exactly like one that did not. Nothing downstream can tell the
        # difference afterwards -- the kernel just runs, and for a resume that means starting over.
        blob = (r.stdout or "") + (r.stderr or "")
        if "not valid dataset sources" in blob or "could not be added to the kernel" in blob:
            sys.exit(f"push FAILED SILENTLY: Kaggle rejected one or more dataset sources.\n"
                     f"  {blob.strip().splitlines()[-1] if blob.strip() else ''}\n"
                     f"  The kernel version exists but is missing an input it needs. Delete or re-push it; "
                     f"do NOT let a resume run without its checkpoint.")
        print(f"\nwatch: python scripts/kaggle_run.py status {args.arm}")
        print(f"logs : python scripts/kaggle_run.py output {args.arm}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    """Carry a run's checkpoint from one account to another, then start it there.

    WHY THIS EXISTS: Kaggle's GPU quota is per account and per week (30 h), so a run pinned to one
    account advances under three sessions a week while the other two idle. A private dataset cannot be
    read by another account and one account cannot attach another's kernel output, so the state has to
    physically move: download the source kernel's output, publish it as a dataset the TARGET account
    owns, and start the target's own kernel with that dataset attached. rehydrate() needs no change --
    it globs /kaggle/input for resume_<arm>.pt and does not care which mount it came from.

    Only the resume artifacts are carried, not the whole output. The output also contains the staged
    source tree, which is 167 KB of noise here and, worse, a second copy of the code whose presence on
    the mount is exactly the ambiguity the cell's staged-trainer assert exists to resolve.
    """
    arm = ARMS[args.arm]
    if "resume_slug" not in arm:
        sys.exit(f"handoff: arm {args.arm!r} is not configured for rotation (no resume_slug).")
    src, dst = args.src, args.dst
    if src == dst:
        sys.exit("handoff: source and destination are the same account -- use `push --relay` instead, "
                 "which attaches the kernel's own output without moving anything.")
    suser, duser = USER[src], USER[dst]

    scratch = Path(os.environ.get("KAGGLE_ZIPTMP") or (REPO / "kaggle_rung1" / "_ziptmp"))
    scratch.mkdir(parents=True, exist_ok=True)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(scratch)

    staging = Path(tempfile.mkdtemp(prefix=f"handoff_{args.arm}_", dir=str(scratch)))
    dl = staging / "_dl"
    dl.mkdir()
    try:
        print(f"=== handoff {args.arm}: {suser} -> {duser} ===")
        print(f"  [1/4] downloading {suser}/{arm['slug']} output ...", flush=True)
        _run_cli(src, "kernels", "output", f"{suser}/{arm['slug']}", "-p", str(dl))

        # Identify what to carry by looking, not by trusting a name written in this file. The cell's
        # internal run name is not the ARMS key, and duplicating it here would be a second place to
        # keep in sync -- the class of drift the arm7 "w16" filename is already a monument to.
        resumes = sorted(dl.glob("resume_*.pt"))
        if len(resumes) != 1:
            sys.exit(f"handoff: expected exactly one resume_*.pt in the source output, found "
                     f"{len(resumes)}: {[p.name for p in resumes]}\n"
                     f"  Refusing: which one is the run's state is not a guess worth making.")
        name = resumes[0].name[len("resume_"):-len(".pt")]
        payload = staging / "payload"
        payload.mkdir()
        carried = []
        shutil.copy(resumes[0], payload / resumes[0].name)
        carried.append(resumes[0].name)
        for extra in (f"{name}.pt", f"{name}.pt.done"):
            if (dl / extra).is_file():
                shutil.copy(dl / extra, payload / extra); carried.append(extra)
        stages = dl / f"stages_{name}"
        if stages.is_dir():
            shutil.copytree(stages, payload / stages.name); carried.append(stages.name + "/")
        size = sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())
        print(f"  [2/4] carrying {carried}  ({size / 2**20:.0f} MiB)")

        # 3. publish it as a dataset the DESTINATION account owns.
        _write_dataset_metadata(payload, duser, arm["resume_slug"], f"phase64 {args.arm} resume state")
        r = kaggle_ops.kaggle(dst, "datasets", "status", f"{duser}/{arm['resume_slug']}", check=False)
        exists = r.returncode == 0 and "ready" in r.stdout.lower()
        print(f"  [3/4] {'versioning' if exists else 'creating'} {duser}/{arm['resume_slug']} ...", flush=True)
        if exists:
            _run_cli(dst, "datasets", "version", "-p", str(payload),
                     "-m", f"resume state carried from {suser}", "--dir-mode", "zip")
        else:
            _run_cli(dst, "datasets", "create", "-p", str(payload), "--dir-mode", "zip")

        if args.no_push:
            print("  [4/4] --no-push: dataset published, kernel NOT started.")
            return 0
        print(f"  [4/4] starting {duser}/{arm['slug']} with the carried state ...", flush=True)
        return cmd_push(argparse.Namespace(arm=args.arm, relay=False, acct=dst, resume_ds=True))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def cmd_status(args: argparse.Namespace) -> int:
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for a in arms:
        arm = ARMS[a]
        # A rotating arm has one kernel PER ACCOUNT, all sharing the slug. Reporting only the primary
        # would hide the session that is actually running.
        for acct in arm.get("accts", (arm["acct"],)):
            user = USER[acct]
            r = kaggle_ops.kaggle(acct, "kernels", "status", f"{user}/{arm['slug']}", check=False)
            line = (r.stdout or r.stderr).strip().replace("\n", " ")
            print(f"{a:9} {user:13} {arm['slug']:20} {line}")
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
    print(f"{'arm':10} {'account':9} {'user':14} {'kernel-slug':22} {'datasets':28} script")
    for a, arm in ARMS.items():
        ds = ",".join(arm["datasets"])
        who = ("ROTATES " + ">".join(USER[x] for x in arm["accts"])) if "accts" in arm else USER[arm["acct"]]
        print(f"{a:10} {arm['acct']:9} {who:14} {arm['slug']:22} {ds:28} {arm['script'].name}")
    print("\ndataset payloads (uploaded per account that needs them):")
    for n, d in DATASETS.items():
        mark = "" if d["folder"].is_dir() else "   [payload folder MISSING]"
        print(f"  {n:12} {d['slug']:22} <- {d['folder']}{mark}")
    return 0


def _safe_console() -> None:
    """Never let PRINTING a message be the thing that crashes the driver.

    The two halves of this are easy to fix separately and wrong separately. Reading the CLI's output as
    utf-8 with errors="replace" (kaggle_ops) can produce U+FFFD; writing U+FFFD to a cp1252 console
    then raises UnicodeEncodeError -- which is how `status` died mid-table while reporting a kernel
    perfectly correctly. Decoding leniently and encoding strictly just moves the failure downstream.
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    _safe_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("upload"); u.add_argument("account", choices=USER)
    u.add_argument("which", choices=[*DATASETS, "all"]); u.set_defaults(func=cmd_upload)

    pu = sub.add_parser("push"); pu.add_argument("arm", choices=ARMS)
    pu.add_argument("--relay", action="store_true")
    pu.add_argument("--acct", choices=USER, help="run on this account instead of the arm's primary")
    pu.add_argument("--resume-ds", action="store_true",
                    help="attach the arm's carried-over resume dataset (rotation); excludes --relay")
    pu.set_defaults(func=cmd_push)

    ho = sub.add_parser("handoff", help="carry a run's checkpoint to another account and start it there")
    ho.add_argument("arm", choices=ARMS)
    ho.add_argument("--src", required=True, choices=USER); ho.add_argument("--dst", required=True, choices=USER)
    ho.add_argument("--no-push", action="store_true", help="publish the resume dataset but do not start")
    ho.set_defaults(func=cmd_handoff)

    st = sub.add_parser("status"); st.add_argument("arm", choices=[*ARMS, "all"]); st.set_defaults(func=cmd_status)
    ou = sub.add_parser("output"); ou.add_argument("arm", choices=ARMS)
    ou.add_argument("-p", "--path"); ou.set_defaults(func=cmd_output)
    sub.add_parser("plan").set_defaults(func=cmd_plan)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
