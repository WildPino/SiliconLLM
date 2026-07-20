#!/usr/bin/env python3
"""Acceptance for --branch-from: forking from a stage artefact must reproduce the baseline EXACTLY.

This is the prerequisite for declaring the rung-1 branch-from-D A/B valid. If the fork is not byte-identical
to the point where the baseline was, the two upcycle arms are not being compared under equal conditions and
the experiment measures the fork as much as the variable.

Method: run the full curriculum once with --save-stage-ckpt (the baseline), then fork from the stage-D
artefact and run only E->F. Every stage the branch re-runs must match the baseline's row to the last digit.

What makes this non-trivial (and what an earlier model-only artefact would have failed):
  * gstep -- batch sampling is a pure function of (seed, rank, gstep, micro);
  * kdc.pos -- which teacher-logit chunks are resident;
  * GradScaler state -- created once and carried across stages; a fresh one restarts at 65536.
Optimizer moments are correctly absent: stage entry rebuilds the optimizer on the baseline path too.

Run: python benchmarks/phase64/mve/ws4_branch_check.py
"""
import os, re, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PY): PY = sys.executable
DATA = os.path.join(ROOT, "kaggle_mve", "account_1", "mve_data", "data")
work = tempfile.mkdtemp(prefix="ws4_branch_")

BASE = [PY, os.path.join(HERE, "mve_train.py"), "--tag", "full", "--arm", "kd", "--kd", "span",
        "--recall", "on", "--sparse-moe", "--qat-alpha", "8", "--smoke", "--device", "cuda",
        "--data-dir", DATA]


def rows_of(txt):
    return {m[0]: m for m in re.findall(r"stage (\w) done: val BPB ([\d.]+) -> ([\d.]+) \(([-+\d.]+)\)", txt)}


try:
    stg = os.path.join(work, "stages")
    r1 = subprocess.run(BASE + ["--save-stage-ckpt", stg, "--ckpt-dir", work,
                                "--out", os.path.join(work, "base.pt"),
                                "--resume-ckpt", os.path.join(work, "base_r.pt")],
                        capture_output=True, text=True)
    base = rows_of(r1.stdout + r1.stderr)
    print("baseline stages:", {k: v[2] for k, v in base.items()} or "(none -- run failed)")
    if not base:
        print("\n".join((r1.stdout + r1.stderr).strip().splitlines()[-15:])); sys.exit(1)

    dckpt = os.path.join(stg, "stage_D_kd_span_on_full.pt")
    assert os.path.isfile(dckpt), f"missing stage-D artefact at {dckpt}"

    r2 = subprocess.run(BASE + ["--branch-from", dckpt, "--ckpt-dir", work,
                                "--out", os.path.join(work, "br.pt"),
                                "--resume-ckpt", os.path.join(work, "br_r.pt")],
                        capture_output=True, text=True)
    br = rows_of(r2.stdout + r2.stderr)
    print("branch   stages:", {k: v[2] for k, v in br.items()} or "(none -- run failed)")
    if not br:
        print("\n".join((r2.stdout + r2.stderr).strip().splitlines()[-15:])); sys.exit(1)

    ok = True
    print(f"\n  {'stage':6s} {'baseline in->out':26s} {'branch in->out':26s}  match")
    for st in ("E", "F"):
        if st not in br:
            print(f"  {st:6s} MISSING from branch"); ok = False; continue
        b, c = base[st], br[st]
        same = (b[1], b[2]) == (c[1], c[2])
        print(f"  {st:6s} {b[1]} -> {b[2]:18s} {c[1]} -> {c[2]:18s}  {'IDENTICAL' if same else 'DIFFER'}")
        ok &= same
    # stages before the fork must NOT be re-run: that is the whole economic point
    rerun = [st for st in ("C", "D") if st in br]
    print(f"\n  stages re-run before the fork: {rerun or 'none'}  "
          f"({'correct -- E+F only' if not rerun else 'WRONG: the fork saved nothing'})")
    ok &= not rerun
    print("\n==== branch-from acceptance: " + ("PASS" if ok else "FAIL") + " ====")
    sys.exit(0 if ok else 1)
finally:
    shutil.rmtree(work, ignore_errors=True)
