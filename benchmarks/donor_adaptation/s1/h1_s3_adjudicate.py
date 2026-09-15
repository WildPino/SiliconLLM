#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 session-3 terminal guard and immutable invocation of the existing CPU evaluator.

This file does not change H1's scientific judge.  It freezes and verifies the evaluator and
its inputs before invoking the already-published ``h1_eval.py`` exactly once.  Addendum R of
BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md was written while the Kaggle job was still RUNNING.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results" / "h1"
EVALUATOR = HERE / "h1_eval.py"
FACTORS = HERE / "_h1_bundle" / "h0_trained3.npz"
LABELS = HERE / "_h1_bundle" / "labels_E256.npz"
STATS = HERE / "_h1_bundle" / "h1_actstats.npz"
ROUTERS = Path("D:/_ktmp/e37/e37_routers_E256.npz")
APPLIED = RESULTS / "h1_applied_8L.json"
DEFAULT_OUT = RESULTS / "h1_eval_h1_s3.json"
DEFAULT_AUDIT = RESULTS / "h1_eval_h1_s3_adjudication.json"

FROZEN = {
    "benchmarks/donor_adaptation/s1/h1_eval.py": {
        "bytes": 20771,
        "sha256": "aa2cad66f18a0471e7e1b863c58c4e8bbfb0f380f6687e6a27d8af84a8aa8396",
        "git_blob": "78dce2ff0d75bbf0dba7698705b3cda8f7d34790",
    },
    "benchmarks/donor_adaptation/density/common.py": {
        "bytes": 12666,
        "sha256": "7ac00b31e91d0018c775e2e9b26120c0ec025075d7913218ea3cdcfe3196e461",
        "git_blob": "608a563bc7f30e21c8941e3703b40ce25fb0960e",
    },
    "benchmarks/donor_adaptation/s1/h1_qat.py": {
        "bytes": 48685,
        "sha256": "7876d6032c3fd46f4d836d88aaf7ebae087f0437dc743f310f0a3a4652bd30ad",
        "git_blob": "4caeae0b77f72a0b389b0cb583aebff841063857",
    },
}

INPUTS = {
    str(APPLIED): {
        "bytes": 1605,
        "sha256": "6ebc333f74f0f337567a99e527cdfa7750d092161cc5ca2d1836a510557f1b17",
    },
    str(FACTORS): {
        "bytes": 352967686,
        "sha256": "dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8",
    },
    str(LABELS): {
        "bytes": 859537,
        "sha256": "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c",
    },
    str(STATS): {
        "bytes": 1189630,
        "sha256": "49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe",
    },
    str(ROUTERS): {
        "bytes": 44046858,
        "sha256": "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a",
    },
}

EXPECTED_LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_file(path: Path, expected: dict, git_blob: bool = False) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing frozen file: {path}")
    actual = {"path": str(path), "bytes": path.stat().st_size,
              "sha256": sha256_file(path)}
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise SystemExit(f"frozen file identity mismatch: {path}: {actual} != {expected}")
    if git_blob:
        actual["git_blob"] = git("hash-object", str(path))
        if actual["git_blob"] != expected["git_blob"]:
            raise SystemExit(f"frozen Git blob mismatch: {path}")
    return actual


def verify_guard_provenance() -> dict:
    rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    worktree_blob = git("hash-object", str(Path(__file__).resolve()))
    head_blob = git("rev-parse", "HEAD:" + rel)
    if worktree_blob != head_blob:
        raise SystemExit("H1 adjudication guard is not the committed HEAD blob -- STOP")
    return {"head": git("rev-parse", "HEAD"), "path": rel, "git_blob": worktree_blob,
            "sha256": sha256_file(Path(__file__).resolve())}


def basename(value) -> str | None:
    return Path(str(value).replace("\\", "/")).name if value else None


def validate_training_meta(meta: dict) -> dict:
    wrong = []
    expected = {
        "complete": True, "layers": EXPECTED_LAYERS, "k": 16, "E": 256,
        "steps_requested": 4000, "bs": 2, "accum": 8, "lr": 2e-4,
        "router_lr": 3e-4, "aux": 0.01, "seed": 3141,
        "nonfinite_microbatches": 0, "adam_state_restarted": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            wrong.append((key, meta.get(key), value))
    steps = meta.get("steps_completed")
    if not isinstance(steps, int) or not 250 <= steps <= 4000:
        wrong.append(("steps_completed", steps, "integer in [250, 4000]"))
    if meta.get("stop_reason") not in ("time-cap", "steps-exhausted"):
        wrong.append(("stop_reason", meta.get("stop_reason"), "time-cap|steps-exhausted"))
    if basename(meta.get("resumed_qo_from")) != "h0_trained3.npz":
        wrong.append(("resumed_qo_from", meta.get("resumed_qo_from"), "h0_trained3.npz"))
    if basename(meta.get("resumed_ffn_from")) != "h1_trained_s2.npz":
        wrong.append(("resumed_ffn_from", meta.get("resumed_ffn_from"), "h1_trained_s2.npz"))
    gates = meta.get("gates") or {}
    if not (gates.get("G_H1a", {}).get("hard_gate_any_router", {}).get("fires") is True
            and gates.get("G_H1a", {}).get("soft_gate_zero_router", {}).get("fires") is True):
        wrong.append(("gates.G_H1a", gates.get("G_H1a"), "both identity controls fire"))
    if gates.get("carve_is_live", {}).get("differs") is not True:
        wrong.append(("gates.carve_is_live", gates.get("carve_is_live"), "differs=true"))
    if gates.get("G_H1c", {}).get("fires") is not True:
        wrong.append(("gates.G_H1c", gates.get("G_H1c"), "fires=true"))
    first_applied = gates.get("G_H1b", {}).get("first_applied_step")
    if not isinstance(first_applied, int) or not 1 <= first_applied <= 40:
        wrong.append(("gates.G_H1b.first_applied_step", first_applied, "integer in [1, 40]"))
    if wrong:
        raise ValueError("session-3 metadata differs from addendum M: %r" % (wrong,))
    return {"steps_completed": steps, "stop_reason": meta["stop_reason"],
            "seconds": meta.get("seconds"), "seconds_per_step": meta.get("seconds_per_step")}


def selftest() -> None:
    good = {
        "complete": True, "layers": EXPECTED_LAYERS, "k": 16, "E": 256,
        "steps_requested": 4000, "steps_completed": 765, "stop_reason": "time-cap",
        "bs": 2, "accum": 8, "lr": 2e-4, "router_lr": 3e-4, "aux": 0.01,
        "seed": 3141, "nonfinite_microbatches": 0, "adam_state_restarted": True,
        "resumed_qo_from": "h0_trained3.npz", "resumed_ffn_from": "h1_trained_s2.npz",
        "gates": {"G_H1a": {"hard_gate_any_router": {"fires": True},
                              "soft_gate_zero_router": {"fires": True}},
                  "carve_is_live": {"differs": True},
                  "G_H1b": {"first_applied_step": 1}, "G_H1c": {"fires": True}},
    }
    validate_training_meta(good)
    for key, bad_value in (("seed", 2718), ("steps_completed", 249),
                           ("stop_reason", "in-progress"), ("nonfinite_microbatches", 1)):
        bad = json.loads(json.dumps(good))
        bad[key] = bad_value
        try:
            validate_training_meta(bad)
        except ValueError:
            continue
        raise AssertionError(f"metadata control failed to reject {key}={bad_value!r}")
    bad = json.loads(json.dumps(good))
    bad["gates"]["G_H1c"]["fires"] = False
    try:
        validate_training_meta(bad)
    except ValueError:
        return
    raise AssertionError("metadata control failed to reject a non-moving trained object")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained",
                        help="freshly downloaded final h1_trained_s3.npz")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
        print("H1 S3 adjudication guard self-test PASS")
        return 0
    if not args.trained:
        parser.error("--trained is required unless --self-test is used")

    trained = Path(args.trained).resolve()
    sidecar = trained.with_suffix(".json")
    out = Path(args.out).resolve()
    audit = Path(args.audit).resolve()
    existing = [str(path) for path in (out, audit) if path.exists()]
    if existing:
        raise SystemExit("write-once H1 adjudication output already exists: %s" % existing)
    if trained.name != "h1_trained_s3.npz" or not trained.is_file() or not sidecar.is_file():
        raise SystemExit("expected the final h1_trained_s3.npz plus same-basename JSON -- STOP")

    provenance = verify_guard_provenance()
    dependencies = []
    for rel, expected in FROZEN.items():
        dependencies.append(verify_file(ROOT / rel, expected, git_blob=True))
    inputs = [verify_file(Path(path), expected) for path, expected in INPUTS.items()]
    with sidecar.open(encoding="utf-8") as handle:
        train_meta = json.load(handle)
    try:
        summary = validate_training_meta(train_meta)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    trained_ids = [
        {"path": str(trained), "bytes": trained.stat().st_size,
         "sha256": sha256_file(trained)},
        {"path": str(sidecar), "bytes": sidecar.stat().st_size,
         "sha256": sha256_file(sidecar)},
    ]
    print("H1 S3 terminal preflight PASS: %s" % summary, flush=True)
    if args.preflight_only:
        return 0

    command = [sys.executable, str(EVALUATOR), "--trained", str(trained),
               "--factors", str(FACTORS), "--labels", str(LABELS), "--stats", str(STATS),
               "--routers", str(ROUTERS), "--tag", "h1_s3", "--out", str(out)]
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in (0, 1):
        raise SystemExit("h1_eval.py operational failure %d; no scientific verdict" %
                         completed.returncode)
    if not out.is_file():
        raise SystemExit("h1_eval.py returned without its result -- STOP")
    with out.open(encoding="utf-8") as handle:
        result = json.load(handle)
    passes = result.get("G_H1", {}).get("passes")
    if passes is not (completed.returncode == 0):
        raise SystemExit("evaluator exit/result disagreement -- STOP")
    audit.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": "H1_S3_CPU_ADJUDICATION", "status": "PASS" if passes else "FAIL",
        "guard_provenance": provenance, "frozen_dependencies": dependencies,
        "frozen_inputs": inputs, "trained_outputs": trained_ids,
        "training_summary": summary, "command": command,
        "evaluator_returncode": completed.returncode,
        "result": {"path": str(out), "bytes": out.stat().st_size,
                   "sha256": sha256_file(out), "G_H1": result["G_H1"]},
        "seconds": time.time() - started,
        "scope": "H1 ternary 8-layer carve only; no rank, rate, one-byte, 10B or scale claim",
    }
    with audit.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
