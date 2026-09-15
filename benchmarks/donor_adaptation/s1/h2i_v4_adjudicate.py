#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2I Phase-B v4 terminal guard and immutable invocation of the CPU evaluator.

This file does not alter H2I's scientific judge.  It makes addendum K's external freeze
executable: the final v4 bundle, raw log, training metadata, evaluator dependencies and
scientific inputs are verified before invoking the already-published ``h2i_eval.py`` once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results" / "h2i"
EVALUATOR = HERE / "h2i_eval.py"
FACTORS = Path("D:/_ktmp/h0_run3_final/h0_trained3.npz")
LABELS = ROOT / "benchmarks/donor_adaptation/density/results/d0c_labels/labels_E256.npz"
STATS = HERE / "results/h1/h1_actstats.npz"
ROUTERS = Path("D:/_ktmp/e37/e37_routers_E256.npz")
H1_RESULT = HERE / "results/h1/h1_applied_8L.json"
HELDOUT = HERE / "results/h1/h1_heldout.npz"
PHASE_A = RESULTS / "h2i_applied_8L.json"
DEFAULT_OUT = RESULTS / "h2i_eval_h2i_trained_s1.json"
DEFAULT_AUDIT = RESULTS / "h2i_eval_h2i_trained_s1_adjudication.json"

EXPECTED_LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]
EXPECTED_MODEL = "Qwen/Qwen2.5-1.5B"
EXPECTED_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
EXPECTED_RULE = "R8 per-output-row amax/127 RTN, codes [-127,127], STE"
EXPECTED_MANIFEST_SHA256 = "330fa237ebe3ea85c48447f52628a3f77f1f4cf0e7275ef7e122e9eb8b93ba56"
SCORE_THRESHOLD = 1.0190764652622473

# Frozen while Kaggle v4 was RUNNING.  These are the seven records in
# h2i_phase_b_v4_adjudication_freeze.json, copied here so the freeze is enforced.
FROZEN_DEPENDENCIES = {
    "benchmarks/donor_adaptation/s1/h2i_eval.py": {
        "bytes": 16328,
        "sha256": "977d89abbf4567566814eccde9690c058e13da643a959e0672ba1badbec991bb",
        "git_blob": "0813f32f8c165ae92d347f26e30e85ae88eb7a7f",
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
    "benchmarks/donor_adaptation/s1/h2i_applied.py": {
        "bytes": 23767,
        "sha256": "fcb717c10fe5fbcf26762f5ae6e2324484e2b4a02195caee6088a9c3c3339900",
        "git_blob": "40c188cc769575b8a9704359c68899ad4d8b368e",
    },
    "benchmarks/donor_adaptation/ternary/t2_rules.py": {
        "bytes": 18895,
        "sha256": "10037c87fb4403d5f0f97bde9736c7f198485cc20722e5e937037bb8a2c2e8c4",
        "git_blob": "8be4464e7d81cd89686a4fe2179c42f0f4f9a9e4",
    },
    "benchmarks/donor_adaptation/engine/e6_generate.py": {
        "bytes": 11004,
        "sha256": "410353b083c9a98e6a3044551212b5c8aae8995ed1cbaac0f03b99016726824e",
        "git_blob": "5a4a5484832de5ed2d241fcaf4d813f692f0a8ac",
    },
    "benchmarks/donor_adaptation/engine/results/e6/engine.json": {
        "bytes": 10827,
        "sha256": "6e865cf72675335c1844d14985502798c2437aa106562b92c34adcfeb3535965",
        "git_blob": "7f4a82b08904a420e2dff870f73aff58569f708e",
    },
}

FROZEN_INPUTS = {
    str(FACTORS): (352967686, "dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8"),
    str(LABELS): (859537, "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c"),
    str(STATS): (1189630, "49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe"),
    str(ROUTERS): (44046858, "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a"),
    str(H1_RESULT): (1605, "6ebc333f74f0f337567a99e527cdfa7750d092161cc5ca2d1836a510557f1b17"),
    str(HELDOUT): (98564, "110a90ab358efcacdc090fdf0fce62bb74d6d06df02d1083cf4cf7e752529d35"),
    str(PHASE_A): (12239, "c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc"),
}

TRAINING_APPARATUS = {
    "h2i_qat.py": "f8df56204e564ca0ced074059d0418fcb16682b5d200cc18a1785d46290d6c02",
    "h2i_applied.py": "fcb717c10fe5fbcf26762f5ae6e2324484e2b4a02195caee6088a9c3c3339900",
    "h1_qat.py": "7876d6032c3fd46f4d836d88aaf7ebae087f0437dc743f310f0a3a4652bd30ad",
    "common.py": "7ac00b31e91d0018c775e2e9b26120c0ec025075d7913218ea3cdcfe3196e461",
    "t2_rules.py": "10037c87fb4403d5f0f97bde9736c7f198485cc20722e5e937037bb8a2c2e8c4",
    "e6_generate.py": "410353b083c9a98e6a3044551212b5c8aae8995ed1cbaac0f03b99016726824e",
}

TRAINING_INPUTS = {
    "factors": (352967686, "dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8"),
    "labels": (859537, "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c"),
    "stats": (1189630, "49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe"),
    "routers": (44046858, "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a"),
    "phase_a": (12239, "c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc"),
    "train": (64000260, "0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9"),
    "heldout": (98564, "110a90ab358efcacdc090fdf0fce62bb74d6d06df02d1083cf4cf7e752529d35"),
    "calib": (131332, "b8d4184db0988f4d86ae3089e99ba167b8e05e6dcdbbc504f7679729fe7aa019"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def file_identity(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file(path: Path, expected_bytes: int, expected_sha: str,
                expected_blob: str | None = None) -> dict:
    actual = file_identity(path)
    if (actual["bytes"], actual["sha256"]) != (expected_bytes, expected_sha):
        raise SystemExit(f"frozen file identity mismatch: {path}: {actual}")
    if expected_blob is not None:
        actual["git_blob"] = git("hash-object", str(path))
        if actual["git_blob"] != expected_blob:
            raise SystemExit(f"frozen Git blob mismatch: {path}")
    return actual


def verify_guard_provenance() -> dict:
    path = Path(__file__).resolve()
    rel = path.relative_to(ROOT).as_posix()
    worktree_blob = git("hash-object", str(path))
    head_blob = git("rev-parse", "HEAD:" + rel)
    if worktree_blob != head_blob:
        raise SystemExit("H2I v4 adjudication guard is not the committed HEAD blob -- STOP")
    return {"head": git("rev-parse", "HEAD"), "path": rel, "git_blob": worktree_blob,
            "sha256": sha256_file(path)}


def _finite_positive_mapping(value) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(item, (int, float)) and math.isfinite(item) and item > 0
        for item in value.values()
    )


def validate_training_meta(meta: dict) -> dict:
    wrong = []
    expected = {
        "complete": True, "phase": "B", "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION, "layers": EXPECTED_LAYERS, "E": 256, "k": 16,
        "ffn_rule": EXPECTED_RULE, "sdpa": "sdpa", "steps_requested": 4000,
        "seed": 4242, "lr": 2e-4, "router_lr": 3e-4, "aux": 0.01,
        "bs": 2, "accum": 8, "nonfinite_microbatches": 0,
        "router_init_sha256": TRAINING_INPUTS["routers"][1], "resumed_ffn_from": None,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            wrong.append((key, meta.get(key), value))

    steps = meta.get("steps_completed")
    applied = meta.get("optimizer_applied_updates")
    if not isinstance(steps, int) or not 250 <= steps <= 4000:
        wrong.append(("steps_completed", steps, "integer in [250, 4000]"))
    if not isinstance(applied, int) or not 250 <= applied <= (steps if isinstance(steps, int) else 4000):
        wrong.append(("optimizer_applied_updates", applied, "integer in [250, steps_completed]"))
    if meta.get("stop_reason") not in ("time-cap", "steps-exhausted"):
        wrong.append(("stop_reason", meta.get("stop_reason"), "time-cap|steps-exhausted"))
    if not _finite_positive_mapping(meta.get("optimizer_moved_at_first_applied")):
        wrong.append(("optimizer_moved_at_first_applied",
                      meta.get("optimizer_moved_at_first_applied"), "all watched deltas finite > 0"))

    manifest = meta.get("manifest") or {}
    manifest_expected = {"sha256": EXPECTED_MANIFEST_SHA256, "files": 15, "validated": True}
    for key, value in manifest_expected.items():
        if manifest.get(key) != value:
            wrong.append(("manifest." + key, manifest.get(key), value))

    provenance = meta.get("provenance") or {}
    if provenance.get("mode") != "manifest-only-no-git" or provenance.get("head_commit") is not None:
        wrong.append(("provenance mode/head", provenance, "manifest-only-no-git / null"))
    provenance_files = provenance.get("files") or {}
    for name, digest in TRAINING_APPARATUS.items():
        if (provenance_files.get(name) or {}).get("sha256") != digest:
            wrong.append(("provenance.files." + name,
                          (provenance_files.get(name) or {}).get("sha256"), digest))

    inputs = meta.get("inputs") or {}
    for name, (size, digest) in TRAINING_INPUTS.items():
        row = inputs.get(name) or {}
        if row.get("bytes") != size or row.get("sha256") != digest:
            wrong.append(("inputs." + name, (row.get("bytes"), row.get("sha256")),
                          (size, digest)))

    controls = meta.get("controls") or {}
    toy = controls.get("G_H2Ic_toy") or {}
    gradients = toy.get("ste_gradients_finite_nonzero") or {}
    seeded = controls.get("device_safe_seeded_probe") or {}
    boundary = controls.get("gate_boundary_test") or {}
    real = controls.get("G_H2Ic_real") or {}
    clauses = {
        "G_H2Ib.r8_exact_t2_rules": (controls.get("G_H2Ib") or {}).get("r8_exact_t2_rules") is True,
        "G_H2Ic_toy.gradients": all(gradients.get(name) is True for name in ("gate", "up", "down")),
        "G_H2Ic_toy.hard_identity": toy.get("hard_k_equals_E_any_router") is True,
        "G_H2Ic_toy.soft_identity": toy.get("soft_k_equals_E_zero_router") is True,
        "G_H2Ic_toy.carve_live": toy.get("carve_live") is True,
        "device_safe_seeded_probe": all(seeded.get(name) is True for name in
                                        ("repeat_exact", "global_rng_unchanged", "shape_ok", "finite")),
        "gate_boundary_test": boundary.get("fires") is True,
        "G_H2Ic_real": real.get("fires") is True,
        "G_H2Ic_real.router_init": real.get("router_init_exact") is True,
        "G_H2Ic_real.cardinality": real.get("cardinality_fires") is True,
        "G_H2Ic_real.carve_live": real.get("carve_live") is True,
    }
    for name, fires in clauses.items():
        if not fires:
            wrong.append(("controls." + name, False, True))

    progress = meta.get("progress")
    if not isinstance(progress, list) or not progress or progress[0].get("step") != 0:
        wrong.append(("progress", progress, "non-empty and starts at step 0"))
    else:
        last_step = -1
        for index, row in enumerate(progress):
            row_step = row.get("step")
            finite_fields = [row.get(key) for key in ("bpb_soft", "bpb_hard")]
            if not isinstance(row_step, int) or row_step <= last_step or \
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in finite_fields):
                wrong.append(("progress[%d]" % index, row, "strict step order and finite BPB"))
                break
            last_step = row_step
        if isinstance(steps, int) and last_step > steps:
            wrong.append(("progress.last_step", last_step, "<= steps_completed"))

    if wrong:
        raise ValueError("H2I v4 metadata differs from the frozen Phase-B object: %r" % (wrong,))
    return {"steps_completed": steps, "optimizer_applied_updates": applied,
            "stop_reason": meta["stop_reason"], "seconds": meta.get("seconds"),
            "seconds_per_step": meta.get("seconds_per_step"),
            "manifest_sha256": manifest["sha256"]}


def validate_result(result: dict, returncode: int) -> dict:
    expected_status = {0: "PASS", 2: "FAIL-SCORE", 3: "SCORE-ONLY"}
    if returncode not in expected_status:
        raise ValueError("operational evaluator return code %d" % returncode)
    score = (result.get("G_H2I_score") or {}).get("fires")
    rank = result.get("G_H2I_rank") or {}
    combined = (result.get("G_H2I") or {}).get("fires")
    expected = {
        0: (True, False, True, True),
        2: (False, True, False, False),
        3: (True, False, False, False),
    }[returncode]
    actual = (score, rank.get("not_run"), rank.get("fires"), combined)
    if result.get("status") != expected_status[returncode] or actual != expected:
        raise ValueError("evaluator exit/result disagreement: rc=%d status=%r gates=%r" %
                         (returncode, result.get("status"), actual))
    value = (result.get("G_H2I_score") or {}).get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("evaluator did not record a finite trained-hard score")
    if (value < SCORE_THRESHOLD) is not bool(score):
        raise ValueError("score threshold/result disagreement")
    return {"status": result["status"], "score_bpb": value,
            "score_fires": score, "rank_fires": rank.get("fires"),
            "combined_fires": combined}


def _good_meta() -> dict:
    controls = {
        "G_H2Ib": {"r8_exact_t2_rules": True},
        "G_H2Ic_toy": {"ste_gradients_finite_nonzero": {"gate": True, "up": True, "down": True},
                         "hard_k_equals_E_any_router": True,
                         "soft_k_equals_E_zero_router": True, "carve_live": True},
        "device_safe_seeded_probe": {"repeat_exact": True, "global_rng_unchanged": True,
                                       "shape_ok": True, "finite": True},
        "gate_boundary_test": {"fires": True},
        "G_H2Ic_real": {"fires": True, "router_init_exact": True,
                          "cardinality_fires": True, "carve_live": True},
    }
    return {
        "complete": True, "phase": "B", "model": EXPECTED_MODEL, "revision": EXPECTED_REVISION,
        "layers": EXPECTED_LAYERS, "E": 256, "k": 16, "ffn_rule": EXPECTED_RULE,
        "sdpa": "sdpa", "steps_requested": 4000, "steps_completed": 250,
        "stop_reason": "time-cap", "seed": 4242, "lr": 2e-4, "router_lr": 3e-4,
        "aux": 0.01, "bs": 2, "accum": 8, "nonfinite_microbatches": 0,
        "optimizer_applied_updates": 250, "optimizer_moved_at_first_applied": {"x": 1e-6},
        "router_init_sha256": TRAINING_INPUTS["routers"][1], "resumed_ffn_from": None,
        "manifest": {"sha256": EXPECTED_MANIFEST_SHA256, "files": 15, "validated": True},
        "provenance": {"mode": "manifest-only-no-git", "head_commit": None,
                         "files": {name: {"sha256": digest}
                                   for name, digest in TRAINING_APPARATUS.items()}},
        "inputs": {name: {"bytes": size, "sha256": digest}
                   for name, (size, digest) in TRAINING_INPUTS.items()},
        "controls": controls,
        "progress": [{"step": 0, "bpb_soft": 1.0, "bpb_hard": 1.1}],
    }


def selftest() -> None:
    good = _good_meta()
    validate_training_meta(good)
    mutations = [
        ("seed", 2718), ("steps_completed", 249), ("optimizer_applied_updates", 249),
        ("stop_reason", "in-progress"), ("nonfinite_microbatches", 1),
    ]
    for key, bad_value in mutations:
        bad = json.loads(json.dumps(good))
        bad[key] = bad_value
        try:
            validate_training_meta(bad)
        except ValueError:
            continue
        raise AssertionError("metadata control failed to reject %s=%r" % (key, bad_value))
    bad = json.loads(json.dumps(good))
    bad["manifest"]["sha256"] = "0" * 64
    try:
        validate_training_meta(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("metadata control failed to reject a different bundle manifest")
    for returncode, status, score, rank_not_run, rank_fires, combined, value in (
        (0, "PASS", True, False, True, True, SCORE_THRESHOLD - 1e-6),
        (2, "FAIL-SCORE", False, True, False, False, SCORE_THRESHOLD),
        (3, "SCORE-ONLY", True, False, False, False, SCORE_THRESHOLD - 1e-6),
    ):
        validate_result({"status": status, "G_H2I_score": {"fires": score, "value": value},
                         "G_H2I_rank": {"not_run": rank_not_run, "fires": rank_fires},
                         "G_H2I": {"fires": combined}}, returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", help="freshly downloaded final h2i_trained_s1.npz")
    parser.add_argument("--raw-log", help="raw log downloaded with Kaggle v4 output")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
        print("H2I v4 adjudication guard self-test PASS")
        return 0
    if not args.trained or not args.raw_log:
        parser.error("--trained and --raw-log are required unless --self-test is used")

    trained = Path(args.trained).resolve()
    sidecar = trained.with_suffix(".json")
    raw_log = Path(args.raw_log).resolve()
    out = Path(args.out).resolve()
    audit = Path(args.audit).resolve()
    existing = [str(path) for path in (out, audit) if path.exists()]
    if existing:
        raise SystemExit("write-once H2I adjudication output already exists: %s" % existing)
    if trained.name != "h2i_trained_s1.npz" or not trained.is_file() or not sidecar.is_file():
        raise SystemExit("expected final h2i_trained_s1.npz plus same-basename JSON -- STOP")
    if raw_log.name != "h2i-one-byte-phase-b.log" or not raw_log.is_file():
        raise SystemExit("expected preserved raw h2i-one-byte-phase-b.log -- STOP")

    provenance = verify_guard_provenance()
    dependencies = []
    for rel, expected in FROZEN_DEPENDENCIES.items():
        dependencies.append(verify_file(ROOT / rel, expected["bytes"], expected["sha256"],
                                        expected["git_blob"]))
    inputs = [verify_file(Path(path), size, digest)
              for path, (size, digest) in FROZEN_INPUTS.items()]
    with sidecar.open(encoding="utf-8") as handle:
        train_meta = json.load(handle)
    try:
        training_summary = validate_training_meta(train_meta)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    downloaded = [file_identity(trained), file_identity(sidecar), file_identity(raw_log)]
    print("H2I v4 terminal preflight PASS: %s" % training_summary, flush=True)
    if args.preflight_only:
        return 0

    command = [sys.executable, str(EVALUATOR), "--trained", str(trained),
               "--factors", str(FACTORS), "--labels", str(LABELS),
               "--stats", str(STATS), "--routers", str(ROUTERS),
               "--h1-result", str(H1_RESULT), "--heldout", str(HELDOUT),
               "--out", str(out)]
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in (0, 2, 3):
        raise SystemExit("h2i_eval.py operational failure %d; no scientific verdict" %
                         completed.returncode)
    if not out.is_file():
        raise SystemExit("h2i_eval.py returned without its result -- STOP")
    with out.open(encoding="utf-8") as handle:
        result = json.load(handle)
    try:
        verdict = validate_result(result, completed.returncode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    audit.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": "H2I_PHASE_B_V4_CPU_ADJUDICATION", "status": verdict["status"],
        "guard_provenance": provenance, "frozen_dependencies": dependencies,
        "frozen_inputs": inputs, "downloaded_outputs": downloaded,
        "training_summary": training_summary, "command": command,
        "evaluator_returncode": completed.returncode, "verdict": verdict,
        "result": {**file_identity(out), "G_H2I": result["G_H2I"]},
        "seconds": time.time() - started,
        "scope": "H2I one-byte 8-layer carve score+rank; no rate, engine, 10B or scale claim",
    }
    with audit.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
