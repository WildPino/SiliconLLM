#!/usr/bin/env python3
"""Bounded CPU diagnostic: H4 terminal q/o alone, then the trained H2I FFN on it.

This is intentionally a measurement-only crossing.  It makes no gate decision and will not
load a model until the published H4/H2I identities and every runtime input have been checked.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
TERNDIR = os.path.abspath(os.path.join(HERE, "..", "ternary"))
for _path in (DENSDIR, ENGDIR, TERNDIR, HERE):
    sys.path.insert(0, _path)

import common as C  # noqa: E402
from e6_generate import N_NEW, PROMPTS  # noqa: E402
import h1_qat as H1  # noqa: E402
import h2i_applied as H2  # noqa: E402
import h2i_eval as H2E  # noqa: E402
import h4_qat as H4  # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

OUTDIR = os.path.join(HERE, "results", "h5")
DEFAULT_H4 = os.path.join(HERE, "results", "h4", "stage_a_v2_kaggle", "h4_trained.npz")
DEFAULT_H4_EVAL = os.path.join(HERE, "results", "h4", "h4_eval_stage_a_v2_terminal.json")
DEFAULT_H2I_TRAINED = "D:/_ktmp/h2i_kaggle_v4_output/h2i_trained_s1.npz"
DEFAULT_H2I_EVAL = os.path.join(HERE, "results", "h2i", "h2i_eval_h2i_trained_s1.json")
DEFAULT_OUT = os.path.join(OUTDIR, "h5_cross_composition.json")
E6_REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
EXPECTED_LAYERS = [3, 6, 9, 12, 15, 18, 21, 24]
EXPECTED_MODEL = "Qwen/Qwen2.5-1.5B"
EXPECTED_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
EXPECTED_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
EXPECTED_H4_EVAL_SHA = "23c3792f4a04efedf90ec0617654a529f0e2a115419f55cd71ad0d55d48e48f2"
EXPECTED_H2I_EVAL_SHA = "2180e1c5d01d903368b1b9119b99049ac5e0d48abbac8bbdbc460dca2151e805"
EXPECTED_H4_TERMINAL_SHA = "11a483419e6b8795164e4651547f6b3765fc325adaa3283153d9cd5276d4cefb"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_record(path):
    if not os.path.isfile(path):
        raise SystemExit("required input is missing: %s" % path)
    return {"path": os.path.abspath(path), "bytes": os.path.getsize(path),
            "sha256": sha256_file(path)}


def read_json(path, label):
    if not os.path.isfile(path):
        raise SystemExit("%s is missing: %s" % (label, path))
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("cannot read %s: %s" % (label, exc))


def assert_head_provenance():
    """Match h2i_eval's rule: this apparatus must be the committed HEAD blobs."""
    files = [os.path.abspath(__file__), os.path.abspath(C.__file__),
             os.path.abspath(H4.__file__), os.path.abspath(H1.__file__),
             os.path.abspath(H2.__file__), os.path.abspath(H2E.__file__),
             os.path.abspath(H2.T2.__file__), os.path.join(ENGDIR, "e6_generate.py"),
             H2.E6_ENGINE, E6_REF]
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        records = {}
        for path in files:
            rel = os.path.relpath(path, root).replace("\\", "/")
            worktree_blob = subprocess.check_output(
                ["git", "hash-object", path], cwd=root, text=True
            ).strip()
            head_blob = subprocess.check_output(
                ["git", "rev-parse", "HEAD:" + rel], cwd=root, text=True,
                stderr=subprocess.STDOUT
            ).strip()
            if worktree_blob != head_blob:
                raise SystemExit("uncommitted diagnostic dependency %s: %s != %s -- STOP"
                                 % (rel, worktree_blob, head_blob))
            records[rel] = {"git_blob": worktree_blob, "sha256": sha256_file(path)}
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("cannot establish diagnostic provenance: %s" % exc)
    return {"head_commit": head, "files": records}


def check_record(name, path, canonical):
    got = input_record(path)
    expected_sha = canonical.get("sha256")
    expected_bytes = canonical.get("bytes")
    if got["sha256"] != expected_sha or got["bytes"] != expected_bytes:
        raise SystemExit("canonical H2I %s mismatch: got %d %s; expected %s %s" %
                         (name, got["bytes"], got["sha256"], expected_bytes, expected_sha))
    return got


def check_h4_bundle(path, terminal_meta, published_eval):
    got = input_record(path)
    expected = {
        "stage": "H4", "model": EXPECTED_MODEL, "revision": EXPECTED_REVISION,
        "rank": 48, "terminal_checkpoint": True,
    }
    wrong = [(key, terminal_meta.get(key), value) for key, value in expected.items()
             if terminal_meta.get(key) != value]
    if terminal_meta.get("status") not in ("TIME_CAP", "STEPS_COMPLETE"):
        wrong.append(("status", terminal_meta.get("status"), "TIME_CAP|STEPS_COMPLETE"))
    if int(terminal_meta.get("actual_steps") or 0) <= 0:
        wrong.append(("actual_steps", terminal_meta.get("actual_steps"), ">0"))
    if int(terminal_meta.get("applied_steps") or 0) <= 0:
        wrong.append(("applied_steps", terminal_meta.get("applied_steps"), ">0"))
    if terminal_meta.get("checkpoint_sha256") != got["sha256"]:
        wrong.append(("checkpoint_sha256", terminal_meta.get("checkpoint_sha256"), got["sha256"]))
    if published_eval.get("factors_sha256") != got["sha256"]:
        wrong.append(("published factors_sha256", published_eval.get("factors_sha256"), got["sha256"]))
    if wrong:
        raise SystemExit("H4 terminal sidecar/identity mismatch: %r" % wrong)
    if H4.ORGANS != ("q_proj", "o_proj"):
        raise SystemExit("H4 apparatus no longer specifies q/o-only organs: %r" % (H4.ORGANS,))
    with np.load(path) as factors:
        layers = H4.bundle_layers(path)
        if layers != list(range(28)) or len(factors.files) != 280:
            raise SystemExit("H4 terminal must contain all 28 q/o layers, got %r" % layers)
        for layer in layers:
            for organ in H4.ORGANS:
                prefix = "L%02d.%s." % (layer, organ)
                shapes = {"A": (1536, 48), "s": (48,), "B": (48, 1536),
                          "rms_in": (1536,), "rms_A": (48,)}
                for suffix, shape in shapes.items():
                    key = prefix + suffix
                    if key not in factors.files or tuple(factors[key].shape) != shape:
                        raise SystemExit("invalid H4 rank48 q/o factor %s" % key)
    return got


def check_published_identity(h4_eval, h2i_eval):
    for label, record in (("H4 eval", h4_eval), ("H2I eval", h2i_eval)):
        if record.get("model") != EXPECTED_MODEL or record.get("revision") != EXPECTED_REVISION:
            raise SystemExit("%s model/revision is not the pinned Qwen 1.5B object" % label)
        meta = record.get("eval_slice", {})
        if (meta.get("part"), meta.get("n_seq"), meta.get("seq_len"), meta.get("seed"),
                meta.get("ids_sha256")) != ("heldout", 24, 512, 1234, EXPECTED_IDS_SHA):
            raise SystemExit("%s does not identify the frozen 24x512 heldout seed1234 slice" % label)
    if h4_eval.get("stage") != "H4" or h4_eval.get("mode") != "factored_trained" \
            or h4_eval.get("rank") != 48:
        raise SystemExit("published H4 terminal evaluation identity is invalid")
    if h2i_eval.get("layers") != EXPECTED_LAYERS or h2i_eval.get("k") != 16 \
            or h2i_eval.get("E") != 256:
        raise SystemExit("published H2I result is not the registered 8L hard-k16 R8 object")


def preflight(args):
    if os.path.exists(args.out):
        raise SystemExit("H5 cross-composition output already exists; it is write-once: %s" % args.out)
    if (C.MODEL_ID, C.REVISION) != (EXPECTED_MODEL, EXPECTED_REVISION) \
            or (H4.MODEL_ID, H4.REVISION) != (EXPECTED_MODEL, EXPECTED_REVISION) \
            or (H2.EXPECT_MODEL, H2.EXPECT_REVISION) != (EXPECTED_MODEL, EXPECTED_REVISION):
        raise SystemExit("source model/revision constants drifted from the pinned Qwen 1.5B revision")

    provenance = assert_head_provenance()
    for label, path, expected_sha in (
            ("H4 canonical evaluation", args.h4_eval, EXPECTED_H4_EVAL_SHA),
            ("H2I canonical evaluation", args.h2i_eval, EXPECTED_H2I_EVAL_SHA)):
        if input_record(path)["sha256"] != expected_sha:
            raise SystemExit("%s SHA-256 mismatch -- STOP" % label)
    h4_eval = read_json(args.h4_eval, "published H4 evaluation")
    h2i_eval = read_json(args.h2i_eval, "published H2I evaluation")
    check_published_identity(h4_eval, h2i_eval)
    h4_terminal_path = os.path.splitext(args.h4)[0] + ".json"
    h4_terminal = read_json(h4_terminal_path, "H4 terminal sidecar")
    h2i_meta_path = os.path.splitext(args.h2i_trained)[0] + ".json"
    h2i_meta = read_json(h2i_meta_path, "H2I trained sidecar")
    h4_bundle = check_h4_bundle(args.h4, h4_terminal, h4_eval)
    if h4_bundle["sha256"] != EXPECTED_H4_TERMINAL_SHA:
        raise SystemExit("H4 terminal checkpoint SHA-256 mismatch -- STOP")

    canonical_inputs = h2i_eval.get("inputs", {})
    needed = ("labels", "stats", "trained", "trained_json")
    if any(name not in canonical_inputs for name in needed):
        raise SystemExit("published H2I result lacks a canonical input identity")
    inputs = {
        "h4_terminal": h4_bundle,
        "h4_terminal_sidecar": input_record(h4_terminal_path),
        "labels": check_record("labels", args.labels, canonical_inputs["labels"]),
        "stats": check_record("stats", args.stats, canonical_inputs["stats"]),
        "h2i_trained": check_record("trained", args.h2i_trained, canonical_inputs["trained"]),
        "h2i_trained_sidecar": check_record("trained_json", h2i_meta_path,
                                              canonical_inputs["trained_json"]),
        "published_h4_eval": input_record(args.h4_eval),
        "published_h2i_eval": input_record(args.h2i_eval),
        "e6_reference": input_record(E6_REF),
        "e6_engine": input_record(H2.E6_ENGINE),
    }
    if h2i_meta != h2i_eval.get("train_meta"):
        raise SystemExit("H2I trained metadata differs from the canonical published identity")
    return provenance, h4_eval, h2i_eval, inputs


def dense_e6_targets():
    ref = read_json(E6_REF, "dense E6 reference")
    targets = ref.get(EXPECTED_MODEL)
    if not isinstance(targets, list) or len(targets) != len(PROMPTS):
        raise SystemExit("dense E6 reference does not contain five Qwen 1.5B prompts")
    ids = []
    for index, row in enumerate(targets):
        target = row.get("ids") if isinstance(row, dict) else None
        if not isinstance(row, dict) or row.get("prompt") != index \
                or not isinstance(target, list) or len(target) < N_NEW:
            raise SystemExit("dense E6 target %d is not a 32-token reference" % index)
        ids.append(target[-N_NEW:])
    return ids


def measure_arm(model, tok, ids, nbytes, prompt_ids, targets):
    bpb = H2.bpb(model, ids, nbytes)
    rank = H2.rank_against(model, prompt_ids, targets)
    return {
        "bpb": bpb,
        "free": rank["free"],
        "counted": rank["counted"],
        "teacher_forced": rank["teacher_forced_top1"],
        "mean_rank": rank["mean_target_rank"],
        "rank_le5": rank["rank_le5"],
        "per_prompt": rank["per_prompt"],
        "generated_ids": rank["generated_ids"],
        "decoded_generations": [tok.decode(row) for row in rank["generated_ids"]],
    }


def assert_h4_reproduction(actual, published):
    if abs(actual["bpb"] - published["bpb"]) > 1e-5:
        raise SystemExit("H4 BPB does not reproduce canonical terminal evaluation")
    for key in ("free", "teacher_forced", "rank_le5"):
        if actual[key] != published[key]:
            raise SystemExit("H4 %s does not reproduce canonical terminal evaluation" % key)
    if actual["mean_rank"] != published["mean_rank"]:
        raise SystemExit("H4 mean rank does not reproduce canonical terminal evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h4", default=DEFAULT_H4)
    parser.add_argument("--h4-eval", default=DEFAULT_H4_EVAL)
    parser.add_argument("--h2i-trained", default=DEFAULT_H2I_TRAINED)
    parser.add_argument("--h2i-eval", default=DEFAULT_H2I_EVAL)
    parser.add_argument("--labels", default=H2.DEFAULT_LABELS)
    parser.add_argument("--stats", default=H2.DEFAULT_STATS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    provenance, h4_published, h2i_published, inputs = preflight(args)
    if args.preflight_only:
        print("H5 PREFLIGHT PASS: source, checkpoint, sidecar, and canonical-result identities", flush=True)
        return 0
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    if (C.MODEL_ID, C.REVISION) != (EXPECTED_MODEL, EXPECTED_REVISION):
        raise SystemExit("loaded model is not the required pinned Qwen 1.5B revision")
    ids, byts, slice_meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    if slice_meta.get("ids_sha256") != EXPECTED_IDS_SHA or int(byts.sum()) != H2.EXPECT_SCORED_BYTES:
        raise SystemExit("runtime heldout slice differs from frozen H4/H2I instrument")
    prompt_ids = H2.frozen_prompts(tok)
    targets = dense_e6_targets()

    arm_a_started = time.time()
    if H4.build(model, args.h4, list(range(28)), "cpu") != 56:
        raise SystemExit("H4 terminal did not install exactly 56 q/o organs")
    arm_a = measure_arm(model, tok, ids, float(byts.sum()), prompt_ids, targets)
    arm_a["seconds"] = time.time() - arm_a_started
    assert_h4_reproduction(arm_a, h4_published)

    arm_b_started = time.time()
    if H2.build_ffn_int8(model, args.labels, args.stats, EXPECTED_LAYERS, 16, 256, "cpu") \
            != len(EXPECTED_LAYERS):
        raise SystemExit("H2I R8 FFN builder did not install all eight registered layers")
    mods = H1.ffn_mods(model, EXPECTED_LAYERS)
    if H2E.install_trained(mods, args.h2i_trained, args.labels) != len(EXPECTED_LAYERS):
        raise SystemExit("H2I trained bundle did not install all eight registered layers")
    for _, mod in mods:
        mod.k = 16
        mod.hard_gate = True
    arm_b = measure_arm(model, tok, ids, float(byts.sum()), prompt_ids, targets)
    arm_b["seconds"] = time.time() - arm_b_started

    result = {
        "diagnostic": "H5 bounded cross-composition; raw metrics only, no automatic gate adjudication",
        "model": C.MODEL_ID,
        "revision": C.REVISION,
        "provenance": provenance,
        "inputs": inputs,
        "published_references": {
            "h4_terminal_metrics": {key: h4_published[key] for key in
                                      ("bpb", "free", "teacher_forced", "mean_rank", "rank_le5")},
            "h2i_train_meta_sha256": inputs["h2i_trained_sidecar"]["sha256"],
        },
        "eval_slice": slice_meta,
        "e6": {"prompts": PROMPTS, "n_new": N_NEW, "targets": "dense Qwen 1.5B E6 reference"},
        "h4": {"rank": 48, "organs": list(H4.ORGANS), "layers": list(range(28))},
        "h2i": {"ffn_rule": "R8 per-output-row amax/127 RTN, codes [-127,127], STE",
                "layers": EXPECTED_LAYERS, "k": 16, "E": 256, "hard_gate": True},
        "arms": {"A_h4_terminal_qo": arm_a, "B_h4_plus_h2i_trained_ffn": arm_b},
        "seconds_total": time.time() - started,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
