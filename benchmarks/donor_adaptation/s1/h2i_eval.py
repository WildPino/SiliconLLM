#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2I Phase B CPU fp32 adjudicator: score, rank, and router decomposition.

The T4 trains; this file decides.  Gates and boundaries were fixed in addendum B of
BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md before this evaluator or a trained R8 bundle existed.
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
for _p in (DENSDIR, HERE):
    sys.path.insert(0, _p)

import common as C                                           # noqa: E402
import h1_qat as H1                                          # noqa: E402
import h2i_applied as H2                                     # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

OUTDIR = os.path.join(HERE, "results", "h2i")
PHASE_A = os.path.join(OUTDIR, "h2i_applied_8L.json")
PHASE_A_BYTES = 12239
PHASE_A_SHA256 = "c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc"
APPLIED = 1.0190764652622473
UNCARVED = 0.8100055950172889
NEAR_LO, NEAR_HI = UNCARVED - 0.01, UNCARVED + 0.01
RANK_FREE = 9
RANK_TF = 99
RANK_MEAN = 3.675


def log(message):
    print(message, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def committed_provenance():
    files = [os.path.abspath(__file__), os.path.abspath(C.__file__),
             os.path.abspath(H1.__file__), os.path.abspath(H2.__file__),
             os.path.abspath(H2.T2.__file__), os.path.join(H2.ENGDIR, "e6_generate.py"),
             H2.E6_ENGINE]
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        records = {}
        for path in files:
            rel = os.path.relpath(path, root).replace("\\", "/")
            wt = subprocess.check_output(
                ["git", "hash-object", path], cwd=root, text=True
            ).strip()
            blob = subprocess.check_output(
                ["git", "rev-parse", "HEAD:" + rel], cwd=root, text=True,
                stderr=subprocess.STDOUT
            ).strip()
            if wt != blob:
                raise SystemExit("uncommitted evaluator dependency %s: %s != %s -- STOP"
                                 % (rel, wt, blob))
            records[rel] = {"git_blob": blob, "sha256": sha256_file(path)}
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("cannot establish evaluator provenance: %s" % exc)
    return {"head_commit": head, "files": records}


def preflight(args):
    if os.path.exists(args.out):
        raise SystemExit("H2I evaluation output already exists; it is write-once: %s" % args.out)
    if not os.path.isfile(PHASE_A) or os.path.getsize(PHASE_A) != PHASE_A_BYTES \
            or sha256_file(PHASE_A) != PHASE_A_SHA256:
        raise SystemExit("H2I Phase A identity mismatch -- the frozen gate may not move")
    with open(PHASE_A, encoding="utf-8") as fh:
        phase_a = json.load(fh)
    if phase_a.get("bpb", {}).get("applied-int8-8L") != APPLIED \
            or phase_a.get("bpb", {}).get("int8-8L") != UNCARVED \
            or phase_a.get("rank_applied_int8_vs_h0", {}).get("free") != RANK_FREE \
            or phase_a.get("rank_applied_int8_vs_h0", {}).get("teacher_forced_top1") != RANK_TF \
            or phase_a.get("rank_applied_int8_vs_h0", {}).get("mean_target_rank") != RANK_MEAN:
        raise SystemExit("H2I Phase A content differs from addendum B -- STOP")

    frozen = {"factors": args.factors, "labels": args.labels, "stats": args.stats,
              "routers": args.routers, "h1_result": args.h1_result,
              "heldout": args.heldout}
    inputs, _ = H2.validate_inputs(frozen)
    for path in (args.trained, os.path.splitext(args.trained)[0] + ".json"):
        if not os.path.isfile(path):
            raise SystemExit("missing trained bundle or sidecar: %s" % path)
    trained_json = os.path.splitext(args.trained)[0] + ".json"
    with open(trained_json, encoding="utf-8") as fh:
        train_meta = json.load(fh)
    expected = {"layers": H2.LAYERS, "k": H2.K, "E": H2.E_GROUPS,
                "steps_requested": 4000, "bs": 2, "accum": 8,
                "lr": 2e-4, "router_lr": 3e-4, "aux": 0.01, "seed": 4242}
    wrong = [(key, train_meta.get(key), value) for key, value in expected.items()
             if train_meta.get(key) != value]
    if train_meta.get("resumed_ffn_from") not in (None, False):
        wrong.append(("resumed_ffn_from", train_meta.get("resumed_ffn_from"), None))
    if train_meta.get("complete") is not True:
        wrong.append(("complete", train_meta.get("complete"), True))
    if int(train_meta.get("steps_completed") or 0) < 250:
        wrong.append(("steps_completed", train_meta.get("steps_completed"), ">=250"))
    if train_meta.get("ffn_rule") != "R8 per-output-row amax/127 RTN STE":
        wrong.append(("ffn_rule", train_meta.get("ffn_rule"),
                      "R8 per-output-row amax/127 RTN STE"))
    if train_meta.get("router_init_sha256") != H2.PINNED["routers"][1]:
        wrong.append(("router_init_sha256", train_meta.get("router_init_sha256"),
                      H2.PINNED["routers"][1]))
    if wrong:
        raise SystemExit("trained sidecar differs from registered Phase B object: %r" % wrong)
    inputs["phase_a"] = {"path": os.path.abspath(PHASE_A), "bytes": PHASE_A_BYTES,
                         "sha256": PHASE_A_SHA256}
    inputs["trained"] = {"path": os.path.abspath(args.trained),
                         "bytes": os.path.getsize(args.trained),
                         "sha256": sha256_file(args.trained)}
    inputs["trained_json"] = {"path": os.path.abspath(trained_json),
                              "bytes": os.path.getsize(trained_json),
                              "sha256": sha256_file(trained_json)}
    return phase_a, train_meta, inputs


def install_trained(mods, bundle, labels_npz):
    with np.load(bundle) as trained, np.load(labels_npz) as labels:
        for li, mod in mods:
            prefix = "L%02d" % li
            names = ("gate", "up", "down", "router", "rms_in", "rms_h", "labels")
            missing = [prefix + "." + name for name in names
                       if prefix + "." + name not in trained.files]
            if missing:
                raise SystemExit("trained R8 bundle missing %s -- STOP" % missing)
            bundle_labels = torch.from_numpy(trained[prefix + ".labels"]).long()
            frozen_labels = torch.from_numpy(labels["c%d" % li]).long()
            if not torch.equal(bundle_labels, frozen_labels):
                raise SystemExit("trained layer %d labels differ from frozen D0c -- STOP" % li)
            for name in ("gate", "up", "down", "router", "rms_in", "rms_h"):
                src = torch.from_numpy(trained[prefix + "." + name]).float()
                dst = getattr(mod, name)
                if tuple(src.shape) != tuple(dst.shape):
                    raise SystemExit("trained layer %d %s shape %s != %s -- STOP"
                                     % (li, name, tuple(src.shape), tuple(dst.shape)))
                dst.data.copy_(src)
            mod.invalidate()
    return len(mods)


def band_of(value):
    if value >= APPLIED:
        return "TRAINING-DOES-NOT-HELP"
    if value > NEAR_HI:
        return "TRAINING-HELPS"
    if value >= NEAR_LO:
        return "CARVE-IS-NEAR-FREE"
    return "TRAINING-OVERSHOOTS-BASE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", required=True)
    parser.add_argument("--factors", default=H2.DEFAULT_FACTORS)
    parser.add_argument("--labels", default=H2.DEFAULT_LABELS)
    parser.add_argument("--stats", default=H2.DEFAULT_STATS)
    parser.add_argument("--routers", default=H2.DEFAULT_ROUTERS)
    parser.add_argument("--h1-result", default=H2.DEFAULT_H1_RESULT)
    parser.add_argument("--heldout", default=H2.DEFAULT_HELDOUT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    tag = os.path.splitext(os.path.basename(args.trained))[0]
    args.out = args.out or os.path.join(OUTDIR, "h2i_eval_%s.json" % tag)
    started = time.time()

    log("== H2I Phase B CPU fp32 eval: %s ==" % tag)
    provenance = committed_provenance()
    phase_a, train_meta, inputs = preflight(args)
    log("   committed evaluator, frozen Phase A and trained-bundle metadata asserted")

    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    ids, byts, slice_meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    if slice_meta.get("ids_sha256") != H2.EXPECT_IDS_SHA \
            or int(byts.sum()) != H2.EXPECT_SCORED_BYTES:
        raise SystemExit("frozen H2I instrument differs -- STOP")
    nbytes = float(byts.sum())

    rows = {"intact": H2.bpb(model, ids, nbytes)}
    intact_ok = abs(rows["intact"] - H2.ANCHOR_INTACT) <= 1e-5
    log("   intact %.9f: %s" % (rows["intact"], intact_ok))
    if not intact_ok:
        raise SystemExit("intact anchor fails -- STOP")
    n_qo, qo_layers = H1.build_qo(model, args.factors, "cpu")
    rows["h0-run3"] = H2.bpb(model, ids, nbytes)
    h0_ok = abs(rows["h0-run3"] - H2.ANCHOR_H0) <= 1e-5
    log("   h0-run3 %.9f (%d organs/%d layers): %s"
        % (rows["h0-run3"], n_qo, len(qo_layers), h0_ok))
    if not h0_ok:
        raise SystemExit("H0 anchor fails -- STOP")

    H2.build_ffn_int8(model, args.labels, args.stats, H2.LAYERS,
                      H2.K, H2.E_GROUPS, "cpu")
    mods = H1.ffn_mods(model, H2.LAYERS)
    install_trained(mods, args.trained, args.labels)

    probe = torch.randn(1, 6, model.config.hidden_size,
                        generator=torch.Generator().manual_seed(90210))
    first = mods[0][1]
    first.hard_gate = True
    same_h, diff_h = H1.g_h1a(first, probe)
    first.hard_gate = False
    saved_router = first.router.detach().clone()
    first.router.data.zero_()
    first.invalidate()
    same_s, diff_s = H1.g_h1a(first, probe)
    first.router.data.copy_(saved_router)
    first.invalidate()
    first.hard_gate = True
    mask = first.group_mask(probe)
    cardinality = bool(torch.all(mask.sum(dim=-1) == 560))
    live, live_diff = H1.carve_is_live(first, probe)
    wiring = bool(same_h and same_s and diff_h == 0 and diff_s == 0 and cardinality and live)
    log("   wiring hard-kE=%s soft-zero-kE=%s active560=%s live=%s"
        % (same_h, same_s, cardinality, live))
    if not wiring:
        raise SystemExit("trained real-shape wiring control fails -- STOP")

    for _, mod in mods:
        mod.k, mod.hard_gate = H2.K, False
    rows["trained-soft"] = H2.bpb(model, ids, nbytes)
    for _, mod in mods:
        mod.hard_gate = True
    rows["trained-hard"] = H2.bpb(model, ids, nbytes)
    saved_routers = {li: mod.router.detach().clone() for li, mod in mods}

    for _, mod in mods:
        mod.k = H2.E_GROUPS
    rows["trained-kE"] = H2.bpb(model, ids, nbytes)
    for _, mod in mods:
        mod.k = H2.K

    with np.load(args.routers) as routers:
        for li, mod in mods:
            source = torch.from_numpy(routers["r%d" % li]).float()
            if tuple(source.shape) != tuple(mod.router.shape):
                raise SystemExit("E37 router shape mismatch at layer %d -- STOP" % li)
            mod.router.data.copy_(source)
            mod.invalidate()
    rows["trained-experts-e37-router-hard"] = H2.bpb(model, ids, nbytes)
    for li, mod in mods:
        mod.router.data.copy_(saved_routers[li])
        mod.invalidate()
        mod.hard_gate = True

    score = rows["trained-hard"] < APPLIED
    score_gate = {"threshold_exclusive": APPLIED, "value": rows["trained-hard"],
                  "delta": rows["trained-hard"] - APPLIED, "fires": bool(score),
                  "band": band_of(rows["trained-hard"])}
    decomposition = {
        "experts_vs_applied": rows["trained-experts-e37-router-hard"] - APPLIED,
        "router_vs_e37": rows["trained-hard"] - rows["trained-experts-e37-router-hard"],
        "soft_vs_hard": rows["trained-soft"] - rows["trained-hard"],
    }
    log("   trained hard %.9f vs applied %.9f: score %s (%s)"
        % (rows["trained-hard"], APPLIED, score, score_gate["band"]))
    log("   components experts %+.9f router %+.9f soft-hard %+.9f"
        % (decomposition["experts_vs_applied"], decomposition["router_vs_e37"],
           decomposition["soft_vs_hard"]))

    ids_cal, _, calib_meta = C.get_slice(tok, "calib", 32, 512, 42424)
    router_nats, static_nats, router_fires, picks = H1.g_h1e(
        model, mods, ids_cal, ids, "cpu", False)
    router_gate = {"router_nats_per_token": router_nats,
                   "static_nats_per_token": static_nats,
                   "fires": bool(router_fires), "static_groups": picks,
                   "calib_slice": calib_meta, "gate": "hard on both arms"}
    log("   router %.6f vs STATIC %.6f nats/token: %s"
        % (router_nats, static_nats, router_fires))

    rank = None
    rank_gate = {"not_run": not score, "reason": None, "fires": False}
    if score:
        prompt_ids = phase_a["rank_reference"]["prompt_ids"]
        target_ids = phase_a["rank_reference"]["target_ids"]
        if H2.frozen_prompts(tok) != prompt_ids or len(target_ids) != 5 \
                or any(len(row) != 32 for row in target_ids):
            raise SystemExit("Phase A rank reference shape/tokenization differs -- STOP")
        log("-- score passed; running the preregistered rank gate")
        rank = H2.rank_against(model, prompt_ids, target_ids)
        clauses = {"free_gt_9": rank["free"] > RANK_FREE,
                   "teacher_forced_gt_99": rank["teacher_forced_top1"] > RANK_TF,
                   "mean_rank_lt_3.675": rank["mean_target_rank"] < RANK_MEAN}
        rank_gate = {"thresholds": {"free_min_exclusive": RANK_FREE,
                                     "teacher_forced_min_exclusive": RANK_TF,
                                     "mean_rank_max_exclusive": RANK_MEAN},
                     "values": {"free": rank["free"],
                                "teacher_forced": rank["teacher_forced_top1"],
                                "mean_rank": rank["mean_target_rank"]},
                     "clauses": clauses, "fires": all(clauses.values()),
                     "not_run": False, "reason": None}
    else:
        rank_gate["reason"] = "score gate failed; addendum B forbids rank as a rescue claim"

    combined = bool(score and rank_gate["fires"])
    result = {
        "brief": "briefs/BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md addendum B",
        "status": "PASS" if combined else ("SCORE-ONLY" if score else "FAIL-SCORE"),
        "model": C.MODEL_ID, "revision": C.REVISION,
        "provenance": provenance, "inputs": inputs,
        "train_meta": train_meta, "eval_slice": slice_meta,
        "layers": H2.LAYERS, "k": H2.K, "E": H2.E_GROUPS,
        "bpb": rows, "G_H2I_score": score_gate,
        "G_H2I_rank": rank_gate, "G_H2I": {"fires": combined},
        "rank_trained_vs_h0": rank,
        "router_vs_static": router_gate,
        "router_improves_over_e37": decomposition["router_vs_e37"] < 0,
        "decomposition": decomposition,
        "controls": {"intact": intact_ok, "h0": h0_ok,
                     "wiring": {"fires": wiring, "hard_kE_diff": diff_h,
                                "soft_zero_kE_diff": diff_s,
                                "hard_k16_active_560": cardinality,
                                "carve_live": live, "carve_live_diff": live_diff}},
        "seconds": time.time() - started,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    log("-- wrote %s [%.0fs], %s" % (args.out, result["seconds"], result["status"]))
    return 0 if combined else (3 if score else 2)


if __name__ == "__main__":
    sys.exit(main())
