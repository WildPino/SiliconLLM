#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2I Phase A -- matched one-byte FFN baseline before training the carve.

Pre-registration: docs/research/donor_adaptation/briefs/
BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md, commit 970951a.  This runner measures exactly three
new things on the H1 eight-layer shape: uncarved R8 BPB, hard-k16 R8 BPB, and the latter's rank
against H0 run 3's own trajectories.  Published H1 ternary rows are validated, then reused.

The process refuses before model loading unless this file and every imported experiment helper
are committed HEAD blobs and every frozen input has the registered size and sha256.
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
TERNDIR = os.path.abspath(os.path.join(HERE, "..", "ternary"))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
for _p in (DENSDIR, TERNDIR, ENGDIR):
    sys.path.insert(0, _p)

import common as C                                           # noqa: E402
from e6_generate import N_NEW, PROMPTS                       # noqa: E402
import h1_qat as H1                                          # noqa: E402
import t2_rules as T2                                        # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

OUTDIR = os.path.join(HERE, "results", "h2i")
DEFAULT_OUT = os.path.join(OUTDIR, "h2i_applied_8L.json")
DEFAULT_FACTORS = "D:/_ktmp/h0_run3_final/h0_trained3.npz"
DEFAULT_LABELS = os.path.join(HERE, "..", "density", "results", "d0c_labels",
                              "labels_E256.npz")
DEFAULT_STATS = os.path.join(HERE, "results", "h1", "h1_actstats.npz")
DEFAULT_ROUTERS = "D:/_ktmp/e37/e37_routers_E256.npz"
DEFAULT_H1_RESULT = os.path.join(HERE, "results", "h1", "h1_applied_8L.json")
DEFAULT_HELDOUT = os.path.join(HERE, "results", "h1", "h1_heldout.npz")
E6_ENGINE = os.path.join(ENGDIR, "results", "e6", "engine.json")

EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
EXPECT_SCORED_BYTES = 51870
EXPECT_MODEL = "Qwen/Qwen2.5-1.5B"
EXPECT_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
LAYERS = list(H1.H1_LAYERS)
E_GROUPS = 256
K = 16
LN2 = 0.6931471805599453
CHANCE = 4.069819
ANCHOR_INTACT = 0.7675949641196625
ANCHOR_H0 = 0.810022487699936
ANCHOR_TERNARY = 0.9478511380553962
ANCHOR_APPLIED = 1.0966361325809948

PINNED = {
    "factors": (352967686, "dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8"),
    "labels": (859537, "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c"),
    "stats": (1189630, "49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe"),
    "routers": (44046858, "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a"),
    "h1_result": (1605, "6ebc333f74f0f337567a99e527cdfa7750d092161cc5ca2d1836a510557f1b17"),
    "heldout": (98564, "110a90ab358efcacdc090fdf0fce62bb74d6d06df02d1083cf4cf7e752529d35"),
}


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
    """Assert the complete imported apparatus, not only this entry point, equals HEAD."""
    dependencies = [os.path.abspath(__file__), os.path.abspath(C.__file__),
                    os.path.abspath(H1.__file__), os.path.abspath(T2.__file__),
                    os.path.join(ENGDIR, "e6_generate.py"), E6_ENGINE]
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        rows = {}
        for path in dependencies:
            rel = os.path.relpath(path, root).replace("\\", "/")
            worktree_blob = subprocess.check_output(
                ["git", "hash-object", path], cwd=root, text=True
            ).strip()
            head_blob = subprocess.check_output(
                ["git", "rev-parse", "HEAD:" + rel], cwd=root, text=True,
                stderr=subprocess.STDOUT
            ).strip()
            if worktree_blob != head_blob:
                raise SystemExit("APPARATUS IS NOT THE COMMITTED HEAD BLOB: %s worktree %s, "
                                 "HEAD %s -- STOP" % (rel, worktree_blob, head_blob))
            rows[rel] = {"git_blob": worktree_blob, "sha256": sha256_file(path)}
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("cannot establish committed apparatus provenance: %s" % exc)
    return {"head_commit": head, "files": rows}


def validate_inputs(paths):
    actual = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise SystemExit("missing pinned H2I input %s: %s" % (name, path))
        size = os.path.getsize(path)
        digest = sha256_file(path)
        want_size, want_digest = PINNED[name]
        if size != want_size or digest != want_digest:
            raise SystemExit("PINNED INPUT MISMATCH %s: bytes %d sha256 %s; expected %d %s"
                             % (name, size, digest, want_size, want_digest))
        actual[name] = {"path": os.path.abspath(path), "bytes": size, "sha256": digest}

    with open(paths["h1_result"], encoding="utf-8") as fh:
        old = json.load(fh)
    expected = {
        "model": EXPECT_MODEL, "revision": EXPECT_REVISION, "layers": LAYERS,
        "k": K, "E": E_GROUPS, "gate_mode": "hard",
    }
    wrong = [(key, old.get(key), value) for key, value in expected.items()
             if old.get(key) != value]
    expected_bpb = {"intact": ANCHOR_INTACT, "h0-run3": ANCHOR_H0,
                    "ternary-8L": ANCHOR_TERNARY, "applied-8L": ANCHOR_APPLIED}
    wrong.extend(("bpb." + key, old.get("bpb", {}).get(key), value)
                 for key, value in expected_bpb.items()
                 if old.get("bpb", {}).get(key) != value)
    if old.get("eval_slice", {}).get("ids_sha256") != EXPECT_IDS_SHA:
        wrong.append(("eval_slice.ids_sha256",
                      old.get("eval_slice", {}).get("ids_sha256"), EXPECT_IDS_SHA))
    if old.get("eval_slice", {}).get("total_scored_bytes") != EXPECT_SCORED_BYTES:
        wrong.append(("eval_slice.total_scored_bytes",
                      old.get("eval_slice", {}).get("total_scored_bytes"),
                      EXPECT_SCORED_BYTES))
    if wrong:
        raise SystemExit("published H1 control does not match the preregistered object: %r"
                         % wrong)
    return actual, old


def r8_codes(w):
    """The shipped R8 rule: one amax/127 scale per output row."""
    a = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = (w / a).round().clamp(-127, 127)
    return q, a


def r8_ste(w):
    """R8 in the forward, fp32 master and straight-through gradient in the backward."""
    with torch.no_grad():
        q, a = r8_codes(w.detach().float())
        wq = q * a
    return w + (wq - w).detach()


class Int8CarvedFFN(H1.TernaryCarvedFFN):
    """H1's selection apparatus with only the three FFN quantizers changed R3 -> R8."""

    def _quant(self):
        key = (self.gate._version, self.up._version, self.down._version)
        if self._ck == key and not torch.is_grad_enabled():
            return self._cv
        gq, uq, dq = r8_ste(self.gate), r8_ste(self.up), r8_ste(self.down)
        if not torch.is_grad_enabled():
            self._ck, self._cv = key, (gq, uq, dq)
        return gq, uq, dq


def build_ffn_int8(model, labels_npz, stats_npz, layers, k, groups, device):
    """Install the matched R8 FFNs while asserting the inherited labels/statistics."""
    with np.load(labels_npz) as labels, np.load(stats_npz) as stats:
        for li in layers:
            mlp = model.model.layers[li].mlp
            label_key = "c%d" % li
            rin_key, rh_key = "L%02d.rms_in" % li, "L%02d.rms_h" % li
            for key in (label_key, rin_key, rh_key):
                owner = labels if key == label_key else stats
                if key not in owner.files:
                    raise SystemExit("layer %d missing frozen input key %s -- STOP" % (li, key))
            lab = torch.from_numpy(labels[label_key]).long()
            rin = torch.from_numpy(stats[rin_key]).float()
            rh = torch.from_numpy(stats[rh_key]).float()
            ffn = int(mlp.gate_proj.weight.shape[0])
            hidden = int(mlp.gate_proj.weight.shape[1])
            counts = torch.bincount(lab, minlength=groups)
            if tuple(lab.shape) != (ffn,) or int(lab.min()) != 0 \
                    or int(lab.max()) != groups - 1 or not torch.all(counts == ffn // groups):
                raise SystemExit("layer %d labels are not E=%d equal groups over F=%d -- STOP"
                                 % (li, groups, ffn))
            if tuple(rin.shape) != (hidden,) or tuple(rh.shape) != (ffn,):
                raise SystemExit("layer %d activation-stat shapes differ: rin %s rh %s"
                                 % (li, tuple(rin.shape), tuple(rh.shape)))
            mod = Int8CarvedFFN(
                mlp.gate_proj.weight.detach().clone(),
                mlp.up_proj.weight.detach().clone(),
                mlp.down_proj.weight.detach().clone(), lab, rin, rh, k, groups)
            model.model.layers[li].mlp = mod.to(device)
    return len(layers)


def phase_a_decision(rows):
    one_byte_delta = abs(rows["int8-8L"] - rows["h0-run3"])
    carve_delta = rows["applied-int8-8L"] - rows["int8-8L"]
    return {
        "G_H2Ie": {"metric": "abs(int8-8L - h0-run3)", "threshold_max": 0.01,
                    "value": one_byte_delta, "fires": one_byte_delta <= 0.01},
        "G_H2If": {"ordinal": rows["applied-int8-8L"] > rows["int8-8L"],
                    "threshold_min_exclusive": 0.05, "value": carve_delta,
                    "fires": rows["applied-int8-8L"] > rows["int8-8L"]
                             and carve_delta > 0.05},
    }


def selftest():
    """G-H2Ib/c plus boundary tests, all before the donor is loaded."""
    torch.manual_seed(271828)
    planted = torch.tensor([
        [-8.0, -3.25, -1.0, -0.125, 0.0, 0.25, 2.75, 7.0],
        [-0.031, -0.017, -0.004, 0.0, 0.003, 0.011, 0.019, 0.029],
    ])
    q, a = r8_codes(planted)
    tq, ta = T2.r8_int8_rtn(planted)
    exact = torch.equal(q, tq) and torch.equal(a, ta) \
        and torch.equal(q * a, tq * ta) and torch.equal(r8_ste(planted), tq * ta)
    distinct = int(torch.unique(q).numel())
    if not exact or distinct < 5:
        raise SystemExit("G-H2Ib FAILS: R8 parity %s, distinct codes %d" % (exact, distinct))

    d, f, groups, k = 6, 8, 4, 2
    labels = torch.arange(f) % groups
    with torch.enable_grad():
        mod = Int8CarvedFFN(torch.randn(f, d) / 5, torch.randn(f, d) / 5,
                            torch.randn(d, f) / 5, labels, torch.ones(d), torch.ones(f),
                            k, groups)
        mod.router.data.normal_(0, 0.1)
        x = torch.randn(2, 3, d)
        y = mod(x)
        y.square().mean().backward()
    gradients = {}
    for name in ("gate", "up", "down"):
        grad = getattr(mod, name).grad
        gradients[name] = bool(grad is not None and torch.isfinite(grad).all()
                               and torch.count_nonzero(grad) > 0)
    if not all(gradients.values()):
        raise SystemExit("G-H2Ib FAILS: missing/nonfinite/zero STE gradient %r" % gradients)

    mod.hard_gate = True
    same, dmax = H1.g_h1a(mod, x)
    mask = mod.group_mask(x)
    active = mask.sum(dim=-1)
    expected_active = k * (f // groups)
    cardinality = bool(torch.all(active == expected_active))
    live, live_dmax = H1.carve_is_live(mod, x)
    if not same or dmax != 0.0 or not cardinality or not live:
        raise SystemExit("G-H2Ic FAILS: k=E same %s dmax %.9g, active %s want %d, live %s"
                         % (same, dmax, active.tolist(), expected_active, live))

    boundary_rows = {"h0-run3": 0.0, "int8-8L": 0.01,
                     "applied-int8-8L": 0.059}
    at_boundary = phase_a_decision(boundary_rows)
    boundary_ok = at_boundary["G_H2Ie"]["fires"] \
        and not at_boundary["G_H2If"]["fires"]
    if not boundary_ok:
        raise SystemExit("H2I gate boundary selftest FAILS: %r" % at_boundary)
    return {"G_H2Ib": {"r8_exact_t2_rules": exact, "distinct_codes": distinct,
                         "ste_gradients_finite_nonzero": gradients, "fires": True},
            "G_H2Ic_toy": {"k_equals_E_bit_identical": same, "max_abs_diff": dmax,
                             "hard_active_neurons_each": active.tolist(),
                             "expected_each": expected_active, "carve_live": live,
                             "carve_max_abs_diff": live_dmax, "fires": True},
            "gate_boundary_test": {"at_one_byte_0.01_fires": True,
                                    "below_carve_0.05_fires": False,
                                    "fires": boundary_ok}}


@torch.inference_mode()
def bpb(model, ids, nbytes):
    total = torch.zeros((), dtype=torch.float64)
    for i in range(ids.shape[0]):
        ch = ids[i:i + 1]
        logits = model(ch).logits
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        total += -lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1).double().sum()
    return float(total / (LN2 * nbytes))


@torch.inference_mode()
def greedy(model, prompt_ids, n_new):
    ids = list(prompt_ids)
    for _ in range(n_new):
        logits = model(torch.tensor([ids])).logits[0, -1]
        ids.append(int(torch.argmax(logits)))
    return ids[-n_new:]


def frozen_prompts(tok):
    with open(E6_ENGINE, encoding="utf-8") as fh:
        e6 = json.load(fh)
    if e6.get("n_new") != N_NEW or e6.get("prompts") != PROMPTS:
        raise SystemExit("E6 prompt record differs from imported frozen definition -- STOP")
    pids = [tok(prompt)["input_ids"] for prompt in PROMPTS]
    if pids != e6.get("prompt_ids"):
        raise SystemExit("tokenizer prompt IDs differ from E6's stored IDs -- STOP")
    return pids


@torch.inference_mode()
def rank_against(model, prompt_ids, targets):
    generated, per_prompt, all_ranks = [], [], []
    free_total = tf_total = 0
    for i, (prompt, target) in enumerate(zip(prompt_ids, targets)):
        new = greedy(model, prompt, N_NEW)
        generated.append(new)
        matches = [int(x == y) for x, y in zip(new, target)]
        free = sum(matches)
        first = next((j for j, value in enumerate(matches) if not value), None)

        full = torch.tensor([prompt + target])
        logits = model(full).logits[0].float()
        ranks, top1 = [], 0
        for pos, token in enumerate(target):
            row = logits[len(prompt) - 1 + pos]
            top1 += int(int(torch.argmax(row)) == token)
            ranks.append(int((row > row[token]).sum()) + 1)
        per_prompt.append({"prompt": i, "free": free,
                           "first_divergence_zero_based": first,
                           "teacher_forced_top1": top1,
                           "mean_target_rank": sum(ranks) / float(len(ranks)),
                           "target_ranks": ranks})
        free_total += free
        tf_total += top1
        all_ranks.extend(ranks)
        log("     prompt %d: free %d/%d first-div %s tf %d/%d mean-rank %.2f"
            % (i, free, N_NEW, "none" if first is None else first, top1, N_NEW,
               per_prompt[-1]["mean_target_rank"]))
    return {"free": free_total, "counted": len(targets) * N_NEW,
            "teacher_forced_top1": tf_total,
            "mean_target_rank": sum(all_ranks) / float(len(all_ranks)),
            "rank_le5": sum(1 for value in all_ranks if value <= 5),
            "per_prompt": per_prompt, "generated_ids": generated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", default=DEFAULT_FACTORS)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--routers", default=DEFAULT_ROUTERS)
    parser.add_argument("--h1-result", default=DEFAULT_H1_RESULT)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("H2I Phase A output already exists; it is write-once: %s" % args.out)
    started = time.time()
    paths = {"factors": args.factors, "labels": args.labels, "stats": args.stats,
             "routers": args.routers, "h1_result": args.h1_result,
             "heldout": args.heldout}

    log("== H2I Phase A: R8 applied carve on the matched H1 eight-layer shape ==")
    provenance = committed_provenance()
    log("   committed apparatus asserted at HEAD %s" % provenance["head_commit"])
    inputs, published_h1 = validate_inputs(paths)
    log("   six frozen input identities and published H1 controls asserted")
    controls = selftest()
    log("   G-H2Ib/c toy controls FIRE (R8 parity, STE gradients, carve wiring)")

    log("-- loading %s CPU fp32" % EXPECT_MODEL)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    if C.MODEL_ID != EXPECT_MODEL or C.REVISION != EXPECT_REVISION:
        raise SystemExit("model constants drifted: %s @ %s" % (C.MODEL_ID, C.REVISION))
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    if meta.get("ids_sha256") != EXPECT_IDS_SHA \
            or int(meta.get("total_scored_bytes", -1)) != EXPECT_SCORED_BYTES:
        raise SystemExit("runtime frozen slice identity differs -- STOP: %r" % meta)
    with np.load(args.heldout) as held:
        if "ids" not in held.files or not np.array_equal(held["ids"], ids.numpy()):
            raise SystemExit("pinned H1 heldout ids differ from runtime slice -- STOP")
    nbytes = float(byts.sum())
    if int(nbytes) != EXPECT_SCORED_BYTES:
        raise SystemExit("runtime scored-byte total differs: %s" % nbytes)

    rows = {"intact": bpb(model, ids, nbytes)}
    intact_ok = abs(rows["intact"] - ANCHOR_INTACT) <= 1e-5
    log("   intact       %.9f  G-H2Id %s" %
        (rows["intact"], "FIRES" if intact_ok else "*** FAILS ***"))
    if not intact_ok:
        raise SystemExit("G-H2Id intact anchor fails; no H2I cell may be read")

    n_qo, qo_layers = H1.build_qo(model, args.factors, "cpu")
    rows["h0-run3"] = bpb(model, ids, nbytes)
    h0_ok = abs(rows["h0-run3"] - ANCHOR_H0) <= 1e-5
    log("   h0-run3      %.9f  (%d q/o organs, %d layers) G-H2Id %s"
        % (rows["h0-run3"], n_qo, len(qo_layers),
           "FIRES" if h0_ok else "*** FAILS ***"))
    if not h0_ok:
        raise SystemExit("G-H2Id H0 anchor fails; no H2I cell may be read")

    prompt_ids = frozen_prompts(tok)
    log("-- recording H0 run-3's five own 32-token trajectories")
    h0_targets = [greedy(model, prompt, N_NEW) for prompt in prompt_ids]
    log("   H0 trajectories frozen in memory: %d positions" % (len(h0_targets) * N_NEW))

    installed = build_ffn_int8(model, args.labels, args.stats, LAYERS, K, E_GROUPS, "cpu")
    mods = H1.ffn_mods(model, LAYERS)
    with np.load(args.routers) as routers:
        for li, mod in mods:
            key = "r%d" % li
            if key not in routers.files:
                raise SystemExit("E37 router bundle missing %s -- STOP" % key)
            weight = torch.from_numpy(routers[key]).float()
            if tuple(weight.shape) != tuple(mod.router.shape):
                raise SystemExit("router %s shape %s != %s -- STOP"
                                 % (key, tuple(weight.shape), tuple(mod.router.shape)))
            mod.router.data.copy_(weight)
            mod.hard_gate = True

    x = torch.randn(1, 6, model.config.hidden_size)
    same, real_dmax = H1.g_h1a(mods[0][1], x)
    real_mod = mods[0][1]
    mask = real_mod.group_mask(x)
    active = mask.sum(dim=-1)
    active_ok = bool(torch.all(active == 560))
    live, live_dmax = H1.carve_is_live(real_mod, x)
    controls["G_H2Ic_real"] = {
        "layer": mods[0][0], "k_equals_E_bit_identical": same,
        "max_abs_diff": real_dmax, "hard_active_neurons_each": active.tolist(),
        "expected_each": 560, "cardinality_fires": active_ok,
        "carve_live": live, "carve_max_abs_diff": live_dmax,
        "labels_per_group": 35, "installed_layers": installed,
        "fires": bool(same and real_dmax == 0.0 and active_ok and live),
    }
    log("   G-H2Ic real layer %d: k=E exact %s; k16 active=560 %s; live %s"
        % (mods[0][0], same, active_ok, live))
    if not controls["G_H2Ic_real"]["fires"]:
        raise SystemExit("G-H2Ic real-shape wiring fails -- STOP")

    for _, mod in mods:
        mod.k = E_GROUPS
    rows["int8-8L"] = bpb(model, ids, nbytes)
    log("   int8-8L      %.9f  (k=E, no carve)" % rows["int8-8L"])

    for _, mod in mods:
        mod.k = K
    rows["applied-int8-8L"] = bpb(model, ids, nbytes)
    log("   applied-R8   %.9f  (hard k=%d, E37 router)" %
        (rows["applied-int8-8L"], K))

    decisions = phase_a_decision(rows)
    decisions["G_H2Id"] = {"intact_expected": ANCHOR_INTACT,
                             "intact_abs_error": abs(rows["intact"] - ANCHOR_INTACT),
                             "h0_expected": ANCHOR_H0,
                             "h0_abs_error": abs(rows["h0-run3"] - ANCHOR_H0),
                             "tolerance": 1e-5, "fires": bool(intact_ok and h0_ok)}
    log("   G-H2Ie one-byte delta %.9f <= 0.01: %s"
        % (decisions["G_H2Ie"]["value"], decisions["G_H2Ie"]["fires"]))
    log("   G-H2If carve delta %+.9f > 0.05 and ordinal: %s"
        % (decisions["G_H2If"]["value"], decisions["G_H2If"]["fires"]))

    log("-- rank partner: applied R8 against the just-recorded H0 trajectories")
    rank = rank_against(model, prompt_ids, h0_targets)
    log("   rank total: free %d/%d; teacher-forced %d/%d; mean target rank %.2f"
        % (rank["free"], rank["counted"], rank["teacher_forced_top1"],
           rank["counted"], rank["mean_target_rank"]))

    launch = bool(decisions["G_H2Ie"]["fires"] and decisions["G_H2If"]["fires"])
    result = {
        "brief": "briefs/BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md Phase A",
        "status": "PHASE_B_ELIGIBLE" if launch else "PHASE_B_NOT_ELIGIBLE",
        "model": C.MODEL_ID, "revision": C.REVISION,
        "provenance": provenance, "inputs": inputs, "eval_slice": meta,
        "scored_bytes": nbytes, "layers": LAYERS, "k": K, "E": E_GROUPS,
        "ffn_rule": "R8 per-output-row amax/127 RTN, codes [-127,127], STE",
        "gate_mode": "hard", "bpb": rows,
        "vs_chance": {key: value - CHANCE for key, value in rows.items()},
        "published_h1_reused_not_remeasured": published_h1["bpb"],
        "controls": controls, "decisions": decisions,
        "rank_reference": {"meaning": "H0 run-3 own greedy trajectories captured before R8",
                           "prompt_ids": prompt_ids, "target_ids": h0_targets,
                           "n_new": N_NEW},
        "rank_applied_int8_vs_h0": rank,
        "phase_b_eligible": launch, "seconds": time.time() - started,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    log("-- wrote %s [%.0fs], %s" % (args.out, result["seconds"], result["status"]))
    return 0 if launch else 2


if __name__ == "__main__":
    sys.exit(main())
