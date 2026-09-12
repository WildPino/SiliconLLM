#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E38 -- is there ANY selector at all, or is 1.17% simply not enough capacity?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E38_IS_THERE_ANY_SELECTOR_AT_ALL.md
(pushed before this file existed).

WHY THIS EXISTS.  E37's control found that a router capturing 71.3% of the oracle's group mass
and one capturing 6.1% produce the SAME model.  That says the ridge family is not the lever.
It says nothing about whether ANY selector is -- both routers could be sitting far below a
ceiling a better one would reach.  The two readings send the next GPU-hour to opposite places,
so this measures the ceiling directly by handing the carve an ORACLE it could never have.

NOTHING HERE IS REIMPLEMENTED.  The slice is common.get_slice with h0_eval's pinned
("heldout", 24, 512, 1234) and its hash is asserted; the labels are e19_carve_rank.load_labels;
the fitted routers are the npz e37_fit_routers.py wrote; the random router is
carve_common.router_weights at E37's seed 26.  The BPB loop is h0_eval.py's, line for line.

fp32 and NOT ternary, deliberately: E37 measured selection and format together, and the engine
cannot express an oracle.  The cost -- registered in the brief s2, not discovered here -- is
that E38's absolute BPB is not comparable to E37's.  Only the gaps WITHIN E38 are.

  python e38_oracle_ceiling.py              # the registered sweep
  python e38_oracle_ceiling.py --smoke      # 2 sequences, 2 arms, for the plumbing
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ternary")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))

import common as C                          # noqa: E402
import e19_carve_rank as E19                # noqa: E402
import carve_common as CV                   # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e38_oracle_ceiling.json")

HF = "Qwen/Qwen2.5-1.5B"
REV = "8faed761d45a263340a0528343f099c05c9a4323"
E_GROUPS = 256
LN2 = 0.6931471805599453
CHANCE = 4.069819                           # E12's line, same slice
DENSE_ANCHOR, ANCHOR_TOL = 0.767595, 0.001  # E22 `base`, re-measured 2026-09-12 at 0.7675949641
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
NEV, SEQLEN, SEEDEV = 24, 512, 1234         # h0_eval.py's pinned eval slice
NCAL, SEEDCAL = 8, 424242                   # for `static` only; NOT the eval slice
PARITY_TOL = 1e-6                           # G-E38A: masking with every group kept is identity
RANDOM_SEED = 26                            # E37's synthetic router seed
PART_B_SEED = 38038                         # grouping B, a random equal partition

KS_A = [256, 64, 16, 3, 1]
SELECTORS_A = ["oracle", "fitted", "random", "static"]
KS_B = [64, 3]
SELECTORS_B = ["oracle", "random", "static"]
VERDICT_K, VERDICT_SEL = 3, "oracle"
BAR_LEVER, BAR_DEAD = 0.10, 0.50            # brief s6


def log(*a):
    print(*a, flush=True)


class Carve:
    """Masks the FFN intermediate to the selected groups, which is what ffn_carved computes.

    One pre-hook on gate_proj captures the FFN input x (the router reads the same s->xb the
    engine's router reads); one pre-hook on down_proj masks gate*up before the projection.
    """

    def __init__(self, model, labels, routers, rand_routers, static_sel):
        self.model, self.labels = model, labels
        self.routers, self.rand = routers, rand_routers
        self.static = static_sel
        self.mode, self.k = "off", 0
        self.mass_acc = None
        self.x = {}
        self.handles = []
        for li, lay in enumerate(model.model.layers):
            self.handles.append(lay.mlp.gate_proj.register_forward_pre_hook(self._mk_in(li)))
            self.handles.append(lay.mlp.down_proj.register_forward_pre_hook(self._mk_mask(li)))

    def _mk_in(self, li):
        def f(mod, args):
            self.x[li] = args[0].detach()
        return f

    def _mk_mask(self, li):
        def f(mod, args):
            h = args[0]
            if self.mode == "off":
                return None
            lab = self.labels[li]
            flat = h.reshape(-1, h.shape[-1])
            mass = torch.zeros(flat.shape[0], E_GROUPS, dtype=flat.dtype)
            mass.index_add_(1, lab, flat.float() ** 2)
            if self.mass_acc is not None:                       # calibration pass
                self.mass_acc[li] += mass.sum(0).double()
                return None
            if self.k >= E_GROUPS:
                sel = torch.arange(E_GROUPS).expand(flat.shape[0], E_GROUPS)
            elif self.mode == "oracle":
                sel = mass.topk(self.k, dim=1).indices
            elif self.mode == "static":
                sel = self.static[li][:self.k].expand(flat.shape[0], self.k)
            else:
                R = self.routers[li] if self.mode == "fitted" else self.rand[li]
                xf = self.x[li].reshape(-1, self.x[li].shape[-1]).float()
                sel = (xf @ R).topk(self.k, dim=1).indices
            keep = torch.zeros(flat.shape[0], E_GROUPS, dtype=torch.bool)
            keep.scatter_(1, sel, True)
            return (flat.mul(keep[:, lab].to(flat.dtype)).reshape(h.shape),) + tuple(args[1:])
        return f

    def close(self):
        for h in self.handles:
            h.remove()


def bpb_of(model, ids_ev, b_tot):
    """h0_eval.py's loop, unchanged."""
    nats = []
    with torch.no_grad():
        for i in range(ids_ev.shape[0]):
            ch = ids_ev[i:i + 1]
            lp = torch.nn.functional.log_softmax(model(ch).logits[:, :-1], dim=-1)
            nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double())
    return float(torch.stack(nats).sum() / (LN2 * b_tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()

    log("== E38: the ORACLE ceiling on the carve ==")
    log("  fp32, NOT ternary -- selection isolated from format (brief s2).  Absolute BPB is")
    log("  NOT comparable to E37's; only the gaps within E38 are.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(HF, revision=REV, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    D = model.config.hidden_size
    F = model.config.intermediate_size
    L = model.config.num_hidden_layers
    log("  donor %s  D=%d F=%d L=%d  E=%d (group = %d neurons)" % (HF, D, F, L, E_GROUPS,
                                                                   F // E_GROUPS))

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", NEV, SEQLEN, SEEDEV)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    if a.smoke:
        ids_ev, byts_ev = ids_ev[:2], byts_ev[:2]
    b_tot = float(byts_ev.sum())
    log("  eval slice %d x %d, %d scored bytes, ids_sha %s" % (ids_ev.shape[0], SEQLEN,
                                                               int(b_tot), EXPECT_IDS_SHA[:16]))

    # ---- groupings.  A is E37's label set; B is a random equal partition of the same F.
    labs_np, _ = E19.load_labels(E_GROUPS)
    gA = [torch.from_numpy(labs_np[li].astype(np.int64)) for li in range(L)]
    rng = np.random.default_rng(PART_B_SEED)
    gB = []
    for li in range(L):
        p = rng.permutation(F)
        lb = np.empty(F, dtype=np.int64)
        lb[p] = np.arange(F) // (F // E_GROUPS)
        gB.append(torch.from_numpy(lb))
    for li in (0, L - 1):
        assert torch.bincount(gA[li], minlength=E_GROUPS).min() > 0
        assert (torch.bincount(gB[li], minlength=E_GROUPS) == F // E_GROUPS).all()
    log("  grouping A = D0c labels (E37's); grouping B = random equal partition, seed %d"
        % PART_B_SEED)

    npz = np.load("D:/_ktmp/e37/e37_routers_E%d.npz" % E_GROUPS)
    routers = [torch.from_numpy(npz["r%d" % li]).float().T.contiguous() for li in range(L)]
    rand = [torch.from_numpy(CV.router_weights(D, E_GROUPS, RANDOM_SEED, li)).float().T
            .contiguous() for li in range(L)]
    log("  routers: fitted from e37_fit_routers.py, random = carve_common seed %d" % RANDOM_SEED)

    out = {"brief": "BRIEF_E38 (pre-registered)", "donor": HF, "revision": REV,
           "precision": "fp32", "E": E_GROUPS, "group_neurons": F // E_GROUPS,
           "eval_slice": {"part": "heldout", "n": int(ids_ev.shape[0]), "seqlen": SEQLEN,
                          "seed": SEEDEV, "scored_bytes": b_tot, "ids_sha256": EXPECT_IDS_SHA},
           "chance_bpb": CHANCE, "bands": {"lever": BAR_LEVER, "dead": BAR_DEAD}}

    # ---- G-E38B: the anchor, measured before anything is masked.
    t = time.time()
    dense = bpb_of(model, ids_ev, b_tot)
    gB2 = {"bpb": dense, "anchor": DENSE_ANCHOR, "diff": dense - DENSE_ANCHOR,
           "tol": ANCHOR_TOL, "fires": bool(abs(dense - DENSE_ANCHOR) < ANCHOR_TOL or a.smoke),
           "seconds": time.time() - t}
    out["G_E38B"] = gB2
    log("")
    log("  G-E38B  dense fp32 = %.9f  vs E22 base %.6f  diff %+.2e -> %s"
        % (dense, DENSE_ANCHOR, dense - DENSE_ANCHOR, "FIRES" if gB2["fires"] else "VOID"))
    if not gB2["fires"]:
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E38B VOID -- this session is not measuring E22's object. STOP.")

    carve = Carve(model, gA, routers, rand, None)

    # ---- `static`: top-k by calibration-averaged mass.  A separate slice, never the eval one.
    ids_cal, _, _ = C.get_slice(tok, "calib", 2 if a.smoke else NCAL, SEQLEN, SEEDCAL)
    carve.mass_acc = [torch.zeros(E_GROUPS, dtype=torch.float64) for _ in range(L)]
    carve.mode = "calib"
    for i in range(ids_cal.shape[0]):
        model(ids_cal[i:i + 1])
    static_sel = [m.argsort(descending=True).long() for m in carve.mass_acc]
    carve.static, carve.mass_acc = static_sel, None
    log("  static selection fitted on a SEPARATE calib slice (%d x %d, seed %d), never the eval one"
        % (ids_cal.shape[0], SEQLEN, SEEDCAL))

    def run(grouping, sels, ks, tag):
        carve.labels = gA if grouping == "A" else gB
        rows = {}
        for k in ks:
            for sel in sels:
                if a.smoke and (k, sel) not in ((256, "oracle"), (3, "oracle")):
                    continue
                carve.mode, carve.k = sel, k
                t1 = time.time()
                b = bpb_of(model, ids_ev, b_tot)
                carve.mode = "off"
                rows["%s_k%d" % (sel, k)] = {"selector": sel, "k": k, "bpb": b,
                                             "vs_dense": b - dense, "vs_chance": b - CHANCE,
                                             "activation_pct": 100.0 * k / E_GROUPS,
                                             "seconds": time.time() - t1}
                log("    %-8s k=%-4d  BPB %.6f   vs dense %+.4f   vs chance %+.4f  [%.0fs]"
                    % (sel, k, b, b - dense, b - CHANCE, time.time() - t1))
        out[tag] = rows
        return rows

    log("")
    log("  -- grouping A (D0c labels, E37's) --")
    A = run("A", SELECTORS_A, KS_A, "grouping_A")
    log("")
    log("  -- grouping B (random equal partition) --")
    B = run("B", SELECTORS_B, KS_B, "grouping_B")
    carve.close()

    # ---- G-E38A: at k=E the mask is the identity for every selector.
    per = {}
    for sel in SELECTORS_A:
        key = "%s_k%d" % (sel, E_GROUPS)
        if key in A:
            per[sel] = {"bpb": A[key]["bpb"], "diff": abs(A[key]["bpb"] - dense),
                        "agrees": bool(abs(A[key]["bpb"] - dense) < PARITY_TOL)}
    gA1 = {"per_selector": per, "tol": PARITY_TOL,
           "fires": bool(per and all(v["agrees"] for v in per.values()))}
    out["G_E38A"] = gA1
    log("")
    log("  G-E38A  mask inert at k=E: %s  (worst %.2e, tol %.0e)"
        % ("FIRES" if gA1["fires"] else "VOID",
           max([v["diff"] for v in per.values()] or [float("nan")]), PARITY_TOL))

    # ---- G-E38C: the oracle is an upper bound, by construction.
    viol = []
    for k in KS_A:
        o = A.get("oracle_k%d" % k)
        if not o:
            continue
        for sel in ("fitted", "random", "static"):
            r = A.get("%s_k%d" % (sel, k))
            if r and o["bpb"] > r["bpb"] + 1e-9:
                viol.append({"k": k, "vs": sel, "oracle": o["bpb"], "other": r["bpb"]})
    out["G_E38C"] = {"violations": viol, "fires": bool(not viol)}
    log("  G-E38C  oracle <= every attainable selector: %s%s"
        % ("FIRES" if not viol else "VOID", "" if not viol else "  %d violations" % len(viol)))

    # ---- the verdict.
    vk = A.get("%s_k%d" % (VERDICT_SEL, VERDICT_K))
    rk = A.get("random_k%d" % VERDICT_K)
    if vk and rk:
        gap = rk["bpb"] - vk["bpb"]
        if vk["vs_dense"] <= BAR_LEVER:
            name = "SELECTION-IS-THE-LEVER"
        elif gap > BAR_DEAD:
            name = "SELECTION-PARTIAL"
        else:
            name = "SELECTION-IS-DEAD"
        out["verdict"] = {"cell": "%s_k%d grouping A" % (VERDICT_SEL, VERDICT_K),
                          "bpb": vk["bpb"], "vs_dense": vk["vs_dense"],
                          "oracle_minus_random": -gap, "gap_over_random": gap, "name": name}
        log("")
        log("  VERDICT  oracle k=%d = %.6f  (dense %+.4f, beats random by %.4f) -> %s"
            % (VERDICT_K, vk["bpb"], vk["vs_dense"], gap, name))

    # ---- prediction 3: does `static` tie `random`?
    tie = {}
    for k in KS_A:
        s, r = A.get("static_k%d" % k), A.get("random_k%d" % k)
        if s and r:
            tie["k%d" % k] = {"static": s["bpb"], "random": r["bpb"], "d": s["bpb"] - r["bpb"]}
    out["static_vs_random"] = tie

    # ---- prediction 4: is the damage invariant to the partition?
    inv = {}
    for k in KS_B:
        for sel in SELECTORS_B:
            x, y = A.get("%s_k%d" % (sel, k)), B.get("%s_k%d" % (sel, k))
            if x and y:
                inv["%s_k%d" % (sel, k)] = {"A": x["bpb"], "B": y["bpb"], "d": y["bpb"] - x["bpb"]}
    out["partition_invariance"] = inv

    out["dense_bpb"] = dense
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
