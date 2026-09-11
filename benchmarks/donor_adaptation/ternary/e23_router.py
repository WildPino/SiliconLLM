#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E23 -- a real router, because every carve number here is a ceiling.

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E23_A_REAL_ROUTER.md (25bde22),
pushed before this file existed.  Nothing is exported and NO TIMING IS TAKEN.

D0, D0c, E19 and E22 all install the same ORACLE hook: read the true squared activation mass per
expert group, keep the top k.  D0c says so in its own text -- "a floor no real router can reach".
So V52's +0.141846 and QO512+V52's 126/160 teacher-forced are CEILINGS, and the T4 healing
proposal targets the second of those.  This runner replaces the oracle with a router that only
sees what a router can see.

The router (brief s2, fixed before the run): per layer, ridge regression from the block input x
to sqrt of per-group mass, lambda = 0.01 * mean(diag(X'X)) -- E20's damping.  Closed form, no
gradients, no tuning.  Charged to the budget in the brief at 11.0 M, moving the configuration
from 0.9441 G to 0.9551 G, still inside E18's 0.982-1.060 G.

Env: D_THREADS (6), E23_ONLY (comma list), E23_SMOKE (1)
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
sys.path.insert(0, DENSDIR)
sys.path.insert(0, HERE)
sys.path.insert(0, ENGDIR)

import common as C                                          # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402
import e21_rank as E21                                      # noqa: E402
import e19_carve_rank as E19                                # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E23_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E23_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e23_router%s.json" % ("_smoke" if SMOKE else ""))
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453

CHANCE = 4.069819
FLOOR, MARGIN = 12, 2
AT_FLOOR_MAX, RANKS_MIN = FLOOR + MARGIN, 80
TF_LO, TF_HI = 107, 119
NCAL, SEQCAL, SEEDCAL = 32, 512, 42424
RIDGE_FRAC = 0.01                           # E20's damping, unchanged
RANK = 512                                  # E21's QO-ACT-512
CARVE_E, CARVE_K = 256, 133                 # E19's V52
CARVE_ACT = 0.51953125
ACT_TOL = 0.002
RNG_SEED = 20260911                         # pinned, brief s4

# ---- anchors, quoted from the published runs and never re-derived here
E22_V52 = {"bpb": 0.9094409944151614, "free": 12, "tf": 117}
E22_QO_V52 = {"bpb": 1.0050393404796996, "free": 15, "tf": 126}
BASE_BPB = 0.7675949641196624
REPL_TOL = 1e-9

# retention bands, brief s6
RET_HOLDS, RET_COSTS = 0.80, 0.40

#        tag                   router     lowrank
ARMS = [("base",               None,      False),
        ("V52-ORACLE",         "oracle",  False),
        ("V52-STATIC",         "static",  False),
        ("V52-RANDOM",         "random",  False),
        ("V52-LINEAR",         "linear",  False),
        ("QO512+V52-ORACLE",   "oracle",  True),
        ("QO512+V52-LINEAR",   "linear",  True)]


def log(*a):
    print(*a, flush=True)


def band_free(m):
    if m <= AT_FLOOR_MAX:
        return "AT-FLOOR"
    if m >= RANKS_MIN:
        return "RANKS"
    return "PARTIAL"


def band_tf(m):
    if m > TF_HI:
        return "CHEAPER"
    if m >= TF_LO:
        return "COMPARABLE"
    return "WORSE"


# ====================================================== the routed carve
HOOKS = []
STATS = {"kept": 0.0, "ntok": 0}


def install(model, layer_ids, lab_map, k, kind, routers, static_sets, rnd_sets):
    """Same hook shape as e19_carve_rank.install -- only the SCORE changes.

    oracle: the true per-group squared mass, which is what D0/D0c/E19/E22 all used.
    linear: x @ R_L, the ridge router, which sees only the block input.
    static: a fixed per-layer set (largest mean mass over the calibration slice).
    random: a fixed per-layer random set, seed pinned.

    In every case exactly k of E groups survive, so achieved activation is k/E by construction
    and G-T4 compares all arms at the same cost.
    """
    for L in layer_ids:
        lab = torch.from_numpy(lab_map[L].astype(np.int64))
        E = int(lab.max()) + 1
        oh = torch.zeros(len(lab), E)
        oh[torch.arange(len(lab)), lab] = 1.0
        sizes = oh.sum(0)
        R = routers.get(L) if routers else None
        fixed = None
        if kind == "static":
            fixed = static_sets[L]
        elif kind == "random":
            fixed = rnd_sets[L]

        def mk(lab=lab, oh=oh, sizes=sizes, E=E, k=k, R=R, fixed=fixed, kind=kind, L=L):
            def pre(mod, args):
                h = args[0]
                flat = h.reshape(-1, h.shape[-1])
                n = flat.shape[0]
                if fixed is not None:
                    keep = torch.zeros(n, E, dtype=torch.bool)
                    keep[:, fixed] = True
                else:
                    if kind == "oracle":
                        sc = (flat ** 2) @ oh
                    else:
                        sc = ROUTER_IN[L][:n] @ R           # [n, E], from the block input
                    sel = sc.topk(k, dim=1).indices
                    keep = torch.zeros(n, E, dtype=torch.bool)
                    keep.scatter_(1, sel, True)
                STATS["kept"] += float((keep.float() @ sizes).sum())
                STATS["ntok"] += int(n)
                return ((flat * keep[:, lab]).reshape(h.shape),) + args[1:]
            return pre
        HOOKS.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk()))


# the block input each router reads, captured one layer-step ahead of its down_proj hook
ROUTER_IN = {}
IN_HOOKS = []


def install_input_taps(model, layer_ids):
    def mk(L):
        def f(mod, inp, out):
            ROUTER_IN[L] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        return f
    for L in layer_ids:
        IN_HOOKS.append(model.model.layers[L].mlp.gate_proj.register_forward_hook(mk(L)))


def clear():
    while HOOKS:
        HOOKS.pop().remove()


def clear_taps():
    while IN_HOOKS:
        IN_HOOKS.pop().remove()
    ROUTER_IN.clear()


# ====================================================== fitting the routers
# E24 adds a second TARGET for the same fit.  The accumulation, the damping and the solve are
# untouched; only which function of the per-group mass the ridge regresses onto changes.  Kept
# here rather than copied into e24_depth.py so there is ONE definition of the router fit, and
# E24's G-U1 (reproduce E23's two anchors to 1e-9) is what proves this refactor is inert.
TARGETS = {"sqrt": lambda m: m.sqrt(),
           "log1p": lambda m: torch.log1p(m)}


def fit_routers(model, ids_cal, layer_ids, lab_map):
    """E23's signature, unchanged: the sqrt-target ridge routers."""
    R, diag, acc = fit_routers_multi(model, ids_cal, layer_ids, lab_map, ("sqrt",))
    return R["sqrt"], diag, acc


def fit_routers_multi(model, ids_cal, layer_ids, lab_map, targets=("sqrt",)):
    """Ridge from the block input x to f(per-group mass), accumulated in normal-equation form.

    Accumulating X'X [D,D] and X'Y [D,E] instead of storing X keeps this O(D^2 + D E) per layer
    regardless of the token count -- the scale law applies to the REMEDY too, so nothing here is
    O(tokens).  Several targets share ONE calibration pass: only X'Y is per-target, and the
    solve reuses the same factorable X'X.

    Returns ({target: {layer: R}}, diag, acc).  `diag` is reported off targets[0], which keeps
    its shape exactly what E23 wrote.
    """
    for t in targets:
        assert t in TARGETS, "unknown router target %r" % (t,)
    D = model.config.hidden_size
    XtX, XtY, N, acc = {}, {t: {} for t in targets}, {}, {}
    ohs = {}
    for L in layer_ids:
        lab = torch.from_numpy(lab_map[L].astype(np.int64))
        E = int(lab.max()) + 1
        oh = torch.zeros(len(lab), E)
        oh[torch.arange(len(lab)), lab] = 1.0
        ohs[L] = oh
        XtX[L] = torch.zeros(D, D, dtype=torch.float64)
        for t in targets:
            XtY[t][L] = torch.zeros(D, E, dtype=torch.float64)
        acc[L] = torch.zeros(E, dtype=torch.float64)
        N[L] = 0

    hooks = []
    pend = {}

    def mk_in(L):
        def f(mod, inp, out):
            pend[L] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        return f

    def mk_out(L):
        def f(mod, inp, out):
            h = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()   # post-SwiGLU
            m = ((h ** 2) @ ohs[L]).clamp_min(0)                        # [n, E]
            acc[L] += m.sum(0).double()   # STATIC's target, same pass
            x = pend[L]
            XtX[L] += (x.T @ x).double()
            for t in targets:
                XtY[t][L] += (x.T @ TARGETS[t](m)).double()
            N[L] += x.shape[0]
        return f

    for L in layer_ids:
        lay = model.model.layers[L]
        hooks.append(lay.mlp.gate_proj.register_forward_hook(mk_in(L)))
        hooks.append(lay.mlp.down_proj.register_forward_hook(mk_out(L)))
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            model(ids_cal[i:i + 1])
    for h in hooks:
        h.remove()

    routers = {t: {} for t in targets}
    diag = {}
    for L in layer_ids:
        A = XtX[L]
        lam = RIDGE_FRAC * float(torch.diagonal(A).mean())
        Ainv = A + torch.eye(A.shape[0], dtype=torch.float64) * lam
        for t in targets:
            R = torch.linalg.solve(Ainv, XtY[t][L])
            routers[t][L] = R.float()
        R0 = routers[targets[0]][L]
        # in-sample scale, reported not gated: how much of the mass a linear map can even see
        ss_tot = float((XtY[targets[0]][L] ** 2).sum())
        diag[L] = {"ridge_lambda": lam, "n_tokens": N[L],
                   "router_fro": float(torch.linalg.norm(R0)),
                   "xty_fro": ss_tot ** 0.5}
        if len(targets) > 1:
            diag[L]["router_fro_by_target"] = {t: float(torch.linalg.norm(routers[t][L]))
                                               for t in targets}
    return routers, diag, acc


def fixed_sets(acc, layer_ids, k):
    """STATIC: the k groups with the largest MEAN mass over the calibration slice, per layer.
    RANDOM: a fixed random k-of-E, seed pinned.  Both are token-independent by construction.
    `acc` comes from fit_routers' pass -- the same tokens, one traversal."""
    g = torch.Generator().manual_seed(RNG_SEED)
    static, rnd = {}, {}
    for L in layer_ids:
        static[L] = acc[L].topk(k).indices
        rnd[L] = torch.randperm(acc[L].shape[0], generator=g)[:k]
    return static, rnd


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS
    n_layers = 4 if SMOKE else None

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    layer_ids = list(range(n_layers if n_layers else len(model.model.layers)))
    d_ffn = model.config.intermediate_size

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", 24, 512, 1234)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    if SMOKE:
        ids_ev, byts_ev = ids_ev[:2], byts_ev[:2]
    B_TOT = float(byts_ev.sum())
    log("slice %dx%d, %d scored bytes | ids_sha256 OK"
        % (ids_ev.shape[0], ids_ev.shape[1], int(B_TOT)))

    e6eng = json.load(open(os.path.join(ENGDIR, "results", "e6", "engine.json"), encoding="utf-8"))
    pids = []
    for i, p in enumerate(prompts):
        q = tok(p)["input_ids"]
        assert q == e6eng["prompt_ids"][i], "prompt %d differs from E6's stored ids" % i
        pids.append(q)
    tgts = [ref[i]["ids"][-N_NEW:][:n_new] for i in range(len(prompts))]

    labs, _ = E19.load_labels(CARVE_E)
    log("== D0c partition E=%d loaded for %d layers ==" % (CARVE_E, len(labs)))

    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    assert meta_cal["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "calib and eval must be DIFFERENT corpus halves"
    if SMOKE:
        ids_cal = ids_cal[:2]

    log("== fitting %d ridge routers (D=%d -> E=%d) ==" % (len(layer_ids),
                                                           model.config.hidden_size, CARVE_E))
    t0 = time.time()
    routers, rdiag, mass_acc = fit_routers(model, ids_cal, layer_ids, labs)
    log("   done in %.0fs, lambda %.4g, %d tokens"
        % (time.time() - t0, rdiag[layer_ids[0]]["ridge_lambda"],
           rdiag[layer_ids[0]]["n_tokens"]))

    static, rnd = fixed_sets(mass_acc, layer_ids, CARVE_K)
    log("== fixed sets for the two planted negatives: from the same pass ==")

    # H for the low-rank arms, E21's own capture
    Hs = {}
    need_lr = any(a[2] for a in ARMS if (not ONLY or a[0] in ONLY))
    if need_lr:
        log("== H = X'X for q_proj/o_proj, %d organs ==" % (2 * len(layer_ids)))
        cnt, hooks = {}, []

        def mkh(key):
            def f(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                Hs[key] = (x.T @ x) if key not in Hs else Hs[key] + (x.T @ x)
                cnt[key] = cnt.get(key, 0) + x.shape[0]
            return f
        for li in layer_ids:
            lay = model.model.layers[li]
            hooks.append(lay.self_attn.q_proj.register_forward_hook(mkh((li, "q_proj"))))
            hooks.append(lay.self_attn.o_proj.register_forward_hook(mkh((li, "o_proj"))))
        t0 = time.time()
        with torch.no_grad():
            for i in range(ids_cal.shape[0]):
                model(ids_cal[i:i + 1])
        for h in hooks:
            h.remove()
        log("   done in %.0fs, %d H matrices" % (time.time() - t0, len(Hs)))

    d_model = model.config.hidden_size
    router_params = len(layer_ids) * d_model * CARVE_E
    out = {"brief": "briefs/BRIEF_E23_A_REAL_ROUTER.md (25bde22)",
           "question": "does the carve survive a REAL router?  every carve number in this "
                       "programme (D0, D0c, E19, E22) uses an ORACLE.  Nothing exported, no "
                       "timing taken, 6.79 tok/s untouched.",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "eval_slice": meta_ev, "calib_slice": meta_cal,
           "carve": {"E": CARVE_E, "k": CARVE_K}, "rank": RANK, "rng_seed": RNG_SEED,
           "router": {"kind": "ridge, block input -> sqrt(group mass)",
                      "ridge_frac": RIDGE_FRAC, "params": router_params,
                      "params_M": router_params / 1e6},
           "router_diag": {str(k): v for k, v in rdiag.items()},
           "anchors": {"E22_V52": E22_V52, "E22_QO512+V52": E22_QO_V52,
                       "base_bpb": BASE_BPB, "repl_tol": REPL_TOL},
           "bands": {"free": {"floor": FLOOR, "at_floor_max": AT_FLOOR_MAX,
                              "ranks_min": RANKS_MIN, "ceiling": len(prompts) * n_new},
                     "teacher_forced_from_E20": [TF_LO, TF_HI],
                     "retention": {"holds": RET_HOLDS, "costs": RET_COSTS}},
           "chance_bpb": CHANCE, "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    todo = [a for a in ARMS if (not ONLY or a[0] in ONLY or a[0] == "base")]
    for tag, kind, lowr in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            rr = out["arms"][tag]
            log("  %-18s CACHED  bpb %.6f  free %d  tf %d"
                % (tag, rr["bpb"], rr["matched"], rr["teacher_forced"]))
            continue
        t0 = time.time()
        saved = []
        if lowr:
            for li in layer_ids:
                lay = model.model.layers[li]
                for nm in ("q_proj", "o_proj"):
                    mod = getattr(lay.self_attn, nm)
                    saved.append((mod, mod.weight.data.clone()))
                    Wr, _ = E21.lowrank(mod.weight.data, Hs[(li, nm)], RANK, True)
                    mod.weight.data = Wr.float()
        if kind is not None:
            STATS["kept"], STATS["ntok"] = 0.0, 0
            if kind == "linear":
                install_input_taps(model, layer_ids)
            install(model, layer_ids, labs, CARVE_K, kind, routers, static, rnd)

        with torch.no_grad():
            nats = []
            for i in range(ids_ev.shape[0]):
                ch = ids_ev[i:i + 1]
                lg = model(ch).logits.float()
                lp = torch.nn.functional.log_softmax(lg[:, :-1], dim=-1)
                nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double())
        bpb = float(torch.stack(nats).sum() / (LN2 * B_TOT))

        allids, matched, first_div, per_prompt = [], 0, None, []
        for i in range(len(prompts)):
            new = E21.greedy(model, pids[i], n_new)[-n_new:]
            allids.append(new)
            m = sum(1 for a, b in zip(new, tgts[i]) if a == b)
            d = next((j for j, (a, b) in enumerate(zip(new, tgts[i])) if a != b), None)
            if d is not None and first_div is None:
                first_div = [i, d]
            per_prompt.append({"prompt": i, "free": m, "diverges_at": d})
            matched += m

        tf, ranks = 0, []
        with torch.no_grad():
            for i in range(len(prompts)):
                full = torch.tensor([pids[i] + tgts[i]])
                lg = model(full).logits[0].float()
                h = 0
                for kk in range(n_new):
                    row = lg[len(pids[i]) - 1 + kk]
                    t = tgts[i][kk]
                    h += int(int(torch.argmax(row)) == t)
                    ranks.append(int((row > row[t]).sum()) + 1)
                per_prompt[i]["tf"] = h
                tf += h

        ach = (STATS["kept"] / float(STATS["ntok"] * d_ffn)) if kind else 1.0
        clear()
        clear_taps()
        for mod, W in saved:
            mod.weight.data = W

        counted = len(prompts) * n_new
        rec = {"arm": tag, "router": kind, "lowrank": lowr,
               "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "agree": matched / float(counted),
               "teacher_forced": tf, "tf_frac": tf / float(counted),
               "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
               "rank_le5": int(sum(1 for r in ranks if r <= 5)),
               "achieved_activation": ach,
               "G_T4_activation_matches": (kind is None) or abs(ach - CARVE_ACT) < ACT_TOL,
               "first_div": first_div, "per_prompt": per_prompt,
               "band_free": band_free(matched), "band_tf": band_tf(tf),
               "ids": allids, "text": [tok.decode(x) for x in allids],
               "seconds": time.time() - t0}

        if tag == "base":
            rec["G_T0"] = ("FIRES" if (matched == counted and tf == counted
                                       and (SMOKE or abs(bpb - BASE_BPB) < REPL_TOL))
                           else "VOID")
        if tag == "V52-ORACLE" and not SMOKE:
            rec["G_T1"] = {"bpb_here": bpb, "bpb_E22": E22_V52["bpb"],
                           "abs_diff": abs(bpb - E22_V52["bpb"]),
                           "free_here": matched, "free_E22": E22_V52["free"],
                           "tf_here": tf, "tf_E22": E22_V52["tf"], "activation": ach}
            rec["G_T1"]["verdict"] = ("FIRES" if (abs(bpb - E22_V52["bpb"]) < REPL_TOL
                                                  and matched == E22_V52["free"]
                                                  and tf == E22_V52["tf"]
                                                  and abs(ach - CARVE_ACT) < ACT_TOL) else "VOID")
        if tag == "QO512+V52-ORACLE" and not SMOKE:
            rec["G_T1b"] = {"bpb_here": bpb, "bpb_E22": E22_QO_V52["bpb"],
                            "abs_diff": abs(bpb - E22_QO_V52["bpb"]),
                            "free_here": matched, "free_E22": E22_QO_V52["free"],
                            "tf_here": tf, "tf_E22": E22_QO_V52["tf"]}

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-18s BPB %.6f (%+.6f)  free %3d/%d  tf %3d/%d  mrank %8.2f  act %.4f  %s / %s"
            % (tag, bpb, bpb - CHANCE, matched, counted, tf, counted,
               rec["mean_rank"], ach, rec["band_free"], rec["band_tf"]))
        for g in ("G_T0", "G_T1"):
            if g in rec:
                v = rec[g] if isinstance(rec[g], str) else rec[g]["verdict"]
                log("                      %s: %s" % (g, v))

    # ---- retention and the gates, brief s5/s6
    A = out["arms"]
    ver = {}
    for comp, orc in (("V52-LINEAR", "V52-ORACLE"),
                      ("QO512+V52-LINEAR", "QO512+V52-ORACLE")):
        if comp in A and orc in A and "V52-RANDOM" in A:
            rnd_tf = A["V52-RANDOM"]["teacher_forced"]
            den = A[orc]["teacher_forced"] - rnd_tf
            ret = (A[comp]["teacher_forced"] - rnd_tf) / float(den) if den else None
            lab = None
            if ret is not None:
                if ret >= RET_HOLDS and A[comp]["teacher_forced"] >= TF_LO:
                    lab = "ROUTER-HOLDS"
                elif ret >= RET_COSTS or A[comp]["teacher_forced"] >= 80:
                    lab = "ROUTER-COSTS"
                else:
                    lab = "ROUTER-BREAKS-IT"
            ver[comp] = {"oracle": orc, "tf_linear": A[comp]["teacher_forced"],
                         "tf_oracle": A[orc]["teacher_forced"], "tf_random": rnd_tf,
                         "retention": ret, "verdict": lab,
                         "bpb_linear": A[comp]["bpb"], "bpb_oracle": A[orc]["bpb"],
                         "bpb_cost_of_real_router": A[comp]["bpb"] - A[orc]["bpb"]}
    out["retention"] = ver

    # G-T2: the planted negatives must actually be bad
    gt2 = None
    if all(t in A for t in ("V52-LINEAR", "V52-STATIC", "V52-RANDOM")):
        lin = A["V52-LINEAR"]["teacher_forced"]
        gap_s = lin - A["V52-STATIC"]["teacher_forced"]
        gap_r = lin - A["V52-RANDOM"]["teacher_forced"]
        gt2 = {"tf_linear": lin, "tf_static": A["V52-STATIC"]["teacher_forced"],
               "tf_random": A["V52-RANDOM"]["teacher_forced"],
               "gap_vs_static": gap_s, "gap_vs_random": gap_r,
               "verdict": "FIRES" if (gap_s > 2 and gap_r > 2) else "NO-RESOLUTION"}
    out["G_T2"] = gt2

    voids = []
    for t in ("base", "V52-ORACLE"):
        if t in A:
            v = A[t].get("G_T0") or A[t].get("G_T1", {}).get("verdict")
            if v == "VOID":
                voids.append(t)
    bad_act = [t for t, r in A.items() if not r.get("G_T4_activation_matches", True)]
    out["VOID"] = voids
    out["G_T4_failures"] = bad_act

    log("")
    log("  %-18s %-10s %-9s %-9s %-8s %-9s %s"
        % ("arm", "BPB", "free", "tf", "act", "band-free", "band-tf"))
    for tag, _k, _l in ARMS:
        if tag in A:
            r = A[tag]
            log("    %-18s %9.6f %4d/%-4d %4d/%-4d %7.4f  %-9s %s"
                % (tag, r["bpb"], r["matched"], r["counted"], r["teacher_forced"],
                   r["counted"], r["achieved_activation"], r["band_free"], r["band_tf"]))
    log("")
    for comp, d in ver.items():
        log("  %-18s tf %d  (oracle %d, random %d)  retention %s  -> %s"
            % (comp, d["tf_linear"], d["tf_oracle"], d["tf_random"],
               ("%.4f" % d["retention"]) if d["retention"] is not None else "n/a", d["verdict"]))
        log("                     BPB cost of a real router: %+.6f" % d["bpb_cost_of_real_router"])
    if gt2:
        log("  G-T2 planted negatives: linear %d vs static %d vs random %d -> %s"
            % (gt2["tf_linear"], gt2["tf_static"], gt2["tf_random"], gt2["verdict"]))
    log("  router cost: %d params = %.1f M, charged in the brief at 0.9441 -> 0.9551 G"
        % (router_params, router_params / 1e6))
    log("  G-T4 failures: %s" % (", ".join(bad_act) if bad_act else "none"))
    log("  VOID: %s" % (", ".join(voids) if voids else "none"))

    out["seconds_total"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("wrote %s  [%.0fs]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
