#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E22 -- does cheap compose?  The first configuration inside the 50 tok/s budget.

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E22_DOES_CHEAP_COMPOSE.md (3f8465a),
pushed before this file existed.  Nothing here is exported and NO TIMING IS TAKEN.

E21 gave the cheapest cut on the rank axis (QO-ACT-512: +0.052689 BPB, 144/160 teacher-forced).
E19 gave the deepest carve this donor's own budget requires (V52: +0.141846 BPB, oracle router).
Brief section 2, published before the run: composed they put the 1.5B at 0.9441 G active ternary
weights/token against E18's 0.982-1.060 G budget for 50 tok/s -- the first configuration this
programme has derived INSIDE the budget.  Whether it still works is what this runner measures.

NOTHING IS REIMPLEMENTED.  The low-rank construction is e21_rank.lowrank / _sqrt_pair, the carve
is e19_carve_rank.install over D0c's cached partition, the ternarization is t2_rules.r3_actsearch
driven by t2b_organs.capture, and the teacher-forced instrument is E20 part B's.  That is what
makes G-S1 and G-S2 real replication gates rather than decoration: two independent prior runs,
each of which must be reproduced before any composed arm is read.

Env: D_THREADS (6), E22_ONLY (comma list), E22_SMOKE (1)
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
import t2_rules as T2                                       # noqa: E402
import t2b_organs as T2B                                    # noqa: E402
import e21_rank as E21                                      # noqa: E402
import e19_carve_rank as E19                                # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E22_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E22_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e22_compose%s.json" % ("_smoke" if SMOKE else ""))
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453

CHANCE = 4.069819
FLOOR, MARGIN = 12, 2
AT_FLOOR_MAX, RANKS_MIN = FLOOR + MARGIN, 80
TF_LO, TF_HI = 107, 119                     # E20 part B, eight ternary heads
ADD_TOL = 0.020                             # brief section 6: 4 * sigma_seed
NCAL, SEQCAL, SEEDCAL = 32, 512, 42424

RANK = 512                                  # E21's QO-ACT-512
CARVE_E, CARVE_K = 256, 133                 # E19's V52
CARVE_ACT = 0.51953125                      # E19's achieved value, = 133/256
ACT_TOL = 0.002                             # D0c section 3.2's own tolerance

# ---- the two replication anchors, quoted from the published runs and never re-derived here
E21_QO512 = {"bpb": 0.8202837636996289, "free": 26, "tf": 144}     # results/e21_rank.json
E19_V52_BPB = 0.909440994415161                                    # results/e19_carve_rank.json
E19_V52_FREE = 12
BASE_BPB = 0.7675949641196624
REPL_TOL = 1e-9

#  tag               lowrank ternary-factors carve ffn+head  balanced
#  "-T"  = the factored form EXACTLY as brief section 3 registered it
#  "-TB" = the same with A's column norms folded into B's rows, added after the smoke
ARMS = [("base",          False,  False,  False, False, False),
        ("QO512",         True,   False,  False, False, False),
        ("V52",           False,  False,  True,  False, False),
        ("QO512+V52",     True,   False,  True,  False, False),
        ("QO512-T",       True,   True,   False, False, False),
        ("QO512-TB",      True,   True,   False, False, True),
        ("QO512-T+V52",   True,   True,   True,  False, False),
        ("QO512-TB+V52",  True,   True,   True,  False, True),
        ("STACK",         True,   True,   True,  True,  True)]


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


# ================================================= the factored form, brief section 3
def factors(W, H, r):
    """A (n_out x r) and B (r x n_in) with A.B == e21_rank.lowrank(W, H, r, True).

    Built from e21_rank._sqrt_pair so the basis is bit-for-bit the one E21 measured; the
    identity A.B == Wr is checked by the caller and reported as G-S5.
    """
    Wd = W.double()
    n_in = Wd.shape[1]
    r = min(r, n_in)
    Hh, Hi, lam, cond = E21._sqrt_pair(H)
    G = Hh.mm(Wd.T.mm(Wd)).mm(Hh)
    G = 0.5 * (G + G.T)
    ev, Bv = torch.linalg.eigh(G)
    Br = Bv[:, n_in - r:]
    A = Wd.mm(Hh).mm(Br)            # [n_out, r]
    B = Br.T.mm(Hi)                 # [r, n_in]
    return A, B, {"damping": lam, "H_cond_damped": cond}


def ternary_factors(W, H, r, rms_in, n_cal, balanced=False):
    """Both factors ternarized with the SHIPPED rule (t2_rules.r3_actsearch).

    B sees x, whose per-channel RMS is already captured.  A sees B_t x, and its RMS follows in
    closed form from the SAME H -- rms_A = sqrt(diag(B_t H B_t') / n) -- so no second calibration
    pass is needed and no RMS is guessed.  B_t, not B, is used for that: it is what actually
    feeds A once the first factor is quantized.
    """
    A, B, d = factors(W, H, r)

    # ---- BALANCED variant, added after the smoke and disclosed as such (see the probe).
    # The registered form puts the whole singular-value range into A's COLUMNS, whose norms
    # span 5.2e2 on layer 0's q_proj, while r3_actsearch has only a per-ROW scale -- so 78% of
    # A ternarizes to zero and the product lands at relative error 1.5570, worse than replacing
    # W with zeros.  Folding A's column norms into B's rows is EXACTLY identity-preserving
    # (measured 2.3e-16) and costs no storage, because B's ternary format already carries a
    # per-row scale.  Both arms are run; the registered one stays in the table as the record.
    if balanced:
        c = A.norm(dim=0).clamp_min(1e-30)
        A = A / c
        B = c.unsqueeze(1) * B
    Af, Bf = A.float(), B.float()

    qB, aB = T2.r3_actsearch(Bf, rms_in)
    Bt = (qB * aB)

    Bd = Bt.double()
    varA = torch.diagonal(Bd.mm(H.double()).mm(Bd.T)) / float(n_cal)
    rms_A = varA.clamp_min(1e-16).sqrt().float()

    qA, aA = T2.r3_actsearch(Af, rms_A)
    At = (qA * aA)

    d.update({"zero_frac_A": float((qA == 0).float().mean()),
              "zero_frac_B": float((qB == 0).float().mean()),
              "balanced": balanced, "rank": r})
    return At.mm(Bt), (Af.mm(Bf)), d


# ================================================= apply one arm
def apply_arm(model, lowr, tern, carve, ffnhead, bal, Hs, act_rms, n_cal, labs,
              layer_ids, n_layers_g=None):
    saved, undo, st = [], [], {"n_lowrank": 0, "rel": [], "energy": [],
                               "params_dense": 0, "params_factored": 0, "gs5": []}
    t0 = time.time()

    if lowr:
        for li, lay in enumerate(model.model.layers):
            if li not in layer_ids and layer_ids is not None:
                continue
            for nm in ("q_proj", "o_proj"):
                mod = getattr(lay.self_attn, nm)
                W = mod.weight.data
                saved.append((mod, W.clone()))
                H = Hs[(li, nm)]
                if tern:
                    Wr, Wfp, d = ternary_factors(W, H, RANK, act_rms[(li, nm)],
                                                 n_cal, bal)
                    if not st["gs5"]:      # spot check on the first organ only: another eigh
                        st["gs5"].append(  # per organ would double the arm's cost for nothing
                            float(torch.linalg.norm(Wfp - E21.lowrank(W, H, RANK, True)[0])
                                  / torch.linalg.norm(W)))
                else:
                    Wr, d = E21.lowrank(W, H, RANK, True)
                    st["rel"].append(d["rel_weight_error"])
                    if d["energy_kept_weighted"] is not None:
                        st["energy"].append(d["energy_kept_weighted"])
                mod.weight.data = Wr.float()
                st["n_lowrank"] += 1
                st["params_dense"] += W.shape[0] * W.shape[1]
                st["params_factored"] += RANK * (W.shape[0] + W.shape[1])
                st.setdefault("damping", d["damping"])
                if tern:
                    st.setdefault("zero_frac_A", d["zero_frac_A"])
                    st.setdefault("zero_frac_B", d["zero_frac_B"])

    if ffnhead:
        # exactly the brief: ternary FFN and ternary head, via t2b's own code path.
        # k_proj / v_proj are NOT touched -- 22.0 M, 1.4% of the model -- and that is recorded.
        for tag in ("F", "H"):
            r, s = T2B.apply_arm(model, tag, act_rms, n_layers_g)
            undo.append(r)
            st["ternary_%s_n" % tag] = s["n"]

    if carve:
        E19.STATS["kept"], E19.STATS["ntok"] = 0.0, 0
        E19.install(model, layer_ids, labs, CARVE_K)

    st["seconds_apply"] = time.time() - t0
    st["rel_weight_error"] = float(np.mean(st["rel"])) if st["rel"] else None
    st["energy_kept_weighted"] = float(np.mean(st["energy"])) if st["energy"] else None
    st["gs5_max_rel"] = float(np.max(st["gs5"])) if st["gs5"] else None
    st["saving_vs_dense"] = (1.0 - st["params_factored"] / float(st["params_dense"])
                             if st["params_dense"] else 0.0)
    del st["rel"], st["energy"], st["gs5"]

    def restore():
        E19.clear()
        while undo:
            undo.pop()()
        for mod, W in saved:
            mod.weight.data = W
    return restore, st


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS
    n_layers = 4 if SMOKE else None

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    n_all = len(model.model.layers)
    layer_ids = list(range(n_layers if n_layers else n_all))
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

    # ---- D0c's partition, read from the cache ITS run wrote.  Nothing is re-clustered.
    labs, _nulls = E19.load_labels(CARVE_E)
    log("== D0c partition E=%d loaded for %d layers ==" % (CARVE_E, len(labs)))

    # ---- calibration, two passes over the SAME frozen slice
    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    assert meta_cal["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "calib and eval must be DIFFERENT corpus halves"
    if SMOKE:
        ids_cal = ids_cal[:2]
    n_cal = int(ids_cal.shape[0] * ids_cal.shape[1])

    log("== calibration pass 1/2: act_rms via t2b_organs.capture, %d tokens ==" % n_cal)
    t0 = time.time()
    act_rms = T2B.capture(model, ids_cal, n_layers)
    log("   done in %.0fs, %d organs" % (time.time() - t0, len(act_rms)))

    log("== calibration pass 2/2: H = X'X for q_proj/o_proj, %d organs =="
        % (2 * len(layer_ids)))
    Hs, cnt, hooks = {}, {}, []

    def mk(key):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            Hs[key] = (x.T @ x) if key not in Hs else Hs[key] + (x.T @ x)
            cnt[key] = cnt.get(key, 0) + x.shape[0]
        return f
    for li in layer_ids:
        lay = model.model.layers[li]
        hooks.append(lay.self_attn.q_proj.register_forward_hook(mk((li, "q_proj"))))
        hooks.append(lay.self_attn.o_proj.register_forward_hook(mk((li, "o_proj"))))
    t0 = time.time()
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            model(ids_cal[i:i + 1])
    for h in hooks:
        h.remove()
    n_cal_h = cnt[(layer_ids[0], "q_proj")]
    log("   done in %.0fs, %d H matrices, %d tokens each"
        % (time.time() - t0, len(Hs), n_cal_h))

    # free identity, costs nothing: act_rms should equal sqrt(diag(H)/n) -- E20 measured 3.8e-06
    k0 = (layer_ids[0], "q_proj")
    id_err = float((act_rms[k0] - (torch.diagonal(Hs[k0]) / n_cal_h).sqrt()).abs().max())

    out = {"brief": "briefs/BRIEF_E22_DOES_CHEAP_COMPOSE.md (3f8465a)",
           "question": "does cheap compose?  nothing exported, no timing taken, "
                       "6.79 tok/s untouched; the carve's router is an ORACLE = a ceiling",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "eval_slice": meta_ev, "calib_slice": meta_cal,
           "calib_tokens": n_cal_h, "rank": RANK, "carve": {"E": CARVE_E, "k": CARVE_K},
           "act_rms_vs_diagH_maxabs": id_err,
           "anchors": {"E21_QO512": E21_QO512, "E19_V52_bpb": E19_V52_BPB,
                       "E19_V52_free": E19_V52_FREE, "base_bpb": BASE_BPB,
                       "repl_tol": REPL_TOL},
           "bands": {"free": {"floor": FLOOR, "at_floor_max": AT_FLOOR_MAX,
                              "ranks_min": RANKS_MIN, "ceiling": len(prompts) * n_new},
                     "teacher_forced_from_E20": [TF_LO, TF_HI],
                     "additivity_tol": ADD_TOL},
           "chance_bpb": CHANCE, "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    todo = [a for a in ARMS if (not ONLY or a[0] in ONLY or a[0] == "base")]
    for tag, lowr, tern, carve, ffnhead, bal in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            rr = out["arms"][tag]
            log("  %-13s CACHED  bpb %.6f  free %d  tf %d"
                % (tag, rr["bpb"], rr["matched"], rr["teacher_forced"]))
            continue
        t0 = time.time()
        if tag == "base":
            restore, st = (lambda: None), {"n_lowrank": 0, "seconds_apply": 0.0,
                                           "rel_weight_error": None,
                                           "energy_kept_weighted": None,
                                           "gs5_max_rel": None, "saving_vs_dense": 0.0}
        else:
            restore, st = apply_arm(model, lowr, tern, carve, ffnhead, bal,
                                    Hs, act_rms, n_cal_h, labs, layer_ids, n_layers)

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
                for k in range(n_new):
                    row = lg[len(pids[i]) - 1 + k]
                    t = tgts[i][k]
                    h += int(int(torch.argmax(row)) == t)
                    ranks.append(int((row > row[t]).sum()) + 1)
                per_prompt[i]["tf"] = h
                tf += h

        ach = (E19.STATS["kept"] / float(E19.STATS["ntok"] * d_ffn)) if carve else 1.0
        restore()

        counted = len(prompts) * n_new
        rec = {"arm": tag, "lowrank": lowr, "ternary_factors": tern, "carve": carve,
               "ffn_head_ternary": ffnhead, "balanced_factors": bal,
               "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "agree": matched / float(counted),
               "teacher_forced": tf, "tf_frac": tf / float(counted),
               "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
               "rank_le5": int(sum(1 for r in ranks if r <= 5)),
               "achieved_activation": ach,
               "first_div": first_div, "per_prompt": per_prompt,
               "band_free": band_free(matched), "band_tf": band_tf(tf),
               "ids": allids,
               "text": [tok.decode(x) for x in allids],
               "seconds": time.time() - t0}
        rec.update({k: v for k, v in st.items() if k != "seconds"})

        # ---- replication gates, checked the moment the arm is read
        if tag == "base":
            rec["G_S0"] = ("FIRES" if (matched == counted and tf == counted
                                       and (SMOKE or abs(bpb - BASE_BPB) < REPL_TOL))
                           else "VOID")
        if tag == "QO512" and not SMOKE:
            rec["G_S1"] = {"bpb_here": bpb, "bpb_E21": E21_QO512["bpb"],
                           "abs_diff": abs(bpb - E21_QO512["bpb"]),
                           "free_here": matched, "free_E21": E21_QO512["free"],
                           "tf_here": tf, "tf_E21": E21_QO512["tf"]}
            rec["G_S1"]["verdict"] = ("FIRES" if (abs(bpb - E21_QO512["bpb"]) < REPL_TOL
                                                  and matched == E21_QO512["free"]
                                                  and tf == E21_QO512["tf"]) else "VOID")
        if tag == "V52" and not SMOKE:
            rec["G_S2"] = {"bpb_here": bpb, "bpb_E19": E19_V52_BPB,
                           "abs_diff": abs(bpb - E19_V52_BPB),
                           "free_here": matched, "free_E19": E19_V52_FREE,
                           "activation_here": ach, "activation_E19": CARVE_ACT}
            rec["G_S2"]["verdict"] = ("FIRES" if (abs(bpb - E19_V52_BPB) < REPL_TOL
                                                  and matched == E19_V52_FREE
                                                  and abs(ach - CARVE_ACT) < ACT_TOL) else "VOID")

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-13s BPB %.6f (%+.6f)  free %3d/%d  tf %3d/%d  mrank %8.2f  act %.4f  %s / %s"
            % (tag, bpb, bpb - CHANCE, matched, counted, tf, counted,
               rec["mean_rank"], ach, rec["band_free"], rec["band_tf"]))
        if "G_S1" in rec or "G_S2" in rec or "G_S0" in rec:
            g = rec.get("G_S0") or rec.get("G_S1", {}).get("verdict") or rec["G_S2"]["verdict"]
            log("                 replication gate: %s" % g)

    # ---- additivity, brief section 6
    A = out["arms"]
    add = {}
    for comp, p1, p2 in (("QO512+V52", "QO512", "V52"),
                         ("QO512-T+V52", "QO512-T", "V52"),
                         ("QO512-TB+V52", "QO512-TB", "V52")):
        if comp in A and p1 in A and p2 in A and "base" in A:
            pred = A[p1]["bpb"] + A[p2]["bpb"] - A["base"]["bpb"]
            exc = A[comp]["bpb"] - pred
            tmin = min(A[p1]["teacher_forced"], A[p2]["teacher_forced"])
            add[comp] = {"parts": [p1, p2], "additive_prediction": pred,
                         "measured": A[comp]["bpb"], "excess": exc,
                         "score_band": ("ADDITIVE" if abs(exc) <= ADD_TOL
                                        else ("SUB-ADDITIVE" if exc < 0 else "SUPER-ADDITIVE")),
                         "tf_measured": A[comp]["teacher_forced"], "tf_min_of_parts": tmin,
                         "rank_band": ("RANK-SUB-ADDITIVE"
                                       if A[comp]["teacher_forced"] >= tmin
                                       else "RANK-SUPER-ADDITIVE")}
    out["additivity"] = add

    voids = []
    for t in ("base", "QO512", "V52"):
        if t in A:
            v = (A[t].get("G_S0") or A[t].get("G_S1", {}).get("verdict")
                 or A[t].get("G_S2", {}).get("verdict"))
            if v == "VOID":
                voids.append(t)
    out["VOID"] = voids

    log("")
    log("  %-13s %-10s %-9s %-9s %-8s %-9s %s"
        % ("arm", "BPB", "free", "tf", "act", "band-free", "band-tf"))
    for tag, _l, _t, _c, _f, _b in ARMS:
        if tag in A:
            r = A[tag]
            log("    %-13s %9.6f %4d/%-4d %4d/%-4d %7.4f  %-9s %s"
                % (tag, r["bpb"], r["matched"], r["counted"], r["teacher_forced"],
                   r["counted"], r["achieved_activation"], r["band_free"], r["band_tf"]))
    log("")
    for comp, d in add.items():
        log("  additivity %-13s pred %.6f  meas %.6f  excess %+.6f  %s | tf %d vs min(parts) %d  %s"
            % (comp, d["additive_prediction"], d["measured"], d["excess"], d["score_band"],
               d["tf_measured"], d["tf_min_of_parts"], d["rank_band"]))
    log("")
    log("  act_rms vs sqrt(diag(H)/n) max abs: %.3e" % id_err)
    log("  VOID: %s" % (", ".join(voids) if voids else "none"))

    out["seconds_total"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("wrote %s  [%.0fs]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
