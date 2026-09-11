#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E24 -- the depth the budget permits, with a router that exists.

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E24_THE_DEPTH_THE_BUDGET_PERMITS.md
(b78ce4d), pushed before this file existed.  Nothing is exported and NO TIMING IS TAKEN.

E23 split the verdict: the carve alone holds with a real ridge router (V52-LINEAR tf 110/160,
retention 0.9067) and the COMPOSED configuration does not (QO512+V52-LINEAR tf 102/160,
retention 0.7143, ROUTER-COSTS, band WORSE).  ROUTER-COSTS obliges re-deriving the healing
target at a shallower depth.  This runner is that re-derivation: not a search for a depth that
works, but a measurement at the three depths the 50 tok/s budget permits once the 11.0 M router
is CHARGED -- k = 139, 148, 156 of 256, fixed by arithmetic in E23 s7 before this brief existed.
k = 133 (E19's depth, carried unexamined by E22 and E23) is 2.8% BELOW the budget floor and is
carried here only as the replication anchor.

Every fit, hook and low-rank factorisation is IMPORTED from e23_router / e21_rank.  That import
is what makes G-U1 meaningful: if this file reproduces E23's two composed arms to < 1e-9, the
machinery is the same machinery and the depth axis is the only thing that moved.

Env: D_THREADS (6), E24_ONLY (comma list), E24_SMOKE (1)
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
import e23_router as E23                                    # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E24_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E24_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e24_depth%s.json" % ("_smoke" if SMOKE else ""))
E23OUT = os.path.join(ENGDIR, "results", "e23_router.json")
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453

CHANCE = 4.069819
FLOOR, MARGIN = 12, 2
AT_FLOOR_MAX, RANKS_MIN = FLOOR + MARGIN, 80
TF_LO, TF_HI = 107, 119
NCAL, SEQCAL, SEEDCAL = 32, 512, 42424
CARVE_E = 256
RANK = 512                                  # E21's QO-ACT-512, as in E22 and E23
ACT_TOL = 0.002

# ---- anchors, quoted from E23's PUBLISHED RECORD and never re-derived here.
# Note: e23_router.py's E22_QO_V52 literal (1.0050393404796996) is E22's published number;
# E23 MEASURED 1.0050386831300866 for the same arm and that is what this file must reproduce,
# because the reproduction is of the run whose machinery is being reused.
BASE_BPB = 0.7675949641196624
A_K133_LIN = {"bpb": 1.2014771810176486, "free": 9, "tf": 102, "e23_arm": "QO512+V52-LINEAR"}
A_K133_ORC = {"bpb": 1.0050386831300866, "free": 15, "tf": 126, "e23_arm": "QO512+V52-ORACLE"}
REPL_TOL = 1e-9

# ---- the budget, s1, recomputed here from the shape rather than copied as prose
BUDGET_LO, BUDGET_HI = 0.982e9, 1.060e9      # E18 s31
K_BUDGET = (139, 148, 156)
K_ANCHOR = 133
G_U3_TOL = 3                                 # tokens of non-monotonicity that count as a finding
STATIC_TIE = 8                               # prediction 4's band, reported not gated

#        tag             k     router    target  lowrank
ARMS = [("base",         None, None,     None,    False),
        ("K133-LINEAR",  133,  "linear", "sqrt",  True),
        ("K133-ORACLE",  133,  "oracle", None,    True),
        ("K139-LINEAR",  139,  "linear", "sqrt",  True),
        ("K148-LINEAR",  148,  "linear", "sqrt",  True),
        ("K156-LINEAR",  156,  "linear", "sqrt",  True),
        ("K156-ORACLE",  156,  "oracle", None,    True),
        ("K148-STATIC",  148,  "static", None,    True),
        ("K148-LOG1P",   148,  "linear", "log1p", True)]

TARGETS_NEEDED = ("sqrt", "log1p")


def log(*a):
    print(*a, flush=True)


def budget(D, F, L, NKV, HD, V, k, with_router):
    """s1's table, as arithmetic.  The router is CHARGED unless the arm does not read one."""
    qo = 2 * (2 * D * RANK) * L
    kv = 2 * (D * NKV * HD) * L
    head = V * D
    rt = (D * CARVE_E * L) if with_router else 0
    ffn_full = 3 * D * F * L
    ffn = int(round(ffn_full * k / float(CARVE_E)))
    tot = qo + kv + head + rt + ffn
    return {"q_o_rank": qo, "k_v": kv, "head": head, "router": rt,
            "ffn_full": ffn_full, "ffn_at_k": ffn, "total": tot, "total_G": tot / 1e9,
            "activation": k / float(CARVE_E),
            "in_budget": bool(BUDGET_LO <= tot <= BUDGET_HI)}


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS
    n_layers = 4 if SMOKE else None

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]

    # ---- the anchors must match E23's own record on disk, or the anchor is not the anchor
    anchor_check = None
    if os.path.exists(E23OUT):
        e23 = json.load(open(E23OUT, encoding="utf-8"))["arms"]
        anchor_check = {}
        for tag, A in (("K133-LINEAR", A_K133_LIN), ("K133-ORACLE", A_K133_ORC)):
            r = e23[A["e23_arm"]]
            ok = (r["bpb"] == A["bpb"] and r["matched"] == A["free"]
                  and r["teacher_forced"] == A["tf"])
            anchor_check[tag] = {"e23_arm": A["e23_arm"], "bpb_on_disk": r["bpb"],
                                 "free_on_disk": r["matched"], "tf_on_disk": r["teacher_forced"],
                                 "matches_literal": bool(ok)}
            if not ok:
                raise SystemExit("ANCHOR MISMATCH vs %s -- STOP" % E23OUT)
        log("== E23 anchors verified against %s ==" % os.path.basename(E23OUT))

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    layer_ids = list(range(n_layers if n_layers else len(model.model.layers)))
    cfg = model.config
    d_ffn = cfg.intermediate_size

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

    todo = [a for a in ARMS if (not ONLY or a[0] in ONLY or a[0] == "base")]

    # ---- ONE calibration pass for BOTH router targets and the static accumulator
    need_rt = any(a[2] in ("linear", "static") for a in todo)
    routers, rdiag, mass_acc = {}, {}, {}
    if need_rt:
        log("== fitting %d x %d ridge routers (D=%d -> E=%d), targets %s =="
            % (len(TARGETS_NEEDED), len(layer_ids), cfg.hidden_size, CARVE_E,
               ",".join(TARGETS_NEEDED)))
        t0 = time.time()
        routers, rdiag, mass_acc = E23.fit_routers_multi(model, ids_cal, layer_ids, labs,
                                                         TARGETS_NEEDED)
        log("   done in %.0fs, lambda %.4g, %d tokens"
            % (time.time() - t0, rdiag[layer_ids[0]]["ridge_lambda"],
               rdiag[layer_ids[0]]["n_tokens"]))

    # STATIC's set depends on k, so it is built per depth, not once
    fixed_cache = {}

    def get_fixed(k):
        if k not in fixed_cache:
            fixed_cache[k] = E23.fixed_sets(mass_acc, layer_ids, k)
        return fixed_cache[k]

    # ---- the low-rank q/o weights, factorised ONCE and reused by all eight composed arms.
    # E21.lowrank over 56 organs is the expensive part of an arm; doing it per-arm would run it
    # eight times for eight identical results.
    ORIG, LR = {}, {}
    need_lr = any(a[4] for a in todo)
    if need_lr:
        log("== H = X'X for q_proj/o_proj, %d organs ==" % (2 * len(layer_ids)))
        Hs, cnt, hooks = {}, {}, []

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

        log("== rank-%d q/o, factorised ONCE for all composed arms ==" % RANK)
        t0 = time.time()
        for li in layer_ids:
            lay = model.model.layers[li]
            for nm in ("q_proj", "o_proj"):
                mod = getattr(lay.self_attn, nm)
                ORIG[(li, nm)] = mod.weight.data.clone()
                Wr, _ = E21.lowrank(mod.weight.data, Hs[(li, nm)], RANK, True)
                LR[(li, nm)] = Wr.float()
        del Hs
        log("   done in %.0fs, %d organs cached" % (time.time() - t0, len(LR)))

    lr_on = [False]

    def set_lowrank(on):
        if on == lr_on[0]:
            return
        for li in layer_ids:
            lay = model.model.layers[li]
            for nm in ("q_proj", "o_proj"):
                mod = getattr(lay.self_attn, nm)
                mod.weight.data = LR[(li, nm)] if on else ORIG[(li, nm)]
        lr_on[0] = on

    d_model = cfg.hidden_size
    out = {"brief": "briefs/BRIEF_E24_THE_DEPTH_THE_BUDGET_PERMITS.md (b78ce4d)",
           "question": "E23's ROUTER-COSTS was measured at k=133, a depth 2.8% BELOW the "
                       "budget floor and inherited from before routers cost anything.  At the "
                       "three depths the 50 tok/s budget actually permits with the router "
                       "charged -- 139, 148, 156 of 256 -- does the composed configuration "
                       "recover?  Nothing exported, no timing taken.",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "eval_slice": meta_ev, "calib_slice": meta_cal,
           "carve_E": CARVE_E, "rank": RANK, "targets": list(TARGETS_NEEDED),
           "router": {"kind": "ridge, block input -> f(group mass)",
                      "ridge_frac": E23.RIDGE_FRAC,
                      "params": len(layer_ids) * d_model * CARVE_E,
                      "params_M": len(layer_ids) * d_model * CARVE_E / 1e6},
           "router_diag": {str(k): v for k, v in rdiag.items()},
           "anchors": {"base_bpb": BASE_BPB, "K133-LINEAR": A_K133_LIN,
                       "K133-ORACLE": A_K133_ORC, "repl_tol": REPL_TOL,
                       "verified_against_e23_json": anchor_check},
           "bands": {"free": {"floor": FLOOR, "at_floor_max": AT_FLOOR_MAX,
                              "ranks_min": RANKS_MIN, "ceiling": len(prompts) * n_new},
                     "teacher_forced_from_E20": [TF_LO, TF_HI],
                     "budget_G": [BUDGET_LO / 1e9, BUDGET_HI / 1e9]},
           "budget": {}, "chance_bpb": CHANCE, "arms": {}}

    nl_full = len(model.model.layers)
    hd = cfg.hidden_size // cfg.num_attention_heads
    for k in (K_ANCHOR,) + K_BUDGET:
        out["budget"][str(k)] = budget(d_model, d_ffn, nl_full, cfg.num_key_value_heads, hd,
                                       cfg.vocab_size, k, True)
    out["budget"]["148_no_router"] = budget(d_model, d_ffn, nl_full, cfg.num_key_value_heads,
                                            hd, cfg.vocab_size, 148, False)
    log("== budget, recomputed from the shape ==")
    for kk, b in out["budget"].items():
        log("   k=%-14s act %.4f  total %.4f G  in-budget %s"
            % (kk, b["activation"], b["total_G"], b["in_budget"]))

    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    for tag, k, kind, target, lowr in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            rr = out["arms"][tag]
            log("  %-14s CACHED  bpb %.6f  free %d  tf %d"
                % (tag, rr["bpb"], rr["matched"], rr["teacher_forced"]))
            continue
        t0 = time.time()
        set_lowrank(bool(lowr))
        if kind is not None:
            E23.STATS["kept"], E23.STATS["ntok"] = 0.0, 0
            st, rnd = (get_fixed(k) if kind == "static" else (None, None))
            R = routers[target] if kind == "linear" else None
            if kind == "linear":
                E23.install_input_taps(model, layer_ids)
            E23.install(model, layer_ids, labs, k, kind, R, st, rnd)

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

        ach = (E23.STATS["kept"] / float(E23.STATS["ntok"] * d_ffn)) if kind else 1.0
        E23.clear()
        E23.clear_taps()

        counted = len(prompts) * n_new
        want_act = (k / float(CARVE_E)) if kind else 1.0
        rec = {"arm": tag, "k": k, "router": kind, "target": target, "lowrank": lowr,
               "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "agree": matched / float(counted),
               "teacher_forced": tf, "tf_frac": tf / float(counted),
               "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
               "rank_le5": int(sum(1 for r in ranks if r <= 5)),
               "achieved_activation": ach, "charged_activation": want_act,
               "G_U2": bool(abs(ach - want_act) < ACT_TOL),
               "charged_G": (out["budget"][str(k)]["total_G"] if k else None),
               "first_div": first_div, "per_prompt": per_prompt,
               "band_free": E23.band_free(matched), "band_tf": E23.band_tf(tf),
               "ids": allids, "text": [tok.decode(x) for x in allids],
               "seconds": time.time() - t0}
        if tag == "K148-STATIC":
            # registered in s3: STATIC needs no per-token score, so it is charged at 0 M router
            rec["charged_G"] = out["budget"]["148_no_router"]["total_G"]
            rec["router_charged_M"] = 0.0

        if tag == "base":
            rec["G_U0"] = ("FIRES" if (matched == counted and tf == counted
                                       and (SMOKE or abs(bpb - BASE_BPB) < REPL_TOL))
                           else "VOID")
        for anch, A in (("K133-LINEAR", A_K133_LIN), ("K133-ORACLE", A_K133_ORC)):
            if tag == anch and not SMOKE:
                rec["G_U1"] = {"bpb_here": bpb, "bpb_anchor": A["bpb"],
                               "abs_diff": abs(bpb - A["bpb"]),
                               "free_here": matched, "free_anchor": A["free"],
                               "tf_here": tf, "tf_anchor": A["tf"], "e23_arm": A["e23_arm"],
                               "verdict": ("FIRES" if (abs(bpb - A["bpb"]) < REPL_TOL
                                                       and matched == A["free"]
                                                       and tf == A["tf"]) else "VOID")}

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-14s BPB %.6f (%+.6f)  free %3d/%d  tf %3d/%d  mrank %8.2f  act %.4f  %s / %s"
            % (tag, bpb, bpb - CHANCE, matched, counted, tf, counted,
               rec["mean_rank"], ach, rec["band_free"], rec["band_tf"]))
        for g in ("G_U0", "G_U1"):
            if g in rec:
                log("                  %s: %s"
                    % (g, rec[g] if isinstance(rec[g], str) else rec[g]["verdict"]))

    set_lowrank(False)

    # ================================================== the gates
    A = out["arms"]
    voids = []
    if A.get("base", {}).get("G_U0") == "VOID":
        voids.append("base")
    for t in ("K133-LINEAR", "K133-ORACLE"):
        if A.get(t, {}).get("G_U1", {}).get("verdict") == "VOID":
            voids.append(t)
    out["VOID"] = voids
    out["G_U2_failures"] = [t for t, r in A.items() if not r.get("G_U2", True)]

    # G-U3: monotonicity is NOT assumed
    curve = [(k, A["K%d-LINEAR" % k]["teacher_forced"]) for k in K_BUDGET
             if ("K%d-LINEAR" % k) in A]
    gu3 = None
    if len(curve) == len(K_BUDGET):
        drops = [curve[i][1] - curve[i + 1][1] for i in range(len(curve) - 1)]
        worst = max(drops) if drops else 0
        gu3 = {"curve": curve, "steps": drops, "worst_drop": worst,
               "monotone_within_tol": bool(worst <= G_U3_TOL),
               "verdict": ("MONOTONE" if worst <= G_U3_TOL else
                           "NON-MONOTONIC -- the 'deeper is better' framing is withdrawn")}
    out["G_U3"] = gu3

    # G-U4: the verdict
    gu4 = None
    if len(curve) == len(K_BUDGET) and not voids:
        best_k, best_tf = max(curve, key=lambda p: p[1])
        if best_tf >= TF_LO:
            lab = "DEPTH-RECOVERS"
        elif best_tf >= 103:
            lab = "DEPTH-HELPS-NOT-ENOUGH"
        else:
            lab = "DEPTH-IS-NOT-THE-LEVER"
        gu4 = {"best_k": best_k, "best_tf": best_tf, "anchor_tf_k133": A_K133_LIN["tf"],
               "gain_over_k133": best_tf - A_K133_LIN["tf"],
               "verdict": ("SMOKE -- 4 layers, 2 prompts, 8 tokens: NOT A VERDICT"
                           if SMOKE else lab)}
    out["G_U4"] = gu4

    # the registered THIRD OUTCOME: is the 11.0 M router worth its budget?
    third = None
    if "K148-STATIC" in A and "K148-LINEAR" in A:
        st, ln = A["K148-STATIC"]["teacher_forced"], A["K148-LINEAR"]["teacher_forced"]
        third = {"tf_static": st, "tf_linear": ln, "gap": ln - st,
                 "within_prediction_4": bool(abs(ln - st) <= STATIC_TIE),
                 "static_ties_or_beats": bool(st >= ln),
                 "note": ("STATIC ties or beats the ridge: the 11.0 M router is not worth its "
                          "budget and this depth grid is obsolete on its own result"
                          if st >= ln else "the ridge earns its 11.0 M at this depth")}
    out["router_worth_its_budget"] = third

    lg1 = None
    if "K148-LOG1P" in A and "K148-LINEAR" in A:
        lg1 = {"tf_log1p": A["K148-LOG1P"]["teacher_forced"],
               "tf_sqrt": A["K148-LINEAR"]["teacher_forced"],
               "gain": A["K148-LOG1P"]["teacher_forced"] - A["K148-LINEAR"]["teacher_forced"],
               "bpb_log1p": A["K148-LOG1P"]["bpb"], "bpb_sqrt": A["K148-LINEAR"]["bpb"]}
    out["target_axis"] = lg1

    ceil156 = None
    if "K156-ORACLE" in A:
        ceil156 = {"tf_oracle_k156": A["K156-ORACLE"]["teacher_forced"],
                   "tf_oracle_k133": A_K133_ORC["tf"],
                   "tf_linear_k156": A.get("K156-LINEAR", {}).get("teacher_forced"),
                   "oracle_ridge_gap": (A["K156-ORACLE"]["teacher_forced"]
                                        - A["K156-LINEAR"]["teacher_forced"]
                                        if "K156-LINEAR" in A else None)}
    out["ceiling_at_deepest"] = ceil156
    out["seconds_total"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    # ================================================== the table
    log("")
    log("  %-14s %-4s %-10s %-9s %-9s %-8s %-8s %-9s %s"
        % ("arm", "k", "BPB", "free", "tf", "act", "G", "band-free", "band-tf"))
    for tag, k, _kd, _tg, _l in ARMS:
        if tag in A:
            r = A[tag]
            log("    %-14s %-4s %9.6f %4d/%-4d %4d/%-4d %7.4f %7s  %-9s %s"
                % (tag, k if k else "-", r["bpb"], r["matched"], r["counted"],
                   r["teacher_forced"], r["counted"], r["achieved_activation"],
                   ("%.4f" % r["charged_G"]) if r.get("charged_G") else "-",
                   r["band_free"], r["band_tf"]))
    log("")
    log("  VOID: %s" % (", ".join(voids) if voids else "none"))
    log("  G-U2 failures: %s" % (", ".join(out["G_U2_failures"]) or "none"))
    if gu3:
        log("  G-U3 depth curve (k, tf): %s  worst drop %d -> %s"
            % (gu3["curve"], gu3["worst_drop"], gu3["verdict"]))
    if gu4:
        log("  G-U4 best in-budget depth k=%d tf %d (k=133 read %d, %+d) -> %s"
            % (gu4["best_k"], gu4["best_tf"], gu4["anchor_tf_k133"],
               gu4["gain_over_k133"], gu4["verdict"]))
    if third:
        log("  router worth its budget: static %d vs ridge %d (gap %+d) -> %s"
            % (third["tf_static"], third["tf_linear"], third["gap"], third["note"]))
    if lg1:
        log("  target axis: log1p %d vs sqrt %d (%+d)"
            % (lg1["tf_log1p"], lg1["tf_sqrt"], lg1["gain"]))
    if ceil156:
        log("  ceiling at k=156: oracle %s, ridge %s, gap %s"
            % (ceil156["tf_oracle_k156"], ceil156["tf_linear_k156"],
               ceil156["oracle_ridge_gap"]))
    log("  wrote %s in %.0fs" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
