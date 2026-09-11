#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E29 -- does the residual RANK layers, or is E27's finding just "don't touch the ends"?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E29_DOES_THE_RESIDUAL_RANK.md, pushed at
d804b88 before this file existed.

THIS IS A CONTROL ON E27'S BEST RESULT AND IT IS BUILT TO BE ABLE TO KILL IT.

E27 reported L21-MINRES 113/160 against L21-LAST 57/160 at identical depth, parameter count and
charged cost, and I put that in the INDEX and proposed it to the T4 proposal.  The comparison is
not fair: LAST drops 21-27 and layer 27 carries the SECOND-HIGHEST residual of all 28 (0.7618
against a 12-18 band of 0.28-0.36), so LAST is a rule that deletes the most active block in the
back half.  Beating it establishes almost nothing.

THE THREE-POINT TEST.  Same 7 layers dropped from the same interior band [3, n-4], so depth,
parameter count and charged cost are IDENTICAL and only the selection moves:
    MINRES      the 7 smallest residuals -- E27's rule, unchanged (all seven already lie in the
                band, so the restriction does not re-tune it)
    RANDMID     7 drawn uniformly at random from the band, three seeds -- THE NULL
    MAXRES-IN   the 7 largest residuals inside the band -- THE ANTI-RULE, which must FAIL
"the residual ranks layers" predicts MAXRES-IN < RANDMID < MINRES.  The outcome that matters is
RANDMID ~= MINRES, which would mean the residual carries no information.

MAXRES-IN is restricted to the interior on purpose: the global 7 largest would include layer 0
(residual 19.05, the block that builds the representation), and demolishing the model with that
proves nothing anyone doubted.  Restricting it makes it a FAIR anti-rule.

NO TIMING.  Quality is deterministic and may be measured under load, which is why this runs while
the box is busy and E28's timing cannot.

  python e29_depth_rule.py
  E29_SMOKE=1 python e29_depth_rule.py         # 8 layers, 2 prompts, every code path
  E29_ONLY=L21-RANDMID-s1 python e29_depth_rule.py
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
sys.path.insert(0, DENSDIR)
sys.path.insert(0, HERE)
sys.path.insert(0, ENGDIR)

import common as C                                          # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402
import e21_rank as E21                                      # noqa: E402
# E27's machinery is IMPORTED, never reimplemented: if the anchors do not come back
# bit-identical then something moved underneath and no new arm here is readable.
from e27_floor import (Depth, measure_residuals, keep_set, band_free, band_tf,   # noqa: E402
                       active, t10_transpose, CHANCE, BASE_BPB, REPL_TOL,
                       TF_LO, TF_HI, NCAL, SEQCAL, SEEDCAL, EXPECT_IDS_SHA, HF)

SMOKE = os.environ.get("E29_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E29_ONLY", "").split(",") if x.strip()]
OUT = os.path.join(ENGDIR, "results", "e29_depth_rule%s.json" % ("_smoke" if SMOKE else ""))
E27OUT = os.path.join(ENGDIR, "results", "e27_floor.json")
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
LN2 = float(np.log(2.0))

MARGIN = 3                  # the interior band is [MARGIN, n-1-MARGIN]
RANDMID_NEAR = 8            # brief s5: within this many tf tokens of MINRES kills the finding

# E27's own rows, hardcoded AND verified against e27_floor.json on disk before the model loads.
A_L21_MINRES = {"bpb": 0.9934455803226115, "free": 14, "tf": 113}
A_L21_LAST = {"bpb": 2.5000829558275610, "free": 1, "tf": 57}

#        tag                drop  rule         seed
ARMS = [("base",              0,   None,        None),
        ("L21-MINRES",        7,   "MINRES",    None),
        ("L21-LAST",          7,   "LAST",      None),
        ("L21-RANDMID-s1",    7,   "RANDMID",   1),
        ("L21-RANDMID-s2",    7,   "RANDMID",   2),
        ("L21-RANDMID-s3",    7,   "RANDMID",   3),
        ("L21-MAXRES-IN",     7,   "MAXRES-IN", None),
        ("L21-NOADJ",         7,   "NOADJ",     None),
        ("L18-MINRES",        10,  "MINRES",    None),
        ("L17-MINRES",        11,  "MINRES",    None)]

SMOKE_DROP = {0: 0, 7: 2, 10: 3, 11: 3}


def log(*a):
    print(*a, flush=True)
    sys.stdout.flush()


def band(n):
    """The interior: the layers that are candidates for dropping in every E29 condition."""
    return list(range(MARGIN, n - MARGIN))


def keep_set29(n, drop, rule, residual, seed=None):
    """MINRES and LAST delegate to E27's own function so the anchors run E27's code path.
    The three new rules all draw from the SAME interior band, so every 7-drop arm has
    identical depth, parameter count and charged cost and only the selection differs."""
    if drop <= 0:
        return list(range(n)), None
    if rule in ("MINRES", "LAST"):
        keep = keep_set(n, drop, rule, residual)
        return keep, None

    cand = band(n)
    if len(cand) < drop:
        raise SystemExit("interior band has %d layers, cannot drop %d" % (len(cand), drop))

    if rule == "RANDMID":
        rnd = random.Random(seed)
        gone = sorted(rnd.sample(cand, drop))
        note = "uniform from the interior band [%d,%d], seed %s" % (cand[0], cand[-1], seed)
    elif rule == "MAXRES-IN":
        order = sorted(cand, key=lambda i: (-residual[i], i))
        gone = sorted(order[:drop])
        note = "the %d LARGEST residuals inside [%d,%d]" % (drop, cand[0], cand[-1])
    elif rule == "NOADJ":
        # ascending residual over the whole stack, but never two adjacent -- forces a scattered
        # set where MINRES happened to produce a contiguous block.
        order = sorted(range(n), key=lambda i: (residual[i], i))
        gone = []
        for i in order:
            if len(gone) == drop:
                break
            if all(abs(i - g) > 1 for g in gone):
                gone.append(i)
        if len(gone) < drop:
            raise SystemExit("NOADJ could not find %d non-adjacent layers" % drop)
        gone = sorted(gone)
        note = "ascending residual, no two adjacent"
    else:
        raise SystemExit("unknown E29 rule %r" % (rule,))

    return [i for i in range(n) if i not in set(gone)], {"dropped": gone, "note": note}


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS

    # ---- the anchors must match E27's own record on disk, or they are not anchors
    anchor_check = None
    if os.path.exists(E27OUT):
        a27 = json.load(open(E27OUT, encoding="utf-8"))["arms"]
        anchor_check = {}
        for tag, lit in (("L21-MINRES", A_L21_MINRES), ("L21-LAST", A_L21_LAST)):
            r = a27[tag]
            ok = (abs(r["bpb"] - lit["bpb"]) < REPL_TOL and r["matched"] == lit["free"]
                  and r["teacher_forced"] == lit["tf"])
            anchor_check[tag] = {"bpb_on_disk": r["bpb"], "free_on_disk": r["matched"],
                                 "tf_on_disk": r["teacher_forced"], "matches_literal": bool(ok)}
            if not ok:
                raise SystemExit("ANCHOR MISMATCH for %s vs %s -- STOP" % (tag, E27OUT))
        log("== E27 anchors verified against %s ==" % os.path.basename(E27OUT))

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    cfg = model.config
    D = cfg.hidden_size
    HDIM = D // cfg.num_attention_heads
    QD = cfg.num_attention_heads * HDIM
    KD = cfg.num_key_value_heads * HDIM
    F = cfg.intermediate_size
    V = cfg.vocab_size
    if SMOKE:
        model.model.layers = nn.ModuleList(list(model.model.layers)[:8])
        model.config.num_hidden_layers = 8
    depth = Depth(model)
    NL = depth.n

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", 24, 512, 1234)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    if SMOKE:
        ids_ev, byts_ev = ids_ev[:2], byts_ev[:2]
    B_TOT = float(byts_ev.sum())
    log("slice %dx%d, %d scored bytes | ids_sha256 OK  (G-E29e)"
        % (ids_ev.shape[0], ids_ev.shape[1], int(B_TOT)))

    e6eng = json.load(open(os.path.join(ENGDIR, "results", "e6", "engine.json"), encoding="utf-8"))
    pids = []
    for i, p in enumerate(prompts):
        q = tok(p)["input_ids"]
        assert q == e6eng["prompt_ids"][i], "prompt %d differs from E6's stored ids" % i
        pids.append(q)
    tgts = [ref[i]["ids"][-N_NEW:][:n_new] for i in range(len(prompts))]

    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    assert meta_cal["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "calib and eval must be DIFFERENT corpus halves"
    if SMOKE:
        ids_cal = ids_cal[:2]

    todo = [a for a in ARMS if (not ONLY or a[0] in ONLY or a[0] == "base")]

    log("== MINRES: mean relative residual per layer, one pass (E27's function) ==")
    t0 = time.time()
    residual = measure_residuals(model, ids_cal)
    log("   done in %.0fs" % (time.time() - t0))

    # ---- G-E29d: the profile must reproduce E27's
    gd = None
    if not SMOKE and os.path.exists(E27OUT):
        r27 = json.load(open(E27OUT, encoding="utf-8"))["minres_residual"]
        worst = max(abs(a - b) for a, b in zip(residual, r27))
        gd = {"worst_abs_diff": worst, "tol": 1e-6, "fires": bool(worst < 1e-6)}
        log("   G-E29d: profile vs E27, worst abs diff %.3e -> %s"
            % (worst, "FIRES" if gd["fires"] else "MISMATCH"))
        if not gd["fires"]:
            raise SystemExit("G-E29d MISMATCH -- the residual profile moved, STOP")

    cand = band(NL)
    log("   interior band [%d,%d], %d candidates; dropping 7 leaves depth/cost identical"
        % (cand[0], cand[-1], len(cand)))
    for rule, seed in (("MINRES", None), ("MAXRES-IN", None), ("NOADJ", None),
                       ("RANDMID", 1), ("RANDMID", 2), ("RANDMID", 3)):
        d = SMOKE_DROP.get(7, 7) if SMOKE else 7
        k, nt = keep_set29(NL, d, rule, residual, seed)
        gone = [i for i in range(NL) if i not in set(k)]
        log("     %-12s%s drops %s" % (rule, ("-s%d" % seed) if seed else "    ", gone))

    donor_full = active(D, QD, KD, F, NL, V, None, None)

    out = {"brief": "briefs/BRIEF_E29_DOES_THE_RESIDUAL_RANK.md (d804b88)",
           "question": "E27 reported MINRES beating LAST by 56 tf tokens, but LAST deletes layer "
                       "27 which carries the second-highest residual of all 28.  Does the "
                       "residual RANK layers, or is the finding just 'don't touch the ends'?",
           "model": HF, "smoke": SMOKE, "n_new": n_new, "n_prompts": len(prompts),
           "eval_slice": meta_ev, "calib_slice": meta_cal,
           "shape": {"D": D, "QD": QD, "KD": KD, "F": F, "L": NL, "V": V},
           "interior_band": [cand[0], cand[-1]], "margin": MARGIN,
           "randmid_near_bar": RANDMID_NEAR,
           "anchors": {"base_bpb": BASE_BPB, "L21-MINRES": A_L21_MINRES,
                       "L21-LAST": A_L21_LAST, "repl_tol": REPL_TOL,
                       "verified_against_e27_json": anchor_check},
           "minres_residual": residual, "G_E29d": gd,
           "bars": {"tf_comparable": [TF_LO, TF_HI]},
           "chance_bpb": CHANCE, "donor_full_active": donor_full, "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    for tag, drop, rule, seed in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            rr = out["arms"][tag]
            log("  %-16s CACHED  bpb %.6f  free %d  tf %d"
                % (tag, rr["bpb"], rr["matched"], rr["teacher_forced"]))
            continue
        t0 = time.time()
        d = SMOKE_DROP.get(drop, drop) if SMOKE else drop

        keep, note = keep_set29(NL, d, rule, residual, seed) if rule else \
            (list(range(NL)), None)
        keep = depth.set(keep) if rule else list(range(NL))
        n_keep = len(keep)

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
            dv = next((j for j, (a, b) in enumerate(zip(new, tgts[i])) if a != b), None)
            if dv is not None and first_div is None:
                first_div = [i, dv]
            per_prompt.append({"prompt": i, "free": m, "diverges_at": dv})
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

        depth.restore()

        counted = len(prompts) * n_new
        don = active(D, QD, KD, F, n_keep, V, None, None)
        t10 = t10_transpose(D, None, None, n_keep, NL)
        rec = {"arm": tag, "n_dropped": d, "depth_rule": rule, "seed": seed,
               "rule_note": note, "layers_kept": keep if rule else None,
               "layers_dropped": ([i for i in range(NL) if i not in set(keep)] if rule else []),
               "n_layers": n_keep, "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "agree": matched / float(counted),
               "teacher_forced": tf, "tf_frac": tf / float(counted),
               "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
               "rank_le5": int(sum(1 for r in ranks if r <= 5)),
               "donor_active": don, "donor_active_G": don["total"] / 1e9,
               "T10": t10, "first_div": first_div, "per_prompt": per_prompt,
               "band_free": band_free(matched), "band_tf": band_tf(tf),
               "ids": allids, "text": [tok.decode(x) for x in allids],
               "seconds": time.time() - t0}

        # ---- the replication gates
        if tag == "base":
            rec["G_E29a"] = ("SMOKE" if SMOKE else
                             ("FIRES" if (matched == counted and tf == counted
                                          and abs(bpb - BASE_BPB) < REPL_TOL) else "VOID"))
        for gname, lit in (("G_E29b", ("L21-MINRES", A_L21_MINRES)),
                           ("G_E29c", ("L21-LAST", A_L21_LAST))):
            if tag == lit[0] and not SMOKE:
                rec[gname] = {"bpb_here": bpb, "bpb_anchor": lit[1]["bpb"],
                              "abs_diff": abs(bpb - lit[1]["bpb"]),
                              "free_here": matched, "free_anchor": lit[1]["free"],
                              "tf_here": tf, "tf_anchor": lit[1]["tf"],
                              "verdict": ("FIRES" if (abs(bpb - lit[1]["bpb"]) < REPL_TOL
                                                      and matched == lit[1]["free"]
                                                      and tf == lit[1]["tf"]) else "VOID")}

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-16s BPB %.6f  free %3d/%d  tf %3d/%d  L %2d  drop %-22s %s"
            % (tag, bpb, matched, counted, tf, counted, n_keep,
               str(rec["layers_dropped"]), rec["band_tf"]))

    # ================================================== the three-point test
    A = out["arms"]
    verdict = None
    need = ["L21-MINRES", "L21-MAXRES-IN", "L21-RANDMID-s1", "L21-RANDMID-s2", "L21-RANDMID-s3"]
    if all(t in A for t in need):
        mr = A["L21-MINRES"]["teacher_forced"]
        mx = A["L21-MAXRES-IN"]["teacher_forced"]
        rs = [A["L21-RANDMID-s%d" % i]["teacher_forced"] for i in (1, 2, 3)]
        rm = sum(rs) / 3.0
        gap = mr - rm
        ordered = mx < rm < mr
        kills = gap <= RANDMID_NEAR
        verdict = {
            "MINRES_tf": mr, "RANDMID_tf": rs, "RANDMID_mean": rm,
            "RANDMID_spread": max(rs) - min(rs), "MAXRES_IN_tf": mx,
            "gap_MINRES_minus_RANDMID": gap, "near_bar": RANDMID_NEAR,
            "ordering_MAXRES_lt_RAND_lt_MINRES": bool(ordered),
            "name": ("SMOKE -- NOT A VERDICT" if SMOKE else
                     ("RESIDUAL-DOES-NOT-RANK" if kills else
                      ("RESIDUAL-RANKS" if ordered else "MIXED")))}
        log("")
        log("  == THE THREE-POINT TEST, same 7 layers from the same band ==")
        log("     MAXRES-IN  tf %3d   (the anti-rule: must fail)" % mx)
        log("     RANDMID    tf %s  mean %.1f  spread %d   (the null)"
            % (rs, rm, max(rs) - min(rs)))
        log("     MINRES     tf %3d   (E27's rule)" % mr)
        log("     MINRES - RANDMID = %+.1f tokens   (bar: <= %d kills the finding)"
            % (gap, RANDMID_NEAR))
        log("     ordering MAXRES < RANDMID < MINRES: %s" % ordered)
        log("     -> %s" % verdict["name"])
        if kills and not SMOKE:
            log("")
            log("     *** THE REGISTERED ALTERNATIVE FIRES.  The residual does not rank layers.")
            log("         E27 s3's headline is an artifact of comparing against a bad rule, the")
            log("         T4 proposal's L21-MINRES recommendation is WITHDRAWN, and what")
            log("         survives is: a quarter of the layers can go PROVIDED THE FIRST AND")
            log("         LAST FEW ARE SPARED. ***")
    out["three_point_test"] = verdict

    out["seconds_total"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("wrote %s  [%.0fs]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    sys.exit(main())
