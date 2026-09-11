#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E21 -- can RANK buy what precision and sparsity could not?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E21_CAN_RANK_BUY_IT.md (24b8832),
pushed before any arm ran.  Nothing here is exported and NO TIMING IS TAKEN.

E19: every carve built here is FFN-only and FFN-only carving cannot reach 50 tok/s at any depth,
because on the 7B `attn+head` alone are 1.367 G against a 0.982-1.060 G budget.  Cutting attention
and the head has never been attempted.  Brief section 2 prices it: at rank 256 they fall to
0.245 G.  This runner measures whether that configuration is usable.

The verdict arms are ACTIVATION-WEIGHTED low-rank, minimising ||(W - Wr) X||_F rather than
||W - Wr||_F, because D2's spectra say these matrices are not low-rank in weight space (q_proj
needs r=425-547 for 90% energy) while D4 and E20 say the data covariance is extremely anisotropic.
Plain SVD is carried as the control for exactly that contrast.

  Wr = W H^(1/2) Br Br' H^(-1/2),  Br = top-r eigenvectors of H^(1/2) W'W H^(1/2)

which is the exact minimiser at rank r, and reduces to plain SVD when H = I.  Computed through the
1536x1536 Gram rather than a 151936x1536 SVD -- same answer, far cheaper.

Env: D_THREADS (6), E21_ONLY (comma list), E21_SMOKE (1)
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
sys.path.insert(0, HERE)
sys.path.insert(0, ENGDIR)

import common as C                                          # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E21_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E21_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e21_rank%s.json" % ("_smoke" if SMOKE else ""))
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453

CHANCE = 4.069819
FLOOR, MARGIN = 12, 2
AT_FLOOR_MAX, RANKS_MIN = FLOOR + MARGIN, 80
TF_TERNARY_LO, TF_TERNARY_HI = 107, 119     # E20 part B: eight ternary heads, measured
NCAL, SEQCAL, SEEDCAL = 32, 512, 42424
LAM_FRAC = 0.01                             # E20's damping, unchanged

# tag -> (organs, rank, weighted)   organs: "head", "qo", "both"
ARMS = [
    ("base",         None,   None, None),
    ("FULL",         "head", None, False),   # planted control: r = full, plain
    ("H-SVD-256",    "head",  256, False),
    ("H-SVD-512",    "head",  512, False),
    ("H-ACT-256",    "head",  256, True),
    ("H-ACT-512",    "head",  512, True),
    ("QO-SVD-256",   "qo",    256, False),
    ("QO-ACT-256",   "qo",    256, True),
    ("QO-ACT-512",   "qo",    512, True),
    ("BOTH-ACT-256", "both",  256, True),
]


def log(*a):
    print(*a, flush=True)


def band_free(m):
    if m <= AT_FLOOR_MAX:
        return "AT-FLOOR"
    if m >= RANKS_MIN:
        return "RANKS"
    return "PARTIAL"


def band_tf(m):
    if m > TF_TERNARY_HI:
        return "RANK-IS-CHEAPER"
    if m >= TF_TERNARY_LO:
        return "RANK-IS-COMPARABLE"
    return "RANK-IS-WORSE"


# ============================================================ the construction
def _sqrt_pair(H):
    """H^(1/2) and H^(-1/2) from the damped eigendecomposition, damping fixed at E20's."""
    Hd = H.double()
    lam = LAM_FRAC * float(torch.diagonal(Hd).mean())
    Hd = Hd + torch.eye(Hd.shape[0], dtype=torch.float64) * lam
    ev, U = torch.linalg.eigh(Hd)
    ev = ev.clamp_min(1e-12)
    s = ev.sqrt()
    return (U * s).mm(U.T), (U * (1.0 / s)).mm(U.T), lam, float(ev[-1] / ev[0])


def lowrank(W, H, r, weighted):
    """Wr at rank r; exact minimiser of ||(W - Wr) H^(1/2)||_F.  Returns (Wr, diagnostics)."""
    Wd = W.double()
    n_in = Wd.shape[1]
    r = min(r, n_in) if r is not None else n_in
    if weighted:
        Hh, Hi, lam, cond = _sqrt_pair(H)
    else:
        Hh = torch.eye(n_in, dtype=torch.float64)
        Hi = Hh
        lam, cond = 0.0, 1.0
    G = Hh.mm(Wd.T.mm(Wd)).mm(Hh)                  # [n_in, n_in], symmetric PSD
    G = 0.5 * (G + G.T)
    ev, B = torch.linalg.eigh(G)                   # ascending
    Br = B[:, n_in - r:]                           # top r
    T = Hh.mm(Br).mm(Br.T).mm(Hi)
    Wr = Wd.mm(T)
    tot = float(ev.clamp_min(0).sum())
    kept = float(ev[n_in - r:].clamp_min(0).sum())
    return Wr.float(), {"rank": r, "energy_kept_weighted": (kept / tot) if tot > 0 else None,
                        "damping": lam, "H_cond_damped": cond,
                        "rel_weight_error": float(torch.linalg.norm(Wr.float() - W)
                                                  / torch.linalg.norm(W))}


# ============================================================ apply / restore
def untie(model):
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())


def apply_arm(model, organs, r, weighted, Hs, n_layers=None):
    saved, st = [], {"n": 0, "rel": [], "energy": [], "params_dense": 0, "params_factored": 0}
    t0 = time.time()

    def do(mod, key):
        W = mod.weight.data
        saved.append((mod, W.clone()))
        Wr, d = lowrank(W, Hs[key], r, weighted)
        mod.weight.data.copy_(Wr)
        st["n"] += 1
        st["rel"].append(d["rel_weight_error"])
        if d["energy_kept_weighted"] is not None:
            st["energy"].append(d["energy_kept_weighted"])
        st["params_dense"] += W.shape[0] * W.shape[1]
        st["params_factored"] += d["rank"] * (W.shape[0] + W.shape[1])
        st.setdefault("damping", d["damping"])
        st.setdefault("H_cond_damped", d["H_cond_damped"])

    if organs in ("head", "both"):
        untie(model)
        do(model.lm_head, ("head", "lm_head"))
    if organs in ("qo", "both"):
        for li, lay in enumerate(model.model.layers):
            if n_layers is not None and li >= n_layers:
                break
            do(lay.self_attn.q_proj, (li, "q_proj"))
            do(lay.self_attn.o_proj, (li, "o_proj"))

    st["seconds"] = time.time() - t0
    st["rel_weight_error"] = sum(st["rel"]) / len(st["rel"]) if st["rel"] else 0.0
    st["energy_kept_weighted"] = (sum(st["energy"]) / len(st["energy"])) if st["energy"] else None
    st["saving_vs_dense"] = (1.0 - st["params_factored"] / float(st["params_dense"])
                             if st["params_dense"] else 0.0)
    del st["rel"], st["energy"]

    def restore():
        for mod, W in saved:
            mod.weight.data.copy_(W)
    return restore, st


def greedy(model, prompt_ids, n_new):
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(n_new):
            lg = model(torch.tensor([ids])).logits[0, -1].float()
            ids.append(int(torch.argmax(lg)))
    return ids


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS
    n_layers = 4 if SMOKE else None

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]
    refids = {i: r["ids"][-N_NEW:] for i, r in enumerate(ref)}

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()

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

    # ---- one calibration pass: H = X'X for lm_head and for every q_proj / o_proj
    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    assert meta_cal["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "calib and eval must be DIFFERENT corpus halves"
    if SMOKE:
        ids_cal = ids_cal[:2]
    Hs, cnt, hooks = {}, {}, []

    def mk(key):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            Hs[key] = (x.T @ x) if key not in Hs else Hs[key] + (x.T @ x)
            cnt[key] = cnt.get(key, 0) + x.shape[0]
        return f
    hooks.append(model.lm_head.register_forward_hook(mk(("head", "lm_head"))))
    for li, lay in enumerate(model.model.layers):
        if n_layers is not None and li >= n_layers:
            break
        hooks.append(lay.self_attn.q_proj.register_forward_hook(mk((li, "q_proj"))))
        hooks.append(lay.self_attn.o_proj.register_forward_hook(mk((li, "o_proj"))))
    log("== calibration pass: H for %d organs, %d tokens =="
        % (len(hooks), ids_cal.shape[0] * ids_cal.shape[1]))
    t0 = time.time()
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            model(ids_cal[i:i + 1])
    for h in hooks:
        h.remove()
    log("   done in %.0fs, %d H matrices, %d tokens each"
        % (time.time() - t0, len(Hs), cnt[("head", "lm_head")]))

    out = {"brief": "briefs/BRIEF_E21_CAN_RANK_BUY_IT.md (24b8832)",
           "question": "can RANK buy what precision and sparsity could not? nothing exported, "
                       "no timing taken, 6.79 tok/s untouched",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "eval_slice": meta_ev, "calib_slice": meta_cal,
           "calib_tokens": cnt[("head", "lm_head")], "damping_frac": LAM_FRAC,
           "bands": {"free": {"floor": FLOOR, "at_floor_max": AT_FLOOR_MAX,
                              "ranks_min": RANKS_MIN, "ceiling": len(prompts) * n_new},
                     "teacher_forced_from_E20": [TF_TERNARY_LO, TF_TERNARY_HI]},
           "chance_bpb": CHANCE, "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    base_ids, base_bpb = None, None
    todo = [a for a in ARMS if (not ONLY or a[0] in ONLY or a[0] == "base")]
    for tag, organs, r, weighted in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            rr = out["arms"][tag]
            log("  %-13s CACHED  free %d  tf %d" % (tag, rr["matched"], rr["teacher_forced"]))
            if tag == "base":
                base_ids, base_bpb = rr["ids"], rr["bpb"]
            continue
        t0 = time.time()
        if tag == "base":
            restore, st = (lambda: None), {"n": 0, "rel_weight_error": 0.0,
                                           "energy_kept_weighted": None, "saving_vs_dense": 0.0,
                                           "seconds": 0.0}
        else:
            restore, st = apply_arm(model, organs, r, weighted, Hs, n_layers)

        with torch.no_grad():
            nats = []
            for i in range(ids_ev.shape[0]):
                ch = ids_ev[i:i + 1]
                lg = model(ch).logits.float()
                lp = torch.nn.functional.log_softmax(lg[:, :-1], dim=-1)
                nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double())
        bpb = float(torch.stack(nats).sum() / (LN2 * B_TOT))

        # free-running
        allids, matched, first_div, per_prompt = [], 0, None, []
        for i in range(len(prompts)):
            new = greedy(model, pids[i], n_new)[-n_new:]
            allids.append(new)
            m = sum(1 for a, b in zip(new, tgts[i]) if a == b)
            d = next((j for j, (a, b) in enumerate(zip(new, tgts[i])) if a != b), None)
            if d is not None and first_div is None:
                first_div = [i, d]
            per_prompt.append({"prompt": i, "free": m, "diverges_at": d})
            matched += m

        # teacher-forced (E20 part B's instrument)
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
        restore()

        counted = len(prompts) * n_new
        rec = {"arm": tag, "organs": organs, "rank": r, "weighted": weighted,
               "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "first_div": first_div,
               "teacher_forced": tf, "mean_rank_of_donor_token": sum(ranks) / float(len(ranks)),
               "rank_le_5": sum(1 for x in ranks if x <= 5),
               "per_prompt": per_prompt, "ids": allids, "seconds": time.time() - t0,
               "text": [tok.decode(x) for x in allids]}
        rec.update(st)
        if tag == "base":
            base_ids, base_bpb = allids, bpb
            rec["G_R0"] = "FIRES" if (matched == counted and tf == counted) else "VOID"
        elif tag == "FULL":
            same = (allids == base_ids)
            rec["G_R1_token_identical"] = bool(same)
            rec["G_R1_bpb_diff"] = abs(bpb - base_bpb)
            rec["G_R1"] = "FIRES" if (same and abs(bpb - base_bpb) < 1e-9) else "VOID"
        else:
            rec["G_R3_free"] = band_free(matched) if not SMOKE else "n/a (smoke)"
            rec["G_R4_tf"] = band_tf(tf) if not SMOKE else "n/a (smoke)"

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-13s BPB %.6f (%+.6f)  relerr %.4f  energy %s  free %3d/%-3d  tf %3d/%-3d  "
            "mrank %7.2f  %s [%.0fs]"
            % (tag, bpb, bpb - CHANCE, st["rel_weight_error"],
               ("%.4f" % st["energy_kept_weighted"]) if st["energy_kept_weighted"] else "  n/a ",
               matched, counted, tf, counted, rec["mean_rank_of_donor_token"],
               rec.get("G_R4_tf", rec.get("G_R0", rec.get("G_R1", ""))), rec["seconds"]))

    A = out["arms"]
    void = []
    if "base" in A and A["base"].get("G_R0") != "FIRES":
        void.append("G-R0: base did not reproduce E6's reference on both metrics")
    if "FULL" in A and A["FULL"].get("G_R1") != "FIRES":
        void.append("G-R1: the factorise-and-reassemble path is not lossless at full rank "
                    "(token_identical=%s, bpb_diff=%.3e)"
                    % (A["FULL"].get("G_R1_token_identical"), A["FULL"].get("G_R1_bpb_diff", -1)))
    out["VOID"] = void

    # G-R2: does activation weighting beat plain SVD at equal rank, in BPB?
    g2 = {}
    for a, b, nm in (("H-SVD-256", "H-ACT-256", "head r=256"),
                     ("H-SVD-512", "H-ACT-512", "head r=512"),
                     ("QO-SVD-256", "QO-ACT-256", "q/o r=256")):
        if a in A and b in A:
            g2[nm] = {"svd": A[a]["bpb"], "act": A[b]["bpb"], "act_minus_svd": A[b]["bpb"] - A[a]["bpb"],
                      "act_wins": bool(A[b]["bpb"] < A[a]["bpb"])}
    if g2:
        out["G_R2"] = {"pairs": g2, "holds": all(v["act_wins"] for v in g2.values()),
                       "note": "sanity, not a verdict: a construction that cannot win on the "
                               "objective it optimises has not been demonstrated to work"}
    out["seconds_total"] = time.time() - t_start

    log("\n  bands  free: floor %d, AT-FLOOR <= %d, RANKS >= %d, ceiling %d"
        % (FLOOR, AT_FLOOR_MAX, RANKS_MIN, len(prompts) * n_new))
    log("         teacher-forced (E20 part B, eight ternary heads): %d-%d = RANK-IS-COMPARABLE"
        % (TF_TERNARY_LO, TF_TERNARY_HI))
    log("\n  arm           BPB        free      tf        saving  band")
    for tag, _, _, _ in ARMS:
        if tag in A:
            rr = A[tag]
            log("    %-13s %.6f  %3d/%-3d  %3d/%-3d   %5.1f%%  %s"
                % (tag, rr["bpb"], rr["matched"], rr["counted"], rr["teacher_forced"],
                   rr["counted"], 100.0 * rr.get("saving_vs_dense", 0.0),
                   rr.get("G_R4_tf", "")))
    if "G_R2" in out:
        log("\n  G_R2 holds=%s" % out["G_R2"]["holds"])
        for k, v in out["G_R2"]["pairs"].items():
            log("       %-12s SVD %.6f  ACT %.6f  -> %+.6f" % (k, v["svd"], v["act"],
                                                               v["act_minus_svd"]))
    log("\n  VOID: %s" % (", ".join(void) if void else "none"))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("wrote %s  [%.0fs]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
