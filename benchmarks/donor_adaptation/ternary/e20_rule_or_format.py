#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E20 -- is it the RULE or the FORMAT?  A data-optimal ternary head.

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E20_RULE_OR_FORMAT.md (e5d438c),
pushed before any arm ran.

Every ternarization this programme has run is `scale = |w|.mean(dim=1); round; clamp(-1,1)` --
BitLinear-1.58 round-to-nearest, a weight-space rule that never looks at a token.  E20 holds the
FORMAT exactly fixed at what QWENDON1 stores (ternary codes, ONE fp32 scale per output row, same
bytes, same kernel) and changes only how the codes and scales are CHOSEN.

  R3H    scale = mean|w| per row, RTN                     no data   == T2b's H arm
  OPTH   scale searched to minimise ||w - s*q||^2         no data   isolates scale choice
  GPTQH  scale searched to minimise (w-w^)'H(w-w^),       DATA      the verdict arm
         with GPTQ error compensation across input columns

Env: D_THREADS (6), E20_ONLY (comma list), E20_SMOKE (1 = 2 prompts, 8 tokens, 2 eval seqs)
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
import d0_coactivation as DC                                # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E20_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E20_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e20_rule_or_format%s.json" % ("_smoke" if SMOKE else ""))
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453

ARMS = ["base", "ID", "R3H", "OPTH", "GPTQH"]
T2B_H_BPB = 1.106584          # T2b / E18: the H arm, quoted and never re-derived
T2B_H_AGREE = 9               # E18 part B
REPL_TOL_BPB = 1e-5

CHANCE = 4.069819
FLOOR, MARGIN = 12, 2
AT_FLOOR_MAX, RANKS_MIN = FLOOR + MARGIN, 80
NCAL, SEQCAL, SEEDCAL = 32, 512, 42424
LAM_FRAC = 0.01               # damping, fixed in the brief s2 -- not tuned after the fact
N_SCALE_GRID = 40             # scale search resolution, fixed here
GPTQ_BLOCK = 128              # GPTQ lazy-batch block, the reference value


def log(*a):
    print(*a, flush=True)


def band(m):
    if m <= AT_FLOOR_MAX:
        return "AT-FLOOR"
    if m >= RANKS_MIN:
        return "RANKS"
    return "PARTIAL"


# ============================================================ quantizers (format held FIXED)
def q_r3(W, H=None):
    """The shipped rule: per-output-row mean-|w| scale, round to nearest of {-1,0,+1}."""
    s = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    return (W / s).round().clamp(-1, 1) * s


def _best_scale_weightspace(W, grid):
    """Per row, the scale minimising ||w - s*q||^2 over a grid of multiples of mean|w|."""
    base = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    best = base.clone()
    besterr = torch.full((W.shape[0], 1), float("inf"))
    for g in grid:
        s = base * g
        q = (W / s).round().clamp(-1, 1)
        err = ((W - q * s) ** 2).sum(dim=1, keepdim=True)
        m = err < besterr
        besterr = torch.where(m, err, besterr)
        best = torch.where(m, s, best)
    return best


def q_opt(W, H=None):
    """Still no data: only the SCALE is chosen better than mean|w|."""
    grid = [0.5 + 0.05 * i for i in range(N_SCALE_GRID)]
    s = _best_scale_weightspace(W, grid)
    return (W / s).round().clamp(-1, 1) * s


def _best_scale_hessian(W, dH, grid):
    """Per row, the scale minimising the DIAGONALLY H-weighted error sum_j H_jj (w_j - s q_j)^2.
    The diagonal is used for the scale search only; GPTQ below handles the off-diagonal."""
    base = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    best = base.clone()
    besterr = torch.full((W.shape[0], 1), float("inf"))
    for g in grid:
        s = base * g
        q = (W / s).round().clamp(-1, 1)
        err = (((W - q * s) ** 2) * dH.unsqueeze(0)).sum(dim=1, keepdim=True)
        m = err < besterr
        besterr = torch.where(m, err, besterr)
        best = torch.where(m, s, best)
    return best


def q_gptq(W, H):
    """GPTQ / OBQ, the standard blocked form: quantize input column by column, pushing each
    column's error onto the not-yet-quantized columns through the inverse Hessian.  The FORMAT is
    unchanged -- the output is still ternary codes with ONE fp32 scale per output row; only WHICH
    code each weight gets, and which scale each row gets, is chosen differently."""
    W = W.clone().float()
    n_out, n_in = W.shape
    Hd = H.clone().float()

    dead = torch.diagonal(Hd) == 0
    if bool(dead.any()):
        Hd[dead, dead] = 1.0
        W[:, dead] = 0.0

    grid = [0.5 + 0.05 * i for i in range(N_SCALE_GRID)]
    s = _best_scale_hessian(W, torch.diagonal(Hd).clone(), grid)      # [n_out, 1], fixed upfront

    lam = LAM_FRAC * float(torch.diagonal(Hd).mean())
    Hd += torch.eye(n_in) * lam
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)

    Q = torch.zeros_like(W)
    sc = s[:, 0]
    for i1 in range(0, n_in, GPTQ_BLOCK):
        i2 = min(i1 + GPTQ_BLOCK, n_in)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi1 = Hinv[i1:i2, i1:i2]
        for k in range(i2 - i1):
            w = W1[:, k]
            q = (w / sc).round().clamp(-1, 1) * sc                     # ternary, per-row scale
            Q1[:, k] = q
            e = (w - q) / Hi1[k, k]
            W1[:, k:] -= e.unsqueeze(1) * Hi1[k, k:].unsqueeze(0)
            E1[:, k] = e
        Q[:, i1:i2] = Q1
        if i2 < n_in:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


QUANT = {"ID": None, "R3H": q_r3, "OPTH": q_opt, "GPTQH": q_gptq}


# ============================================================ apply / restore
def apply_head(model, tag, H):
    """Untie, clone, requantize the head.  tie_word_embeddings=True on this donor, so converting in
    place would silently convert the embedding too -- t2b_organs.py:149 does exactly this clone."""
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
    W0 = model.lm_head.weight.data.clone()
    fn = QUANT[tag]
    t = time.time()
    Wq = W0.clone() if fn is None else fn(W0, H)
    secs = time.time() - t
    model.lm_head.weight.data.copy_(Wq)
    zero = float((Wq == 0).float().mean())
    rel = float(torch.linalg.norm(Wq - W0) / torch.linalg.norm(W0))

    def restore():
        model.lm_head.weight.data.copy_(W0)
    return restore, {"quantize_seconds": secs, "zero_fraction": zero,
                     "rel_weight_error": rel,
                     "identical_to_donor": bool(torch.equal(Wq, W0))}


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

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]
    refids = {i: r["ids"][-N_NEW:] for i, r in enumerate(ref)}

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    assert model.config.vocab_size == 151936

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", DC.N_EVAL, DC.SEQ_LEN_EVAL, DC.SEED_EVAL)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    if SMOKE:
        ids_ev, byts_ev = ids_ev[:2], byts_ev[:2]
    B_TOT = float(byts_ev.sum())
    log("slice %dx%d, %d scored bytes  |  ids_sha256 OK"
        % (ids_ev.shape[0], ids_ev.shape[1], int(B_TOT)))

    e6eng = json.load(open(os.path.join(ENGDIR, "results", "e6", "engine.json"), encoding="utf-8"))
    pids = []
    for i, p in enumerate(prompts):
        q = tok(p)["input_ids"]
        assert q == e6eng["prompt_ids"][i], "prompt %d differs from E6's stored ids" % i
        pids.append(q)

    # ---- H = E[h h'] on the frozen calibration slice; h is the exact input to lm_head
    ids_cal, _, _ = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    if SMOKE:
        ids_cal = ids_cal[:2]
    d_model = model.config.hidden_size
    H = torch.zeros(d_model, d_model, dtype=torch.float64)
    ntok = 0
    box = {}

    def hook(mod, args):
        box["h"] = args[0].detach()
    hh = model.lm_head.register_forward_pre_hook(hook)
    log("== capturing head Hessian on %d calib tokens ==" % (ids_cal.shape[0] * ids_cal.shape[1]))
    t0 = time.time()
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            model(ids_cal[i:i + 1])
            h = box["h"].reshape(-1, d_model).double()
            H += h.T @ h
            ntok += h.shape[0]
    hh.remove()
    H /= float(ntok)
    H = H.float()
    ev = torch.linalg.eigvalsh(H.double())
    # E[h h'] is PSD by construction; a tiny negative eigenvalue here is float round-off in the
    # accumulation, so the RAW condition number is meaningless.  The number that governs the solve
    # is the DAMPED one, and that is what is reported.
    lam_rep = LAM_FRAC * float(torch.diagonal(H).double().mean())
    cond_damped = (float(ev[-1]) + lam_rep) / (float(ev[0]) + lam_rep)
    log("   H over %d tokens in %.0fs  |  eig min %.3e max %.3e  lambda %.4e  cond(damped) %.3e"
        % (ntok, time.time() - t0, float(ev[0]), float(ev[-1]), lam_rep, cond_damped))

    out = {"brief": "briefs/BRIEF_E20_RULE_OR_FORMAT.md (e5d438c)",
           "question": "is the constraint the RULE or the FORMAT? format held fixed at what "
                       "QWENDON1 stores: ternary codes, one fp32 scale per output row",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "calib_tokens": ntok,
           "hessian": {"damping_frac": LAM_FRAC, "scale_grid_points": N_SCALE_GRID,
                       "gptq_block": GPTQ_BLOCK,
                       "eig_min": float(ev[0]), "eig_max": float(ev[-1]),
                       "lambda": lam_rep, "cond_damped": cond_damped},
           "bands": {"floor_E18_partA": FLOOR, "margin_E17": MARGIN,
                     "at_floor_max": AT_FLOOR_MAX, "ranks_min": RANKS_MIN,
                     "ceiling": len(prompts) * n_new},
           "chance_bpb": CHANCE, "t2b_H_reference": {"bpb": T2B_H_BPB, "agree": T2B_H_AGREE},
           "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    base_ids, base_bpb = None, None
    todo = [a for a in ARMS if (not ONLY or a in ONLY or a == "base")]
    for tag in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            r = out["arms"][tag]
            log("  %-6s CACHED  %d/%d  BPB %.9f" % (tag, r["matched"], r["counted"], r["bpb"]))
            if tag == "base":
                base_ids, base_bpb = r["ids"], r["bpb"]
            continue
        t0 = time.time()
        if tag == "base":
            restore, st = (lambda: None), {"quantize_seconds": 0.0, "zero_fraction": 0.0,
                                           "rel_weight_error": 0.0, "identical_to_donor": True}
        else:
            restore, st = apply_head(model, tag, H)

        with torch.no_grad():
            nats = []
            for i in range(ids_ev.shape[0]):
                ch = ids_ev[i:i + 1]
                lg = model(ch).logits.float()
                lp = torch.nn.functional.log_softmax(lg[:, :-1], dim=-1)
                nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double())
        bpb = float(torch.stack(nats).sum() / (LN2 * B_TOT))

        allids, matched, counted, first_div, per_prompt = [], 0, 0, None, []
        for i in range(len(prompts)):
            ids = greedy(model, pids[i], n_new)
            new = ids[-n_new:]
            allids.append(new)
            theirs = refids[i][:n_new]
            m = sum(1 for a, b in zip(new, theirs) if a == b)
            d = next((j for j, (a, b) in enumerate(zip(new, theirs)) if a != b), None)
            if d is not None and first_div is None:
                first_div = [i, d]
            per_prompt.append({"prompt": i, "matched": m, "of": n_new, "diverges_at": d})
            matched += m
            counted += n_new
        restore()

        rec = {"arm": tag, "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "agree": matched / float(counted),
               "first_div": first_div, "per_prompt": per_prompt, "ids": allids,
               "seconds": time.time() - t0, "text": [tok.decode(x) for x in allids]}
        rec.update(st)
        if tag == "base":
            base_ids, base_bpb = allids, bpb
            rec["G_Q0"] = "FIRES" if matched == counted else "VOID"
        elif tag == "ID":
            same = (allids == base_ids)
            rec["G_Q1_token_identical"] = bool(same)
            rec["G_Q1_bpb_diff"] = abs(bpb - base_bpb)
            rec["G_Q1"] = "FIRES" if (same and abs(bpb - base_bpb) == 0.0) else "VOID"
        elif tag == "R3H" and not SMOKE:
            db = abs(bpb - T2B_H_BPB)
            rec["G_Q2"] = {"t2b_bpb": T2B_H_BPB, "here": bpb, "abs_diff": db,
                           "bpb_ok": bool(db < REPL_TOL_BPB),
                           "e18_agree": T2B_H_AGREE, "agree_here": matched,
                           "agree_ok": bool(matched == T2B_H_AGREE)}
        if tag not in ("base", "ID"):
            rec["G_Q3"] = band(matched) if not SMOKE else "NOT-APPLICABLE (smoke)"

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-6s BPB %.6f (%+.6f)  zero %.4f  relerr %.4f  %3d/%-3d %6.2f%%  div %-8s %s [%.0fs]"
            % (tag, bpb, bpb - CHANCE, st["zero_fraction"], st["rel_weight_error"],
               matched, counted, 100.0 * matched / counted, str(first_div),
               rec.get("G_Q3", rec.get("G_Q0", rec.get("G_Q1", ""))), rec["seconds"]))

    A = out["arms"]
    void = []
    if "base" in A and A["base"].get("G_Q0") != "FIRES":
        void.append("G-Q0: base did not reproduce the reference")
    if "ID" in A and A["ID"].get("G_Q1") != "FIRES":
        void.append("G-Q1: the requantize path is not lossless under the identity quantizer")
    if "R3H" in A and "G_Q2" in A["R3H"]:
        g = A["R3H"]["G_Q2"]
        if not g["bpb_ok"] or not g["agree_ok"]:
            void.append("G-Q2: R3H did not replicate T2b/E18 (bpb_ok=%s agree_ok=%s)"
                        % (g["bpb_ok"], g["agree_ok"]))
    out["VOID"] = void

    # G-Q4: sanity on the objective the quantizers optimise -- reported, never a verdict
    if all(t in A for t in ("R3H", "OPTH", "GPTQH")):
        b = {t: A[t]["bpb"] for t in ("R3H", "OPTH", "GPTQH")}
        out["G_Q4"] = {"bpb": b,
                       "gptq_beats_opt": bool(b["GPTQH"] < b["OPTH"]),
                       "opt_beats_or_equals_r3": bool(b["OPTH"] <= b["R3H"]),
                       "gptq_margin_over_r3": b["R3H"] - b["GPTQH"],
                       "holds": bool(b["GPTQH"] < b["OPTH"] <= b["R3H"]),
                       "note": "sanity, not a verdict: a quantizer that cannot win on the "
                               "objective it optimises has not been demonstrated to work"}
    out["G_Q3"] = {t: A[t]["G_Q3"] for t in A if "G_Q3" in A[t]}
    out["seconds_total"] = time.time() - t_start

    log("\n  bands: floor %d, margin %d, AT-FLOOR <= %d, RANKS >= %d, ceiling %d"
        % (FLOOR, MARGIN, AT_FLOOR_MAX, RANKS_MIN, len(prompts) * n_new))
    if "G_Q4" in out:
        log("  G_Q4  %s" % json.dumps(out["G_Q4"]["bpb"]))
        log("        holds=%s  GPTQH beats R3H by %+.6f BPB"
            % (out["G_Q4"]["holds"], out["G_Q4"]["gptq_margin_over_r3"]))
    log("  G_Q3  %s" % json.dumps(out["G_Q3"]))
    log("\n  arm ladder (how much the quantizer knows -> BPB vs chance -> agreement):")
    for tag in ARMS:
        if tag in A:
            r = A[tag]
            log("    %-6s %+.6f  ->  %3d/%d" % (tag, r["vs_chance"], r["matched"], r["counted"]))
    log("\n  VOID: %s" % (", ".join(void) if void else "none"))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("wrote %s  [%.0fs total]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
