#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E20 -- is it the RULE or the FORMAT?  Every rule this repo owns, applied to the head.

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E20_RULE_OR_FORMAT.md (e5d438c),
pushed before any arm ran.  Run 1 (commit 6c7b7a4) is VOID: G-Q2 caught the brief's premise.

WHAT RUN 1 GOT WRONG.  BRIEF section 0 says every ternarization here is `mean|w|` round-to-nearest,
citing t1_ternarize.py:97-98.  That is R0.  The SHIPPED rule is `qwen_export.quantize` under
`--rule R3` = `t2_rules.r3_actsearch`, a per-row threshold search minimising the
ACTIVATION-RMS-WEIGHTED error over the calibration slice -- and qwen_export.py:148 hooks lm_head
too, so the shipped head quantization already looks at tokens.  And GPTQ is not untried:
`t2_rules.r4_gptq` has existed since T2, which ran it on the FFN (R4 4.299819, R5 2.027495).

WHAT RUN 2 DOES.  The format is still held EXACTLY fixed at what QWENDON1 stores -- ternary codes
with one fp32 scale per output row, same bytes, same kernel, 0.500000 B/weight.  Only the choice of
codes and scales changes.  Every rule is IMPORTED from t2_rules, never restated, and the shipped
arm is `t2b_organs.apply_arm(model, "H", ...)` itself, so the anchor gate compares like with like.

  R0H   mean|w|, RTN                               no data    = t1_ternarize.ternarize
  R1H   TWN, Delta = 0.7 E|w|                      no data
  R2H   threshold search, unweighted L2            no data
  R3H   threshold search, act-RMS-weighted L2      DATA       THE SHIPPED RULE (gate G-Q2)
  R4H   GPTQ, mean|w| scale                        DATA
  R5H   GPTQ, act-searched scale                   DATA       T2's best rule, never shipped
  OPTH  scale grid, unweighted L2, RTN             no data    E20's own, carried from run 1
  GPTQH GPTQ, H-diag-searched scale                DATA       E20's own, carried from run 1

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
import t2_rules as T2                                       # noqa: E402
from t2b_organs import capture, apply_arm                   # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E20_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E20_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e20_rules_on_the_head%s.json" % ("_smoke" if SMOKE else ""))
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453
HEADKEY = ("head", "lm_head")                 # t2b_organs.capture's own key for the head

ARMS = ["base", "ID", "R0H", "R1H", "R2H", "R3H", "R4H", "R5H", "OPTH", "GPTQH"]
KNOWS_DATA = {"R0H": False, "R1H": False, "R2H": False, "R3H": True,
              "R4H": True, "R5H": True, "OPTH": False, "GPTQH": True}

# The two published anchors for the shipped rule on the head, quoted and never re-derived:
T2B_H_BPB = 1.1065836079824596      # results/t2_arms -> t2b_organs.json, arm H
E18_H_AGREE = 9                     # results/e18_ranking_ladder.json, arm H
REPL_TOL_BPB = 1e-5

CHANCE = 4.069819                   # E12: log2(V)/bytes_per_token on this slice
FLOOR, MARGIN = 12, 2               # E18 part A floor, E17 margin -- fixed before run 1
AT_FLOOR_MAX, RANKS_MIN = FLOOR + MARGIN, 80
NCAL, SEQCAL, SEEDCAL = 32, 512, 42424
LAM_FRAC = 0.01
N_SCALE_GRID = 40
GPTQ_BLOCK = 128


def log(*a):
    print(*a, flush=True)


def band(m):
    if m <= AT_FLOOR_MAX:
        return "AT-FLOOR"
    if m >= RANKS_MIN:
        return "RANKS"
    return "PARTIAL"


# ====================================================== E20's own two quantizers (run-1 carryover)
def _best_scale(W, grid, dH=None):
    """Per row, the scale minimising sum_j w_j (w_j - s q_j)^2, weights dH or 1."""
    base = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    best, besterr = base.clone(), torch.full((W.shape[0], 1), float("inf"))
    for g in grid:
        s = base * g
        q = (W / s).round().clamp(-1, 1)
        e = (W - q * s) ** 2
        if dH is not None:
            e = e * dH.unsqueeze(0)
        err = e.sum(dim=1, keepdim=True)
        m = err < besterr
        besterr = torch.where(m, err, besterr)
        best = torch.where(m, s, best)
    return best


GRID = [0.5 + 0.05 * i for i in range(N_SCALE_GRID)]


def q_opth(W, H, rms):
    s = _best_scale(W, GRID)
    return (W / s).round().clamp(-1, 1), s


def q_gptqh(W, H, rms):
    """E20's own variant: GPTQ with the scale chosen against the Hessian DIAGONAL.
    The solve itself is t2_rules.r4_gptq -- one definition, as qwen_export states."""
    s = _best_scale(W, GRID, torch.diagonal(H).float().clone())
    return T2.r4_gptq(W, H, percdamp=LAM_FRAC, blocksize=GPTQ_BLOCK, alpha=s)


def q_r0(W, H, rms):
    return T2.r0_bitlinear(W)


def q_r1(W, H, rms):
    return T2.r1_twn(W)


def q_r2(W, H, rms):
    return T2.r2_search(W)


def q_r4(W, H, rms):
    return T2.r4_gptq(W, H, percdamp=LAM_FRAC, blocksize=GPTQ_BLOCK)


def q_r5(W, H, rms):
    _, a3 = T2.r3_actsearch(W, rms)                      # t2_rules.py:207, verbatim
    return T2.r4_gptq(W, H, percdamp=LAM_FRAC, blocksize=GPTQ_BLOCK, alpha=a3)


QUANT = {"ID": None, "R0H": q_r0, "R1H": q_r1, "R2H": q_r2,
         "R4H": q_r4, "R5H": q_r5, "OPTH": q_opth, "GPTQH": q_gptqh}


# ============================================================ apply / restore
def untie(model):
    """tie_word_embeddings=True here, so converting the head in place would convert the embedding
    too.  t2b_organs.py:149 does exactly this clone; apply_arm does it itself for its own arms."""
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())


def apply_head(model, tag, H, rms):
    untie(model)
    W0 = model.lm_head.weight.data.clone()
    fn = QUANT[tag]
    t = time.time()
    if fn is None:
        Wq, zero = W0.clone(), 0.0
    else:
        q, a = fn(W0, H, rms)
        Wq = (q * a).float()
        zero = float((q == 0).float().mean())
    secs = time.time() - t
    model.lm_head.weight.data.copy_(Wq)
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

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", 24, 512, 1234)
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

    # ---- one calibration pass: t2b's act_rms (all organs incl. head) + H = X'X for the head.
    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    assert meta_cal["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "calib and eval must be DIFFERENT corpus halves"
    if SMOKE:
        ids_cal = ids_cal[:2]
    d_model = model.config.hidden_size
    acc = {"H": torch.zeros(d_model, d_model), "n": 0}

    def hH(mod, inp, out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        acc["H"] += x.T @ x                      # fp32 accumulate, promoted below -- T2's own form
        acc["n"] += x.shape[0]
    hh = model.lm_head.register_forward_hook(hH)
    log("== calibration pass: act_rms (t2b_organs.capture) + head H, %d tokens =="
        % (ids_cal.shape[0] * ids_cal.shape[1]))
    t0 = time.time()
    act_rms = capture(model, ids_cal, None)      # IMPORTED -- this is T2b/E18's own capture
    hh.remove()
    H = acc["H"].double()
    rms = act_rms[HEADKEY]
    ev = torch.linalg.eigvalsh(H)
    lam = LAM_FRAC * float(torch.diagonal(H).mean())
    log("   done in %.0fs, %d organ keys, H over %d tokens  |  eig min %.3e max %.3e  "
        "cond(damped) %.3e" % (time.time() - t0, len(act_rms), acc["n"],
                               float(ev[0]), float(ev[-1]),
                               (float(ev[-1]) + lam) / (float(ev[0]) + lam)))
    # act_rms is sqrt(mean x^2) and diag(H)/n is mean x^2 -- an identity, so it is a free check
    # that the two statistics came from the same tokens.
    chk = float((rms - torch.sqrt(torch.diagonal(H).float() / acc["n"]).clamp_min(1e-8))
                .abs().max())
    log("   act_rms vs sqrt(diag(H)/n) max abs diff %.3e" % chk)

    out = {"brief": "briefs/BRIEF_E20_RULE_OR_FORMAT.md (e5d438c)",
           "supersedes": "run 1, commit 6c7b7a4, VOID on G-Q2: its R3H arm was R0, not the "
                         "shipped rule. Its base/ID/OPTH/GPTQH arms are re-measured here.",
           "question": "is the constraint the RULE or the FORMAT? format held fixed at what "
                       "QWENDON1 stores: ternary codes, one fp32 scale per output row",
           "rules_imported_from": "t2_rules.py (r0_bitlinear, r1_twn, r2_search, r3_actsearch, "
                                  "r4_gptq); the R3H arm is t2b_organs.apply_arm(model,'H',...) "
                                  "itself, so the anchor gate compares like with like",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "eval_slice": meta_ev, "calib_slice": meta_cal,
           "calib_tokens": acc["n"], "knows_data": KNOWS_DATA,
           "hessian": {"damping_frac": LAM_FRAC, "scale_grid_points": N_SCALE_GRID,
                       "gptq_block": GPTQ_BLOCK, "eig_min": float(ev[0]),
                       "eig_max": float(ev[-1]), "lambda": lam,
                       "act_rms_vs_diagH_max_abs_diff": chk},
           "bands": {"floor_E18_partA": FLOOR, "margin_E17": MARGIN,
                     "at_floor_max": AT_FLOOR_MAX, "ranks_min": RANKS_MIN,
                     "ceiling": len(prompts) * n_new},
           "chance_bpb": CHANCE,
           "anchors": {"t2b_H_bpb": T2B_H_BPB, "e18_H_agree": E18_H_AGREE},
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
        elif tag == "R3H":
            untie(model)
            W0 = model.lm_head.weight.data.clone()
            tq = time.time()
            restore, s2 = apply_arm(model, "H", act_rms, None)      # T2b's own code path
            st = {"quantize_seconds": time.time() - tq,
                  "zero_fraction": float(s2["zero_frac"][0]) if s2.get("zero_frac") else None,
                  "rel_weight_error": float(torch.linalg.norm(model.lm_head.weight.data - W0)
                                            / torch.linalg.norm(W0)),
                  "identical_to_donor": False, "via": "t2b_organs.apply_arm(model,'H',act_rms)",
                  "t2b_stats": {k: v for k, v in s2.items() if k != "zero_frac"}}
            del W0
        else:
            restore, st = apply_head(model, tag, H, rms)

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

        rec = {"arm": tag, "knows_data": KNOWS_DATA.get(tag), "bpb": bpb,
               "vs_chance": bpb - CHANCE, "matched": matched, "counted": counted,
               "agree": matched / float(counted), "first_div": first_div,
               "per_prompt": per_prompt, "ids": allids, "seconds": time.time() - t0,
               "text": [tok.decode(x) for x in allids]}
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
                           "e18_agree": E18_H_AGREE, "agree_here": matched,
                           "agree_ok": bool(matched == E18_H_AGREE)}
        if tag not in ("base", "ID"):
            rec["G_Q3"] = band(matched) if not SMOKE else "NOT-APPLICABLE (smoke)"

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-6s BPB %.6f (%+.6f)  zero %-6s relerr %.4f  %3d/%-3d %6.2f%%  div %-8s %-8s [%.0fs]"
            % (tag, bpb, bpb - CHANCE,
               ("%.4f" % st["zero_fraction"]) if st["zero_fraction"] is not None else "n/a",
               st["rel_weight_error"], matched, counted, 100.0 * matched / counted,
               str(first_div), rec.get("G_Q3", rec.get("G_Q0", rec.get("G_Q1", ""))),
               rec["seconds"]))

    A = out["arms"]
    void = []
    if "base" in A and A["base"].get("G_Q0") != "FIRES":
        void.append("G-Q0: base did not reproduce E6's reference")
    if "ID" in A and A["ID"].get("G_Q1") != "FIRES":
        void.append("G-Q1: the requantize path is not lossless under the identity quantizer")
    if "R3H" in A and "G_Q2" in A["R3H"]:
        g = A["R3H"]["G_Q2"]
        if not g["bpb_ok"] or not g["agree_ok"]:
            void.append("G-Q2: R3H did not replicate T2b/E18 (bpb_ok=%s agree_ok=%s)"
                        % (g["bpb_ok"], g["agree_ok"]))
    out["VOID"] = void

    # G-Q4: sanity on the objective the data-aware arms optimise. Reported, never a verdict.
    if all(t in A for t in ("R0H", "R3H", "R5H")):
        b = {t: A[t]["bpb"] for t in A if t not in ("base", "ID")}
        out["G_Q4"] = {"bpb": b, "best_rule": min(b, key=b.get),
                       "best_is_data_aware": bool(KNOWS_DATA.get(min(b, key=b.get))),
                       "r5_beats_r3": bool(b["R5H"] < b["R3H"]),
                       "r5_minus_r3": b["R5H"] - b["R3H"],
                       "T2_ffn_r5_minus_r3": 2.027495180363716 - 2.4769674487677693,
                       "note": "sanity, not a verdict: a rule that cannot win on the objective "
                               "it optimises has not been demonstrated to work"}
    out["G_Q3"] = {t: A[t]["G_Q3"] for t in A if "G_Q3" in A[t]}

    # the axis the brief asked about: does knowing the data buy RANKING?
    kn = [(A[t]["bpb"], A[t]["matched"]) for t in A if KNOWS_DATA.get(t) is True]
    un = [(A[t]["bpb"], A[t]["matched"]) for t in A if KNOWS_DATA.get(t) is False]
    if kn and un:
        out["data_axis"] = {
            "data_aware": {"n": len(kn), "best_bpb": min(x[0] for x in kn),
                           "best_agree": max(x[1] for x in kn)},
            "weight_space": {"n": len(un), "best_bpb": min(x[0] for x in un),
                             "best_agree": max(x[1] for x in un)}}
    if len(A) >= 4:
        xs = [A[t]["bpb"] for t in A if t not in ("base", "ID")]
        ys = [float(A[t]["matched"]) for t in A if t not in ("base", "ID")]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
        out["r_bpb_agreement"] = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
                                  if sx > 0 and sy > 0 else None)
        out["bpb_span"] = max(xs) - min(xs)
    out["seconds_total"] = time.time() - t_start

    log("\n  bands: floor %d, margin %d, AT-FLOOR <= %d, RANKS >= %d, ceiling %d"
        % (FLOOR, MARGIN, AT_FLOOR_MAX, RANKS_MIN, len(prompts) * n_new))
    log("\n  arm    knows   BPB        vs chance   greedy   band")
    for tag in ARMS:
        if tag in A:
            r = A[tag]
            log("    %-6s %-6s %.6f  %+.6f  %3d/%-3d %s"
                % (tag, {True: "DATA", False: "no", None: "--"}[r.get("knows_data")],
                   r["bpb"], r["vs_chance"], r["matched"], r["counted"], r.get("G_Q3", "")))
    if "G_Q4" in out:
        log("\n  G_Q4  best rule = %s (data-aware: %s), R5H-R3H = %+.6f here vs %+.6f on T2's FFN"
            % (out["G_Q4"]["best_rule"], out["G_Q4"]["best_is_data_aware"],
               out["G_Q4"]["r5_minus_r3"], out["G_Q4"]["T2_ffn_r5_minus_r3"]))
    if "data_axis" in out:
        log("  data axis  data-aware best %d/160 at BPB %.6f  |  weight-space best %d/160 at "
            "BPB %.6f" % (out["data_axis"]["data_aware"]["best_agree"],
                          out["data_axis"]["data_aware"]["best_bpb"],
                          out["data_axis"]["weight_space"]["best_agree"],
                          out["data_axis"]["weight_space"]["best_bpb"]))
    if out.get("r_bpb_agreement") is not None:
        log("  r(BPB, agreement) = %+.4f over a BPB span of %.6f"
            % (out["r_bpb_agreement"], out["bpb_span"]))
    log("\n  VOID: %s" % (", ".join(void) if void else "none"))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("wrote %s  [%.0fs total]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
