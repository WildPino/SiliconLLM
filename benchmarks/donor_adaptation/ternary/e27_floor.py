#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E27 -- the floor, not the FFN.

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E27_THE_FLOOR_NOT_THE_FFN.md
(808ba43), pushed before this file existed.  Nothing is exported and NO TIMING IS TAKEN.

E24 closed with an instruction: the lever that opens the goal's shape is the one that shrinks the
non-FFN FLOOR, not more FFN depth.  The floor has exactly two components any lever can touch --
how WIDE a layer is (rank on q/o, on k/v) and how MANY layers there are.  Width has been measured
at one fraction only (r/D = 1/3, E21, at D = 1536).  Depth has never been attacked here at all.

NO CARVE AND NO ROUTER ANYWHERE.  Mixing the FFN axis back in would make every cell unreadable.

Env: D_THREADS (6), E27_ONLY (comma list), E27_SMOKE (1)
"""
import json
import os
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

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E27_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E27_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e27_floor%s.json" % ("_smoke" if SMOKE else ""))
E21OUT = os.path.join(ENGDIR, "results", "e21_rank.json")
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453

CHANCE = 4.069819
FLOOR_FREE, MARGIN = 12, 2
AT_FLOOR_MAX, RANKS_MIN = FLOOR_FREE + MARGIN, 80
TF_LO, TF_HI = 107, 119
NCAL, SEQCAL, SEEDCAL = 32, 512, 42424
CARVE_E = 256                       # charged in the T10 transposition, never executed here

# ---- anchors, quoted from the published records and never re-derived here
BASE_BPB = 0.7675949641196624
A_QO512 = {"bpb": 0.8202837636996289, "free": 26, "tf": 144, "e21_arm": "QO-ACT-512"}
REPL_TOL = 1e-9

# ---- T10, the goal's shape.  Brief s1.
T10 = dict(D=4096, QD=4096, KD=1024, F=14336, L=48, V=32768)
THROUGHPUT_G = 49.9                 # E25's measured charged throughput, G active weights/s
BUDGET_G = THROUGHPUT_G / 50.0      # active weights/token that 50 tok/s allows
K_BAR_OPENS, K_BAR_HELPS = 133, 40  # brief s5
MINRES_MARGIN = 3                   # brief s3: FLOOR-MIN's registered decision procedure

#        tag                rqo   rkv   drop  rule
ARMS = [("base",            None, None, 0,    None),
        ("L28-PASSTHROUGH", None, None, 0,    "LAST"),    # planted control on the new code
        ("QO-512",          512,  None, 0,    None),      # planted positive, replicates E21
        ("QO-192",          192,  None, 0,    None),
        ("QO-96",           96,   None, 0,    None),
        ("KV-96",           None, 96,   0,    None),
        ("QO192+KV96",      192,  96,   0,    None),
        ("L24-LAST",        None, None, 4,    "LAST"),
        ("L21-LAST",        None, None, 7,    "LAST"),
        ("L21-MINRES",      None, None, 7,    "MINRES"),
        ("L14-MINRES",      None, None, 14,   "MINRES"),
        ("FLOOR-MIN",       192,  96,   7,    "REGISTERED")]


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


# ====================================================== the two floor levers, as arithmetic
def fac(out_, in_, r):
    """Active weights of an [out, in] matrix: r*(out+in) factored, out*in dense (r None/0)."""
    return r * (out_ + in_) if r else out_ * in_


def active(D, QD, KD, F, L, V, rqo, rkv, k=None, router=False):
    """Active weights/token.  k is None for a full FFN; the router is charged only when asked
    for, and E27 never executes one -- it appears in the T10 transposition because every budget
    table this is read against charges it."""
    qo = 2 * fac(QD, D, rqo) * L
    kv = 2 * fac(KD, D, rkv) * L
    head = V * D
    rt = (D * CARVE_E * L) if router else 0
    ffn = 3 * D * F * L if k is None else int(round(3 * D * F * L * k / float(CARVE_E)))
    return {"q_o": qo, "k_v": kv, "head": head, "router": rt, "ffn": ffn,
            "total": qo + kv + head + rt + ffn}


def t10_transpose(D_donor, rqo, rkv, n_keep, n_total):
    """The same FRACTIONS at the goal's shape: r/D is carried across, and so is the fraction of
    layers kept.  This is the mapping E27 exists to test, and it is the only one that means
    anything across two widths."""
    T = T10
    rq = int(round(rqo / float(D_donor) * T["D"])) if rqo else None
    rk = int(round(rkv / float(D_donor) * T["D"])) if rkv else None
    L = int(round(T["L"] * n_keep / float(n_total)))
    fl = active(T["D"], T["QD"], T["KD"], T["F"], L, T["V"], rq, rk, k=0, router=True)["total"]
    ffn_full = 3 * T["D"] * T["F"] * L
    allow = BUDGET_G * 1e9 - fl
    k = allow / float(ffn_full) * CARVE_E
    return {"r_qo": rq, "r_kv": rk, "L": L, "floor": fl, "floor_G": fl / 1e9,
            "ffn_full_G": ffn_full / 1e9, "k_permitted": k,
            "activation_permitted": k / float(CARVE_E),
            "param_fraction_kept": n_keep / float(n_total)}


# ====================================================== depth
class Depth(object):
    """Drop decoder layers and put them back.

    The KV cache addresses layers by self_attn.layer_idx, so a gap in it is a SILENT WRONG
    ANSWER and not an error.  G-F1 -- dropping zero layers must be bit-identical to base -- is
    what proves the reindexing here is right, and it runs before any depth cell is read.
    """

    def __init__(self, model):
        self.model = model
        self.orig = list(model.model.layers)
        self.n = len(self.orig)

    def set(self, keep):
        keep = sorted(int(i) for i in keep)
        layers = [self.orig[i] for i in keep]
        for new_i, lay in enumerate(layers):
            lay.self_attn.layer_idx = new_i
        self.model.model.layers = nn.ModuleList(layers)
        self.model.config.num_hidden_layers = len(layers)
        return keep

    def restore(self):
        for i, lay in enumerate(self.orig):
            lay.self_attn.layer_idx = i
        self.model.model.layers = nn.ModuleList(self.orig)
        self.model.config.num_hidden_layers = self.n


def measure_residuals(model, ids_cal):
    """MINRES, defined in the brief before it was run: mean over positions of
    ||block_out - block_in|| / ||block_in||, one forward pass, one number per layer."""
    n = len(model.model.layers)
    tot = [0.0] * n
    cnt = [0] * n
    hooks = []

    def mk(li):
        def f(mod, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            y = output[0] if isinstance(output, tuple) else output
            xf = x.detach().reshape(-1, x.shape[-1]).float()
            yf = y.detach().reshape(-1, y.shape[-1]).float()
            den = torch.linalg.norm(xf, dim=1).clamp_min(1e-12)
            tot[li] += float((torch.linalg.norm(yf - xf, dim=1) / den).sum())
            cnt[li] += xf.shape[0]
        return f

    for li, lay in enumerate(model.model.layers):
        hooks.append(lay.register_forward_hook(mk(li), with_kwargs=True))
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            model(ids_cal[i:i + 1])
    for h in hooks:
        h.remove()
    return [tot[i] / max(cnt[i], 1) for i in range(n)]


def keep_set(n_total, drop, rule, residual):
    if drop <= 0:
        return list(range(n_total))
    if rule == "LAST":
        return list(range(n_total - drop))
    if rule == "MINRES":
        order = sorted(range(n_total), key=lambda i: (residual[i], i))
        gone = set(order[:drop])
        return [i for i in range(n_total) if i not in gone]
    raise SystemExit("unknown depth rule %r" % (rule,))


# ====================================================== width
def capture_H(model, ids_cal, organs):
    """H = X'X per organ, from the frozen calibration slice.  Same capture E21 and E24 use."""
    Hs, cnt, hooks = {}, {}, []

    def mkh(key):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            Hs[key] = (x.T @ x) if key not in Hs else Hs[key] + (x.T @ x)
            cnt[key] = cnt.get(key, 0) + x.shape[0]
        return f
    for li, lay in enumerate(model.model.layers):
        for nm in organs:
            hooks.append(getattr(lay.self_attn, nm).register_forward_hook(mkh((li, nm))))
    with torch.no_grad():
        for i in range(ids_cal.shape[0]):
            model(ids_cal[i:i + 1])
    for h in hooks:
        h.remove()
    return Hs


def apply_rank(model, Hs, rqo, rkv):
    """Activation-weighted rank on the CURRENT layer stack, via E21's own lowrank.  Returns a
    restore closure and the per-organ diagnostics."""
    saved, rel, ener = [], [], []
    for li, lay in enumerate(model.model.layers):
        jobs = []
        if rqo:
            jobs += [("q_proj", rqo), ("o_proj", rqo)]
        if rkv:
            jobs += [("k_proj", rkv), ("v_proj", rkv)]
        for nm, r in jobs:
            mod = getattr(lay.self_attn, nm)
            saved.append((mod, mod.weight.data.clone()))
            Wr, d = E21.lowrank(mod.weight.data, Hs[(li, nm)], r, True)
            mod.weight.data = Wr.float()
            rel.append(d["rel_weight_error"])
            if d["energy_kept_weighted"] is not None:
                ener.append(d["energy_kept_weighted"])

    def restore():
        for mod, W in saved:
            mod.weight.data = W
    st = {"n_organs": len(saved),
          "rel_weight_error": float(np.mean(rel)) if rel else 0.0,
          "energy_kept_weighted": float(np.mean(ener)) if ener else None}
    return restore, st


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]

    # ---- the anchor must match E21's own record on disk, or the anchor is not the anchor
    anchor_check = None
    if os.path.exists(E21OUT):
        r = json.load(open(E21OUT, encoding="utf-8"))["arms"][A_QO512["e21_arm"]]
        ok = (r["bpb"] == A_QO512["bpb"] and r["matched"] == A_QO512["free"]
              and r["teacher_forced"] == A_QO512["tf"])
        anchor_check = {"e21_arm": A_QO512["e21_arm"], "bpb_on_disk": r["bpb"],
                        "free_on_disk": r["matched"], "tf_on_disk": r["teacher_forced"],
                        "matches_literal": bool(ok)}
        if not ok:
            raise SystemExit("ANCHOR MISMATCH vs %s -- STOP" % E21OUT)
        log("== E21 anchor verified against %s ==" % os.path.basename(E21OUT))

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
        # The smoke keeps 8 layers so a depth arm is still a depth arm, and scales the drops
        # with the stack.  Its numbers are not evidence; it exists to run every code path.
        model.model.layers = nn.ModuleList(list(model.model.layers)[:8])
        model.config.num_hidden_layers = 8
    depth = Depth(model)
    NL = depth.n
    SMOKE_DROP = {4: 1, 7: 2, 14: 4}

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

    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    assert meta_cal["corpus_sha256"] != meta_ev["corpus_sha256"], \
        "calib and eval must be DIFFERENT corpus halves"
    if SMOKE:
        ids_cal = ids_cal[:2]

    todo = [a for a in ARMS if (not ONLY or a[0] in ONLY or a[0] == "base")]

    # ---- one calibration pass for the four attention organs, one for the residuals
    need_w = any(a[1] or a[2] for a in todo)
    Hs = {}
    if need_w:
        log("== H = X'X for q/k/v/o, %d organs ==" % (4 * NL))
        t0 = time.time()
        Hs = capture_H(model, ids_cal, ("q_proj", "k_proj", "v_proj", "o_proj"))
        log("   done in %.0fs, %d H matrices" % (time.time() - t0, len(Hs)))

    need_r = any(a[4] in ("MINRES", "REGISTERED") for a in todo)
    residual = None
    if need_r:
        log("== MINRES: mean relative residual per layer, one pass ==")
        t0 = time.time()
        residual = measure_residuals(model, ids_cal)
        order = sorted(range(NL), key=lambda i: residual[i])
        log("   done in %.0fs | smallest first: %s"
            % (time.time() - t0, ", ".join("L%d=%.4f" % (i, residual[i]) for i in order[:8])))

    # ---- G-F3: the two budget tables come out of ONE function
    donor_full = active(D, QD, KD, F, NL, V, None, None)
    gf3 = t10_transpose(D, 192, 96, NL, NL)          # -> T10 r_qo 512, r_kv 256, L 48
    gf3_ok = abs(gf3["floor_G"] - 0.7130) < 5e-4
    log("== G-F3: T10 floor at r_qo=%d r_kv=%d L=%d is %.4f G (brief s1 says 0.7130) -> %s"
        % (gf3["r_qo"], gf3["r_kv"], gf3["L"], gf3["floor_G"], "OK" if gf3_ok else "MISMATCH"))
    if not gf3_ok:
        raise SystemExit("G-F3 MISMATCH -- the runner and the brief disagree about a floor")

    out = {"brief": "briefs/BRIEF_E27_THE_FLOOR_NOT_THE_FFN.md (808ba43)",
           "question": "E24 said the lever is the non-FFN FLOOR, not more FFN depth.  The floor "
                       "is how WIDE a layer is and how MANY layers there are.  Width is measured "
                       "at one fraction only; depth has never been attacked here.  Does either "
                       "open the goal's shape?  No carve, no router, no timing.",
           "model": HF, "smoke": SMOKE, "threads": THREADS, "n_new": n_new,
           "n_prompts": len(prompts), "eval_slice": meta_ev, "calib_slice": meta_cal,
           "shape": {"D": D, "QD": QD, "KD": KD, "F": F, "L": NL, "V": V},
           "donor_full_active": donor_full,
           "T10": dict(T10, throughput_G_per_s=THROUGHPUT_G, budget_G=BUDGET_G),
           "bars": {"k_opens": K_BAR_OPENS, "k_helps": K_BAR_HELPS,
                    "tf_comparable": [TF_LO, TF_HI]},
           "anchors": {"base_bpb": BASE_BPB, "QO-512": A_QO512, "repl_tol": REPL_TOL,
                       "verified_against_e21_json": anchor_check},
           "minres_residual": residual, "G_F3": {"floor_G": gf3["floor_G"], "ok": gf3_ok},
           "chance_bpb": CHANCE, "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    for tag, rqo, rkv, drop, rule in todo:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            rr = out["arms"][tag]
            log("  %-16s CACHED  bpb %.6f  free %d  tf %d"
                % (tag, rr["bpb"], rr["matched"], rr["teacher_forced"]))
            continue
        t0 = time.time()
        if SMOKE:
            drop = SMOKE_DROP.get(drop, drop)

        # FLOOR-MIN's depth rule is a REGISTERED DECISION PROCEDURE (brief s3), resolved here
        # from arms already on the record rather than chosen after seeing this arm.
        rule_used, rule_note = rule, None
        if rule == "REGISTERED":
            a, b = out["arms"].get("L21-MINRES"), out["arms"].get("L21-LAST")
            if a and b:
                rule_used = ("MINRES" if a["teacher_forced"] > b["teacher_forced"] + MINRES_MARGIN
                             else "LAST")
                rule_note = ("MINRES %d vs LAST %d, margin %d -> %s"
                             % (a["teacher_forced"], b["teacher_forced"], MINRES_MARGIN,
                                rule_used))
            else:
                rule_used = "LAST"
                rule_note = "the two 21-layer arms are not both on the record; defaulting to LAST"

        keep = depth.set(keep_set(NL, drop, rule_used or "LAST", residual)) if rule else \
            list(range(NL))
        n_keep = len(keep)

        # FLOOR-MIN composes depth and width, so its H is recaptured on the PRUNED stack: the
        # input distribution of a surviving layer is not the one it had in the full model, and
        # reusing the full model's H would put a second, uncontrolled difference in the one arm
        # whose job is to compose cleanly.
        Huse, restore_w, st_w = Hs, None, None
        if (rqo or rkv) and n_keep != NL:
            log("     recapturing H on the %d-layer stack" % n_keep)
            Huse = capture_H(model, ids_cal, ("q_proj", "k_proj", "v_proj", "o_proj"))
        if rqo or rkv:
            restore_w, st_w = apply_rank(model, Huse, rqo, rkv)

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

        if restore_w:
            restore_w()
        depth.restore()

        counted = len(prompts) * n_new
        don = active(D, QD, KD, F, n_keep, V, rqo, rkv)
        t10 = t10_transpose(D, rqo, rkv, n_keep, NL)
        rec = {"arm": tag, "r_qo": rqo, "r_kv": rkv, "n_dropped": drop,
               "depth_rule": rule_used, "depth_rule_note": rule_note,
               "layers_kept": keep if drop else None, "n_layers": n_keep,
               "bpb": bpb, "vs_chance": bpb - CHANCE,
               "matched": matched, "counted": counted, "agree": matched / float(counted),
               "teacher_forced": tf, "tf_frac": tf / float(counted),
               "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
               "rank_le5": int(sum(1 for r in ranks if r <= 5)),
               "donor_active": don, "donor_active_G": don["total"] / 1e9,
               "donor_saving_vs_base": 1.0 - don["total"] / float(donor_full["total"]),
               "T10": t10, "width": st_w,
               "first_div": first_div, "per_prompt": per_prompt,
               "band_free": band_free(matched), "band_tf": band_tf(tf),
               "ids": allids, "text": [tok.decode(x) for x in allids],
               "seconds": time.time() - t0}

        if tag == "base":
            # Under SMOKE the stack is truncated to 8 layers, so `base` is a BROKEN model by
            # construction and 160/160 is not the right thing to ask of it.  The smoke exists to
            # run every path, and G-F4 stamps itself NOT A VERDICT there anyway.
            rec["G_F0"] = ("SMOKE" if SMOKE else
                           ("FIRES" if (matched == counted and tf == counted
                                        and abs(bpb - BASE_BPB) < REPL_TOL) else "VOID"))
        if tag == "L28-PASSTHROUGH" and "base" in out["arms"]:
            b = out["arms"]["base"]
            rec["G_F1"] = {"bpb_here": bpb, "bpb_base": b["bpb"],
                           "abs_diff": abs(bpb - b["bpb"]),
                           "free_here": matched, "free_base": b["matched"],
                           "tf_here": tf, "tf_base": b["teacher_forced"],
                           "verdict": ("FIRES" if (bpb == b["bpb"] and matched == b["matched"]
                                                   and tf == b["teacher_forced"]) else "VOID")}
        if tag == "QO-512" and not SMOKE:
            rec["G_F2"] = {"bpb_here": bpb, "bpb_anchor": A_QO512["bpb"],
                           "abs_diff": abs(bpb - A_QO512["bpb"]),
                           "free_here": matched, "free_anchor": A_QO512["free"],
                           "tf_here": tf, "tf_anchor": A_QO512["tf"],
                           "verdict": ("FIRES" if (abs(bpb - A_QO512["bpb"]) < REPL_TOL
                                                   and matched == A_QO512["free"]
                                                   and tf == A_QO512["tf"]) else "VOID")}

        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-16s BPB %.6f  free %3d/%d  tf %3d/%d  L %2d  donor %.4f G  |  T10: r_qo %-5s "
            "r_kv %-5s L %2d floor %.4f G  k %6.1f  %s / %s"
            % (tag, bpb, matched, counted, tf, counted, n_keep, rec["donor_active_G"],
               t10["r_qo"] or "dense", t10["r_kv"] or "dense", t10["L"], t10["floor_G"],
               t10["k_permitted"], rec["band_free"], rec["band_tf"]))
        for g in ("G_F0", "G_F1", "G_F2"):
            if g in rec:
                log("                    %s: %s"
                    % (g, rec[g] if isinstance(rec[g], str) else rec[g]["verdict"]))

    # ================================================== the gates
    A = out["arms"]
    voids = []
    if A.get("base", {}).get("G_F0") == "VOID":
        voids.append("base")
    if A.get("L28-PASSTHROUGH", {}).get("G_F1", {}).get("verdict") == "VOID":
        voids.append("L28-PASSTHROUGH(depth machinery)")
    if A.get("QO-512", {}).get("G_F2", {}).get("verdict") == "VOID":
        voids.append("QO-512(width machinery)")
    out["VOID"] = voids

    # G-F4: the verdict, scored on BOTH halves at once
    CONTROLS = ("base", "L28-PASSTHROUGH")
    cands = [(t, r) for t, r in A.items()
             if t not in CONTROLS and r["teacher_forced"] >= TF_LO]
    gf4 = None
    if not voids:
        best = max(cands, key=lambda p: p[1]["T10"]["k_permitted"]) if cands else None
        if best is None:
            lab = "FLOOR-IS-NOT-ENOUGH"
            det = "no arm reads COMPARABLE at the donor at all"
        else:
            kp = best[1]["T10"]["k_permitted"]
            if kp >= K_BAR_OPENS:
                lab = "FLOOR-OPENS"
            elif kp >= K_BAR_HELPS:
                lab = "FLOOR-HELPS"
            else:
                lab = "FLOOR-IS-NOT-ENOUGH"
            det = ("%s: tf %d, T10 permits k = %.1f of 256 (%.2f%% activation), params kept "
                   "%.1f%%" % (best[0], best[1]["teacher_forced"], kp, 100 * kp / CARVE_E,
                               100 * best[1]["T10"]["param_fraction_kept"]))
        gf4 = {"verdict": ("SMOKE -- NOT A VERDICT" if SMOKE else lab), "detail": det,
               "best_comparable_arm": best[0] if best else None,
               "k_bar_opens": K_BAR_OPENS, "k_bar_helps": K_BAR_HELPS,
               "shallowest_measured_k": K_BAR_OPENS}
    out["G_F4"] = gf4

    # the two levers, side by side, as the brief asked
    out["levers"] = {
        "width_only": {t: {"tf": A[t]["teacher_forced"], "k": A[t]["T10"]["k_permitted"]}
                       for t in ("QO-512", "QO-192", "QO-96", "KV-96", "QO192+KV96") if t in A},
        "depth_only": {t: {"tf": A[t]["teacher_forced"], "k": A[t]["T10"]["k_permitted"],
                           "params_kept": A[t]["T10"]["param_fraction_kept"]}
                       for t in ("L24-LAST", "L21-LAST", "L21-MINRES", "L14-MINRES") if t in A},
        "composed": {t: {"tf": A[t]["teacher_forced"], "k": A[t]["T10"]["k_permitted"],
                         "params_kept": A[t]["T10"]["param_fraction_kept"]}
                     for t in ("FLOOR-MIN",) if t in A}}
    if "L21-MINRES" in A and "L21-LAST" in A:
        out["minres_vs_last"] = {"minres": A["L21-MINRES"]["teacher_forced"],
                                 "last": A["L21-LAST"]["teacher_forced"],
                                 "gain": (A["L21-MINRES"]["teacher_forced"]
                                          - A["L21-LAST"]["teacher_forced"])}
    out["seconds_total"] = time.time() - t_start
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    # ================================================== the table
    log("")
    log("  %-16s %-9s %-9s %-9s %-4s %-9s | %-20s %-9s %s"
        % ("arm", "BPB", "free", "tf", "L", "donor G", "T10 floor", "k of 256", "band-tf"))
    for tag, _q, _k, _d, _r in ARMS:
        if tag in A:
            r = A[tag]
            log("    %-16s %8.6f %4d/%-4d %4d/%-4d %3d %8.4f  | r%-5s/%-5s L%-3d %7.4f %7.1f  %s"
                % (tag, r["bpb"], r["matched"], r["counted"], r["teacher_forced"], r["counted"],
                   r["n_layers"], r["donor_active_G"], r["T10"]["r_qo"] or "dns",
                   r["T10"]["r_kv"] or "dns", r["T10"]["L"], r["T10"]["floor_G"],
                   r["T10"]["k_permitted"], r["band_tf"]))
    log("")
    log("  VOID: %s" % (", ".join(voids) if voids else "none"))
    if "minres_vs_last" in out:
        m = out["minres_vs_last"]
        log("  MINRES vs LAST at 21 layers: %d vs %d (%+d)" % (m["minres"], m["last"], m["gain"]))
    if gf4:
        log("  G-F4: %s" % gf4["verdict"])
        log("        %s" % gf4["detail"])
        log("        bars: k >= %d OPENS (shallowest k ever measured on quality), k >= %d HELPS"
            % (K_BAR_OPENS, K_BAR_HELPS))
    log("  wrote %s in %.0fs" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
