#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 step 4 (CPU, FREE) -- decide G-H1, on this machine, in fp32.

Brief: docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md s5,
addendum C (the anchors and the REPAIRED bands) and addendum E (the router setting).

GPU TRAINS, CPU MEASURES -- the same split H0 used, for the same reason.  h1_qat.py reports an
fp16 held-out BPB every --every steps so the T4 run is watchable, and that number is a PROGRESS
metric.  The gate is decided here with the instrument every published number in this programme
used: the frozen 24x512 held-out slice, ids sha a1a48dc9..., 51,870 scored bytes,
4.22945205479452 B/token, fp32, this box.

  THE GATE -- G-H1, ORDINAL, no tolerance:

      trained-8L  <  applied-8L = 1.096636

  applied-8L is the MATCHED control measured by h1_applied.py before the hours were spent
  (addendum C): the same 8 layers, the same k, the same rule, APPLIED post-hoc instead of
  trained.  It is read back out of its own result file here and checked against the registered
  constant, so that re-measuring it later cannot silently move the gate.

THE DECOMPOSITION, AND IT IS DIAGNOSTIC, NOT A GATE.  The trained arm differs from applied-8L
in THREE ways at once -- the experts trained, the router trained, and the gate is the
renormalised soft gate instead of E37's hard {0,1}.  A win could come from the gate form alone
and would then say nothing about training.  So two intermediate arms are measured, chosen so
the three contributions are EXACTLY ADDITIVE:

  applied-8L   untrained experts + E37 router   + hard gate      1.096636   (registered)
  arm-E        TRAINED experts   + E37 router   + hard gate      -> the EXPERTS' contribution
  arm-ER       TRAINED experts   + TRAINED rtr  + hard gate      -> the ROUTER's contribution
  trained-8L   TRAINED experts   + TRAINED rtr  + SOFT gate      -> the GATE FORM's

  (arm-E - applied) + (arm-ER - arm-E) + (trained - arm-ER) = trained - applied, identically.

E14 s6 forbids promoting a post-hoc metric to a gate, so none of the three may pass or fail
anything.  They exist so that the sentence written about G-H1 names a mechanism instead of
asserting one.

G-H1e IS RE-MEASURED HERE TOO, in fp32.  h1_qat.py reads it on the GPU at the end of training;
this is the same comparison on the frozen slice, with STATIC calibrated on the disjoint calib
half ("calib", 32, 512, 42424) exactly as E23/E37 pinned it.

Env: D_THREADS (6).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
for _p in (DENSDIR, ENGDIR):
    sys.path.insert(0, _p)

import common as C                                          # noqa: E402
import h1_qat as H1                                         # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

OUTDIR = os.path.join(HERE, "results", "h1")
APPLIED_JSON = os.path.join(OUTDIR, "h1_applied_8L.json")
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453
CHANCE = 4.069819
INTACT = 0.767595                       # E12's dense anchor -- the planted control
H0_RUN3 = 0.810022                      # H1's start line
TERNARY_8L = 0.947851                   # matched, addendum C
APPLIED_8L = 1.096636                   # THE GATE's threshold, addendum C
E37_ROUTERS = "D:/_ktmp/e37/e37_routers_E256.npz"

# The REPAIRED band table -- addendum C.  Every edge is a MATCHED 8-layer number measured by
# h1_applied.py on this box.  The version this replaced used E37's all-28-layer 3.475707 as an
# edge, which made TRAINING-HELPS empty and overlapped two other bands.
BANDS = (("CARVE-IS-FREE", None, H0_RUN3),             # <= h0-run3
         ("CARVE-IS-TRAINABLE", H0_RUN3, TERNARY_8L),  # (h0-run3, ternary-8L)
         ("TRAINING-HELPS", TERNARY_8L, APPLIED_8L),   # [ternary-8L, applied-8L)
         ("CARVE-NOT-TRAINABLE", APPLIED_8L, None))    # >= applied-8L


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


def band_of(b):
    if b <= H0_RUN3:
        return "CARVE-IS-FREE"
    if b < TERNARY_8L:
        return "CARVE-IS-TRAINABLE"
    if b < APPLIED_8L:
        return "TRAINING-HELPS"
    return "CARVE-NOT-TRAINABLE"


@torch.no_grad()
def bpb(model, ids, nbytes):
    nats = []
    for i in range(ids.shape[0]):
        ch = ids[i:i + 1]
        lg = model(ch).logits
        lp = torch.nn.functional.log_softmax(lg[:, :-1].float(), dim=-1)
        nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double())
    return float(torch.stack(nats).sum() / (LN2 * nbytes))


def install_trained(mods, bundle, labels_npz):
    """Load h1_qat.py's bundle into the carved modules, and DERIVE nothing on trust.

    The bundle carries its own `labels`; they are asserted equal to the D0c partition the
    control was measured against.  A trained model carved on a DIFFERENT partition would be a
    perfectly plausible artefact -- it would train, it would report a BPB, and the comparison
    with applied-8L would be meaningless.
    """
    z = np.load(bundle)
    lz = np.load(labels_npz)
    n = 0
    for li, m in mods:
        p = "L%02d" % li
        need = [p + s for s in (".gate", ".up", ".down", ".router", ".rms_in", ".rms_h",
                                ".labels")]
        miss = [q for q in need if q not in z.files]
        if miss:
            raise SystemExit("bundle is missing %s -- STOP" % miss)
        lab_b = torch.from_numpy(z[p + ".labels"]).long()
        lab_f = torch.from_numpy(lz["c%d" % li]).long()
        if not torch.equal(lab_b, lab_f):
            raise SystemExit(
                "layer %d: the bundle's partition DIFFERS from %s.  The trained model was "
                "carved on a different grouping than applied-8L, so the gate would compare "
                "two different experiments.  STOP." % (li, labels_npz))
        for nm in ("gate", "up", "down", "router", "rms_in", "rms_h"):
            src = torch.from_numpy(z[p + "." + nm]).float()
            dst = getattr(m, nm)
            if tuple(src.shape) != tuple(dst.shape):
                raise SystemExit("layer %d %s is %s, module wants %s -- STOP"
                                 % (li, nm, tuple(src.shape), tuple(dst.shape)))
            dst.data.copy_(src)
        m.invalidate()          # .data writes do not bump _version; see TernaryCarvedFFN
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True, help="h1_qat.py's bundle (h1_trained.npz)")
    ap.add_argument("--factors", required=True, help="h0_trained3.npz -- the frozen q/o base")
    ap.add_argument("--labels", required=True, help="labels_E256.npz")
    ap.add_argument("--stats", default=None, help="h1_actstats.npz (shape only; the bundle's "
                                                  "own rms overwrites it)")
    ap.add_argument("--routers", default=E37_ROUTERS, help="E37's routers, for arm-E")
    ap.add_argument("--layers", default=",".join(str(x) for x in H1.H1_LAYERS))
    ap.add_argument("--k", type=int, default=H1.K_DEFAULT)
    ap.add_argument("--groups", type=int, default=H1.E_GROUPS)
    ap.add_argument("--no-decomp", action="store_true", help="skip arm-E / arm-ER")
    ap.add_argument("--no-gh1e", action="store_true", help="skip the fp32 G-H1e re-measurement")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",") if x.strip() != ""]
    tag = a.tag or os.path.splitext(os.path.basename(a.trained))[0]
    t0 = time.time()

    log("== H1 eval: %s ==  CPU fp32, layers %s, k %d of E %d" % (tag, layers, a.k, a.groups))

    # ---- the threshold is read back, not retyped -------------------------------------------
    applied = APPLIED_8L
    if os.path.exists(APPLIED_JSON):
        aj = json.load(open(APPLIED_JSON, encoding="utf-8"))
        applied = aj["bpb"]["applied-8L"]
        if abs(applied - APPLIED_8L) > 5e-7:
            raise SystemExit(
                "applied-8L on disk is %.6f but addendum C registered %.6f.  The gate's "
                "threshold was registered before the hours were spent and may not move.  STOP."
                % (applied, APPLIED_8L))
        if aj["layers"] != layers or aj["k"] != a.k or aj["E"] != a.groups:
            raise SystemExit("the control was measured on layers %s k %d E %d, this run is "
                             "%s k %d E %d -- NOT matched.  STOP."
                             % (aj["layers"], aj["k"], aj["E"], layers, a.k, a.groups))
        log("   threshold read back from %s: applied-8L %.6f  (matches the registered %.6f)"
            % (os.path.basename(APPLIED_JSON), applied, APPLIED_8L))
    else:
        log("   *** %s ABSENT -- falling back to the registered constant %.6f"
            % (APPLIED_JSON, APPLIED_8L))

    tj = os.path.splitext(a.trained)[0] + ".json"
    meta_train = json.load(open(tj, encoding="utf-8")) if os.path.exists(tj) else {}
    if meta_train:
        log("   bundle: %s steps requested, complete=%s, router_lr %s, aux %s, %.2f h"
            % (meta_train.get("steps_requested"), meta_train.get("complete"),
               meta_train.get("router_lr"), meta_train.get("aux"),
               (meta_train.get("seconds") or 0) / 3600.0))
        if meta_train.get("complete") is False:
            log("   NOTE: the bundle is a mid-run checkpoint.  That is legitimate (the trainer "
                "saves every --every steps) and it is recorded, not hidden.")

    # ---- the instrument --------------------------------------------------------------------
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    if meta["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP.  This is not the frozen instrument.")
    nb = float(byts.sum())
    log("   slice %d seqs, %d scored bytes, %.14f B/token"
        % (ids.shape[0], int(nb), nb / float(ids[:, 1:].numel())))

    rows = {}
    rows["intact"] = bpb(model, ids, nb)
    ok = abs(rows["intact"] - INTACT) < 1e-5
    log("   intact      %.6f   (anchor %.6f)  PLANTED CONTROL %s"
        % (rows["intact"], INTACT, "FIRES" if ok else "*** FAILS ***"))
    if not ok:
        raise SystemExit("The intact control does not reproduce E12's dense anchor.  The "
                         "instrument has drifted and NOTHING after this line counts.  STOP.")

    n_qo, qo_layers = H1.build_qo(model, a.factors, "cpu")
    rows["h0-run3"] = bpb(model, ids, nb)
    log("   h0-run3     %.6f   (%d organs over %d layers; anchor %.6f)"
        % (rows["h0-run3"], n_qo, len(qo_layers), H0_RUN3))

    H1.build_ffn(model, a.labels, a.stats, layers, a.k, a.groups, "cpu")
    mods = H1.ffn_mods(model, layers)
    n_ffn = install_trained(mods, a.trained, a.labels)
    log("   installed the trained carve on %d layers (partition asserted against %s)"
        % (n_ffn, os.path.basename(a.labels)))

    # ---- the wiring gates, on the REAL shape ------------------------------------------------
    # G-H1a IS ONLY BIT-IDENTICAL WHERE THE GATES ARE EXACTLY 1, and that is TWO situations,
    # not one.  Under the HARD gate every selected group passes at 1.0 whatever the router
    # says, so k=E must reproduce the uncarved forward for ANY router.  Under the SOFT gate
    # g_e = k*p_e / sum_sel p_j, which is 1 at k=E only when p is uniform -- i.e. only at a
    # ZERO router.  Checking a fitted router under the soft gate asks for an identity that
    # cannot hold, and the first version of this file did exactly that: the NULL CONTROL
    # caught it on its first use, before any trained weights existed.  Both are checked.
    m0 = mods[0][1]
    x = torch.randn(1, 6, model.config.hidden_size)
    m0.hard_gate = True
    same_h, dmax_h = H1.g_h1a(m0, x)
    m0.hard_gate = False
    saved_r = m0.router.detach().clone()
    m0.router.data.zero_()
    m0.invalidate()
    same_s, dmax_s = H1.g_h1a(m0, x)
    m0.router.data.copy_(saved_r)
    m0.invalidate()
    live, ldmax = H1.carve_is_live(m0, x)
    log("   G-H1a  k=E identical, HARD gate, any router : %s  (max|d| %.1e)"
        % ("FIRES" if same_h else "*** FAILS ***", dmax_h))
    log("   G-H1a  k=E identical, SOFT gate, router=0   : %s  (max|d| %.1e)"
        % ("FIRES" if same_s else "*** FAILS ***", dmax_s))
    log("   carve actually masks at k=%-3d               : %s  (max|d| %.1e)"
        % (a.k, "FIRES" if live else "*** FAILS ***", ldmax))
    same = same_h and same_s
    if not same or not live:
        raise SystemExit("A wiring gate FAILS on the real shape.  The number this run would "
                         "print is not a measurement of the carve.  STOP.")

    # ---- THE GATE, and it is the HARD-gated arm -- addendum F --------------------------------
    # The soft gate costs +0.801377 BPB by ITSELF on an untrained bundle (the null control),
    # and engine.c runs the hard gate.  Scoring G-H1 soft would charge H1 ~0.80 BPB of gate
    # form before training counted, against a control that is hard, for an object that cannot
    # ship.  Both are measured; the HARD one is the gate.
    for _, m in mods:
        m.k, m.hard_gate = a.k, False
    rows["trained-8L-soft"] = bpb(model, ids, nb)
    log("   trained-8L-soft %.6f   (k = %d, trained router, SOFT gate -- DIAGNOSTIC)"
        % (rows["trained-8L-soft"], a.k))
    for _, m in mods:
        m.hard_gate = True
    rows["trained-8L"] = bpb(model, ids, nb)
    log("   trained-8L      %.6f   (k = %d, trained router, HARD gate)  <- THE GATE NUMBER"
        % (rows["trained-8L"], a.k))
    log("   train/eval gap  %+.6f   (soft minus hard: what the engine DISCARDS)"
        % (rows["trained-8L-soft"] - rows["trained-8L"]))

    # ---- the additive decomposition, DIAGNOSTIC ----------------------------------------------
    decomp = None
    if not a.no_decomp:
        if not os.path.exists(a.routers):
            log("   decomposition SKIPPED: %s absent" % a.routers)
        else:
            saved = {li: m.router.detach().clone() for li, m in mods}
            # arm-ER IS the hard-gated trained arm measured just above -- same weights, same
            # router, same gate.  Reusing it costs one fewer pass and, more to the point,
            # makes the decomposition exact by construction rather than by luck.
            rows["arm-ER"] = rows["trained-8L"]
            log("   arm-ER      %.6f   (trained experts + trained router + HARD gate "
                "-- same as the gate arm)" % rows["arm-ER"])
            rz = np.load(a.routers)
            for li, m in mods:
                w = torch.from_numpy(rz["r%d" % li]).float()
                if tuple(w.shape) != tuple(m.router.shape):
                    raise SystemExit("router r%d is %s, module wants %s -- STOP"
                                     % (li, tuple(w.shape), tuple(m.router.shape)))
                m.router.data.copy_(w)
                m.invalidate()
            rows["arm-E"] = bpb(model, ids, nb)
            log("   arm-E       %.6f   (trained experts + E37 router    + HARD gate)"
                % rows["arm-E"])
            for li, m in mods:
                m.router.data.copy_(saved[li])
                m.invalidate()
                m.hard_gate = False
            decomp = {"experts": rows["arm-E"] - applied,
                      "router": rows["arm-ER"] - rows["arm-E"],
                      "gate_form": rows["trained-8L-soft"] - rows["arm-ER"]}
            tot = rows["trained-8L-soft"] - applied
            resid = abs(sum(decomp.values()) - tot)
            log("")
            log("   DECOMPOSITION of %+.6f (trained-8L-SOFT - applied-8L), DIAGNOSTIC:"
                % tot)
            log("      experts trained   %+.6f" % decomp["experts"])
            log("      router trained    %+.6f" % decomp["router"])
            log("      soft vs hard gate %+.6f" % decomp["gate_form"])
            log("      (additive by construction; residual %.1e)" % resid)
            log("      the first two terms are the GATE's own decomposition (both hard);")
            log("      the third is the train/eval mismatch the engine discards.")
            if decomp["experts"] >= 0 and decomp["router"] >= 0:
                log("      NOTE: neither training term is negative, so nothing H1 trained")
                log("      improved the deployable arm.  G-H1 cannot pass on this reading.")

    # ---- G-H1e, re-measured in fp32 ------------------------------------------------------------
    gh1e = None
    if not a.no_gh1e:
        # addendum F: BOTH arms hard.  STATIC always had gates exactly 1, so a soft router arm
        # made G-H1e a gate-form comparison -- the null control read 5.564279 vs 3.435791 with
        # ZERO training.  Only the SELECTION may differ.
        for _, m in mods:
            m.hard_gate = True
        ids_cal, _, meta_cal = C.get_slice(tok, "calib", 32, 512, 42424)
        r, s, fires_e, picks = H1.g_h1e(model, mods, ids_cal, ids, "cpu", False)
        gh1e = {"router_nats": r, "static_nats": s, "fires": bool(fires_e),
                "gate": "hard on BOTH arms (addendum F)",
                "calib_slice": meta_cal, "static_groups": picks}
        log("")
        log("   G-H1e (fp32, HARD both arms)  router %.6f vs STATIC %.6f nats/token -> %s"
            % (r, s, "FIRES" if fires_e else "FAILS"))
        log("      addendum E registered the expectation that this FAILS: the CPU smoke gave")
        log("      2 of 8 layers and +2.27%% aggregate error against STATIC.  A PASS here is a")
        log("      result about joint training, not a hyper-parameter.")

    # ---- the verdict ----------------------------------------------------------------------------
    trained = rows["trained-8L"]
    passes = trained < applied
    band = band_of(trained)
    log("")
    log("   G-H1  trained-8L < applied-8L   (ORDINAL, no tolerance)")
    log("         %.6f  %s  %.6f   ->  %s   (delta %+.6f)"
        % (trained, "<" if passes else ">=", applied,
           "PASS" if passes else "FAIL", trained - applied))
    log("")
    log("   the REPAIRED bands (addendum C), every edge a matched 8-layer number:")
    for nm, lo, hi in BANDS:
        mark = "  <-- HERE" if nm == band else ""
        if lo is None:
            log("     %-21s BPB <= %.6f%s" % (nm, hi, mark))
        elif hi is None:
            log("     %-21s BPB >= %.6f%s" % (nm, lo, mark))
        else:
            log("     %-21s %.6f <= BPB < %.6f%s" % (nm, lo, hi, mark))
    log("   -> %s" % band)

    rec = {"brief": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md s5 + add. C, E",
           "tag": tag, "trained": a.trained, "factors": a.factors, "labels": a.labels,
           "stats": a.stats, "model": C.MODEL_ID, "revision": C.REVISION,
           "eval_slice": meta, "scored_bytes": nb,
           "layers": layers, "k": a.k, "E": a.groups,
           "bpb": rows, "vs_chance": {k: v - CHANCE for k, v in rows.items()},
           "anchors": {"intact": INTACT, "h0_run3": H0_RUN3, "ternary_8L": TERNARY_8L,
                       "applied_8L": applied, "chance": CHANCE},
           "G_H1": {"metric": "heldout BPB, CPU fp32, frozen slice, HARD gate (add. F)",
                    "threshold": applied, "value": trained,
                    "delta": trained - applied, "passes": bool(passes),
                    "ordinal": True, "band": band},
           "G_H1a_real_shape": {"bit_identical": bool(same),
                                "hard_gate_any_router": {"bit_identical": bool(same_h),
                                                         "max_abs_diff": dmax_h},
                                "soft_gate_zero_router": {"bit_identical": bool(same_s),
                                                          "max_abs_diff": dmax_s}},
           "carve_is_live": {"differs": bool(live), "max_abs_diff": ldmax},
           "G_H1e_fp32": gh1e,
           "decomposition_DIAGNOSTIC_NOT_A_GATE": decomp,
           "train_eval_gap_soft_minus_hard": rows["trained-8L-soft"] - rows["trained-8L"],
           "train_meta": meta_train, "seconds": time.time() - t0}
    out = a.out or os.path.join(OUTDIR, "h1_eval_%s.json" % tag.replace(os.sep, "_"))
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (out, rec["seconds"]))
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
