#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 step 0 (CPU, FREE) -- measure `applied-8L`, the control H1's gate argues with.

Brief: docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md s4,
plus `G-H1d` in s5.1.  Brief s10 item 3 requires this number BEFORE the T4 runs, recorded in
an addendum so it cannot be adjusted afterwards.

WHY IT DOES NOT ALREADY EXIST.  E37's anchors are measured over ALL 28 layers; H1 can only
train 8 of them (17.2 GB of AdamW state does not fit a 16 GB T4 -- brief s2).  Comparing a
trained-8-layer model against an applied-28-layer number would credit training with the 20
layers H1 never touched.  `applied-8L` is the matched control: **the same 8 layers, the same
rule, the same k, APPLIED post-hoc instead of trained.**

MATCHED MEANS MATCHED, AND THAT DECIDES THREE THINGS.
  1. The base is **H0 run 3's state** (`h0_trained3.npz`), not the bare donor -- that is H1's
     start line, so the control must start there too.
  2. The router is **E37's OWN fitted ridge router** (`D:/_ktmp/e37/e37_routers_E256.npz`,
     [E, D] per layer, written by e37_fit_routers.py off e23_router.fit_routers).  Re-fitting
     one here would measure a different post-hoc router than the one E37's number came from.
  3. The gate is **HARD** -- selected groups pass at exactly 1.0, unselected at 0.  That is
     what carve_common does in the engine and what E37 measured.  H1's TRAINED arm uses the
     renormalised soft gate, and the difference is part of what H1 is testing; the control must
     not borrow it.

WHAT IS MEASURED, all on the frozen 24x512 slice in fp32 on this machine:

  intact        the donor untouched -- PLANTED CONTROL, must reproduce 0.767595
  h0-run3       q/o from H0 run 3 installed, FFN untouched -- H1's start line, ~0.810022
  ternary-8L    + ternary FFN on the 8 layers, k = E (NO carve)
  applied-8L    + the carve at k = 16, E37's router, hard gate     <- THE NUMBER

  G-H1d (ORDINAL, no tolerance): applied-8L > ternary-8L.  Carving the 8 layers must be
  strictly worse than not carving them -- E37's direction, reproduced on the matched shape.
  If it does not hold, the control is measuring the 8-layer restriction and not the carve.

Grad is OFF throughout and that is correct here: nothing is trained.  Env: D_THREADS (6).
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
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453
CHANCE = 4.069819
INTACT = 0.767595                       # E12's dense anchor, re-measured here as a control
H0_RUN3 = 0.810022                      # H0 run 3, re-measured here
E37_ROUTERS = "D:/_ktmp/e37/e37_routers_E256.npz"


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


@torch.no_grad()
def bpb(model, ids, nbytes):
    nats = []
    for i in range(ids.shape[0]):
        ch = ids[i:i + 1]
        lg = model(ch).logits
        lp = torch.nn.functional.log_softmax(lg[:, :-1].float(), dim=-1)
        nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double())
    return float(torch.stack(nats).sum() / (LN2 * nbytes))


def load_model(tok_out=False):
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    return (model, tok) if tok_out else model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", required=True, help="h0_trained3.npz -- the q/o base")
    ap.add_argument("--labels", required=True, help="labels_E256.npz")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--routers", default=E37_ROUTERS, help="E37's OWN fitted routers")
    ap.add_argument("--layers", default=",".join(str(x) for x in H1.H1_LAYERS))
    ap.add_argument("--k", type=int, default=H1.K_DEFAULT)
    ap.add_argument("--groups", type=int, default=H1.E_GROUPS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",") if x.strip() != ""]
    t0 = time.time()

    log("== H1 applied-8L ==  CPU fp32, layers %s, k %d of E %d" % (layers, a.k, a.groups))

    model, tok = load_model(tok_out=True)
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
                         "instrument has drifted and NOTHING measured after this line counts.")

    n_qo, qo_layers = H1.build_qo(model, a.factors, "cpu")
    rows["h0-run3"] = bpb(model, ids, nb)
    log("   h0-run3     %.6f   (%d organs over %d layers; H0 run 3 read %.6f)"
        % (rows["h0-run3"], n_qo, len(qo_layers), H0_RUN3))

    H1.build_ffn(model, a.labels, a.stats, layers, a.k, a.groups, "cpu")
    mods = H1.ffn_mods(model, layers)

    # E37's own fitted routers, installed verbatim.  Shape is asserted, not assumed.
    rz = np.load(a.routers)
    for li, m in mods:
        w = torch.from_numpy(rz["r%d" % li]).float()
        if tuple(w.shape) != tuple(m.router.shape):
            raise SystemExit("router r%d is %s, module wants %s -- STOP"
                             % (li, tuple(w.shape), tuple(m.router.shape)))
        m.router.data.copy_(w)
        m.hard_gate = True                 # E23/E37's applied carve: gates exactly {0,1}

    # G-H1a on the REAL shape, not just the toy: k = E must be bit-identical to uncarved.
    x = torch.randn(1, 6, model.config.hidden_size)
    same, dmax = H1.g_h1a(mods[0][1], x)
    log("   G-H1a (real shape, layer %d)  bit-identical at k=E: %s  (max|d| %.1e)"
        % (mods[0][0], "FIRES" if same else "*** FAILS ***", dmax))
    if not same:
        raise SystemExit("G-H1a FAILS on the real shape.  STOP.")

    for _, m in mods:
        m.k = a.groups
    rows["ternary-8L"] = bpb(model, ids, nb)
    log("   ternary-8L  %.6f   (k = E = %d, NO carve)" % (rows["ternary-8L"], a.groups))

    for _, m in mods:
        m.k = a.k
    rows["applied-8L"] = bpb(model, ids, nb)
    log("   applied-8L  %.6f   (k = %d, E37's router, hard gate)  <- THE NUMBER"
        % (rows["applied-8L"], a.k))

    fires = rows["applied-8L"] > rows["ternary-8L"]
    log("")
    log("   G-H1d  carving the 8 layers is strictly worse than not carving them (ORDINAL)")
    log("          applied-8L %.6f  >  ternary-8L %.6f   ->  %s  (delta %+.6f)"
        % (rows["applied-8L"], rows["ternary-8L"],
           "FIRES" if fires else "*** FAILS ***",
           rows["applied-8L"] - rows["ternary-8L"]))
    if not fires:
        log("          A FAILURE HERE IS NOT H1's NULL: it says the matched control is reading")
        log("          the 8-layer restriction rather than the carve, and the anchor is unusable.")

    log("")
    log("   H1's bands, with applied-8L now filled in:")
    log("     CARVE-NOT-TRAINABLE   BPB >= %.6f" % rows["applied-8L"])
    log("     TRAINING-HELPS        3.475707 <= BPB < %.6f" % rows["applied-8L"])
    log("     CARVE-IS-TRAINABLE    0.810022 <  BPB < 3.475707")
    log("     CARVE-IS-FREE         BPB <= 0.810022")

    rec = {"brief": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md s4 + G-H1d",
           "model": C.MODEL_ID, "revision": C.REVISION, "eval_slice": meta,
           "scored_bytes": nb, "layers": layers, "k": a.k, "E": a.groups,
           "factors": a.factors, "routers": a.routers, "gate_mode": "hard",
           "bpb": rows, "vs_chance": {k: v - CHANCE for k, v in rows.items()},
           "anchors_expected": {"intact": INTACT, "h0_run3": H0_RUN3,
                                "e37_ternary_all_28L": 3.475707,
                                "e37_carved_k16_28L": 3.986801, "chance": CHANCE},
           "G_H1a_real_shape": {"bit_identical": bool(same), "max_abs_diff": dmax},
           "G_H1d": {"applied_8L": rows["applied-8L"], "ternary_8L": rows["ternary-8L"],
                     "delta": rows["applied-8L"] - rows["ternary-8L"], "fires": bool(fires)},
           "seconds": time.time() - t0}
    out = a.out or os.path.join(OUTDIR, "h1_applied_8L.json")
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (out, rec["seconds"]))
    log("  RECORD THIS IN AN ADDENDUM BEFORE THE T4 RUNS (brief s10 item 3).")
    return 0 if fires and same else 1


if __name__ == "__main__":
    sys.exit(main())
