#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 step 2 (CPU, free) -- emit the two ids streams the T4 job needs besides H0's.

Brief: docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md.

H1's TRAINING stream is H0's, reused verbatim (`results/h0/h0_train.npz`, 31,250 x 512 drawn
from `calib.txt` at seed 90011).  Re-drawing one would change what "trained on" means between
H0 and H1 for no reason, and H0's provenance argument -- disjoint corpus half, seed
deliberately not 42424, offsets over a globally shuffled file -- carries over unchanged.

What H0 did NOT need and H1 does:

  h1_heldout.npz   the FROZEN 24x512 eval slice, ids only.  h1_qat.py prints a BPB off this
                   every --every steps so the run is watchable.  That number is a PROGRESS
                   metric; the gate is re-measured on this box in fp32 by h1_eval.py.
  h1_calib.npz     the 32x512 calibration slice ("calib", 32, 512, 42424) -- E23's and E37's
                   pinned draw.  G-H1e's STATIC control picks its fixed top-k groups by global
                   activation mass over THIS stream, then is scored on the held-out one.  If
                   STATIC were calibrated on the same tokens it is scored on it would be a
                   fitted control, and beating it would mean nothing.

THE HELD-OUT STREAM IS THE GATE'S OWN SLICE, AND THAT IS A TRAP WORTH NAMING.  Shipping it as
the progress metric is safe only as long as nothing SELECTS on it.  The run is time-capped, not
early-stopped, and RUN.md says to bring back the final bundle -- picking the best checkpoint by
the printed BPB would be selection on the gate's slice and would void G-H1.

Both files carry their slice metadata and are hash-checked here against the frozen constants.

Env: D_THREADS (6)
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
sys.path.insert(0, DENSDIR)
import common as C                                          # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

OUTDIR = os.path.join(HERE, "results", "h1")
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
BPT = 4.22945205479452                  # the frozen slice's own bytes/token
SCORED_BYTES = 51870


def log(m):
    print(m, flush=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # the tokenizer alone -- C.load_model would pull 6 GB of weights this file never touches,
    # and the pin is the same one it uses
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.REVISION)

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", 24, 512, 1234)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH on the held-out slice -- STOP.")
    nb = int(byts_ev.sum())
    if nb != SCORED_BYTES:
        raise SystemExit("held-out slice scores %d bytes, the frozen instrument is %d -- STOP."
                         % (nb, SCORED_BYTES))
    bpt = nb / float(ids_ev[:, 1:].numel())
    if abs(bpt - BPT) > 1e-12:
        raise SystemExit("bytes/token is %.14f, the frozen value is %.14f -- STOP." % (bpt, BPT))
    np.savez(os.path.join(OUTDIR, "h1_heldout.npz"), ids=ids_ev.numpy())
    json.dump({"what": "the FROZEN 24x512 eval slice, ids only -- h1_qat.py's PROGRESS metric, "
                       "NOT the gate.  Nothing may select on it: see h1_data.py's docstring.",
               "model": C.MODEL_ID, "revision": C.REVISION, "slice": meta_ev,
               "scored_bytes": nb, "bytes_per_token": bpt},
              open(os.path.join(OUTDIR, "h1_heldout.json"), "w", encoding="utf-8"), indent=1)
    log("  h1_heldout.npz  %s  ids sha %s  %d bytes  %.14f B/token"
        % (tuple(ids_ev.shape), meta_ev["ids_sha256"][:16], nb, bpt))

    ids_cal, _, meta_cal = C.get_slice(tok, "calib", 32, 512, 42424)
    if meta_cal["ids_sha256"] == EXPECT_IDS_SHA:
        raise SystemExit("the calib slice hashes to the EVAL slice -- STOP, they must differ.")
    np.savez(os.path.join(OUTDIR, "h1_calib.npz"), ids=ids_cal.numpy())
    json.dump({"what": "the 32x512 calibration slice (calib, 32, 512, 42424) -- E23/E37's "
                       "pinned draw.  G-H1e's STATIC control is calibrated HERE and scored on "
                       "the held-out slice; calibrating and scoring on the same tokens would "
                       "make STATIC a fitted control.",
               "model": C.MODEL_ID, "revision": C.REVISION, "slice": meta_cal},
              open(os.path.join(OUTDIR, "h1_calib.json"), "w", encoding="utf-8"), indent=1)
    log("  h1_calib.npz    %s  ids sha %s  (part %s, disjoint half)"
        % (tuple(ids_cal.shape), meta_cal["ids_sha256"][:16], meta_cal["part"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
