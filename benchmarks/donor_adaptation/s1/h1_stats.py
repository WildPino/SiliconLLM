#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 step 0a (CPU, FREE) -- capture the FFN activation RMS that rule R3 needs.

Brief: docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md s3.

WHY THIS FILE EXISTS, AND IT IS A DEFECT I FOUND IN MY OWN RUN.  h1_qat.build_ffn falls back
to `ones` when no stats file is given, and its own docstring says what that means: *"the R3
rule is activation-weighted and using ones here would silently change the format from the one
the engine ships."*  The first applied-8L run was launched WITHOUT --stats, so it measured a
DIFFERENT quantization than E37's.  It is kept as a sensitivity arm and is NOT the registered
anchor.  This file produces the missing input.

NOTHING IS REIMPLEMENTED.  The capture is `qwen_export.capture_act_rms`, imported -- the ONE
definition the exporter and E1's runner already share, over E23/T2's pinned calibration slice
(`calib`, 32, 512, 42424), which build_calib.py guarantees is a DISJOINT corpus half from the
eval slice and which that function asserts.  Re-deriving it here would compare two
implementations instead of using the rule.

WHAT MAPS TO WHAT.  capture_act_rms keys by (layer, organ) and hooks the organ INPUTS:
    (li, "gate_proj")  = RMS of the FFN input x   -> `rms_in`  [D]   (up_proj shares it)
    (li, "down_proj")  = RMS of the hidden h      -> `rms_h`   [F]
which is exactly the pair TernaryCarvedFFN quantizes gate/up against and down against.

CAPTURED ON THE PLAIN DONOR, ON PURPOSE.  E37 calibrated on the dense donor and applied-8L has
to match E37 or it is not a control.  The H0 q/o organs are NOT installed for the capture.

Env: D_THREADS (6)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(HERE, "..", "density")),
           os.path.abspath(os.path.join(HERE, "..", "ternary")),
           os.path.abspath(os.path.join(HERE, "..", "engine"))):
    sys.path.insert(0, _p)

import common as C                                          # noqa: E402
import qwen_export as Q                                     # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))
OUTDIR = os.path.join(HERE, "results", "h1")
CALIB_SEQS = 32                                             # T2/E23/E37's pinned value


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-seqs", type=int, default=CALIB_SEQS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()

    if a.calib_seqs != CALIB_SEQS:
        log("  *** WARNING: --calib-seqs %d != %d.  E37's numbers were calibrated at %d; this "
            "is a DIFFERENT quantization and applied-8L would stop being a matched control."
            % (a.calib_seqs, CALIB_SEQS, CALIB_SEQS))

    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    L = model.config.num_hidden_layers
    log("== H1 activation stats ==  CPU fp32, %d layers, %d calib seqs" % (L, a.calib_seqs))

    act = Q.capture_act_rms(model, tok, a.calib_seqs, L)

    store, rows = {}, []
    for li in range(L):
        rin = act[(li, "gate_proj")].float().cpu()
        rh = act[(li, "down_proj")].float().cpu()
        # up_proj must SHARE the gate_proj input, or the two organs are being quantized
        # against different calibrations of the same tensor.  capture_act_rms aliases them;
        # asserted here rather than trusted.
        if not torch.equal(act[(li, "up_proj")].float().cpu(), rin):
            raise SystemExit("L%02d: up_proj and gate_proj do not share an input RMS -- STOP"
                             % li)
        store["L%02d.rms_in" % li] = rin.numpy()
        store["L%02d.rms_h" % li] = rh.numpy()
        rows.append({"layer": li, "D": int(rin.numel()), "F": int(rh.numel()),
                     "rms_in_mean": float(rin.mean()), "rms_in_max": float(rin.max()),
                     "rms_h_mean": float(rh.mean()), "rms_h_max": float(rh.max())})

    out = a.out or os.path.join(OUTDIR, "h1_actstats.npz")
    np.savez(out, **store)
    json.dump({"brief": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md s3",
               "source": "qwen_export.capture_act_rms, imported -- not reimplemented",
               "model": C.MODEL_ID, "revision": C.REVISION,
               "calib_slice": ["calib", a.calib_seqs, 512, 42424],
               "captured_on": "the PLAIN donor (no H0 q/o installed) -- matches E37",
               "layers": L, "rows": rows, "npz": out, "seconds": time.time() - t0},
              open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8"), indent=1)

    sp = [r["rms_in_mean"] for r in rows]
    log("   rms_in  mean over layers %.4f   min-layer %.4f   max-layer %.4f"
        % (sum(sp) / len(sp), min(sp), max(sp)))
    sp = [r["rms_h_mean"] for r in rows]
    log("   rms_h   mean over layers %.4f   min-layer %.4f   max-layer %.4f"
        % (sum(sp) / len(sp), min(sp), max(sp)))
    log("   wrote %s  [%.0fs]" % (out, time.time() - t0))
    log("   Now re-run h1_applied.py WITH --stats %s -- the run without it is not E37's format."
        % os.path.basename(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
