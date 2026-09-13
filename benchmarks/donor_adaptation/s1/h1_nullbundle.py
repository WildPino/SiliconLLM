#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the NULL BUNDLE -- h1_eval.py's planted control.  CPU, seconds, free.

A bundle in h1_qat.py's exact key layout whose contents are UNTRAINED: the donor's own
gate/up/down, the measured activation RMS, the D0c partition, and E37's OWN fitted router.
That is, by construction, precisely the model h1_applied.py measured as `applied-8L`.

WHY THIS EXISTS.  h1_eval.py's decomposition attributes a BPB delta to three causes.  An
attribution is exactly the kind of plausible artefact this programme keeps failing on: it will
happily print three numbers that sum correctly and mean nothing, if the arms are wired to the
wrong thing.  Run through h1_eval.py this bundle must give

    arm-E  == arm-ER == applied-8L  to the last bit        experts 0.000000, router 0.000000

because nothing was trained and the router IS E37's.  Any other reading means the arms are not
isolating what their names claim, and the decomposition is void BEFORE the T4 hours are spent.
The remaining term, `soft vs hard gate`, is then the whole delta -- and that number is itself
worth having: it prices the gate form alone, with zero training in the model.

Usage:
  python h1_nullbundle.py --labels ../density/results/d0c_labels/labels_E256.npz \
      --stats results/h1/h1_actstats.npz --out results/h1/h1_null_bundle.npz
"""
import argparse
import json
import os
import sys

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
E37_ROUTERS = "D:/_ktmp/e37/e37_routers_E256.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--routers", default=E37_ROUTERS)
    ap.add_argument("--layers", default=",".join(str(x) for x in H1.H1_LAYERS))
    ap.add_argument("--k", type=int, default=H1.K_DEFAULT)
    ap.add_argument("--groups", type=int, default=H1.E_GROUPS)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "h1", "h1_null_bundle.npz"))
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",") if x.strip() != ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    model, _ = C.load_model(dtype=torch.float32)
    model.eval()
    H1.build_ffn(model, a.labels, a.stats, layers, a.k, a.groups, "cpu")
    mods = H1.ffn_mods(model, layers)

    rz = np.load(a.routers)
    store = {}
    for li, m in mods:
        w = torch.from_numpy(rz["r%d" % li]).float()
        if tuple(w.shape) != tuple(m.router.shape):
            raise SystemExit("router r%d is %s, module wants %s -- STOP"
                             % (li, tuple(w.shape), tuple(m.router.shape)))
        m.router.data.copy_(w)
        p = "L%02d" % li
        store[p + ".gate"] = m.gate.detach().float().cpu().numpy()
        store[p + ".up"] = m.up.detach().float().cpu().numpy()
        store[p + ".down"] = m.down.detach().float().cpu().numpy()
        store[p + ".router"] = m.router.detach().float().cpu().numpy()
        store[p + ".rms_in"] = m.rms_in.detach().float().cpu().numpy()
        store[p + ".rms_h"] = m.rms_h.detach().float().cpu().numpy()
        store[p + ".labels"] = m.labels.detach().cpu().numpy()
    np.savez(a.out, **store)
    json.dump({"brief": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md -- h1_eval control",
               "what": "NULL BUNDLE: donor experts + E37 routers, ZERO training.  Through "
                       "h1_eval.py it must give experts 0.000000 and router 0.000000, and "
                       "arm-E == applied-8L.  NOT a model, NOT a result.",
               "complete": True, "layers": layers, "k": a.k, "E": a.groups,
               "steps_requested": 0, "router_lr": 0.0, "aux": 0.0, "seconds": 0.0,
               "model": C.MODEL_ID, "revision": C.REVISION, "routers": a.routers},
              open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), indent=1)
    print("wrote %s  (%d layers, %d arrays)" % (a.out, len(mods), len(store)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
