#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E25 GATE -- does donor_engine.c's FACTORED path compute what PyTorch computes?

Phase 60's law: kernel-bit-exact does NOT compose to system-correctness, so a new kernel is
gated on END-TO-END parity and never on a microbenchmark.  This is that gate for the one kind
the engine did not have until now.

WHAT IS BEING COMPARED.  Both sides run the SAME object: Qwen2.5-1.5B at the pinned revision,
fp32 everywhere, except q_proj and o_proj on all 28 layers, which are the ternary low-rank
factors from h0_factors.npz -- E22's `QO512-TB`, the arm that reads BPB 2.812226 / tf 28 / free
1, and the state H0 starts from.  The reference installs h0_qat.TernaryLowRank, i.e. the SAME
module the T4 trains and the SAME module h0_eval.py measured the start state with; the engine
reads the exported .bin.  Nothing here re-derives the quantization: if the exporter's two
r3_actsearch calls and TernaryLowRank._quant ever disagree, this gate is what catches it.

WHY IT MATTERS MORE THAN THE USUAL PARITY RUN.  E21 and E22 measured the rank cut by computing
A.B and installing a DENSE matrix.  Nothing in this programme had ever EXECUTED the factored
form as two GEMVs with an intermediate of size r until h0_eval.py did it on CPU, and nothing had
ever executed it in the engine at all.  Every tok/s consequence of the rank axis was arithmetic.
This gate is the precondition for making it a measurement.

  python e25_parity_factored.py --weights D:/_ktmp/e25/qwen15b_qo512tb.bin \
      --engine ./donor_engine_e25.exe --factors ../s1/results/h0/h0_factors.npz --n 10
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(HERE, "..", "density")),
           os.path.abspath(os.path.join(HERE, "..", "s1"))):
    sys.path.insert(0, _p)

import common as C          # noqa: E402  -- the pinned MODEL_ID/REVISION, one definition
import h0_qat as H0         # noqa: E402  -- the module the T4 trains and h0_eval measures

# parity_gate.py's bar, unchanged and not re-argued: a runtime that is fast and wrong is worth
# nothing, and 2e-3 relative l2 on fp32 logits is roundoff-scale for a 1536-wide reduction.
REL_TOL = 2e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--engine", default="./donor_engine_e25.exe")
    ap.add_argument("--factors", default=os.path.join(HERE, "..", "s1", "results", "h0",
                                                      "h0_factors.npz"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e25_parity_factored.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()

    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    ids = np.array(tok("The capital of France is Paris, and the capital of Italy is")
                   ["input_ids"][: a.n], dtype="<i4")
    n = len(ids)
    tmp = os.path.dirname(a.weights) or "."
    idf, lof = os.path.join(tmp, "_e25_ids.bin"), os.path.join(tmp, "_e25_logits.bin")
    ids.tofile(idf)
    print("tokens (%d): %s" % (n, ids.tolist()), flush=True)

    # ---- the engine
    r = subprocess.run([a.engine, "--weights", a.weights, "--threads", str(a.threads),
                        "--logits", idf, str(n), lof], capture_output=True)
    print(r.stderr.decode(errors="replace").strip(), flush=True)
    if r.returncode != 0:
        raise SystemExit("engine failed")
    V = model.config.vocab_size
    ours = np.fromfile(lof, dtype="<f4")
    assert ours.size == n * V, "engine returned %d floats, expected %d" % (ours.size, n * V)
    ours = ours.reshape(n, V)

    # ---- the reference: the SAME factored modules, installed by h0_qat's own builder
    n_org = H0.build(model, a.factors, list(range(model.config.num_hidden_layers)), "cpu")
    print("installed %d factored organs in the PyTorch reference from %s"
          % (n_org, os.path.basename(a.factors)), flush=True)
    with torch.no_grad():
        ref = model(torch.tensor(ids.astype(np.int64))[None, :]).logits[0].float().numpy()

    print("\n pos |    max|diff|  |  rel l2   | our argmax | ref argmax | top1 tok")
    worst = 0.0
    for t in range(n):
        d = np.abs(ours[t] - ref[t])
        rel = float(np.linalg.norm(ours[t] - ref[t]) / (np.linalg.norm(ref[t]) + 1e-30))
        worst = max(worst, rel)
        ao, ar = int(ours[t].argmax()), int(ref[t].argmax())
        print(" %3d | %12.6f | %9.2e | %10d | %10d | %s"
              % (t, float(d.max()), rel, ao, ar, repr(tok.decode([ar]))[:18]))
    top1 = float(np.mean(ours.argmax(1) == ref.argmax(1)))
    ok = (worst < REL_TOL and top1 == 1.0)
    print("\n worst relative l2 = %.3e   top-1 agreement = %.4f   [%.0fs]"
          % (worst, top1, time.time() - t0))
    print(" GATE E25-P: " + ("FIRES -- the engine's factored path reproduces PyTorch on the "
                             "same factors" if ok else
                             "FAILS -- the factored path does NOT reproduce PyTorch; no speed "
                             "or quality number may be quoted from it"))
    json.dump({"gate": "E25-P", "weights": a.weights, "factors": a.factors,
               "model": C.MODEL_ID, "revision": C.REVISION,
               "n": n, "organs_installed": n_org, "worst_rel_l2": worst,
               "top1_agreement": top1, "rel_tol": REL_TOL, "fires": bool(ok),
               "tokens": ids.tolist(), "seconds": time.time() - t0},
              open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
