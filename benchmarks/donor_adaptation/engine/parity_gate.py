#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PARITY GATE -- does donor_engine.c compute the same thing as PyTorch?

Runs the SAME tokens through both and compares logits. PyTorch is given exactly the weights the
exporter wrote: for --quant ternary that means every q/k/v/o/gate/up/down replaced by its
dequantized ternary form (the tied fp32 head and all norms/biases untouched), which is what
donor_engine.c reconstructs from codes+scales.

A runtime that is fast and wrong is worth nothing, and a BPB number from an unverified runtime is
worse than no number. This gate runs before any speed or quality claim.

  python parity_gate.py --weights D:/_ktmp/qwen05b_tern.bin --engine bin/donor_engine.exe \
                        --model Qwen/Qwen2.5-0.5B --quant ternary --n 8
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ternary")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
from t1_ternarize import ternarize  # noqa: E402

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--engine", default="bin/donor_engine.exe")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--quant", choices=("fp32", "ternary"), default="ternary")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ids", default=None, help="int32 token-id file; default: a fixed short prompt")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.ids:
        ids = np.fromfile(a.ids, dtype="<i4")[: a.n]
    else:
        ids = np.array(tok("The capital of France is Paris, and the capital of Italy is")
                       ["input_ids"][: a.n], dtype="<i4")
    n = len(ids)
    idf = os.path.join(os.path.dirname(a.weights) or ".", "_parity_ids.bin")
    ids.astype("<i4").tofile(idf)
    print("tokens:", ids.tolist())

    # ---- our engine
    lof = os.path.join(os.path.dirname(a.weights) or ".", "_parity_logits.bin")
    r = subprocess.run([a.engine, "--weights", a.weights, "--threads", str(a.threads),
                        "--logits", idf, str(n), lof], capture_output=True)
    if r.returncode != 0:
        print(r.stderr.decode(errors="replace")[-2000:])
        raise SystemExit("engine failed")
    print(r.stderr.decode(errors="replace").strip())
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
                                             attn_implementation="eager").eval()
    V = m.config.vocab_size
    ours = np.fromfile(lof, dtype="<f4")
    assert ours.size == n * V, "engine returned %d floats, expected %d" % (ours.size, n * V)
    ours = ours.reshape(n, V)

    # ---- PyTorch, given the SAME weights.
    # The exporter's sidecar says whether the head was ternarized too; reading it here rather
    # than taking a flag stops the reference and the engine drifting apart. On 2026-09-04 this
    # gate FAILED at rel l2 0.48 purely because the engine had a ternary head and the reference
    # did not -- the gate was right, the invocation was wrong.
    side = a.weights + ".json"
    head_tern = False
    if os.path.exists(side):
        meta = json.load(open(side))
        head_tern = bool(meta.get("head_ternary"))
        print("sidecar: quant=%s head_ternary=%s" % (meta.get("quant"), head_tern))
    n_sub = 0
    if a.quant == "ternary":
        for lay in m.model.layers:
            for organ in ATTN:
                mod = getattr(lay.self_attn, organ)
                mod.weight.data = ternarize(mod.weight.data)
                n_sub += 1
            for organ in FFN:
                mod = getattr(lay.mlp, organ)
                mod.weight.data = ternarize(mod.weight.data)
                n_sub += 1
        if head_tern:
            m.lm_head.weight = torch.nn.Parameter(ternarize(m.lm_head.weight.data.clone()),
                                                  requires_grad=False)
            n_sub += 1
    print("ternarized %d tensors in the PyTorch reference (head_ternary=%s)"
          % (n_sub, head_tern))
    with torch.no_grad():
        ref = m(torch.tensor(ids.astype(np.int64))[None, :]).logits[0].float().numpy()

    # ---- compare
    print("\n pos |    max|diff|  |  rel l2   | our argmax | ref argmax | top1 tok")
    ok = True
    worst_rel = 0.0
    for t in range(n):
        d = np.abs(ours[t] - ref[t])
        rel = float(np.linalg.norm(ours[t] - ref[t]) / (np.linalg.norm(ref[t]) + 1e-30))
        worst_rel = max(worst_rel, rel)
        am_o, am_r = int(ours[t].argmax()), int(ref[t].argmax())
        if am_o != am_r:
            ok = False
        print(" %3d | %12.6f | %9.2e | %10d | %10d | %s"
              % (t, float(d.max()), rel, am_o, am_r,
                 repr(tok.decode([am_r]))[:18]))
    top1 = float(np.mean(ours.argmax(1) == ref.argmax(1)))
    verdict = ("PASS -- the runtime reproduces PyTorch on the same weights"
               if (worst_rel < 2e-3 and top1 == 1.0) else
               "FAIL -- the runtime does NOT reproduce PyTorch; fix it before any speed or "
               "quality number is quoted")
    print("\n worst relative l2 = %.3e   top-1 agreement = %.4f" % (worst_rel, top1))
    print(" VERDICT: " + verdict)
    json.dump({"model": a.model, "quant": a.quant, "n": n, "worst_rel_l2": worst_rel,
               "top1_agreement": top1, "verdict": verdict, "tokens": ids.tolist()},
              open(os.path.join(HERE, "parity_%s.json" % a.quant), "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
