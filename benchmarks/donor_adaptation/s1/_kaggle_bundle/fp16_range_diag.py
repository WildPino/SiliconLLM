#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why does the 1.5B GPU run come back NaN when the 0.5B one does not?

Diagnosis, not a probe. Runs the donor in fp32 on CPU over the SHARED eval slice and records,
per layer, how close the tensors the S1 probe actually touches come to the fp16 limits:

    fp16 max normal      65504
    fp16 min normal      6.104e-05   (below this, precision degrades into subnormals)

The S1 probe masks on |h| where h = act_fn(gate(x)) * up(x) -- the FFN intermediate. If |h|
exceeds 65504 anywhere, that tensor is +/-Inf in fp16; Inf - Inf or 0*Inf downstream is NaN,
and every subsequent comparison against a threshold is False. That is a RANGE fact about the
donor, and it is size-specific: it needs no GPU to establish.

Reports the two rival explanations separately so they can be told apart:
  (a) OVERFLOW  -- the fp32 values genuinely exceed the fp16 range  -> a real numeric problem
  (b) NO-OVERFLOW -- they do not, and the NaN came from the masking code -> a code problem

Usage:  D_THREADS=6 python fp16_range_diag.py [--model qwen2.5-1.5b] [--seqs 4]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
import common as C  # noqa: E402

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--seqs", type=int, default=4, help="sequences from the shared slice")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

    import s1_sparsity_bpb as S
    repo, rev = S.MODELS[a.model]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32,
                                                 revision_kwargs=None) \
        if False else AutoModelForCausalLM.from_pretrained(repo, revision=rev,
                                                           torch_dtype=torch.float32)
    model.eval()

    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    ids = ids[: a.seqs]
    print("slice ids_sha256 =", meta.get("ids_sha256"))

    layers = model.model.layers
    stats = {}

    def mk_hook(li):
        def hook(mod, inp, out):
            # mod is the MLP; recompute its intermediate exactly as the probe defines it
            x = inp[0]
            h = mod.act_fn(mod.gate_proj(x)) * mod.up_proj(x)
            am = h.abs()
            d = stats.setdefault(li, {"max_abs_h": 0.0, "n_over_fp16max": 0, "n": 0,
                                      "max_abs_gate": 0.0, "max_abs_up": 0.0})
            d["max_abs_h"] = max(d["max_abs_h"], float(am.max()))
            d["n_over_fp16max"] += int((am > FP16_MAX).sum())
            d["n"] += int(am.numel())
            d["max_abs_gate"] = max(d["max_abs_gate"], float(mod.gate_proj(x).abs().max()))
            d["max_abs_up"] = max(d["max_abs_up"], float(mod.up_proj(x).abs().max()))
        return hook

    hs = [layers[i].mlp.register_forward_hook(mk_hook(i)) for i in range(len(layers))]
    with torch.no_grad():
        for i in range(ids.shape[0]):
            model(ids[i: i + 1])
    for h in hs:
        h.remove()

    print("\n  layer |    max|h|  |  over fp16max |   max|gate|  |    max|up|   | fp16?")
    worst = 0.0
    n_over_layers = 0
    for li in sorted(stats):
        d = stats[li]
        worst = max(worst, d["max_abs_h"])
        over = d["n_over_fp16max"]
        if over:
            n_over_layers += 1
        print("  %5d | %11.1f | %13d | %12.1f | %12.1f | %s"
              % (li, d["max_abs_h"], over, d["max_abs_gate"], d["max_abs_up"],
                 "OVERFLOW" if over else "ok"))

    verdict = ("OVERFLOW -- the donor's FFN intermediate genuinely exceeds the fp16 range; "
               "the NaN is a REAL numeric problem of running this donor in fp16"
               if n_over_layers else
               "NO-OVERFLOW -- every value fits in fp16 with margin (worst %.1f vs %.1f). "
               "The NaN did NOT come from the range of h, so it came from the masking/"
               "threshold code, and that is a CODE bug to find rather than a numeric one"
               % (worst, FP16_MAX))
    print("\n  worst |h| over %d sequences = %.1f   (fp16 max = %.1f, headroom x%.1f)"
          % (ids.shape[0], worst, FP16_MAX, FP16_MAX / max(worst, 1e-9)))
    print("  layers overflowing: %d / %d" % (n_over_layers, len(stats)))
    print("  VERDICT: " + verdict)

    out = a.out or os.path.join(C.RESULTS, "fp16_range_diag_%s.json" % a.model)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "repo": repo, "revision": rev,
                   "n_sequences": int(ids.shape[0]), "slice_meta": meta,
                   "fp16_max": FP16_MAX, "fp16_min_normal": FP16_MIN_NORMAL,
                   "worst_abs_h": worst, "n_layers_overflowing": n_over_layers,
                   "per_layer": stats, "verdict": verdict}, fh, indent=1)
    print("  wrote", out)


if __name__ == "__main__":
    main()
