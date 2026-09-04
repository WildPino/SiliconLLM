#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where does the fp16 NaN actually enter? Find the FIRST non-finite tensor in the forward.

fp16_range_diag.py refuted the obvious hypothesis: the FFN intermediate |h| peaks at 4541 against
an fp16 max of 65504, i.e. 14.4x headroom, so the overflow is not there. That leaves everything
upstream, and this walks the whole stack in fp16 ON CPU -- no GPU needed, because whether a value
is representable in fp16 is a property of the number, not of the device.

Hooks every submodule, in execution order, and reports the FIRST one whose output is non-finite,
plus the fp16 headroom of every module that stayed finite. Also runs the same pass in fp32 so the
two can be compared module by module: a module that is finite in fp32 and non-finite in fp16 is
the conversion casualty, and its fp32 magnitude says whether it overflowed or something else
(0*inf, inf-inf, a division) produced the NaN.

Usage:  python fp16_first_nan.py [--model qwen2.5-1.5b] [--tokens 128]
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


def walk(model, ids, dtype, limit_report=12):
    """Returns (ordered list of module records, index of first non-finite)."""
    recs = []
    order = []

    def hook(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            tf = t.float()
            finite = bool(torch.isfinite(tf).all())
            mx = float(tf.abs().max()) if finite else float("nan")
            recs.append({"i": len(recs), "name": name, "cls": type(mod).__name__,
                         "finite": finite, "max_abs": mx,
                         "n_nonfinite": 0 if finite else int((~torch.isfinite(tf)).sum()),
                         "numel": int(t.numel())})
            order.append(name)
        return f

    hs = []
    for name, mod in model.named_modules():
        if name and len(list(mod.children())) == 0:      # leaves only
            hs.append(mod.register_forward_hook(hook(name)))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    first = next((r["i"] for r in recs if not r["finite"]), None)
    return recs, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

    import s1_sparsity_bpb as S
    repo, rev = S.MODELS[a.model]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    ids = ids[:1, : a.tokens]
    print("slice ids_sha256 =", meta.get("ids_sha256"), "| tokens =", ids.shape[1])

    out = {"model": a.model, "repo": repo, "revision": rev, "tokens": int(ids.shape[1])}
    per = {}
    for dt in (torch.float32, torch.float16):
        m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dt).eval()
        recs, first = walk(m, ids, dt)
        per[str(dt)] = {"first_nonfinite_index": first, "n_modules": len(recs)}
        print("\n== %s ==  modules traced: %d" % (dt, len(recs)))
        if first is None:
            top = sorted(recs, key=lambda r: -r["max_abs"])[:8]
            print("   all finite. largest outputs:")
            for r in top:
                print("      %-52s %12.1f  (fp16 headroom x%.1f)"
                      % (r["name"], r["max_abs"], FP16_MAX / max(r["max_abs"], 1e-9)))
            per[str(dt)]["largest"] = top
        else:
            lo = max(0, first - 6)
            print("   FIRST NON-FINITE at module #%d" % first)
            print("   context (the six before it, then it):")
            for r in recs[lo:first + 1]:
                print("      #%-4d %-52s %-9s %12s"
                      % (r["i"], r["name"], "FINITE" if r["finite"] else "NON-FINITE",
                         ("%.1f" % r["max_abs"]) if r["finite"]
                         else "%d bad" % r["n_nonfinite"]))
            per[str(dt)]["context"] = recs[lo:first + 1]
        del m

    f32, f16 = per["torch.float32"], per["torch.float16"]
    if f16["first_nonfinite_index"] is None:
        v = ("NO-NAN-ON-CPU -- the fp16 forward is clean here over %d tokens. The Kaggle NaN is "
             "then NOT a pure representability fact: it needs the GPU kernels (SDPA/flash "
             "backend, or a fused path) or a longer sequence to appear. Next: reproduce with the "
             "same attn_implementation the Kaggle run used, and at 512 tokens." % ids.shape[1])
    elif f32["first_nonfinite_index"] is None:
        v = ("FP16-ONLY -- module #%d is finite in fp32 and non-finite in fp16. That module is "
             "the conversion casualty and the fix belongs there (upcast it, or run that organ "
             "in fp32)." % f16["first_nonfinite_index"])
    else:
        v = "NON-FINITE IN FP32 TOO -- the problem is not the dtype. Look at the slice or the model."
    print("\nVERDICT: " + v)
    out["per_dtype"] = per
    out["verdict"] = v
    p = os.path.join(C.RESULTS, "fp16_first_nan_%s.json" % a.model)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
