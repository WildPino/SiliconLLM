#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE_NAN_HUNT_1P5B.md sec.4 -- where does the 1.5B fp16 NaN enter, ON THE GPU.

fp16_range_diag.py and fp16_first_nan.py (both CPU) refuted "the numbers overflow fp16" and
"fp16 arithmetic is the problem" -- everything finite on CPU, in fp16, at 128 and 512 tokens.
Same weights, same slice, same dtype: finite on CPU, NaN on the T4. So the remaining variable is
the GPU EXECUTION PATH: which attention backend runs, and whether fp32-on-GPU is also affected.

This reuses `walk()` from fp16_first_nan.py UNCHANGED (import, not copy) and loops it over:
  eager (fp16), sdpa-default (fp16), sdpa+MATH (fp16), sdpa+EFFICIENT_ATTENTION (fp16),
  sdpa+FLASH_ATTENTION (fp16), and one fp32-on-GPU control (sdpa-default, matching what the
  failing Kaggle run used).

512 tokens -- the real slice length the production run used (not the 128-token CPU sanity check).

Reads only the JSON it writes back. Prints per-config verdict and dumps the pre-registered
outcome-table row it matches (handoff sec.4) so the reading cannot be picked after the fact.

Usage (on Kaggle, GPU on):  python fp16_gpu_nan_hunt.py [--model qwen2.5-1.5b] [--tokens 512]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
import common as C  # noqa: E402
from fp16_first_nan import walk  # noqa: E402  -- SAME walk(), not re-implemented

FP16_MAX = 65504.0

# (label, dtype, attn_implementation, sdpa_backend_or_None)
CONFIGS = [
    ("eager_fp16",              torch.float16, "eager", None),
    ("sdpa_default_fp16",       torch.float16, "sdpa",  None),
    ("sdpa_MATH_fp16",          torch.float16, "sdpa",  "MATH"),
    ("sdpa_EFFICIENT_fp16",     torch.float16, "sdpa",  "EFFICIENT_ATTENTION"),
    ("sdpa_FLASH_fp16",         torch.float16, "sdpa",  "FLASH_ATTENTION"),
    ("sdpa_default_fp32_CTRL",  torch.float32, "sdpa",  None),   # A1.3-irrelevant: diagnostic only
]

_BACKEND_ENUM = None  # resolved lazily; absent entirely on CPU-only torch builds


def _backend(name):
    global _BACKEND_ENUM
    if _BACKEND_ENUM is None:
        import torch.nn.attention as A
        _BACKEND_ENUM = A.SDPBackend
    return getattr(_BACKEND_ENUM, name)


def run_one(label, dtype, attn_impl, sdpa_backend, repo, rev, ids, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, dtype=dtype, attn_implementation=attn_impl
    ).to(device).eval()
    dtypes_seen = sorted({str(p.dtype) for p in m.parameters()})
    ids_dev = ids.to(device)

    def _do():
        return walk(m, ids_dev, dtype)

    if sdpa_backend is None:
        recs, first = _do()
    else:
        import torch.nn.attention as A
        with A.sdpa_kernel(_backend(sdpa_backend)):
            recs, first = _do()

    rec = {"label": label, "dtype_requested": str(dtype), "attn_implementation": attn_impl,
           "sdpa_backend_forced": sdpa_backend, "param_dtypes_achieved": dtypes_seen,
           "n_modules": len(recs), "first_nonfinite_index": first}
    if first is None:
        top = sorted(recs, key=lambda r: -r["max_abs"])[:8]
        rec["finite"] = True
        rec["largest"] = top
        print("  %-26s FINITE  (%d modules, dtypes=%s)" % (label, len(recs), dtypes_seen))
    else:
        lo = max(0, first - 6)
        rec["finite"] = False
        rec["first_nonfinite_module"] = recs[first]["name"]
        rec["context"] = recs[lo:first + 1]
        print("  %-26s NON-FINITE at #%d %s  (dtypes=%s)"
              % (label, first, recs[first]["name"], dtypes_seen))
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--tokens", type=int, default=512)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "this job is meaningless without a GPU -- refusing to run on CPU"
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0), "| capability",
          torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor,
          "| torch", torch.__version__, "| bf16_supported_achieved(unused flag, sec.3)",
          torch.cuda.is_bf16_supported())

    import s1_sparsity_bpb as S
    repo, rev = S.MODELS[a.model]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    ids, byts, meta = C.get_slice(tok, "heldout", 24, 512, 1234)
    ids = ids[:1, : a.tokens]
    print("slice ids_sha256 =", meta.get("ids_sha256"), "| tokens =", ids.shape[1])
    assert meta.get("ids_sha256") == "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65", \
        "slice hash mismatch -- not the pinned span, refusing to report a reading on it"

    out = {"model": a.model, "repo": repo, "revision": rev, "tokens": int(ids.shape[1]),
           "gpu": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "bf16_supported_achieved": bool(torch.cuda.is_bf16_supported())},
           "slice_ids_sha256": meta.get("ids_sha256"), "configs": []}

    print("\nrunning %d configurations, %d tokens each:" % (len(CONFIGS), ids.shape[1]))
    for label, dtype, attn_impl, backend in CONFIGS:
        try:
            rec = run_one(label, dtype, attn_impl, backend, repo, rev, ids, device)
        except Exception as exc:
            rec = {"label": label, "dtype_requested": str(dtype), "attn_implementation": attn_impl,
                   "sdpa_backend_forced": backend, "error": repr(exc)}
            print("  %-26s ERROR: %r" % (label, exc))
        out["configs"].append(rec)

    # ---- pre-registered reading (handoff sec.4) -- classify, do not editorialise
    fp16_labels = [c["label"] for c in out["configs"]
                   if c.get("dtype_requested") == "torch.float16"]
    fp16_finite = {c["label"]: c.get("finite") for c in out["configs"] if c["label"] in fp16_labels}
    fp32_ctrl = next((c for c in out["configs"] if c["label"] == "sdpa_default_fp32_CTRL"), None)
    eager = fp16_finite.get("eager_fp16")
    sdpa_any_finite = any(v for k, v in fp16_finite.items() if k != "eager_fp16" and v is not None)
    sdpa_all_finite = all(v for k, v in fp16_finite.items() if k != "eager_fp16" and v is not None)
    all_fp16_nonfinite = not any(v for v in fp16_finite.values() if v is not None) and \
                         all(v is not None for v in fp16_finite.values())

    if fp32_ctrl is not None and fp32_ctrl.get("finite") is False:
        reading = ("FP32-ON-GPU ALSO NON-FINITE -- not a dtype problem. Something else is wrong; "
                   "do not attribute this to fp16 or to an SDPA backend.")
    elif eager and not sdpa_any_finite:
        reading = ("EAGER FINITE, ALL SDPA NON-FINITE -- an SDPA backend on sm_75 is the culprit. "
                   "Fix: pin attn_implementation='eager' for the whole ladder, re-run. Does not "
                   "touch the measurement.")
    elif eager is False and sdpa_any_finite:
        reading = ("EAGER NON-FINITE, AT LEAST ONE SDPA BACKEND FINITE -- unexpected inversion of "
                   "the naive hypothesis; report exactly which backend(s) were finite.")
    elif eager and sdpa_any_finite and not sdpa_all_finite:
        reading = ("EAGER FINITE, SOME SDPA BACKENDS NON-FINITE -- pin the working backend "
                   "explicitly rather than relying on the dispatcher's default choice.")
    elif all_fp16_nonfinite:
        reading = ("ALL FP16 CONFIGURATIONS NON-FINITE (fp32-on-GPU finite) -- fp16 on this GPU, "
                   "not fp16 as such. A1.3's 'fp16 on GPU' is not achievable on a T4 for this "
                   "donor. That is itself the finding; do not work around it.")
    else:
        reading = ("DOES NOT MATCH A PRE-REGISTERED ROW CLEANLY -- report the raw per-config table "
                   "verbatim rather than force-fitting a reading.")

    out["reading"] = reading
    print("\nREADING: " + reading)

    outdir = os.environ.get("S1_RESULTS", C.RESULTS)
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "fp16_gpu_nan_hunt_%s.json" % a.model)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print("wrote", p)


if __name__ == "__main__":
    main()
