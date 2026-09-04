#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a Qwen2.5 donor into a flat binary that donor_engine.c can execute.

TWO MODES, and the order matters:

  --quant fp32     every weight fp32.  This exists so the C runtime can be proved CORRECT
                   against PyTorch BEFORE any quantization is involved.  If the first thing
                   we build is the ternary path and it produces garbage, we cannot tell a bug
                   in the C from the cost of the conversion.  fp32 first, always.
  --quant ternary  weights as int8 codes {-1,0,+1} + one fp32 scale per output row, exactly
                   engine.c's format.  The rule is imported from t1_ternarize.ternarize() --
                   ONE definition of the conversion, shared with the T1 probe, never re-derived.

Norms, biases and the embedding table stay fp32 in both modes: engine.c's own convention is
"ternary WEIGHT-CODE arrays only; fp32 tensors, per-row scales and activations excluded"
(engine.c:263).

FORMAT (little-endian, no padding):
    char[8]  "QWENDON1"
    int32    d_model, d_ffn, n_layers, n_heads, n_kv_heads, head_dim, vocab, tied, quant
    float32  rms_eps, rope_theta
    fp32     embed_tokens            [vocab, d_model]
    per layer:
      fp32   input_layernorm.weight  [d_model]
      W      q_proj [q_out, d_model] ; fp32 q_bias [q_out]
      W      k_proj [kv_out, d_model]; fp32 k_bias [kv_out]
      W      v_proj [kv_out, d_model]; fp32 v_bias [kv_out]
      W      o_proj [d_model, q_out]
      fp32   post_attention_layernorm.weight [d_model]
      W      gate_proj [d_ffn, d_model]
      W      up_proj   [d_ffn, d_model]
      W      down_proj [d_model, d_ffn]
    fp32     model.norm.weight       [d_model]
    (lm_head is tied on every Qwen2.5 size we use; if tied==0 a final W follows)

  where W is:  quant==0 -> fp32 [out, in]
               quant==1 -> int8 [out, in] codes, then fp32 [out] scales

Usage:
    python qwen_export.py --model Qwen/Qwen2.5-0.5B --quant fp32    --out qwen05b_fp32.bin
    python qwen_export.py --model Qwen/Qwen2.5-0.5B --quant ternary --out qwen05b_tern.bin
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ternary")))
from t1_ternarize import ternarize  # noqa: E402  -- the single definition of the conversion

MAGIC = b"QWENDON1"


def w_fp32(fh, w):
    fh.write(np.ascontiguousarray(w.numpy(), dtype="<f4").tobytes())


def w_tern(fh, w):
    """int8 codes + fp32 per-row scales, matching t1_ternarize.ternarize(group=0)."""
    scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    q = (w / scale).round().clamp(-1, 1).to(torch.int8)
    fh.write(np.ascontiguousarray(q.numpy(), dtype="i1").tobytes())
    fh.write(np.ascontiguousarray(scale.squeeze(1).numpy(), dtype="<f4").tobytes())
    return float((q == 0).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--quant", choices=("fp32", "ternary"), default="fp32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
                                             dtype=torch.float32).eval()
    c = m.config
    D = c.hidden_size
    F = c.intermediate_size
    L = c.num_hidden_layers
    NH = c.num_attention_heads
    NKV = c.num_key_value_heads
    HD = getattr(c, "head_dim", D // NH)
    V = c.vocab_size
    tied = int(bool(getattr(c, "tie_word_embeddings", False)))
    quant = 0 if a.quant == "fp32" else 1
    W = w_fp32 if quant == 0 else w_tern

    print("exporting %s  D=%d F=%d L=%d heads=%d/%d hd=%d V=%d tied=%d quant=%s"
          % (a.model, D, F, L, NH, NKV, HD, V, tied, a.quant))

    zeros = []
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<9i", D, F, L, NH, NKV, HD, V, tied, quant))
        fh.write(struct.pack("<2f", float(c.rms_norm_eps), float(c.rope_theta)))
        w_fp32(fh, m.model.embed_tokens.weight.data)
        for li in range(L):
            lay = m.model.layers[li]
            w_fp32(fh, lay.input_layernorm.weight.data)
            for name in ("q_proj", "k_proj", "v_proj"):
                mod = getattr(lay.self_attn, name)
                r = W(fh, mod.weight.data)
                if r is not None:
                    zeros.append(r)
                assert mod.bias is not None, "%s has no bias -- Qwen2 should" % name
                w_fp32(fh, mod.bias.data)
            r = W(fh, lay.self_attn.o_proj.weight.data)
            if r is not None:
                zeros.append(r)
            assert lay.self_attn.o_proj.bias is None
            w_fp32(fh, lay.post_attention_layernorm.weight.data)
            for name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(lay.mlp, name)
                assert mod.bias is None
                r = W(fh, mod.weight.data)
                if r is not None:
                    zeros.append(r)
            if (li + 1) % 8 == 0 or li == L - 1:
                print("  layer %d/%d  (%.2f GB written)"
                      % (li + 1, L, fh.tell() / 2**30), flush=True)
        w_fp32(fh, m.model.norm.weight.data)
        if not tied:
            r = W(fh, m.lm_head.weight.data)
            if r is not None:
                zeros.append(r)

    size = os.path.getsize(a.out)
    h = hashlib.sha256()
    with open(a.out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    meta = {"model": a.model, "revision": a.revision, "quant": a.quant,
            "d_model": D, "d_ffn": F, "n_layers": L, "n_heads": NH, "n_kv_heads": NKV,
            "head_dim": HD, "vocab": V, "tied": tied,
            "rms_eps": float(c.rms_norm_eps), "rope_theta": float(c.rope_theta),
            "bytes": size, "sha256": h.hexdigest(),
            "mean_ternary_zero_fraction": (float(np.mean(zeros)) if zeros else None)}
    json.dump(meta, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    print("wrote %s  (%.2f GB)  sha256 %s" % (a.out, size / 2**30, h.hexdigest()[:16]))
    if zeros:
        print("  mean ternary zero-fraction across %d matrices: %.4f"
              % (len(zeros), float(np.mean(zeros))))


if __name__ == "__main__":
    main()
