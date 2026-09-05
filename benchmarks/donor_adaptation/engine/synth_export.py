#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3 -- write a QWENDON1 file of an arbitrary SHAPE, with synthetic weights.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E3_ENGINE_AT_TARGET_SCALE.md (d4937a2),
pre-registered before this file existed.

WHY THIS IS ALLOWED TO EXIST.  A speed measurement needs shape and format, not training.  The
packed kernels in donor_engine.c (matvec / matvec_lut / matvec_lut_g) are pshufb over packed
bytes plus FMA accumulation with no early exit on a zero code, so timing should depend on shape
and format only.  That is a READING of the code and this programme does not accept readings:
brief §4 Gate V1 plants it as a control (--codes zero vs --codes dense on one shape) and the
probe stops if it fires.

WHAT THIS IS NOT.  It is not an exporter.  The weights are noise, so nothing it writes has a BPB
worth reading, and it deliberately does not compute one.  Its only claim is that the BYTES are
laid out exactly as qwen_export.py lays them out -- which Gate V3 checks against
e1_bpb_through_engine.layout(), a definition written independently of this file.

LAYOUT, mirrored from qwen_export.py:277-311:
    "QWENDON1", <9i (D,F,L,NH,NKV,HD,V,tied,quant)>, <2f (rms_eps, rope_theta)>,
    embed fp32,
    per layer: input_layernorm fp32,
               q,k,v  (codes + per-row scales) each followed by an fp32 BIAS,
               o_proj (codes + scales)          -- no bias,
               post_attention_layernorm fp32,
               gate,up,down (codes + scales)    -- no bias,
    model.norm fp32,
    lm_head (codes + scales) if not tied.
Packed codes are two trits per byte, v = (t0+1) + 3*(t1+1), row-major over pairs of columns.
"""
import argparse, hashlib, json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAGIC = b"QWENDON1"
QUANT_PACKED = 2

# Real donor shapes, read off benchmarks/donor_adaptation/configs/*.json.  T10 is synthetic and
# is marked so everywhere it appears.  (D, F, L, NH, NKV, HD, V, tied, source)
SHAPES = {
    "S05": (896, 4864, 24, 14, 2, 64, 151936, 1, "Qwen2.5-0.5B"),
    "S15": (1536, 8960, 28, 12, 2, 128, 151936, 1, "Qwen2.5-1.5B"),
    "S3":  (2048, 11008, 36, 16, 2, 128, 151936, 1, "Qwen2.5-3B"),
    "M7":  (4096, 14336, 32, 32, 8, 128, 32768, 0, "Mistral-7B-v0.3"),
    "Q8":  (4096, 12288, 36, 32, 8, 128, 151936, 0, "Qwen3-8B"),
    "T10": (4096, 14336, 48, 32, 8, 128, 32768, 0, "SYNTHETIC -- the goal's \"es 10B\""),
}


def active_weights(D, F, L, NH, NKV, HD, V):
    """Weights touched by a matvec on every decoded token.  The embedding is a gather, not a
    matvec, so it is not here; the head is, tied or not."""
    per = NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D + 3 * D * F
    return per * L + V * D


def w_packed(fh, rng, n_out, n_in, mode):
    """Codes for one [n_out, n_in] matrix, packed two trits per byte, plus n_out fp32 scales.

    mode 'dense' -> no code is zero (every trit +-1);  'zero' -> every code is zero.
    'mixed' draws the zero fraction E2 measured on real R3 exports (~0.47), so the default file
    looks like a real artifact to anything that inspects it.  None of the three may change the
    TIMING -- that is Gate V1.
    """
    assert n_in % 2 == 0, "packed layout pairs columns; odd n_in is not expressible"
    # Written in row chunks: an embed- or head-sized tensor materialised whole would allocate a
    # multi-GB float64 temporary inside the rng call before anything is cast down.
    CH = max(1, (1 << 24) // max(1, n_in))
    for r0 in range(0, n_out, CH):
        r1 = min(n_out, r0 + CH)
        if mode == "zero":
            qn = np.ones((r1 - r0, n_in), dtype=np.int16)                  # trit 0 -> code 1
        elif mode == "dense":
            qn = rng.integers(0, 2, size=(r1 - r0, n_in), dtype=np.int16) * 2   # {-1,+1} -> {0,2}
        else:
            u = rng.random((r1 - r0, n_in), dtype=np.float32)
            qn = np.where(u < 0.47, 1, np.where(u < 0.735, 0, 2)).astype(np.int16)
        packed = (qn[:, 0::2] + 3 * qn[:, 1::2]).astype(np.uint8)
        fh.write(np.ascontiguousarray(packed).tobytes())
    scale = (rng.random(n_out, dtype=np.float32) * 0.01 + 0.005)
    fh.write(np.ascontiguousarray(scale, dtype="<f4").tobytes())


def w_fp32(fh, rng, *shape):
    if len(shape) == 1:
        a = rng.random(shape[0], dtype=np.float32) * 0.04 - 0.02
        fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
        return
    n_out, n_in = shape
    CH = max(1, (1 << 24) // max(1, n_in))            # row chunks, see w_packed
    for r0 in range(0, n_out, CH):
        a = rng.random((min(n_out, r0 + CH) - r0, n_in), dtype=np.float32) * 0.04 - 0.02
        fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def w_ones(fh, n):
    fh.write(np.ascontiguousarray(np.ones(n, dtype=np.float32), dtype="<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=sorted(SHAPES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--codes", default="mixed", choices=("mixed", "zero", "dense"))
    # RUN 1 WROTE THE WRONG ARM.  SHAPES carries the donor's own `tied` flag, and a tied model runs
    # its head as the fp32 EMBEDDING: 544 MB/token at S05, 13.8 ms, 52% of the token.  Brief s3 asks
    # for "a ternary head (the runnable configuration E2 settled)", which is UNTIED and packed --
    # 68.1 MB/token, the configuration SPEED_LEDGER s12.2 actually measured its 56.1 tok/s on.
    # Default is therefore `ternary`; `donor` reproduces run 1 and is kept so run 1 stays replayable.
    ap.add_argument("--head", default="ternary", choices=("ternary", "donor"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rms-eps", type=float, default=1e-6)
    ap.add_argument("--rope-theta", type=float, default=1000000.0)
    a = ap.parse_args()

    D, F, L, NH, NKV, HD, V, tied, src = SHAPES[a.shape]
    if a.head == "ternary":
        tied = 0                       # untie so the head is written as packed codes, not fp32
    rng = np.random.default_rng(a.seed)
    act = active_weights(D, F, L, NH, NKV, HD, V)
    print("[%s] %s  D=%d F=%d L=%d H=%d/%d hd=%d V=%d tied=%d head=%s  active=%.3f B  codes=%s"
          % (a.shape, src, D, F, L, NH, NKV, HD, V, tied, a.head, act / 1e9, a.codes), flush=True)

    QD, KD = NH * HD, NKV * HD
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<9i", D, F, L, NH, NKV, HD, V, tied, QUANT_PACKED))
        fh.write(struct.pack("<2f", float(a.rms_eps), float(a.rope_theta)))
        w_fp32(fh, rng, V, D)                                  # embed
        for li in range(L):
            w_ones(fh, D)                                      # input_layernorm
            for n_out in (QD, KD, KD):                         # q, k, v  -- each with a bias
                w_packed(fh, rng, n_out, D, a.codes)
                w_fp32(fh, rng, n_out)
            w_packed(fh, rng, D, QD, a.codes)                  # o_proj, no bias
            w_ones(fh, D)                                      # post_attention_layernorm
            w_packed(fh, rng, F, D, a.codes)                   # gate
            w_packed(fh, rng, F, D, a.codes)                   # up
            w_packed(fh, rng, D, F, a.codes)                   # down
            if (li + 1) % 8 == 0 or li == L - 1:
                print("  layer %d/%d  (%.2f GB written)" % (li + 1, L, fh.tell() / 2 ** 30),
                      flush=True)
        w_ones(fh, D)                                          # model.norm
        if not tied:
            w_packed(fh, rng, V, D, a.codes)                   # lm_head

    size = os.path.getsize(a.out)

    # --- Gate V3: the byte layout, checked against a definition written independently of this
    # file.  E1's layout() walks the same format from the header alone; if my write order or any
    # tensor size disagrees with it, the totals cannot match.
    import e1_bpb_through_engine as E1
    hdr = E1.read_header(a.out)
    for k, want in (("D", D), ("F", F), ("L", L), ("NH", NH), ("NKV", NKV),
                    ("HD", HD), ("V", V), ("tied", tied), ("quant", QUANT_PACKED)):
        assert hdr[k] == want, "header %s: wrote %r, read back %r" % (k, want, hdr[k])
    lay = E1.layout(D, F, L, NH, NKV, HD, V, tied, QUANT_PACKED)
    want = hdr["off0"] + sum(E1.nbytes(kind, o, i, QUANT_PACKED) for (_, kind, o, i) in lay)
    assert want == size, ("GATE V3 FAILED: E1's layout says %d bytes, the file is %d (%+d). "
                          "The write order or a tensor size disagrees with the format."
                          % (want, size, size - want))
    print("  GATE V3: %d tensors, %d bytes, matches E1's independent layout exactly"
          % (len(lay), size), flush=True)

    meta = {"synthetic": True, "brief": "BRIEF_E3_ENGINE_AT_TARGET_SCALE.md @ d4937a2",
            "shape": a.shape, "shape_source": src, "codes": a.codes, "seed": a.seed,
            "d_model": D, "d_ffn": F, "n_layers": L, "n_heads": NH, "n_kv_heads": NKV,
            "head_dim": HD, "vocab": V, "tied": tied, "head": a.head, "quant": "packed",
            "active_weights_per_token": int(act), "bytes": size,
            "WARNING": "weights are NOISE -- this file has no meaningful BPB and none is computed"}
    json.dump(meta, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    print("wrote %s  (%.2f GB)  active %.3f B weights/token" % (a.out, size / 2 ** 30, act / 1e9))


if __name__ == "__main__":
    main()
