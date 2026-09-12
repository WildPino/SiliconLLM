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
QUANT_TAGGED = 3
QUANT_V4 = 4                      # E26: tagged + a per-layer int32 ffn_kind
PT_BLK = 64                       # E26: the transposed kind's block size, one cache line
MK_PACKED, MK_FACTORED, MK_F32, MK_PACKED_T = 0, 1, 2, 3
FK_DENSE, FK_CARVED = 0, 1

# Real donor shapes, read off benchmarks/donor_adaptation/configs/*.json.  T10 is synthetic and
# is marked so everywhere it appears.  (D, F, L, NH, NKV, HD, V, tied, source)
SHAPES = {
    "S05": (896, 4864, 24, 14, 2, 64, 151936, 1, "Qwen2.5-0.5B"),
    "S15": (1536, 8960, 28, 12, 2, 128, 151936, 1, "Qwen2.5-1.5B"),
    "S3":  (2048, 11008, 36, 16, 2, 128, 151936, 1, "Qwen2.5-3B"),
    "M7":  (4096, 14336, 32, 32, 8, 128, 32768, 0, "Mistral-7B-v0.3"),
    "Q8":  (4096, 12288, 36, 32, 8, 128, 151936, 0, "Qwen3-8B"),
    "T10": (4096, 14336, 48, 32, 8, 128, 32768, 0, "SYNTHETIC -- the goal's \"es 10B\""),
    # E35: T10 with DEPTH as the only variable.  Every field but L is T10's, so a difference
    # between these and T10 cannot be anything but depth.  Added without touching any entry
    # above -- every earlier probe's shape must keep its bytes.
    "T10L32": (4096, 14336, 32, 32, 8, 128, 32768, 0, "E35 -- T10 at 32 layers"),
    "T10L24": (4096, 14336, 24, 32, 8, 128, 32768, 0, "E35 -- T10 at 24 layers"),
    "T10L16": (4096, 14336, 16, 32, 8, 128, 32768, 0, "E35 -- T10 at 16 layers"),
    "T10L12": (4096, 14336, 12, 32, 8, 128, 32768, 0, "E35 -- T10 at 12 layers"),
    # E36: the first shape in this programme that is ACTUALLY ~10 B on disk and whose ACTIVE
    # slice is sized to E35's measured envelope.  L and D are E35's crossing arm; F is whatever
    # makes the parameter count 10 B.  F = 46080 = 256*180, so --carve 256 gives GSZ=180 and the
    # down_proj runs are 11,520 B -- the FLAT part of E31's curve, not the steep part every
    # earlier carve arm sat on.  Total 9,999,220,736 weights; charged 822,083,584 + 35,389,440*k.
    "A10B": (4096, 46080, 16, 32, 8, 128, 32768, 0, "E36 -- SYNTHETIC, ~10 B at the E35 envelope"),
    # E39: the SAME 9,999,220,736 parameters as A10B and the same per-layer total to the
    # parameter, with weight moved out of q/o (write it with --rank 512) and into the FFN,
    # where the carve makes it nearly free.  F=48128 is 256*188, so --carve 256 divides it.
    "A10B-R512": (4096, 48128, 16, 32, 8, 128, 32768, 0,
                  "E39 -- SYNTHETIC, A10B's exact parameter count, put somewhere else"),
    "A10B-NKV2": (4096, 48640, 16, 32, 2, 128, 32768, 0, "E40 -- SYNTHETIC, A10B's exact parameter count, k/v heads 8 -> 2"),
    "A10B-R128": (4096, 49152, 16, 32, 2, 128, 32768, 0, "E40 addendum A -- SYNTHETIC, A10B's exact parameter count, every lever that is REAL on this engine"),
}


def total_weights(D, F, L, NH, NKV, HD, V, tied, rank=0):
    """Every weight IN THE FILE, charged or not.  E36 needs this and active_weights() cannot
    give it: a carved file's whole point is that most of its parameters are never read on a
    given token, so "how big is this model" and "how much does a token cost" are two different
    numbers and this programme has only ever computed the second one.  Norms and biases are
    left out -- they are 5 numbers in 10^10 and counting them would flatter the total."""
    QO = NH * HD
    qo = 2 * (rank * (QO + D)) if rank else 2 * (QO * D)
    per = qo + 2 * (NKV * HD * D)
    return (per + 3 * D * F) * L + V * D * (1 if tied else 2)


def active_weights(D, F, L, NH, NKV, HD, V, E=0, k=0, rank=0):
    """Weights touched by a matvec on every decoded token.  The embedding is a gather, not a
    matvec, so it is not here; the head is, tied or not.

    E26: with a carve (E groups, k kept) the FFN term is E*D for the router plus 3*D*(F/E)*k
    for the kept neurons.  The router is CHARGED -- leaving it out would flatter the carve by
    50.3 M weights a token at the goal's shape.

    E39: with `rank` set, q_proj and o_proj are the FACTORED kind, so each costs r*(QO+D)
    instead of QO*D.  k_proj and v_proj stay dense.  This used to be open-coded in main() and
    OVERWROTE the carve's number, which is why the two axes could not be combined; there is one
    definition now and the carve and the rank compose.
    """
    QO = NH * HD
    qo = 2 * (rank * (QO + D)) if rank else 2 * (QO * D)
    per = qo + 2 * (NKV * HD * D)
    ffn = 3 * D * F if not E else (E * D + 3 * D * (F // E) * k)
    return (per + ffn) * L + V * D


def _codes(rng, r, c, mode):
    """[r, c] of trit codes SHIFTED to {0,1,2}.  Split out of w_packed so the transposed
    writer packs along the other axis without a second definition of the value distribution."""
    if mode == "zero":
        return np.ones((r, c), dtype=np.int16)                             # trit 0 -> code 1
    if mode == "dense":
        return rng.integers(0, 2, size=(r, c), dtype=np.int16) * 2         # {-1,+1} -> {0,2}
    u = rng.random((r, c), dtype=np.float32)
    return np.where(u < 0.47, 1, np.where(u < 0.735, 0, 2)).astype(np.int16)


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
        qn = _codes(rng, r1 - r0, n_in, mode)
        packed = (qn[:, 0::2] + 3 * qn[:, 1::2]).astype(np.uint8)
        fh.write(np.ascontiguousarray(packed).tobytes())
    scale = (rng.random(n_out, dtype=np.float32) * 0.01 + 0.005)
    fh.write(np.ascontiguousarray(scale, dtype="<f4").tobytes())


def w_tag_packed(fh, rng, n_out, n_in, mode):
    fh.write(struct.pack("<i", MK_PACKED))
    w_packed(fh, rng, n_out, n_in, mode)


def w_tag_packed_t(fh, rng, n_out, n_in, mode):
    """kind 3: the same [n_out, n_in] matrix stored TRANSPOSED -- code[n_in][n_out/2], two
    trits per byte along OUT, per-output-row scale unchanged at [n_out].

    This is what lets a carved FFN skip a down_proj neuron.  In [D, F] a neuron is a column,
    half a byte inside a 64-byte line, and skipping it saves nothing; transposed it is a whole
    contiguous row of D/2 bytes.  Exactly the same number of bytes either way.
    """
    assert n_out % 2 == 0, "the transposed layout pairs OUTPUT rows; odd n_out is not expressible"
    assert (n_out // 2) % PT_BLK == 0, "PT_BLK must divide n_out/2"
    fh.write(struct.pack("<ii", MK_PACKED_T, PT_BLK))
    nb = (n_out // 2) // PT_BLK
    # Block-major, so the writer allocates the whole [n_in, n_out/2] plane once rather than
    # streaming rows: at the shapes here that is 29 MB (T10), not a reason to chunk.
    qn = _codes(rng, n_in, n_out, mode)
    packed = (qn[:, 0::2] + 3 * qn[:, 1::2]).astype(np.uint8)            # [n_in, n_out/2]
    bm = packed.reshape(n_in, nb, PT_BLK).transpose(1, 0, 2)             # [nb, n_in, blk]
    fh.write(np.ascontiguousarray(bm).tobytes())
    scale = (rng.random(n_out, dtype=np.float32) * 0.01 + 0.005)
    fh.write(np.ascontiguousarray(scale, dtype="<f4").tobytes())


def w_tag_factored(fh, rng, n_out, n_in, mode, r):
    """kind 1: A [n_out, r] packed, s [r] fp32, B [r, n_in] packed.

    WHY THIS BELONGS IN A SYNTHETIC WRITER.  Time depends on shape and format, not on values
    (brief s4 Gate V1 plants that as a control).  The factored form changes the SHAPE of the
    work -- r*(out+in) weights instead of out*in, and two matvec calls instead of one -- and
    that is exactly what has never been measured: E21 and E22 installed the dense product A.B,
    so every rank number this programme published priced the cut's QUALITY and never its COST.
    """
    assert r > 0 and r % 2 == 0 and n_in % 2 == 0
    fh.write(struct.pack("<ii", MK_FACTORED, r))
    w_tag_packed(fh, rng, n_out, r, mode)                          # A
    fh.write(np.ascontiguousarray(rng.random(r, dtype=np.float32) * 0.5 + 0.75,
                                  dtype="<f4").tobytes())          # s
    w_tag_packed(fh, rng, r, n_in, mode)                           # B


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
    ap.add_argument("--rank", type=int, default=0,
                    help="E25: write q_proj and o_proj as the FACTORED kind at this rank, which "
                         "forces the quant==3 tagged layout.  0 keeps every matrix dense-packed. "
                         "--rank 0 --tagged is the matched control: same layout, same tags, same "
                         "bytes per matrix, so a --rank R arm differs from it in ONE thing.")
    ap.add_argument("--tagged", action="store_true",
                    help="write the quant==3 tagged layout even at --rank 0")
    ap.add_argument("--carve", type=int, default=0,
                    help="E26: write a CARVED FFN with this many neuron groups (E), which "
                         "forces the quant==4 layout.  The neurons are already group-major "
                         "here because they are noise; on a real donor the exporter permutes "
                         "the F axis.  Implies --v4.")
    ap.add_argument("--carve-k", type=int, default=0,
                    help="the k stored IN the file; the engine's --carve-k overrides it, which "
                         "is how one artifact serves every cell of the sweep.  Default E.")
    ap.add_argument("--v4", action="store_true",
                    help="write the quant==4 container with FK_DENSE on every layer -- the "
                         "matched control for a --carve arm: same container, same tags, same "
                         "bytes, one thing different.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rms-eps", type=float, default=1e-6)
    ap.add_argument("--rope-theta", type=float, default=1000000.0)
    a = ap.parse_args()

    D, F, L, NH, NKV, HD, V, tied, src = SHAPES[a.shape]
    if a.head == "ternary":
        tied = 0                       # untie so the head is written as packed codes, not fp32
    rng = np.random.default_rng(a.seed)
    v4 = bool(a.v4 or a.carve)
    tagged = bool(a.tagged or a.rank or v4)
    quant = QUANT_V4 if v4 else (QUANT_TAGGED if tagged else QUANT_PACKED)
    if a.carve:
        if F % a.carve:
            sys.exit("--carve %d does not divide F=%d" % (a.carve, F))
        if D % 2:
            sys.exit("the transposed down_proj pairs OUTPUT rows; D=%d is odd" % D)
        # E26 kept --carve and --rank apart because it was PRICING one axis at a time, and
        # a two-axis arm cannot price either.  E39's question IS the combination -- the same
        # ten billion with weight moved out of q/o and into the carved FFN -- so the guard is
        # scoped to the probe that needs it rather than to the format, which never cared:
        # donor_engine.c reads q/k/v/o with the same tagged reader in a quant==4 file (:841)
        # and matvec dispatches on m->rank (:448).  No accounting is loosened; active_weights()
        # composes both terms and the runner checks it against its own closed form.
        if a.rank:
            print("  --carve WITH --rank: two axes at once, which E26 forbade and E39 requires. "
                  "Any arm built this way prices the COMBINATION and neither axis alone.",
                  flush=True)
    carve_k = a.carve_k or a.carve
    if a.carve and not (1 <= carve_k <= a.carve):
        sys.exit("--carve-k must be in [1, %d]" % a.carve)
    if a.rank and (a.rank % 2 or a.rank <= 0):
        sys.exit("--rank must be positive and even (two trits per byte)")
    if a.rank and a.rank >= min(D, NH * HD):
        print("  NOTE: rank %d is not a REDUCTION at this shape (q is [%d,%d], o is [%d,%d]) -- "
              "the factored form moves MORE weights than the dense one here."
              % (a.rank, NH * HD, D, D, NH * HD), flush=True)
    act = active_weights(D, F, L, NH, NKV, HD, V)
    act_r = active_weights(D, F, L, NH, NKV, HD, V, a.carve, carve_k, a.rank)
    print("[%s] %s  D=%d F=%d L=%d H=%d/%d hd=%d V=%d tied=%d head=%s  active=%.3f B  codes=%s"
          % (a.shape, src, D, F, L, NH, NKV, HD, V, tied, a.head, act_r / 1e9, a.codes), flush=True)
    if a.carve:
        print("  --carve E=%d, group=%d neurons, k=%d in the file: active %.4f B -> %.4f B "
              "(%+.2f%%), router charged at E*D = %.1f M/token"
              % (a.carve, F // a.carve, carve_k, act / 1e9, act_r / 1e9,
                 100.0 * (act_r - act) / act, a.carve * D * L / 1e6), flush=True)
    if a.rank:
        print("  --rank %d on q_proj/o_proj: active %.4f B -> %.4f B  (%+.2f%%), "
              "matvec calls/token %d -> %d"
              % (a.rank, act / 1e9, act_r / 1e9, 100.0 * (act_r - act) / act,
                 L * 7 + 1, L * 7 + 1 + 2 * L), flush=True)

    QD, KD = NH * HD, NKV * HD
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<9i", D, F, L, NH, NKV, HD, V, tied, quant))
        fh.write(struct.pack("<2f", float(a.rms_eps), float(a.rope_theta)))
        wp = (lambda o, i: w_tag_packed(fh, rng, o, i, a.codes)) if tagged else \
             (lambda o, i: w_packed(fh, rng, o, i, a.codes))
        w_fp32(fh, rng, V, D)                                  # embed
        for li in range(L):
            w_ones(fh, D)                                      # input_layernorm
            for nm, n_out in (("q", QD), ("k", KD), ("v", KD)):   # each with a bias
                if nm == "q" and a.rank:
                    w_tag_factored(fh, rng, n_out, D, a.codes, a.rank)
                else:
                    wp(n_out, D)
                w_fp32(fh, rng, n_out)
            if a.rank:
                w_tag_factored(fh, rng, D, QD, a.codes, a.rank)   # o_proj, no bias
            else:
                wp(D, QD)
            w_ones(fh, D)                                      # post_attention_layernorm
            if v4:
                fh.write(struct.pack("<i", FK_CARVED if a.carve else FK_DENSE))
            if a.carve:
                fh.write(struct.pack("<2i", a.carve, carve_k))
                wp(a.carve, D)                                 # router [E, D]
                wp(F, D)                                       # gate, group-major rows
                wp(F, D)                                       # up,   group-major rows
                w_tag_packed_t(fh, rng, D, F, a.codes)         # down, TRANSPOSED
            else:
                wp(F, D)                                       # gate
                wp(F, D)                                       # up
                wp(D, F)                                       # down
            if (li + 1) % 8 == 0 or li == L - 1:
                print("  layer %d/%d  (%.2f GB written)" % (li + 1, L, fh.tell() / 2 ** 30),
                      flush=True)
        w_ones(fh, D)                                          # model.norm
        if not tied:
            wp(V, D)                                           # lm_head

    size = os.path.getsize(a.out)

    # --- Gate V3: the byte layout, checked against a definition written independently of this
    # file.  E1's layout() walks the same format from the header alone; if my write order or any
    # tensor size disagrees with it, the totals cannot match.
    import e1_bpb_through_engine as E1
    hdr = E1.read_header(a.out)
    for k, want in (("D", D), ("F", F), ("L", L), ("NH", NH), ("NKV", NKV),
                    ("HD", HD), ("V", V), ("tied", tied), ("quant", quant)):
        assert hdr[k] == want, "header %s: wrote %r, read back %r" % (k, want, hdr[k])
    lay = E1.layout(D, F, L, NH, NKV, HD, V, tied, quant)
    if v4:
        want = E1.layout_bytes_v4(D, F, L, NH, NKV, HD, V, tied, a.carve, a.rank)
    elif tagged:
        def spec_of(name, o, i):
            return ("factored", a.rank) if (a.rank and (name.endswith(".q_proj")
                                                        or name.endswith(".o_proj"))) else "packed"
        want = E1.layout_bytes_tagged(D, F, L, NH, NKV, HD, V, tied, spec_of)
    else:
        want = hdr["off0"] + sum(E1.nbytes(kind, o, i, QUANT_PACKED) for (_, kind, o, i) in lay)
    assert want == size, ("GATE V3 FAILED: E1's layout says %d bytes, the file is %d (%+d). "
                          "The write order or a tensor size disagrees with the format."
                          % (want, size, size - want))
    print("  GATE V3: %d tensors, %d bytes, matches E1's independent layout exactly"
          % (len(lay), size), flush=True)

    meta = {"synthetic": True, "brief": "BRIEF_E3_ENGINE_AT_TARGET_SCALE.md @ d4937a2",
            "shape": a.shape, "shape_source": src, "codes": a.codes, "seed": a.seed,
            "d_model": D, "d_ffn": F, "n_layers": L, "n_heads": NH, "n_kv_heads": NKV,
            "head_dim": HD, "vocab": V, "tied": tied, "head": a.head,
            "quant": ("tagged-v2" if v4 else "tagged") if tagged else "packed",
            "rank": a.rank, "carve_E": a.carve, "carve_k_in_file": carve_k,
            "carve_group_size": (F // a.carve) if a.carve else 0,
            "total_weights": int(total_weights(D, F, L, NH, NKV, HD, V, tied)),
            "active_weights_per_token": int(act_r),
            "active_weights_per_token_dense_qo": int(act), "bytes": size,
            "WARNING": "weights are NOISE -- this file has no meaningful BPB and none is computed"}
    json.dump(meta, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    print("wrote %s  (%.2f GB)  active %.3f B weights/token" % (a.out, size / 2 ** 30, act_r / 1e9))


if __name__ == "__main__":
    main()
