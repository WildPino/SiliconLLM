#!/usr/bin/env python3
# Inventor / S4 - silicon-native tokenizer feasibility: on the pinned P62 code-val, how uneven is the
# INFORMATION per BPE token, and what would constant-entropy patches (SEE-style, K4/BLT lineage) look like?
#
# Method (read-only, CPU, deterministic):
#   1. Adaptive interpolated order-2 byte model (KT-style counts, o2->o1->o0 escape mixing) = a crude,
#      self-contained proxy of the SEE's online entropy meter. One left-to-right adaptive pass over
#      code_val.txt gives per-byte code lengths l(b) (bits).
#   2. BPE side: code_val.u16 + code.meta give the exact byte span of every BPE token -> bits per token.
#   3. Patch side: greedy segmentation "accumulate l(b) until >= B bits" -> constant-entropy patches.
#   4. Compare at MATCHED unit count (B = total_bits / n_tokens): the variance/CV of bits-per-unit is the
#      claim under test - BPE spends one forward pass per token regardless of information; patches equalize
#      information per forward pass.
#
# Honest scope: the meter is NOT the SEE (no LZ/MoE experts) and model-BPB over new units requires
# TRAINING on them (a v2 probe). These are feasibility statistics, not a quality claim.
#
# Output: prints tables; tee to docs/in_research/s4_entropy_patching_out.txt
import os, struct, math, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P62 = os.path.join(ROOT, "results", "phase62")


def load_meta(path):
    with open(path, "rb") as f:
        mg, V, nt = struct.unpack("<III", f.read(12))
        assert mg == 0x54444D50, "bad meta magic"
        el = np.frombuffer(f.read(V), dtype=np.uint8).astype(np.int64)
        id2bytes = [f.read(int(el[t])) for t in range(V)]
    return V, el, id2bytes


def adaptive_bits(data: np.ndarray):
    """Interpolated adaptive order-2 byte model; returns per-byte code length in bits (float32).
    p2 = (n2+a*p1)/(N2+a), p1 = (n1+a*p0)/(N1+a), p0 = (n0+1/256)/(N0+1); a=1. Deterministic."""
    n0 = np.zeros(256, np.float64); N0 = 0.0
    n1 = np.zeros((256, 256), np.float32); N1 = np.zeros(256, np.float64)
    n2 = np.zeros((65536, 256), np.float32); N2 = np.zeros(65536, np.float64)
    out = np.empty(len(data), np.float32)
    c1 = 0; c2 = 0
    for i, b in enumerate(data):
        b = int(b)
        p0 = (n0[b] + 1.0 / 256) / (N0 + 1.0)
        p1 = (n1[c1, b] + p0) / (N1[c1] + 1.0)
        p2 = (n2[c2, b] + p1) / (N2[c2] + 1.0)
        out[i] = -math.log2(p2)
        n0[b] += 1; N0 += 1
        n1[c1, b] += 1; N1[c1] += 1
        n2[c2, b] += 1; N2[c2] += 1
        c2 = ((c1 << 8) | b) & 0xFFFF
        c1 = b
    return out


def seg_stats(lens_bits, name):
    a = np.asarray(lens_bits, dtype=np.float64)
    return (f"  {name:<26} units={len(a):>7}  bits/unit mean={a.mean():7.2f}  CV={a.std()/a.mean():5.2f}"
            f"  p50={np.percentile(a,50):6.2f}  p90={np.percentile(a,90):6.2f}  p99={np.percentile(a,99):7.2f}"
            f"  max={a.max():7.1f}")


def main():
    t0 = time.time()
    raw = np.fromfile(os.path.join(P62, "code_val.txt"), dtype=np.uint8)
    ids = np.fromfile(os.path.join(P62, "code_val.u16"), dtype=np.uint16).astype(np.int64)
    V, el, _ = load_meta(os.path.join(P62, "code.meta"))
    tok_lens = el[ids]
    nb_tok = int(tok_lens.sum())
    print("=" * 110)
    print("S4 - entropy-patching feasibility on the pinned P62 code-val (adaptive order-2 byte meter)")
    print("=" * 110)
    print(f"code_val: {len(raw)} bytes | BPE V={V}: {len(ids)} tokens covering {nb_tok} bytes "
          f"(fertility {nb_tok/len(ids):.2f} B/tok)")
    n = min(len(raw), nb_tok)                       # align (u16 stream covers the val bytes in order)
    raw = raw[:n]

    L = adaptive_bits(raw)
    print(f"meter: adaptive o2 bits/byte = {L.mean():.3f} (one-pass adaptive; SEE/zstd on this domain ~1.2-2.4;"
          f" crude proxy, fine for RELATIVE unit stats)  [{time.time()-t0:.0f}s]")

    # ---- bits per BPE token (exact spans) ----
    bounds = np.cumsum(tok_lens)                     # end offsets
    bounds = bounds[bounds <= n]
    starts = np.concatenate([[0], bounds[:-1]])
    csum = np.concatenate([[0.0], np.cumsum(L, dtype=np.float64)])
    tok_bits = csum[bounds] - csum[starts]
    print("\n-- information per unit --")
    print(seg_stats(tok_bits, f"BPE-{V} tokens"))
    for thr in (0.5, 1.0, 2.0):
        print(f"    BPE tokens carrying < {thr:>3} bits: {100*(tok_bits<thr).mean():5.1f}%  (forward passes spent on ~no information)")

    # ---- constant-entropy patches ----
    def patch(B):
        cuts = []; acc = 0.0
        for i in range(len(L)):
            acc += L[i]
            if acc >= B:
                cuts.append(i + 1); acc = 0.0
        b = np.asarray(cuts)
        s = np.concatenate([[0], b[:-1]])
        return s, b

    Bmatch = float(L.sum() / len(tok_bits))          # matched unit count vs BPE
    print(f"\n  B_match = {Bmatch:.2f} bits (same total bits / same-order unit count as BPE):")
    s_, b_ = patch(Bmatch)
    pl = (b_ - s_).astype(np.float64)
    pbits = csum[b_] - csum[s_]
    print(seg_stats(pbits, f"patches @B={Bmatch:.1f}"))
    print(f"    patch length bytes: mean={pl.mean():.2f} p50={np.percentile(pl,50):.0f} p90={np.percentile(pl,90):.0f} "
          f"max={pl.max():.0f}  | units vs BPE: {len(pl)/len(tok_bits):.3f}x")

    print("\n  sweep (unit count & length vs B):")
    print(f"  {'B bits':>8}{'units':>9}{'vs BPE':>8}{'mean B/patch':>13}{'p90 len':>9}{'CV bits':>9}")
    for B in (8.0, 12.0, 16.0, 24.0, 32.0):
        s_, b_ = patch(B)
        pl = (b_ - s_).astype(np.float64)
        pbits = csum[b_] - csum[s_]
        print(f"  {B:>8.0f}{len(pl):>9}{len(pl)/len(tok_bits):>8.3f}{pl.mean():>13.2f}{np.percentile(pl,90):>9.0f}"
              f"{pbits.std()/pbits.mean():>9.2f}")

    # ---- boundary coincidence (KD-span relevance) ----
    bpe_set = set(bounds.tolist())
    s_, b_ = patch(Bmatch)
    coin = np.fromiter((x in bpe_set for x in b_.tolist()), bool, len(b_))
    print(f"\n  patch->BPE boundary coincidence @B_match: {100*coin.mean():.1f}%  "
          f"(cf. T2's student->teacher 39-44%: same-order phenomenon, different segmenters)")

    print("\nreading: the BPE CV row vs the patch CV row IS the S4 thesis quantified - how unevenly BPE")
    print("spends forward passes per bit vs a constant-entropy segmenter. Quality over the new units")
    print("requires training on them (v2 probe); this is the feasibility card only.")


if __name__ == "__main__":
    main()
