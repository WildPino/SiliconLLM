#!/usr/bin/env python3
# Sparse Capacity, phase 4 preparation (D-M): "the bank travels ternary". A mechanism measurement, not a protocol.
#
# The GPU uses only the ternary q and the per-row scale s of each expert weight (sc2_model.w_q: s = mean|W| per row,
# detached; q = clamp(round(W / s), -1, 1); straight-through gradient g / s). So the host can keep the fp32 masters and
# AdamW, recompute (q, s) after its optimizer step, and send q packed at 2 bits plus s, instead of fp32 W.
# Bytes per parameter on the bus: 2/8 + 4/row_len  (Wg, Wu rows of D: 0.016; Wd rows of h = 8: 0.5) -> ~0.43 vs 4.
#
# Measured at the m16 shapes of phase 2/3 (8 layers, E = 2048, h = 8, D = 256), on the real host and GPU:
#   1. the phase-3 fetch (index_select fp32 into a pinned buffer, copy to the GPU), per layer and per step;
#   2. the host cost of re-quantizing and packing the whole bank once per step (dense AdamW moves every expert);
#   3. the ternary fetch (index_select of the packed bytes and scales, copy, unpack on the GPU);
#   4. exactness: q * s rebuilt on the GPU against w_q(W) computed on the GPU from the same fp32 W.
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc2_model as SM      # noqa: E402

L, E, H, D = 8, 2048, 8, 256
SHAPES = {"Wg": (E, H, D), "Wu": (E, H, D), "Wd": (E, D, H)}
REPS = 7


def pack(q):
    """q in {-1, 0, 1} (float, last dim a multiple of 4) -> uint8, 4 trits per byte."""
    u = (q + 1).to(torch.uint8).reshape(*q.shape[:-1], q.shape[-1] // 4, 4)
    return u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)


def unpack(b, n):
    out = torch.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(*b.shape[:-1], n)
    return out.float() - 1.0


def host_quant(W):
    s = W.abs().mean(-1, keepdim=True).clamp_min(1e-5)
    q = torch.round(W / s).clamp_(-1, 1)
    return pack(q), s


def med(f):
    ts = []
    for _ in range(REPS):
        torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2], ts[0], ts[-1]


def main():
    torch.set_num_threads(6)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    host = [{n: (torch.randn(*sh, generator=g) * 0.02) for n, sh in SHAPES.items()} for _ in range(L)]
    P = sum(t.numel() for lay in host for t in lay.values())
    U = torch.arange(E)                                                   # m16: U is (almost) every expert
    stage = {n: torch.empty(*sh).pin_memory() for n, sh in SHAPES.items()}
    pk = [{n: host_quant(W) for n, W in lay.items()} for lay in host]
    pstage = {n: (torch.empty_like(pk[0][n][0]).pin_memory(), torch.empty_like(pk[0][n][1]).pin_memory()) for n in SHAPES}

    def fetch_fp32():
        for lay in host:
            for n, W in lay.items():
                st = stage[n]; torch.index_select(W, 0, U, out=st); st.to(dev, non_blocking=True)
            torch.cuda.current_stream().synchronize()

    def quant_all():
        for li, lay in enumerate(host):
            for n, W in lay.items():
                pk[li][n] = host_quant(W)

    def fetch_tern():
        for lay in pk:
            for n, (b, s) in lay.items():
                sb, ss = pstage[n]
                torch.index_select(b, 0, U, out=sb); torch.index_select(s, 0, U, out=ss)
                bg = sb.to(dev, non_blocking=True); sg = ss.to(dev, non_blocking=True)
                unpack(bg, SHAPES[n][-1]) * sg
            torch.cuda.current_stream().synchronize()

    res = dict(shapes=SHAPES, layers=L, bank_params=P, host_threads=6, reps=REPS)
    for k, f in (("fetch_fp32_per_pass", fetch_fp32), ("host_requant_pack_per_step", quant_all),
                 ("fetch_ternary_per_pass", fetch_tern)):
        m, lo, hi = med(f)
        res[k] = dict(median_s=m, min_s=lo, max_s=hi, ns_per_param=m / P * 1e9)
        print(f"{k:28s} median {m:.4f} s (min {lo:.4f}, max {hi:.4f}) = {m / P * 1e9:.3f} ns per bank parameter", flush=True)
    bytes_t = sum(b.numel() + 4 * s.numel() for lay in pk for b, s in lay.values())
    res["bus_bytes_per_param"] = dict(fp32=4.0, ternary=bytes_t / P)
    print(f"bus bytes per parameter: fp32 4.000, ternary {bytes_t / P:.3f}")
    # exactness: the GPU's w_q(W) vs q * s rebuilt from the host's packing, same fp32 W
    flips, worst = 0, 0.0
    for li, lay in enumerate(host):
        for n, W in lay.items():
            qg, sg = SM.w_q(W.to(dev))
            b, s = pk[li][n]
            rq = unpack(b.to(dev), SHAPES[n][-1])
            flips += int((rq != qg).sum())
            worst = max(worst, float(((s.to(dev) - sg).abs() / sg).max()))
    res["exact"] = dict(q_flips=flips, scale_max_rel_diff=worst)
    print(f"exactness vs w_q on the GPU: ternary flips {flips} of {P}, scale max relative difference {worst:.2e}")
    # phase-3 cost model per step: 3 fetches (forward, recompute, backward) + host AdamW + gradients out (fp32, unchanged)
    out = os.path.join(HERE, "..", "..", "results", "sparse_capacity", "f4_ternary_dry_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    json.dump(res, open(out, "w"), indent=1)
    print("->", os.path.normpath(out))


if __name__ == "__main__":
    main()
