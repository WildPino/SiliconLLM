#!/usr/bin/env python3
# Sparse Capacity, phase 1b (D-M): the desk byte model, shared by the prediction (sc1b_predict.py), the floor check and
# the verdict (sc1b_verdict.py). Protocol: docs/research/SPARSE_CAPACITY/F1B_CROSSOVER_PROTOCOL.md, section 3.
#
# Every number is BYTES PER BANK PARAMETER PER TRAINING STEP, weights traffic only (activations are separate terms).
# a(r) = probability that a hidden unit is active (g>0 and u>0) for at least one of its r tokens, with per-token
# density rho. An inactive unit has zero gradient: SGD and row-wise Adagrad skip it exactly, dense Adam does not.
# The lower-bound byte counts (LB) count every write once (no read-for-ownership): they are FLOORS, never estimates.

D = 256            # model width
A = 1024           # active FFN hidden per token per layer (rung-1: k*h = 8*128), held fixed across h
FMT_B = {"f32": 4.0, "bf16": 2.0, "tern": 0.5}

# arm -> (path, fmt, opt, fused, rho_key). rho_key selects the per-token activation density of the bank.
ARMS = {
    "C1":   ("cpu", "f32",  "sgd",    False, "mid"),   # autograd-like: dense gradient written, then optimizer pass
    "C2":   ("cpu", "f32",  "sgd",    True,  "mid"),   # (a) fused backward+update
    "C3":   ("cpu", "f32",  "rowada", True,  "mid"),   # (b) compact state: row-wise Adagrad, one float per row
    "C3s":  ("cpu", "f32",  "rowada", True,  "low"),   #     density as measured on the trained dense rung (E3)
    "C3d":  ("cpu", "f32",  "rowada", True,  "full"),  #     no dReLU skip at all
    "C4":   ("cpu", "f32",  "adam",   True,  "mid"),   # fp32 master + Adam moments (the OOM state format), fused
    "C5":   ("cpu", "bf16", "rowada", True,  "mid"),   # (c) bf16 master, stochastic rounding
    "C6":   ("cpu", "tern", "rowada", True,  "mid"),   # (c) ternary codes + engine LUT forward, fp32 latent
    "S1":   ("str", "f32",  "adam",   None,  "mid"),   # OOM-like: bf16 copy streamed, bf16 grad back, CPU Adam
    "S2":   ("str", "f32",  "rowada", None,  "mid"),
    "S2s":  ("str", "f32",  "rowada", None,  "low"),
    "S2d":  ("str", "f32",  "rowada", None,  "full"),
    "S3":   ("str", "bf16", "rowada", None,  "mid"),   # bf16 master streamed both ways, update on the GPU
    "S4":   ("str", "tern", "rowada", None,  "mid"),   # codes streamed, bf16 grad back, CPU update of the latent
}
RHO = {"low": 0.065, "mid": 0.5, "full": 1.0}
H1_PRIMARY = "full"   # review obj. 4: a routed single-unit (PEER-like) expert is ON for its tokens; h1 reads density 1


def rho_key(arm, h):
    """The density key an arm runs at: the primary 'mid' becomes 'full' at h1; the sensitivity keys stay."""
    k = ARMS[arm][4]
    return H1_PRIMARY if (h == 1 and k == "mid") else k


def rho_of(arm, h):
    return RHO[rho_key(arm, h)]
# matched pairs: same storage format, same optimizer state, same density. The first four are the GATE pairs.
PAIRS = {
    "f32-adam":        ("C4",  "S1"),
    "f32-rowada":      ("C3",  "S2"),
    "bf16-rowada":     ("C5",  "S3"),
    "tern-rowada":     ("C6",  "S4"),
    "f32-rowada-low":  ("C3s", "S2s"),
    "f32-rowada-full": ("C3d", "S2d"),
}
GATE_PAIRS = ("f32-adam", "f32-rowada", "bf16-rowada", "tern-rowada")
GATE_H = (8, 1)


def a_row(r, rho):
    return 1.0 - (1.0 - rho) ** r


def cpu_bytes(arm, r, rho):
    """(fwd, bwd) bytes/param, lower bound. fwd: gate+up rows always read, down row only if active."""
    _, fmt, opt, fused, _ = ARMS[arm]
    a = a_row(r, rho); wb = FMT_B[fmt]
    fwd = wb * (2.0 / 3.0 + a / 3.0)
    if not fused:
        bwd = 4.0 * a + 4.0 + 12.0           # read active rows, write dense G, then read W+G and write W
    elif opt == "adam":
        bwd = 24.0                            # W, m, v read and written for every row
    elif fmt == "f32":
        bwd = 8.0 * a
    elif fmt == "bf16":
        bwd = 4.0 * a
    else:
        bwd = 8.5 * a                         # fp32 latent read+write, codes rewritten
    return fwd, bwd


def str_bytes(arm, r, rho, skip=False):
    """PCIe and host-RAM bytes/param for the streaming arms (lower bound). The weights cross PCIe twice: the bank is
    larger than the VRAM, so what the forward brought is gone by the backward of the same layer.
    skip=True: an IDEAL streamer that, knowing the active mask after the forward, moves only active rows in the
    backward (both PCIe directions and their DMA); dense Adam cannot skip. Review obj. 3."""
    _, fmt, opt, _, _ = ARMS[arm]
    a = a_row(r, rho)
    if skip and opt != "adam":
        b = str_bytes(arm, r, rho)
        dma = b["h2d_bwd"] + b["d2h"]                       # the DMA share of ram_bwd
        b["ram_bwd"] -= (1 - a) * dma
        b["h2d_bwd"] *= a; b["d2h"] *= a
        return b
    if fmt == "f32" and opt == "adam":
        return dict(h2d_fwd=2, h2d_bwd=2, d2h=2, ram_fwd=2, ram_bwd=4 + 2 + 4 + 8 + 4 + 8 + 2)
    if fmt == "f32":
        return dict(h2d_fwd=2, h2d_bwd=2, d2h=2, ram_fwd=2, ram_bwd=4 + a * (2 + 4 + 4 + 2))
    if fmt == "bf16":
        return dict(h2d_fwd=2, h2d_bwd=2, d2h=2, ram_fwd=2, ram_bwd=4)
    return dict(h2d_fwd=0.5, h2d_bwd=0.5, d2h=2, ram_fwd=0.5, ram_bwd=0.5 + 2 + a * (2 + 8 + 0.5))


def cpu_flop_tok(rho):
    """FLOP per bank parameter per routed token on the CPU path: gate/up dots always (4D per hidden unit), the down
    axpy (2D) and the backward (dot 2D + 2 axpy dx 4D + 3 axpy grads 6D) only for active units; per param = /(3D)."""
    return (4.0 + 14.0 * rho) / 3.0


def act_bytes_tok(h):
    """Activation bytes per bank parameter per routed token on the CPU path: x read twice, dy read, two accumulator
    rows read+written = 7 D-vectors of fp32 per (expert, token), over 3hD parameters."""
    return 7.0 * 4.0 / (3.0 * h)


def desk_cpu(arm, h, r, rho, bw_ram, f_cpu, overlap=False):
    """ns per parameter. additive: memory then compute; overlap: the max of the two (optimistic for the CPU)."""
    fwd, bwd = cpu_bytes(arm, r, rho)
    mem = (fwd + bwd) / bw_ram
    _, fmt, _, _, _ = ARMS[arm]
    comp = r * (cpu_flop_tok(rho) / f_cpu + act_bytes_tok(h) / bw_ram)
    return max(mem, comp) if overlap else mem + comp


def desk_str(arm, h, r, rho, bw_ram, bw_h2d, bw_bidir, bw_vram=360.0, ideal=False):
    """ns per parameter, pipelined within each sweep; forward and backward sweeps are separate (other layers run in
    between). The backward sweep moves data both ways at once, so both directions run at the bidirectional rate.
    GPU work is charged only as activation traffic in VRAM (its FLOPs are far below the tensor-core roofline)."""
    b = str_bytes(arm, r, rho, skip=ideal)
    gpu = 0.0 if ideal else r * (20.0 / (3.0 * h)) / bw_vram
    t_f = max(b["h2d_fwd"] / bw_h2d, b["ram_fwd"] / bw_ram, gpu / 2)
    t_b = max(b["h2d_bwd"] / bw_bidir, b["d2h"] / bw_bidir, b["ram_bwd"] / bw_ram, gpu)
    return t_f + t_b


def floor_cpu(arm, r, rho, bw_ram_peak):
    fwd, bwd = cpu_bytes(arm, r, rho)
    return (fwd + bwd) / bw_ram_peak


def floor_str(arm, r, rho, bw_ram_peak, bw_h2d_peak, bw_d2h_peak):
    """Lower bound with the best measured single-direction rates (no bidirectional penalty) and a = a(r, rho)."""
    b = str_bytes(arm, r, rho)
    return (max(b["h2d_fwd"] / bw_h2d_peak, b["ram_fwd"] / bw_ram_peak)
            + max(b["h2d_bwd"] / bw_h2d_peak, b["d2h"] / bw_d2h_peak, b["ram_bwd"] / bw_ram_peak))


def crossover(rs, t_cpu, t_str):
    """Smallest r at which the CPU stops being cheaper, interpolated in log r. Returns (value, tag):
    tag '<' = the CPU loses already at the first r; '>' = the CPU wins at every r; '=' = interpolated."""
    import math
    d = [c - s for c, s in zip(t_cpu, t_str)]
    if d[0] >= 0:
        return rs[0], "<"
    for k in range(1, len(rs)):
        if d[k] >= 0:
            x0, x1 = math.log(rs[k - 1]), math.log(rs[k])
            f = d[k - 1] / (d[k - 1] - d[k])
            return math.exp(x0 + f * (x1 - x0)), "="
    return rs[-1], ">"


def bank_gain(rs, t_cpu, t_str, mean_r, rmax_tail=2048):
    """Review obj. 5: the step-level value of per-expert placement. Experts get Poisson(mean_r) tokens (uniform
    routing); r = 0 is free on both paths; each expert goes where it is cheaper. t at non-grid r is interpolated in
    log-log; above the grid the CPU is charged its last point scaled linearly in r (compute-bound) and streaming its
    last point (flat). Returns 1 - cost(best placement) / cost(all streaming)."""
    import math
    def at(ts, r):
        if r <= rs[0]:
            return ts[0]
        if r >= rs[-1]:
            return ts[-1]
        for k in range(1, len(rs)):
            if r <= rs[k]:
                f = (math.log(r) - math.log(rs[k - 1])) / (math.log(rs[k]) - math.log(rs[k - 1]))
                return math.exp(math.log(ts[k - 1]) + f * (math.log(ts[k]) - math.log(ts[k - 1])))
    p = math.exp(-mean_r); best = alls = 0.0
    for r in range(1, rmax_tail):
        p *= mean_r / r
        if p < 1e-12 and r > mean_r:
            break
        tc = at(t_cpu, r) if r <= rs[-1] else t_cpu[-1] * r / rs[-1]
        ts = at(t_str, r)
        best += p * min(tc, ts); alls += p * ts
    return 1.0 - best / alls if alls else 0.0


# registered regimes (protocol section 3): 60 GB of host RAM, 8 layers, T tokens per step
REGIME_BYTES = {"f32-adam": 14.0, "f32-rowada": 6.0, "bf16-rowada": 2.0, "tern-rowada": 4.5}


def regime_mean_r(pair, tokens_per_step, ram_bytes=60e9, layers=8):
    return r_regime(tokens_per_step, ram_bytes / REGIME_BYTES[pair] / layers)


def r_regime(tokens_per_step, bank_params_per_layer):
    """Mean tokens per expert per step for uniform routing: r = T * (active params / bank params) per layer,
    independent of h at fixed active hidden A."""
    return tokens_per_step * 3.0 * A * D / bank_params_per_layer
