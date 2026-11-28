#!/usr/bin/env python3
# Sparse Capacity, phase 4 (D-M): the bank beyond the VRAM. Protocol: docs/research/SPARSE_CAPACITY/F4_BEYOND_VRAM_PROTOCOL.md.
#
# The fp32 masters of the experts, their gradients and the AdamW moments live in host RAM (12 + 4 bytes per weight).
# The GPU keeps only what the forward uses: the ternary codes, 2 bits per weight, and the per-row scales
# (sc2_model.w_q). After each step the fused kernel (csrc/adamw_tern.c) updates the masters and writes the new codes and
# scales; they go to the GPU once per step (0.43 bytes per weight instead of 4), and stay there for the forward and the
# backward of the next step. Same maths as AdamW on fp32 masters followed by w_q (kernel selftest: bitwise AdamW, codes
# equal to w_q). The backward returns, per layer, the dense gradient of the codes divided by the scales (the
# straight-through gradient of w_q, as sc2_model._RoundSTE) to a pinned host buffer, with its norm for the clip.
import threading, time
import torch
import sc2_model as SM
import sc4_adamw as KA
import sc4_model as M4

NAMES = ("Wg", "Wu", "Wd")


def unpack_codes(b, n):
    """(rows, n/4) uint8 codes (4 trits per byte, code q + 1, little end first) -> (rows, n) fp32 in {-1, 0, 1}."""
    return torch.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(b.shape[0], n).float() - 1.0


class _HostQ(torch.autograd.Function):
    """Returns the integer-valued ternary weights and their scales of one layer from the GPU codes. Backward: the
    straight-through gradient g / s goes to the host (no gradient flows further on the GPU)."""
    @staticmethod
    def forward(ctx, anchor, hw):
        ctx.hw = hw
        out = hw.unpacked()
        ctx.mark_non_differentiable(*out[1::2])                              # the scales are detached, as in w_q
        return out

    @staticmethod
    def backward(ctx, dqg, dsg, dqu, dsu, dqd, dsd):
        ctx.hw.grads_out(dqg, dqu, dqd)
        return None, None


class HostWeights:
    """One layer's bank on the host, with its codes on the GPU. Shapes: Wg, Wu (E, h, D), Wd (E, D, h); rows of w_q are
    the last dim."""
    def __init__(s, E, h, D, k, dev, gen, nthreads=6):
        s.E, s.h, s.D, s.dev, s.nthreads = E, h, D, dev, nthreads
        shapes = dict(Wg=(E, h, D), Wu=(E, h, D), Wd=(E, D, h))
        s.p = {n: torch.empty(sh) for n, sh in shapes.items()}
        M4.init_bank_(s.p["Wg"], s.p["Wu"], s.p["Wd"], k, gen)
        s.m = {n: torch.zeros(sh) for n, sh in shapes.items()}
        s.v = {n: torch.zeros(sh) for n, sh in shapes.items()}
        s.g = {n: torch.zeros(sh).pin_memory() for n, sh in shapes.items()}            # the gradient lands here
        s.codes_h = {n: torch.empty(*sh[:-1], sh[-1] // 4, dtype=torch.uint8).pin_memory() for n, sh in shapes.items()}
        s.scale_h = {n: torch.empty(*sh[:-1], 1).pin_memory() for n, sh in shapes.items()}
        s.codes = {n: torch.empty_like(t, device=dev) for n, t in s.codes_h.items()}
        s.scale = {n: torch.empty_like(t, device=dev) for n, t in s.scale_h.items()}
        s.anchor = torch.zeros((), device=dev, requires_grad=True)
        s.norms = []; s.t = dict(grad_out=0.0, upload=0.0, host_adam=0.0); s.timed = False
        s.ready = threading.Event(); s.ready.set()                           # codes of the last update are on the GPU
        for n in NAMES:
            KA.pack(s.p[n], s.codes_h[n], s.scale_h[n], nthreads)
        s.upload()

    # ------------------------------------------------------------------ GPU side
    def unpacked(s):
        out = []
        for n in NAMES:
            sh = s.p[n].shape
            q = unpack_codes(s.codes[n].view(-1, sh[-1] // 4), sh[-1]).view(sh)
            out += [q, s.scale[n]]
        return tuple(out)

    def get(s):
        s.ready.wait()                                                        # pipelined update: this layer's codes
        return _HostQ.apply(s.anchor, s)

    def grads_out(s, dqg, dqu, dqd):
        """Straight-through gradient of w_q (g / s, the op of sc2_model._RoundSTE.backward), its norm on the GPU for the
        clip, and the copy to the pinned host gradient (each layer's backward runs once per step)."""
        if s.timed:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        G = [d / s.scale[n] for d, n in zip((dqg, dqu, dqd), NAMES)]
        s.norms = list(torch._foreach_norm(G, 2.0))
        for gr, n in zip(G, NAMES):
            s.g[n].copy_(gr, non_blocking=True)                              # read on the host only after the step's sync
        if s.timed:
            torch.cuda.current_stream().synchronize(); s.t["grad_out"] += time.perf_counter() - t0

    def upload(s):
        t0 = time.perf_counter()
        for n in NAMES:
            s.codes[n].copy_(s.codes_h[n], non_blocking=True); s.scale[n].copy_(s.scale_h[n], non_blocking=True)
        s.t["upload"] += time.perf_counter() - t0                            # issue time (the copy runs on the stream)

    # ----------------------------------------------------------------- host side
    def step(s, stepno, lr, gscale, betas, wd, eps, skip=False):
        """Fused AdamW on the masters (gradient times the clip coefficient), new codes and scales, the upload, and the
        gradient zeroed for the next step. The caller has synchronized: the GPU is done with the old codes."""
        t0 = time.perf_counter()
        if not skip:
            for n in NAMES:
                KA.step(s.p[n], s.g[n], s.m[n], s.v[n], stepno, lr, s.codes_h[n], s.scale_h[n], s.nthreads, betas, wd, eps, gscale)
        s.t["host_adam"] += time.perf_counter() - t0
        s.upload()
        s.zero_grad()

    def zero_grad(s):
        for n in NAMES:
            s.g[n].zero_()

    def state(s):
        return dict(p=s.p, m=s.m, v=s.v)

    def load_state(s, st):
        for n in NAMES:
            s.p[n].copy_(st["p"][n]); s.m[n].copy_(st["m"][n]); s.v[n].copy_(st["v"][n])
            KA.pack(s.p[n], s.codes_h[n], s.scale_h[n], s.nthreads)
        s.upload()

    def nbytes(s):
        return dict(host=sum(t.numel() * 4 for d in (s.p, s.m, s.v, s.g) for t in d.values()),
                    gpu=sum(t.numel() * t.element_size() for d in (s.codes, s.scale) for t in d.values()))


def build(V, E, dev, seed, nthreads=6):
    """The phase-4 model with every bank on the host. The host init draws from its own generator (seed), so the core
    init (torch's global RNG) is the same as in the GPU arms built with the same seed."""
    gen = torch.Generator().manual_seed(10_000 + seed)
    hws = []
    def factory(li, E_):
        hw = HostWeights(E_, SM.H_EXP, SM.S0["D"], SM.ACTIVE // SM.H_EXP, dev, gen, nthreads); hws.append(hw); return hw
    m = M4.build(V, E, factory, seed)
    return m, hws


class Updater:
    """The host update of all layers in a background thread, in layer order: layer l's codes reach the GPU (its event
    is set) while the forward of the next step already runs on layers < l. Same operations in the same order as the
    synchronous update; the kernel runs in C with the GIL released (ctypes)."""
    def __init__(s, hws):
        s.hws = hws; s.th = None; s.last = {}

    def start(s, stepno, lr, gscale, betas, wd, eps, skip_layer=None, sync=False):
        s.wait()
        for hw in s.hws:
            hw.ready.clear(); hw.t = dict(grad_out=hw.t.get("grad_out", 0.0), upload=0.0, host_adam=0.0)
        def run():
            t0 = time.perf_counter()
            for li, hw in enumerate(s.hws):
                hw.step(stepno, lr, gscale, betas, wd, eps, skip=(skip_layer == "all" or li == skip_layer))
                hw.ready.set()
            s.last = dict(host_update=time.perf_counter() - t0, host_adam=sum(hw.t["host_adam"] for hw in s.hws))
        if sync:
            run()
        else:
            s.th = threading.Thread(target=run, daemon=True); s.th.start()

    def wait(s):
        if s.th is not None:
            s.th.join(); s.th = None
        torch.cuda.synchronize()


def clip_(model, hws, max_norm):
    """torch.nn.utils.clip_grad_norm_ over the GPU parameters and the host banks: norms per tensor, the 2-norm of the
    stacked norms, coef = clamp(max_norm / (total + 1e-6), 1). GPU gradients are scaled here; the host ones inside the
    kernel (gscale). Returns (total, coef)."""
    gpu = [p.grad for p in model.parameters() if p.grad is not None]
    norms = list(torch._foreach_norm(gpu, 2.0)) + [n for hw in hws for n in hw.norms]
    total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    torch._foreach_mul_(gpu, coef)
    return total, coef
