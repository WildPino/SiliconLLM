#!/usr/bin/env python3
# Sparse Capacity, phase 2 (D-0, D-R): the model. Protocol: docs/research/SPARSE_CAPACITY/F2_BANK_PROTOCOL.md.
#
# Core = the rung-1 S0 shape (D256 N96 H8 L8, SWA at layer 5, dt_rank 16, x_proj low-rank r = 26), unchanged.
# The per-block MLP is the only thing that changes between arms:
#   dense   gated dReLU, hidden 1024                       (the rung-1 MLP before the upcycle)
#   bank M  E = 128*M experts of h = 8, top-k = 128         (active hidden k*h = 1024 = the dense hidden)
# Precision follows the phase-0 audit (F0_QUANT_AUDIT.md section 4): ternary weights with a per-row absmean scale, and
# the engine's activation quantization (act_quant=engine): integers in [-63, 63], one absmax scale per group, round half
# to even; integer product first, scales applied after in fp32 (W1). Groups: x per token over D (A1, shared by every
# expert of the token; the router reads x NOT quantized), the weighted hidden per (token, expert) over h (A2), and for
# the dense MLP the hidden per token over 1024 (A3). Down scales: per (expert, output d) over h (bank), per row (dense).
import math, os, sys
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
from phase55_ssm import ArchA  # noqa: E402

S0 = dict(D=256, N=96, H=8, L=8, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16)
XPROJ_RANK = 26
QA = 63                       # engine.c: |x| <= 63 so a LUT pair sum fits int8 (F0 section 2)
H_EXP = 8                     # expert hidden (granularity choice, protocol section 2)
ACTIVE = 1024                 # active hidden per token = dense hidden = 4*D
LOAD_W = 0.01                 # rung-1 load-balance weight


class _RoundSTE(torch.autograd.Function):
    """q = clamp(round(v / s), -lim, lim) with the straight-through gradient g / s (s is a detached scale)."""
    @staticmethod
    def forward(ctx, v, s, lim):
        ctx.save_for_backward(s)
        return torch.round(v / s).clamp_(-lim, lim)

    @staticmethod
    def backward(ctx, g):
        (s,) = ctx.saved_tensors
        return g / s, None, None


def act_q(x, dim=-1):
    """Engine quant_i8 on the last dim: returns (integer-valued q, scale). Scale = absmax/63 (detached)."""
    s = (x.detach().abs().amax(dim=dim, keepdim=True) / QA).clamp_min(1e-8)
    return _RoundSTE.apply(x, s, QA), s


def w_q(w, dim=-1):
    """Ternary weight with absmean scale over `dim` (per row for (out, in) matrices): (integer-valued q, scale)."""
    s = w.detach().abs().mean(dim=dim, keepdim=True).clamp_min(1e-5)
    return _RoundSTE.apply(w, s, 1.0), s


def mm_exact(a, b):
    """a @ b in full fp32 (autocast off; TF32 must stay off, the default for matmul). Integer-valued operands
    (|q| <= 63, ternary) give an exact sum below 2^24: the engine's int32 product (W1). On the 3060 full fp32 runs at the
    TF32 rate, so exactness costs nothing here."""
    with torch.autocast(a.device.type, enabled=False):
        return a.float() @ b.float()


class DenseMLP(nn.Module):
    """Gated dReLU MLP, ternary, act_quant=engine (A1 on x, A3 on the hidden)."""
    def __init__(s, D, hid):
        super().__init__()
        s.gate = nn.Linear(D, hid, bias=False); s.up = nn.Linear(D, hid, bias=False); s.down = nn.Linear(hid, D, bias=False)
        s.aux = None

    def forward(s, x):
        cd = x.dtype
        x = x.float()
        qx, sx = act_q(x)
        qg, sg = w_q(s.gate.weight); qu, su = w_q(s.up.weight); qd, sd = w_q(s.down.weight)
        g = mm_exact(qx, qg.T) * sx * sg.T
        u = mm_exact(qx, qu.T) * sx * su.T
        h = F.relu(g) * F.relu(u)
        qh, sh = act_q(h)
        y = mm_exact(qh, qd.T) * sh * sd.T
        s.aux = None
        return y.to(cd)


class DenseFP(nn.Module):
    """The same gated dReLU MLP without any quantization (the precision reading, protocol section 3)."""
    def __init__(s, D, hid):
        super().__init__()
        s.gate = nn.Linear(D, hid, bias=False); s.up = nn.Linear(D, hid, bias=False); s.down = nn.Linear(hid, D, bias=False)
        s.aux = None

    def forward(s, x):
        return s.down(F.relu(s.gate(x)) * F.relu(s.up(x)))


class Bank(nn.Module):
    """E experts of hidden h, softmax router over all E in fp32, top-k renormalized, rung-1 load-balance loss.

    Dense products, sparse elementwise: gate and up are one dense product over all E*h units (cheap on tensor-free
    fp32 too), the k selected experts are gathered, every elementwise step (scales, dReLU, router weight, A2) runs on the
    (N, k, h) selection only, and the down is one dense product of the selection scattered back into (N, E*h). The maths
    is the active-only one: unselected experts get exactly zero gradient. Deterministic: top-k indices are unique per
    token, so gather/scatter never collide (no atomics on shared addresses).
    """
    def __init__(s, D, E, h=H_EXP, k=None, load_w=LOAD_W):
        super().__init__()
        s.D, s.E, s.h = D, E, h
        s.k = k if k is not None else ACTIVE // h
        s.load_w = load_w
        s.router = nn.Linear(D, E, bias=True)
        s.Wg = nn.Parameter(torch.empty(E, h, D)); s.Wu = nn.Parameter(torch.empty(E, h, D))
        s.Wd = nn.Parameter(torch.empty(E, D, h))
        b_in = 1.0 / math.sqrt(D)
        for p in (s.Wg, s.Wu): nn.init.uniform_(p, -b_in, b_in)
        # down init at the dense magnitude (1/sqrt(k*h), like the dense down over 1,024): the factor k that the
        # renormalized top-k weights need goes on the weights in the forward (topw * k), not on Wd. With Wd scaled by k
        # an AdamW step (~lr per weight) would move the bank's down k = 128 times slower than the dense one (review
        # item 3). The A2 scale absorbs the factor, so the forward at init is the same either way.
        b_out = 1.0 / math.sqrt(s.k * h)
        nn.init.uniform_(s.Wd, -b_out, b_out)
        s.aux = None
        s.stats = {}
        s.count_acc = None

    def forward(s, x):
        B, T, D = x.shape
        cd = x.dtype
        xf = x.reshape(-1, D).float(); N = xf.shape[0]
        E, h = s.E, s.h
        with torch.autocast(xf.device.type, enabled=False):
            probs = torch.softmax(s.router(xf), dim=-1)                       # (N,E) fp32, x NOT quantized
        topv, topi = probs.topk(s.k, dim=-1)
        topw = topv / topv.sum(-1, keepdim=True) * s.k                       # mean weight 1 per selected expert
        k = s.k
        qx, sx = act_q(xf)                                                     # A1: per token over D
        qg, sg = w_q(s.Wg); qu, su = w_q(s.Wu)                                 # per row (expert, unit) over D
        qd, sd = w_q(s.Wd)                                                     # per (expert, output d) over h
        idx = topi.unsqueeze(-1).expand(N, k, h)
        gi = mm_exact(qx, qg.reshape(E * h, D).T).reshape(N, E, h).gather(1, idx)   # integer sums, selected only
        ui = mm_exact(qx, qu.reshape(E * h, D).T).reshape(N, E, h).gather(1, idx)
        g = gi * sx.unsqueeze(-1) * sg.reshape(E, h)[topi]
        u = ui * sx.unsqueeze(-1) * su.reshape(E, h)[topi]
        hh = F.relu(g) * F.relu(u) * topw.unsqueeze(-1)                         # weight the hidden before the down
        qh, sh = act_q(hh)                                                     # A2: per (token, expert) over h
        # out[n,d] = sum_e sh[n,e] * sd[e,d] * sum_j qh[n,e,j] * qd[e,d,j]: sd folded into the weight and sh into the
        # activation, each one fp32 product per element (the engine forms S*sa*sc in fp32 too)
        act = torch.zeros(N, E, h, device=x.device, dtype=torch.float32).scatter(1, idx, qh * sh)
        wd = (qd * sd).permute(0, 2, 1).reshape(E * h, D)                      # (E*h, D)
        y = mm_exact(act.reshape(N, E * h), wd)
        counts = torch.bincount(topi.reshape(-1), minlength=E).float()
        load = E * ((counts / (N * k)) * probs.mean(0)).sum()
        s.aux = s.load_w * load
        with torch.no_grad():
            s.stats = dict(load=load.detach(), used=(counts > 0).float().mean())
            s.count_acc = counts if s.count_acc is None else s.count_acc + counts
        return y.reshape(B, T, D).to(cd)


class LowRankLinear(nn.Module):
    """x_proj at rank r (the rung-1 parameterization, mve_model.LowRankLinear)."""
    def __init__(s, fin, fout, r):
        super().__init__(); s.Vl = nn.Linear(fin, r, bias=False); s.Ul = nn.Linear(r, fout, bias=False)

    def forward(s, x):
        return s.Ul(s.Vl(x))


class Student(ArchA):
    def __init__(s, V, M=0, quant=True):
        super().__init__(V, **S0)
        D = S0["D"]; s.M = M
        for blk in s.blocks:
            blk.mlp = (DenseMLP(D, ACTIVE) if quant else DenseFP(D, ACTIVE)) if M == 0 else Bank(D, E=128 * M)
            xp = getattr(blk.mix, "x_proj", None)
            if xp is not None:
                blk.mix.x_proj = LowRankLinear(xp.in_features, xp.out_features, XPROJ_RANK)

    def forward(s, idx):
        import torch.utils.checkpoint as ckpt
        x = s.emb(idx)
        for b in s.blocks:
            x = ckpt.checkpoint(b, x, use_reentrant=False) if (s.training and s.use_ckpt) else b(x)
        aux = sum(b.mlp.aux for b in s.blocks if b.mlp.aux is not None) if s.M else None
        return s.head(s.norm_f(x)), aux

    def counts(s):
        n = sum(p.numel() for p in s.parameters())
        mlp = sum(p.numel() for b in s.blocks for p in b.mlp.parameters())
        if s.M == 0:
            return dict(total=n, mlp=mlp, active=n, bank=0)
        router = sum(p.numel() for b in s.blocks for p in b.mlp.router.parameters())
        bank = mlp - router
        k, E = s.blocks[0].mlp.k, s.blocks[0].mlp.E
        return dict(total=n, mlp=mlp, bank=bank, router=router, active=int(n - bank + bank * k / E))
