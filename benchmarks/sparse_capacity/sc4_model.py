#!/usr/bin/env python3
# Sparse Capacity, phase 4 (D-M): the bank with a product-key router and a sparse expert computation, for banks of
# tens of thousands of experts per layer. Protocol: docs/research/SPARSE_CAPACITY/F4_BEYOND_VRAM_PROTOCOL.md.
#
# The expert maths is phase 2's (sc2_model.Bank): ternary weights with a per-row absmean scale, the engine's activation
# quantization (A1 on x per token, A2 on the weighted hidden per (token, expert)), integer products in exact fp32,
# gated dReLU, top-k renormalized weights times k. Two things change:
#
# 1. Router: product keys (Lample et al. 2019, arXiv 1907.05242; PEER, arXiv 2407.04153). q = Wq x (dk = 128, half of D,
#    as Memory Layers at Scale), split in halves; scores s1 = q1 K1^T + b1 (n1), s2 = q2 K2^T + b2 (n2); the score of
#    expert (i, j) is s1_i + s2_j, E = n1 * n2. The softmax over all E factorizes, p(i,j) = softmax(s1)_i softmax(s2)_j,
#    so phase 2's gating (softmax over E, top-k, renormalized, times k = softmax over the top-k scores, times k) and its
#    load-balance loss E * sum_e f_e mean_n p_e(n) are computed exactly, the mean probability as (P1^T P2) / N. The
#    top-k over E is exact: it lies among the top-k1 x top-k2 candidates (PKM section 3.1).
# 2. Expert computation: only the selected (token, expert) pairs. Pairs are grouped by expert and every group padded
#    to a power of two (so padding is below 2x); each group is one batched matmul with the expert's weights. Forward
#    and backward are written by hand so that every sum runs in a fixed order (no atomics): the gate/up integer sums
#    are bitwise phase 2's; the down and the gradients sum the same terms in another order (selftest: 1e-6 relative).
import math, os, sys
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc2_model as SM      # noqa: E402

DK = 128
CHUNK_ELEMS = 1 << 26        # padded elements per batched matmul (x4 bytes): bounds the transient memory
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False    # the exact integer sums need fp32 reductions


def pk_shape(E):
    """n1 x n2 = E, n1 >= n2, both powers of two."""
    b = int(math.log2(E)); assert 1 << b == E
    return 1 << ((b + 1) // 2), 1 << (b // 2)


class _DetGather(torch.autograd.Function):
    """src.gather(1, idx) with a deterministic backward: the gradient of a source column is the sum of its slots in a
    fixed order (one-hot over the n columns, summed over the k slots, in chunks of tokens). torch's gather backward
    is an atomic scatter-add, whose order changes from run to run when a row repeats an index (here, always)."""
    @staticmethod
    def forward(ctx, src, idx):
        ctx.save_for_backward(idx); ctx.n = src.shape[1]
        return src.gather(1, idx)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors; N, k = idx.shape; n = ctx.n
        out = g.new_empty(N, n)
        step = max(1, (1 << 25) // (k * n))
        ar = torch.arange(n, device=g.device)
        for a in range(0, N, step):
            oh = (idx[a:a + step, :, None] == ar).to(g.dtype)                  # (c, k, n)
            out[a:a + step] = (oh * g[a:a + step, :, None]).sum(1)
        return out, None


class PKRouter(nn.Module):
    def __init__(s, D, E, dk=DK):
        super().__init__()
        s.n1, s.n2 = pk_shape(E); s.E = E; s.dk = dk
        s.q = nn.Linear(D, dk, bias=False)
        bound = 1.0 / math.sqrt(dk // 2)
        s.K1 = nn.Parameter(torch.empty(s.n1, dk // 2).uniform_(-bound, bound))
        s.K2 = nn.Parameter(torch.empty(s.n2, dk // 2).uniform_(-bound, bound))
        s.b1 = nn.Parameter(torch.zeros(s.n1)); s.b2 = nn.Parameter(torch.zeros(s.n2))

    def forward(s, xf, k):
        """xf (N, D) fp32. Returns topi (N, k) expert ids i * n2 + j, topw (N, k) = softmax over the selected scores
        times k, and the load-balance term's two factors (p1, p2)."""
        N = xf.shape[0]
        with torch.autocast(xf.device.type, enabled=False):
            q = s.q(xf)
            q1, q2 = q[:, :s.dk // 2], q[:, s.dk // 2:]
            s1 = q1 @ s.K1.T + s.b1; s2 = q2 @ s.K2.T + s.b2
            k1, k2 = min(k, s.n1), min(k, s.n2)
            with torch.no_grad():
                v1, i1 = s1.topk(k1, dim=-1); v2, i2 = s2.topk(k2, dim=-1)
                ii = torch.empty(N, k, dtype=torch.long, device=xf.device); jj = torch.empty_like(ii)
                step = max(1, (1 << 24) // (k1 * k2))
                for a in range(0, N, step):
                    cand = (v1[a:a + step, :, None] + v2[a:a + step, None, :]).reshape(-1, k1 * k2)
                    _, flat = cand.topk(k, dim=-1)
                    ii[a:a + step] = i1[a:a + step].gather(1, flat // k2); jj[a:a + step] = i2[a:a + step].gather(1, flat % k2)
            sel = _DetGather.apply(s1, ii) + _DetGather.apply(s2, jj)
            topw = torch.softmax(sel, dim=-1) * k
            p1 = torch.softmax(s1, dim=-1); p2 = torch.softmax(s2, dim=-1)
        return ii * s.n2 + jj, topw, (p1, p2)

    def load(s, topi, p1, p2, k):
        """Phase 2's loss: E * sum_e (count_e / (N k)) * mean_n p_e(n), with p_e = p1_i p2_j."""
        N = topi.shape[0]
        counts = torch.bincount(topi.reshape(-1), minlength=s.E).float()
        with torch.autocast(p1.device.type, enabled=False):
            pm = (p1.T @ p2).reshape(-1) / N
        return s.E * ((counts / (N * k)) * pm).sum(), counts


# ------------------------------------------------------------------------------------------- grouped pair layout
class Layout:
    """The selected pairs grouped by expert. Pair p = n * k + slot. For each capacity class C = 2^b (experts whose count
    is in (C/2, C]), in chunks of experts: the expert ids (m,), and the pair ids (m, C) padded with P (a zero row)."""
    def __init__(s, topi, E):
        N, k = topi.shape; P = N * k; dev = topi.device
        s.N, s.k, s.P, s.E = N, k, P, E
        e = topi.reshape(-1)
        order = torch.sort(e, stable=True).indices
        es = e[order]
        counts = torch.bincount(e, minlength=E)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(P, device=dev) - starts[es]
        cls = torch.where(counts > 0, torch.ceil(torch.log2(counts.clamp_min(1).float())).long(), torch.full_like(counts, -1))
        s.groups = []
        cpu_cls = cls.cpu()
        for b in sorted(set(cpu_cls.tolist()) - {-1}):
            C = 1 << b
            ex = torch.nonzero(cls == b).squeeze(1)                           # expert ids of the class, ascending
            rowpos = torch.full((E,), -1, dtype=torch.long, device=dev); rowpos[ex] = torch.arange(ex.numel(), device=dev)
            m = (cls[es] == b)
            idx = torch.full((ex.numel() * C,), P, dtype=torch.long, device=dev)
            idx[rowpos[es[m]] * C + rank[m]] = order[m]                     # unique positions
            idx = idx.view(ex.numel(), C)
            per = max(1, CHUNK_ELEMS // (C * 256))
            for a in range(0, ex.numel(), per):
                s.groups.append((ex[a:a + per], idx[a:a + per]))
        s.tok = torch.arange(P + 1, device=dev) // k; s.tok[P] = N            # pair -> token (P -> the zero row)


def _pad_rows(t):
    return torch.cat([t, t.new_zeros(1, *t.shape[1:])], 0)


class _GateUp(torch.autograd.Function):
    """gi, ui (N, k, h): integer sums qx[n] . qg[e, j] over the selected pairs (exact in fp32 in any order)."""
    @staticmethod
    def forward(ctx, qx, qg, qu, lay):
        # Operands are integers (|qx| <= 63, ternary weights): exact in bf16, and the products are summed in fp32 with
        # an fp32 output (out_dtype; reduced-precision reductions off), so the sums are the exact integers, bitwise
        # those of phase 2's fp32 product (selftest), with half the memory traffic and the tensor cores.
        h = qg.shape[1]
        W = torch.cat([qg, qu], 1)                                            # (E, 2h, D) fp32, kept for the backward
        Wb = W.to(torch.bfloat16)
        xz = _pad_rows(qx).to(torch.bfloat16)
        out = qx.new_zeros(lay.P + 1, 2 * h)
        with torch.autocast(qx.device.type, enabled=False):
            for ex, idx in lay.groups:
                X = xz[lay.tok[idx]]                                          # (m, C, D) bf16
                out[idx.reshape(-1)] = torch.bmm(X, Wb[ex].transpose(1, 2), out_dtype=torch.float32).reshape(-1, 2 * h)
        out[lay.P] = 0
        ctx.lay = lay; ctx.save_for_backward(qx, W); ctx.h = h
        o = out[:lay.P].view(lay.N, lay.k, 2 * h)
        return o[..., :h].contiguous(), o[..., h:].contiguous()

    @staticmethod
    def backward(ctx, dgi, dui):
        qx, W = ctx.saved_tensors; lay = ctx.lay; h = ctx.h
        dG = _pad_rows(torch.cat([dgi, dui], -1).reshape(lay.P, 2 * h).float())
        xz = _pad_rows(qx)
        dW = torch.zeros_like(W)
        dpair = qx.new_zeros(lay.P + 1, qx.shape[1])
        with torch.autocast(qx.device.type, enabled=False):
            for ex, idx in lay.groups:
                g = dG[idx]                                                   # (m, C, 2h), padding rows are 0
                X = xz[lay.tok[idx]]
                dW[ex] = torch.bmm(g.transpose(1, 2), X)                      # (m, 2h, D)
                dpair[idx.reshape(-1)] = torch.bmm(g, W[ex]).reshape(-1, qx.shape[1])
        dqx = dpair[:lay.P].view(lay.N, lay.k, -1).sum(1)
        return dqx, dW[:, :h], dW[:, h:], None


class _Down(torch.autograd.Function):
    """y[n] = sum over the selected pairs (n, e) of a[n, slot] (h) times wdf[e] (D, h)^T."""
    @staticmethod
    def forward(ctx, a, wdf, lay):
        h = a.shape[-1]; D = wdf.shape[1]
        az = _pad_rows(a.reshape(lay.P, h))
        pair = a.new_zeros(lay.P + 1, D)
        with torch.autocast(a.device.type, enabled=False):
            for ex, idx in lay.groups:
                pair[idx.reshape(-1)] = torch.bmm(az[idx], wdf[ex].transpose(1, 2)).reshape(-1, D)
        ctx.lay = lay; ctx.save_for_backward(a, wdf)
        return pair[:lay.P].view(lay.N, lay.k, D).sum(1)

    @staticmethod
    def backward(ctx, dy):
        a, wdf = ctx.saved_tensors; lay = ctx.lay; h = a.shape[-1]; D = wdf.shape[1]
        az = _pad_rows(a.reshape(lay.P, h))
        dyz = _pad_rows(dy.float())                                           # row N = 0: padding gets 0
        da = a.new_zeros(lay.P + 1, h); dW = torch.zeros_like(wdf)
        with torch.autocast(a.device.type, enabled=False):
            for ex, idx in lay.groups:
                g = dyz[lay.tok[idx]]                                         # (m, C, D)
                da[idx.reshape(-1)] = torch.bmm(g, wdf[ex]).reshape(-1, h)
                dW[ex] = torch.bmm(g.transpose(1, 2), az[idx])                # (m, D, h)
        return da[:lay.P].view_as(a), dW, None


def experts_sparse(xf, topw, topi, qg, sg, qu, su, qd, sd):
    """Phase 2's bank maths (sc2_model.Bank.forward from the act_q of x to the down) on the selected pairs only.
    q*/s* are the ternary weights and their scales (integer-valued fp32, as sc2_model.w_q returns them)."""
    N, k = topi.shape; E, h, D = qg.shape
    lay = Layout(topi, E)
    qx, sx = SM.act_q(xf)
    gi, ui = _GateUp.apply(qx, qg, qu, lay)
    g = gi * sx.unsqueeze(-1) * sg.reshape(E, h)[topi]
    u = ui * sx.unsqueeze(-1) * su.reshape(E, h)[topi]
    hh = F.relu(g) * F.relu(u) * topw.unsqueeze(-1)
    qh, sh = SM.act_q(hh)
    return _Down.apply(qh * sh, qd * sd, lay)


class WeightSource:
    """Where the ternary weights of a bank come from: phase 2's fp32 masters on the GPU (w_q in the forward, autograd
    down to the masters), or the host (sc4_host.HostWeights). `get()` returns (qg, sg, qu, su, qd, sd)."""
    def __init__(s, bank):
        s.bank = bank

    def get(s):
        b = s.bank
        qg, sg = SM.w_q(b.Wg); qu, su = SM.w_q(b.Wu); qd, sd = SM.w_q(b.Wd)
        return qg, sg, qu, su, qd, sd


class PKBank(nn.Module):
    """sc2_model.Bank with the product-key router and the sparse computation. With `host=None` the masters are GPU
    parameters (Wg, Wu, Wd, phase 2's init); otherwise they live on the host and only the router stays here."""
    def __init__(s, D, E, h=SM.H_EXP, k=None, load_w=SM.LOAD_W, host=None, gen=None):
        super().__init__()
        s.D, s.E, s.h = D, E, h
        s.k = k if k is not None else SM.ACTIVE // h
        s.load_w = load_w
        s.router = PKRouter(D, E)
        s.host = host
        if host is None:
            s.Wg = nn.Parameter(torch.empty(E, h, D)); s.Wu = nn.Parameter(torch.empty(E, h, D))
            s.Wd = nn.Parameter(torch.empty(E, D, h))
            init_bank_(s.Wg, s.Wu, s.Wd, s.k, gen)
            s.src = WeightSource(s)
        else:
            s.src = host
        s.aux = None; s.stats = {}; s.count_acc = None

    def forward(s, x):
        B, T, D = x.shape; cd = x.dtype
        xf = x.reshape(-1, D).float()
        topi, topw, (p1, p2) = s.router(xf, s.k)
        y = experts_sparse(xf, topw, topi, *s.src.get())
        load, counts = s.router.load(topi, p1, p2, s.k)
        s.aux = s.load_w * load
        with torch.no_grad():
            s.stats = dict(load=load.detach(), used=(counts > 0).float().mean())
            s.count_acc = counts if s.count_acc is None else s.count_acc + counts
        return y.reshape(B, T, D).to(cd)


def init_bank_(Wg, Wu, Wd, k, gen=None):
    """Phase 2's bank init (sc2_model.Bank): gate/up U(+-1/sqrt(D)), down U(+-1/sqrt(k h))."""
    E, h, D = Wg.shape
    b_in = 1.0 / math.sqrt(D); b_out = 1.0 / math.sqrt(k * h)
    with torch.no_grad():
        for p in (Wg, Wu):
            p.uniform_(-b_in, b_in, generator=gen)
        Wd.uniform_(-b_out, b_out, generator=gen)


def build(V, E, host_factory=None, seed=0):
    """The phase-2 Student (core unchanged) with a PKBank in every block. host_factory(layer, E) -> a host weight source
    (sc4_host) or None for GPU masters. The expert weights draw from their own generator (10000 + seed), on the host or
    here alike, so that the core and router inits come from torch's global RNG in the same order in every arm."""
    m = SM.Student(V, M=1, quant=True)
    gen = torch.Generator().manual_seed(10_000 + seed)
    Es = list(E) if isinstance(E, (list, tuple)) else [E] * len(m.blocks)   # a list: one E per layer (tests only)
    for li, blk in enumerate(m.blocks):
        blk.mlp = PKBank(SM.S0["D"], Es[li], host=host_factory(li, Es[li]) if host_factory else None, gen=gen)
    return m


def counts(m):
    n = sum(p.numel() for p in m.parameters())
    b0 = m.blocks[0].mlp; E, h, D, k = b0.E, b0.h, b0.D, b0.k
    bank = 3 * E * h * D * len(m.blocks)
    router = sum(p.numel() for blk in m.blocks for p in blk.mlp.router.parameters())
    on_gpu_bank = 0 if b0.host is not None else bank
    core = n - on_gpu_bank
    return dict(bank=bank, router=router, core_and_router=core, total=core + bank, active=core + bank * k // E, E=E)


# ------------------------------------------------------------------------------------------------------- selftest
def selftest(dev):
    res = []
    def check(name, val, tol):
        ok = bool(val <= tol); res.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<80} {val:.3e}  (tol {tol:.0e})")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # 1. router: exact top-k against the brute force over all E; gating and load loss against phase 2's formulas
    for E in (128, 2048, 32768):
        r = PKRouter(256, E).to(dev); xf = torch.randn(512, 256, device=dev)
        topi, topw, (p1, p2) = r(xf, 128)
        with torch.no_grad():
            q = r.q(xf); s1 = q[:, :64] @ r.K1.T + r.b1; s2 = q[:, 64:] @ r.K2.T + r.b2
            full = (s1[:, :, None] + s2[:, None, :]).reshape(512, -1)
            ref = full.topk(128, dim=-1).indices
        same = float((torch.sort(topi, 1).values != torch.sort(ref, 1).values).any(1).float().mean())
        check(f"E={E}: product-key top-k equals the brute-force top-k (fraction of tokens differing)", same, 0)
        probs = torch.softmax(full, -1)
        tv = probs.gather(1, topi); w2 = tv / tv.sum(-1, keepdim=True) * 128
        check(f"E={E}: weights = phase 2's renormalized top-k of the full softmax (max rel)", float(((topw - w2).abs() / w2).max()), 1e-4)
        load, cnt = r.load(topi, p1, p2, 128)
        ref_load = E * ((cnt / (512 * 128)) * probs.mean(0)).sum()
        check(f"E={E}: load loss = phase 2's on the full softmax (rel)", float((load - ref_load).abs() / ref_load), 1e-5)
        if E == 2048:
            r.zero_grad(); x1 = xf.clone().requires_grad_(True)
            ti, tw, (q1_, q2_) = r(x1, 128); (tw.pow(2).sum() + r.load(ti, q1_, q2_, 128)[0]).backward()
            g1 = [p.grad.clone() for p in r.parameters()] + [x1.grad.clone()]
            r.zero_grad(); x2 = xf.clone().requires_grad_(True)
            ti, tw, (q1_, q2_) = r(x2, 128); (tw.pow(2).sum() + r.load(ti, q1_, q2_, 128)[0]).backward()
            g2 = [p.grad.clone() for p in r.parameters()] + [x2.grad.clone()]
            check("router: two backward passes bitwise equal (max abs over every gradient)",
                  max(float((u - v).abs().max()) for u, v in zip(g1, g2)), 0)
    # 2. sparse expert maths = phase 2's dense maths (same topi/topw), forward and every gradient
    for E, N in ((256, 512), (2048, 1024)):
        bank = SM.Bank(256, E).to(dev)
        xf = torch.randn(N, 256, device=dev, requires_grad=True)
        logits = torch.randn(N, E, device=dev)
        topv, topi = torch.softmax(logits, -1).topk(128, -1)
        tw = (topv / topv.sum(-1, keepdim=True) * 128).requires_grad_(True)
        # phase 2 reference (sc3_stream.experts is sc2's maths with W given)
        import sc3_stream as ST
        Wr = {n: getattr(bank, n).detach().clone().requires_grad_(True) for n in ("Wg", "Wu", "Wd")}
        yr = ST.experts(xf, tw, topi, Wr, 8)
        gy = torch.randn_like(yr)
        gr = torch.autograd.grad(yr, [xf, tw] + list(Wr.values()), gy)
        Ws = {n: getattr(bank, n).detach().clone().requires_grad_(True) for n in ("Wg", "Wu", "Wd")}
        qg, sg = SM.w_q(Ws["Wg"]); qu, su = SM.w_q(Ws["Wu"]); qd, sd = SM.w_q(Ws["Wd"])
        x2 = xf.detach().clone().requires_grad_(True); t2 = tw.detach().clone().requires_grad_(True)
        ys = experts_sparse(x2, t2, topi, qg, sg, qu, su, qd, sd)
        gs = torch.autograd.grad(ys, [x2, t2] + list(Ws.values()), gy)
        rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
        check(f"E={E}: sparse forward vs phase 2 (max abs / max)", rel(ys, yr), 1e-6)
        for nm, a, b in zip(("x", "topw", "Wg", "Wu", "Wd"), gs, gr):
            check(f"E={E}: gradient of {nm} vs phase 2 (max abs / max)", rel(a, b), 1e-5)
        # gate/up integer sums bitwise: compare gi directly
        lay = Layout(topi, E); qx, _ = SM.act_q(xf.detach())
        gi, _ = _GateUp.apply(qx, qg.detach(), qu.detach(), lay)
        dense = SM.mm_exact(qx, qg.detach().reshape(E * 8, 256).T).reshape(N, E, 8).gather(1, topi.unsqueeze(-1).expand(N, 128, 8))
        check(f"E={E}: gate integer sums bitwise equal to phase 2's dense product (max abs)", float((gi - dense).abs().max()), 0)
        # unused experts get exactly zero gradient
        used = torch.zeros(E, dtype=torch.bool, device=dev); used[topi.reshape(-1)] = True
        check(f"E={E}: experts never selected have exactly zero gradient (max abs)", float(gs[2][~used].abs().max()) if (~used).any() else 0.0, 0)
        # determinism: the same call twice, bitwise
        qg, sg = SM.w_q(Ws["Wg"]); qu, su = SM.w_q(Ws["Wu"]); qd, sd = SM.w_q(Ws["Wd"])
        ys2 = experts_sparse(x2, t2, topi, qg, sg, qu, su, qd, sd)
        gs2 = torch.autograd.grad(ys2, [x2, t2] + list(Ws.values()), gy)
        check(f"E={E}: two runs bitwise equal (forward and gradients, max abs)",
              max([float((ys2 - ys).abs().max())] + [float((a - b).abs().max()) for a, b in zip(gs2, gs)]), 0)
    # 3. a whole model step runs and its loss is finite (E = 2048)
    m = build(2048, 2048).to(dev)
    xy = torch.randint(0, 2048, (2, 129), device=dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        lg, aux = m(xy[:, :-1]); loss = F.cross_entropy(lg.float().reshape(-1, 2048), xy[:, 1:].reshape(-1)) + aux
    loss.backward()
    nonzero = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in m.parameters())
    check("whole model: finite loss and a gradient on every parameter (missing count)",
          float(sum(1 for p in m.parameters() if p.grad is None or not torch.isfinite(p.grad).all())), 0)
    print("SELFTEST (phase 4 bank)", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


if __name__ == "__main__":
    sys.exit(0 if selftest(torch.device("cuda")) else 1)
