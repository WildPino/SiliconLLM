#!/usr/bin/env python3
# Sparse Capacity, phase 3 (the method): the bank kept in host RAM and streamed to the GPU.
# Protocol: docs/research/SPARSE_CAPACITY/F3_STREAM_PROTOCOL.md.
#
# StreamBank replaces sc2_model.Bank with the same maths. The master weights of the experts (fp32), their gradients and
# their AdamW state live in host RAM. In each forward the router runs on the GPU; the union U of the experts selected by
# the batch is gathered on the host into a pinned buffer and copied to the GPU; the bank maths runs on U only (the
# per-row ternary scales are per expert, so restricting to U changes nothing). The expert weights are NOT kept on the
# GPU between forward and backward: the backward fetches them again, recomputes the bank locally and sends the gradients
# of U straight to the host (the path a bank larger than the VRAM needs; review item 5). The host AdamW then updates the
# whole bank (dense, as torch's AdamW does in phase 2: momentum and weight decay move also the experts with zero
# gradient). Synchronous: no delayed update. One pinned staging buffer per weight name, shared by all layers.
import time
import torch, torch.nn as nn, torch.nn.functional as F
import sc2_model as SM

NAMES = ("Wg", "Wu", "Wd")


def experts(xf, topw, ti, W, h):
    """The bank maths of sc2_model.Bank.forward on the selected experts U (W: Wg, Wu, Wd of U; ti: top-k indices
    remapped into U)."""
    N, D = xf.shape; k = ti.shape[1]; n = W["Wg"].shape[0]
    qx, sx = SM.act_q(xf)
    qg, sg = SM.w_q(W["Wg"]); qu, su = SM.w_q(W["Wu"]); qd, sd = SM.w_q(W["Wd"])
    idx = ti.unsqueeze(-1).expand(N, k, h)
    gi = SM.mm_exact(qx, qg.reshape(n * h, D).T).reshape(N, n, h).gather(1, idx)
    ui = SM.mm_exact(qx, qu.reshape(n * h, D).T).reshape(N, n, h).gather(1, idx)
    g = gi * sx.unsqueeze(-1) * sg.reshape(n, h)[ti]
    u = ui * sx.unsqueeze(-1) * su.reshape(n, h)[ti]
    hh = F.relu(g) * F.relu(u) * topw.unsqueeze(-1)
    qh, sh = SM.act_q(hh)
    act = torch.zeros(N, n, h, device=xf.device, dtype=torch.float32).scatter(1, idx, qh * sh)
    wd = (qd * sd).permute(0, 2, 1).reshape(n * h, D)
    return SM.mm_exact(act.reshape(N, n * h), wd)


class _ExpertsFn(torch.autograd.Function):
    """Forward: fetch U, compute, keep only the small inputs. Backward: fetch U again, recompute with grad, return the
    gradients of x and of the router weights, and send the gradients of the experts to the host."""
    @staticmethod
    def forward(ctx, xf, topw, U, ti, sb):
        W = sb.fetch(U)
        with torch.no_grad():
            y = experts(xf, topw, ti, W, sb.h)
        ctx.sb = sb
        ctx.save_for_backward(xf, topw, U, ti)
        return y

    @staticmethod
    def backward(ctx, gy):
        xf, topw, U, ti = ctx.saved_tensors
        sb = ctx.sb
        W = sb.fetch(U)
        for t in W.values():
            t.requires_grad_(True)
        with torch.enable_grad():
            xg = xf.detach().requires_grad_(True); wg = topw.detach().requires_grad_(True)
            y = experts(xg, wg, ti, W, sb.h)
            gr = torch.autograd.grad(y, [xg, wg] + [W[n] for n in NAMES], gy)
        sb.grads_out(U, dict(zip(NAMES, gr[2:])))
        return gr[0], gr[1], None, None, None


class Stage:
    """One pinned host buffer per weight name, as large as one layer's bank, shared by every layer."""
    def __init__(s, like):
        s.buf = {n: torch.empty_like(t).pin_memory() for n, t in like.items()}


class StreamBank(nn.Module):
    def __init__(s, bank, dev, stage=None):
        super().__init__()
        s.D, s.E, s.h, s.k, s.load_w = bank.D, bank.E, bank.h, bank.k, bank.load_w
        s.router = bank.router                                                # GPU parameter, as in phase 2
        s.dev = dev
        s.host = {n: nn.Parameter(getattr(bank, n).detach().to("cpu", copy=True).contiguous()) for n in NAMES}
        for p in s.host.values():
            p.grad = torch.zeros_like(p)
        s.stage = stage if stage is not None else Stage({n: p.data for n, p in s.host.items()})
        s.norms = []                                                          # per-tensor gradient norms of the step
        s.aux = None; s.stats = {}; s.count_acc = None
        s.timed = False; s.t = dict(fetch=0.0, grad_out=0.0, n_fetch=0); s.n_sel = []

    def fetch(s, U):
        if s.timed:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()                             # the shared buffer is free again
        Uc = U.cpu(); n = Uc.numel()
        out = {}
        for name, p in s.host.items():
            st = s.stage.buf[name][:n]
            torch.index_select(p.data, 0, Uc, out=st)
            out[name] = st.to(s.dev, non_blocking=True)
        if s.timed:
            torch.cuda.synchronize(); s.t["fetch"] += time.perf_counter() - t0
        s.t["n_fetch"] += 1
        return out

    def begin_step(s):
        for p in s.host.values():
            p.grad.zero_()
        s.norms = []; s.n_sel = []

    def grads_out(s, U, G):
        """Gradients of U to the host gradient (index copy: each layer's backward runs once per step). The per-tensor
        norms stay on the GPU for the clip, computed by the kernel the reference's clip uses."""
        if s.timed:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        s.norms = list(torch._foreach_norm([G[n] for n in NAMES], 2.0))
        Uc = U.cpu(); n = Uc.numel()
        torch.cuda.current_stream().synchronize()
        for name in NAMES:
            s.stage.buf[name][:n].copy_(G[name], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        for name in NAMES:
            s.host[name].grad.index_copy_(0, Uc, s.stage.buf[name][:n])
        if s.timed:
            torch.cuda.synchronize(); s.t["grad_out"] += time.perf_counter() - t0

    def forward(s, x):
        B, T, D = x.shape
        cd = x.dtype
        xf = x.reshape(-1, D).float(); N = xf.shape[0]
        E, h, k = s.E, s.h, s.k
        with torch.autocast(xf.device.type, enabled=False):
            probs = torch.softmax(s.router(xf), dim=-1)                       # (N,E) fp32, x NOT quantized
        topv, topi = probs.topk(k, dim=-1)
        topw = topv / topv.sum(-1, keepdim=True) * k
        U = torch.unique(topi)                                                 # sorted
        n = U.numel()
        remap = torch.empty(E, dtype=torch.long, device=x.device)
        remap[U] = torch.arange(n, device=x.device)
        ti = remap[topi]
        if s.training and torch.is_grad_enabled():
            s.n_sel.append(n)
        y = _ExpertsFn.apply(xf, topw, U, ti, s)
        counts = torch.bincount(topi.reshape(-1), minlength=E).float()
        load = E * ((counts / (N * k)) * probs.mean(0)).sum()
        s.aux = s.load_w * load
        with torch.no_grad():
            s.stats = dict(load=load.detach(), used=(counts > 0).float().mean())
            s.count_acc = counts if s.count_acc is None else s.count_acc + counts
        return y.reshape(B, T, D).to(cd)

    def host_params(s):
        return list(s.host.values())


def convert(model, dev):
    """Replace every sc2_model.Bank of a phase-2 Student (built on the host, same seed) by a StreamBank. Returns the
    model with everything but the experts on the GPU."""
    stage = None
    for b in model.blocks:
        if isinstance(b.mlp, SM.Bank):
            b.mlp = StreamBank(b.mlp, dev, stage)
            stage = b.mlp.stage
    model.to(dev)                                                             # host tensors are not registered: stay
    return model


def banks(model):
    return [b.mlp for b in model.blocks if isinstance(b.mlp, StreamBank)]


def gpu_params(model):
    return [p for p in model.parameters()]                                    # the host tensors are not registered


def host_params(model):
    return [p for sb in banks(model) for p in sb.host_params()]


def clip_(model, max_norm):
    """torch.nn.utils.clip_grad_norm_ over the GPU parameters and the bank gradients of the step, same order and same
    formula: a phase-2 Bank yields its own parameters (Wg, Wu, Wd) before its router's, so each bank's norms go right
    before its router's. The experts outside U have zero gradient and add nothing. The GPU gradients are scaled on the
    GPU, the bank's on the host (the same fp32 product)."""
    before = {id(b.mlp.router.weight): b.mlp for b in model.blocks if isinstance(b.mlp, StreamBank)}
    gpu = [p.grad for p in model.parameters() if p.grad is not None]
    it = iter(torch._foreach_norm(gpu, 2.0))
    norms = []
    for p in model.parameters():
        if id(p) in before:
            norms += before[id(p)].norms
        if p.grad is not None:
            norms.append(next(it))
    total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    torch._foreach_mul_(gpu, coef)
    hg = [p.grad for p in host_params(model)]
    if hg:
        torch._foreach_mul_(hg, coef.cpu())
    return total


def full_state_dict(model):
    """The phase-2 Student state dict (loadable by sc2_model.Student strictly): the host experts put back by name."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for i, b in enumerate(model.blocks):
        if isinstance(b.mlp, StreamBank):
            for n, p in b.mlp.host.items():
                sd[f"blocks.{i}.mlp.{n}"] = p.detach().clone()
    return sd
