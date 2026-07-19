#!/usr/bin/env python3
# Phase 64.4 / MVE — the STUDENT: the S0 ladder recipe at pilot scale, with the four curriculum switches.
#   magic: "MVE0" 0x4D564530
#
# One model class, four switches flipped by the curriculum (mve_train.py stages C->D->E->F):
#   C  dense fp MLP (dReLU-gated, fp32 SSM projections = the P61 verdict)          -> build()
#   D  QAT ternary: hot-swap the MLP linears to BitLinear158, KEEPING the weights  -> qat_ternary()
#   E  sparse upcycling: dense MLP -> MoE E32 x h128 top8, experts seeded from the -> upcycle_moe()
#      dense hidden slices; + the recall slot (InfoNCE), gate init 0 = identity    -> RecallSlot
#   F  reverse-KL fine-tune (loss-side only; no model change)
#
# Reused as-is (no forks): ArchA/Block/RMSNorm (phase55), SparseMLP+sparsify_mlp (phase57), BitLinear158 (phase57),
# MoEMLP (phase59). The ONLY new module here is RecallSlot; everything else is composition.
import math, os, sys
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "phase55"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "phase57"))
from phase55_ssm import ArchA                       # noqa: E402
from phase57_sparse import SparseMLP, sparsify_mlp  # noqa: E402
from phase57_ternary import BitLinear158            # noqa: E402
from phase59_moe import MoEMLP                      # noqa: E402


class SparseMoEMLP(MoEMLP):
    """Active-only MoE training: gather the tokens routed to each expert, run that expert on those tokens alone,
    and scatter the weighted results back. Same parameters, same maths, k/E of the work.

    The inherited forward is compute-all: it evaluates every expert on every token and then multiplies the
    non-selected blocks by zero. That costs E/k = 4x the FLOPs and materializes the full (N, E, hid_e) hidden.
    Here nothing that is routed away is ever computed.

    Do NOT read this as the fix for the MVE's memory pressure -- WS2 measured that attribution and it was wrong.
    The SSM scan is the eater (stage C alone peaks at 4.86 GB); the MoE is ~3% of peak, and stage E was merely the
    last drop. This class is a COMPUTE/SCALING investment: 1.5x at E32 but 4.3x at E128, which is what keeps the
    per-step MoE share ~flat as E grows and so keeps the rung-2/3 cost model standing. End-to-end at the rung-1
    config it is worth ~1%.

    DETERMINISM BY CONSTRUCTION -- no atomics anywhere. The obvious scatter-back is `index_add_`, which on CUDA
    accumulates through float atomics: run-to-run non-reproducible, because float addition is not associative and
    the atomic order is not fixed. We rely on bit-identical reruns as a live diagnostic (arms A and C agreeing to
    the last bit is how the apparatus is checked), so that trade is not available. Instead the per-(token,expert)
    rows are produced in expert-sorted order, permuted back with a gather, and summed over a FIXED k axis --
    a plain reduction over a static layout, identical on every run.

    Ordering note: the weighting is applied to the HIDDEN before the down projection, mirroring the compute-all
    path exactly, so the two differ only by float summation order and not by the sequence of operations.
    """
    def forward(s, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D); N = xf.shape[0]
        with torch.autocast(s.dev_type, enabled=False):
            probs = torch.softmax(s.router(xf.float()), dim=-1)          # (N,E)
        topv, topi = probs.topk(s.k, dim=-1)
        topw = topv / topv.sum(dim=-1, keepdim=True)                     # (N,k)

        # group the (token, expert) pairs by expert. stable=True keeps the layout a deterministic function of the
        # routing alone, so two runs with identical routing produce identical arithmetic.
        P = N * s.k
        flat_e = topi.reshape(-1)
        # (flat_t is implicit: pair p belongs to token p // k -- see the structured expand below)
        flat_w = topw.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        sw = flat_w[order]
        counts = torch.bincount(flat_e, minlength=s.E)
        offs = torch.cumsum(counts, 0) - counts

        Wg, Wu = s._tern(s.gate.weight), s._tern(s.up.weight)            # (H,D) ternarized once, then split by expert
        Wd = s._tern(s.Wd)                                               # (E,D,hid_e)

        # PAD-TO-MAX + BATCHED GEMM, not a Python loop over experts. The loop was the first thing measured and it
        # was 20% SLOWER than the compute-all it was meant to replace: E=32 experts x 8 layers = 256 tiny kernel
        # launches per forward, and at this size launch overhead dominates the FLOPs saved. Padding every expert to
        # the largest routed count turns the whole layer into three bmm's. The padded rows are masked to zero and
        # dropped on the way back, so the maths is unchanged -- this is a scheduling fix, not a modelling one.
        C = int(counts.max().item())
        s._last_cap = C; s._last_fill = P / float(s.E * C)                # routing imbalance: 1.0 = perfectly balanced
        pos = torch.arange(P, device=x.device) - offs[flat_e[order]]     # rank of each pair within its expert
        pad_idx = flat_e[order] * C + pos                                # sorted position -> slot in the (E,C) grid

        # The gather from xf must NOT be done with a repeated index. Every token is routed to k experts, so an
        # index_select over token ids has k duplicates per token, and its BACKWARD is an atomic index_add -- which
        # is where determinism was lost the first time this was measured (grads differed by 2.3e-05 between two
        # identical-seed runs, while the forward stayed bit-identical). Instead: expand along a structured k axis
        # first (backward = a fixed-axis sum) and only then apply permutations with UNIQUE indices, whose backward
        # gathers instead of accumulating. Same tensor, deterministic derivative.
        xrep = xf.unsqueeze(1).expand(N, s.k, D).reshape(P, D)
        xs = xrep.index_select(0, order)                                 # order is a permutation -> unique
        xe = xf.new_zeros(s.E * C, D).index_copy(0, pad_idx, xs).reshape(s.E, C, D)
        w_pad = flat_w.new_zeros(s.E * C).index_copy(0, pad_idx, sw)
        Wg3 = Wg.reshape(s.E, s.hid_e, D).transpose(1, 2).to(xe.dtype)   # (E,D,hid_e)
        Wu3 = Wu.reshape(s.E, s.hid_e, D).transpose(1, 2).to(xe.dtype)
        h = F.relu(torch.bmm(xe, Wg3)) * F.relu(torch.bmm(xe, Wu3))      # (E,C,hid_e)
        h = h * w_pad.reshape(s.E, C, 1).to(h.dtype)                     # weight the hidden, as compute-all does
        op = torch.bmm(h, Wd.transpose(1, 2).to(h.dtype))                # (E,C,D)

        cat = op.reshape(s.E * C, D).index_select(0, pad_idx)            # (P,D) back in expert-sorted order
        inv = torch.empty_like(order); inv[order] = torch.arange(P, device=x.device)
        out = cat.index_select(0, inv).reshape(N, s.k, D).sum(1)         # fixed-axis reduction, no atomics

        sel = torch.zeros_like(probs).scatter(-1, topi, torch.ones_like(topw))
        load = s.E * (sel.mean(0) * probs.mean(0)).sum() / s.k
        aux = s.load_w * load
        coh = xf.new_zeros(())
        if s.lam_coh > 0 and T > 1:
            pr = probs.reshape(B, T, s.E); coh = (pr[:, 1:] - pr[:, :-1]).abs().mean(); aux = aux + s.lam_coh * coh
        s._last_load = float(load.detach()); s._last_coh = float(coh.detach())
        if s._cap:
            s._topi = topi.reshape(B, T, s.k).detach().cpu().numpy()
            s._xin = x.detach().float().cpu().numpy()
        return out.reshape(B, T, D), aux


class BitLinear158Alpha(BitLinear158):
    """W_eff = (1-alpha)*W + alpha*ternary(W) -- the epsilon-identity form of the stage-D switch.

    The curriculum has already paid twice for the same missing principle: the MoE upcycle emitted 1/k of the dense
    output until it was magnitude-matched (+0.49 BPB), and the recall slot enters behind a zero gate precisely so
    that insertion is an exact identity. Stage D was the last switch still teleporting: `qat_ternary()` replaced
    W.x with ternary(W).x in a single step, and the MVE measured the price -- a +0.32 BPB shock, identical in all
    three arms, i.e. a property of the ternarization and not of the arm.

    alpha=0 is an exact fp32 identity at the switch; alpha=1 is bit-for-bit BitLinear158, so the END STATE is
    unchanged and only the PATH becomes continuous. STE is untouched (backward is identity either way).

    Honest cost: while 0<alpha<1 the weights are not yet ternary, so the QAT regularization pressure ramps in
    late; keep N_alpha small relative to stage D or the stage effectively shortens. Applying this is a DECLARED
    deviation at a rung boundary -- stage D can no longer be described as 'pure QAT throughout'.
    """
    def __init__(s, fin, fout, per_row=True):
        super().__init__(fin, fout, per_row=per_row)
        s.alpha = 0.0

    def forward(s, x):
        if s.alpha >= 1.0: return super().forward(x)                 # exactly BitLinear158 at the end state
        w = s.weight
        scale = (w.abs().mean(dim=1, keepdim=True) if s.per_row else w.abs().mean()).clamp_min(1e-5)
        wq = (w / scale).round().clamp(-1, 1)
        w_ste = w + (s.alpha * (wq * scale - w)).detach()
        return F.linear(x, w_ste)

# S0 ladder recipe (SCALEUP_ARCHITECTURE §9 frozen config, pilot rung). MoE active hidden = k*hid_e = 8*128 = 1024
# = the dense hidden (4*D) -> the upcycle is active-param-matched by construction (probe-4 discipline).
S0 = dict(D=256, N=96, H=8, L=8, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16)
MOE = dict(E=32, hid_e=128, k=8)


class RecallSlot(nn.Module):
    """The 'knowing' tier at training time: content-addressable read over the sequence's own past.

    Train-time = the EXACT form of what the C engine does approximately (64.3: Hadamard-IVF + 4-bit ADC shortlist +
    exact top-16 rerank). Here the read is exact top-16 over all strictly-past positions -> the C probe is the
    inference-side approximation of THIS operator, not a different one.

    Design commitments (declared, first read at the MVE):
      - dim 128 keys/queries/values (= the 64.3 probe's D=128 and its 512 B/entry value store), L2-normalized k/q.
      - top-16 read, softmax over the 16 scores, value-weighted sum -> out_proj -> residual add through a LEARNED
        SCALAR GATE INITIALIZED TO ZERO. At insertion the slot is a mathematical no-op: the curriculum cannot be
        destabilized *by the insertion itself*, only by what the InfoNCE pressure subsequently learns. That is
        exactly the variable D4-clause-2 wants to read.
      - InfoNCE positives are MINED, not labelled (TinyStories has no MQAR key/value ground truth): the positive for
        query t is the most recent past position p<t that repeats the current bigram (same current token AND same
        next token). That is the associative-recall structure the tier exists to serve. Negatives = all other past
        positions. tau=0.07. This mining rule is a declared choice and is what the MVE puts on trial.
    """
    def __init__(s, D, dim=128, topk=16, tau=0.07):
        super().__init__()
        s.dim, s.topk, s.tau = dim, topk, tau
        s.q = nn.Linear(D, dim, bias=False)
        s.k = nn.Linear(D, dim, bias=False)
        s.v = nn.Linear(D, dim, bias=False)
        s.o = nn.Linear(dim, D, bias=False)
        s.gate = nn.Parameter(torch.zeros(1))               # ZERO init -> identity at insertion
        s._last_nce = 0.0

    def forward(s, x, y=None, idx=None, want_nce=False):
        B, T, _ = x.shape
        with torch.autocast(x.device.type, enabled=False):
            xf = x.float()
            q = F.normalize(s.q(xf), dim=-1)
            k = F.normalize(s.k(xf), dim=-1)
            v = s.v(xf)
            sc = q @ k.transpose(1, 2)                                        # (B,T,T)
            ar = torch.arange(T, device=x.device)
            past = ar[None, :] < ar[:, None]                                  # strictly past (no self-read)
            sc = sc.masked_fill(~past[None], float("-inf"))
            kk = min(s.topk, T)
            tv, ti = sc.topk(kk, dim=-1)                                      # (B,T,kk)
            w = torch.softmax(tv / s.tau, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)                                  # t=0 has no past -> all -inf -> 0
            bi = torch.arange(B, device=x.device)[:, None, None]
            got = torch.einsum('btk,btkd->btd', w, v[bi, ti])                 # (B,T,kk,dim) advanced-index read
            out = s.o(got) * s.gate

            nce = xf.new_zeros(())
            if want_nce and y is not None and idx is not None:
                # mine positives: most recent p<t with (idx[p],y[p]) == (idx[t],y[t])
                same = (idx[:, None, :] == idx[:, :, None]) & (y[:, None, :] == y[:, :, None]) & past[None]
                has = same.any(-1)                                            # (B,T) query has a positive
                pos = torch.where(has, same.float().cumsum(-1).argmax(-1), torch.zeros_like(has, dtype=torch.long))
                if has.any():
                    logits = sc.masked_fill(~past[None], float("-inf")) / s.tau
                    ce = F.cross_entropy(logits[has], pos[has], reduction="mean")
                    nce = ce
            s._last_nce = float(nce.detach())
        return out.to(x.dtype), nce


class MVEStudent(ArchA):
    """ArchA + the recall slot, with the curriculum switches. forward -> (logits, aux) where aux = MoE load + lam*NCE."""
    def __init__(s, V, recall_layer=-1, **cfg):
        super().__init__(V, **cfg)
        D = cfg["D"]
        s.V, s.D, s.hid = V, D, cfg["mlp_mult"] * D
        sparsify_mlp(s, D, s.hid, act="drelu", gated=True, topk=0.0, ternary=False)   # stage C: fp MLP, dReLU-gated
        s.recall = None
        s.recall_layer = recall_layer if recall_layer >= 0 else cfg["L"] - 2
        s.lam_nce = 0.0
        s.is_ternary = False
        s.is_moe = False

    # ---- stage D: KD-first -> QAT. Hot-swap fp linears to ternary STE, CARRYING the trained weights over.
    def qat_ternary(s, alpha_sched=False):
        n = 0
        cls = BitLinear158Alpha if alpha_sched else BitLinear158
        for blk in s.blocks:
            if not getattr(blk, "use_mlp", False): continue
            for name in ("gate", "up", "down"):
                lin = getattr(blk.mlp, name, None)
                if lin is None or isinstance(lin, BitLinear158): continue
                bl = cls(lin.in_features, lin.out_features)
                with torch.no_grad(): bl.weight.copy_(lin.weight)
                setattr(blk.mlp, name, bl); n += 1
        s.is_ternary = True
        return n

    def set_qat_alpha(s, a):
        """Drive the stage-D interpolation. Returns how many modules were touched (0 if not alpha-scheduled)."""
        n = 0
        for m in s.modules():
            if isinstance(m, BitLinear158Alpha): m.alpha = float(a); n += 1
        return n

    # ---- stage E1: sparse upcycling dense -> MoE. Each expert is seeded from a 128-wide slice of the trained dense
    # hidden (1024/128 = 8 distinct slices, each replicated 4x across the 32 experts) + small noise to break the
    # replica symmetry (otherwise the router sees 4 identical experts and cannot separate them).
    def upcycle_moe(s, dev_type="cuda", load_w=0.01, lam_coh=0.0, noise=0.02, seed=0, sparse=False):
        g = torch.Generator(device="cpu").manual_seed(seed)
        E, hid_e, k = MOE["E"], MOE["hid_e"], MOE["k"]
        cls = SparseMoEMLP if sparse else MoEMLP
        n = 0
        for blk in s.blocks:
            if not getattr(blk, "use_mlp", False): continue
            old = blk.mlp
            moe = cls(s.D, hid_e, E, k, load_w, lam_coh, dev_type)
            with torch.no_grad():
                nslice = s.hid // hid_e                                  # 8
                for e in range(E):
                    sl = slice((e % nslice) * hid_e, (e % nslice + 1) * hid_e)
                    for src, dst in ((old.gate.weight, moe.gate.weight), (old.up.weight, moe.up.weight)):
                        w = src[sl].clone()
                        w += noise * w.std() * torch.randn(w.shape, generator=g).to(w.device)
                        dst[e * hid_e:(e + 1) * hid_e].copy_(w)
                    # MAGNITUDE MATCH (the upcycle's one real trap): the dense MLP sums ALL 1024 hidden units into
                    # down. The MoE weights each selected expert by topw_e, and the k top-k weights sum to 1 -> the
                    # seeded MoE would emit ~1/k of the dense output and the curriculum would take a gratuitous hit
                    # at the switch. Scale the seeded down by k so E[sum_e topw_e * down_e] matches the dense.
                    wd = old.down.weight[:, sl].clone() * k              # (D, hid_e)
                    wd += noise * wd.std() * torch.randn(wd.shape, generator=g).to(wd.device)
                    moe.Wd[e].copy_(wd)
            blk.mlp = moe; n += 1
        s.is_moe = True
        return n

    # ---- stage E2: the recall tier
    def add_recall(s, dim=128, topk=16, lam_nce=0.0):
        s.recall = RecallSlot(s.D, dim=dim, topk=topk)
        s.lam_nce = lam_nce
        return sum(p.numel() for p in s.recall.parameters())

    def forward(s, idx, y=None):
        import torch.utils.checkpoint as ckpt
        x = s.emb(idx)
        aux = x.new_zeros(())
        for i, b in enumerate(s.blocks):
            if s.training and s.use_ckpt:
                out = ckpt.checkpoint(b, x, use_reentrant=False)
            else:
                out = b(x)
            x, a = out if isinstance(out, tuple) else (out, None)
            if a is not None: aux = aux + a
            if s.recall is not None and i == s.recall_layer:
                r, nce = s.recall(x, y=y, idx=idx, want_nce=(s.lam_nce > 0 and s.training))
                x = x + r
                aux = aux + s.lam_nce * nce
        return s.head(s.norm_f(x)), aux


def _block_forward_with_aux(blk, x):
    """Block.forward returns a tensor; MoEMLP returns (out, aux). Patch Block to propagate the aux."""
    x = x + blk.mix(blk.norm(x))
    if blk.use_mlp:
        o = blk.mlp(blk.norm2(x))
        if isinstance(o, tuple):
            m, aux = o
            return x + m, aux
        return x + o, None
    return x, None


# Block.forward must propagate the MoE aux once experts exist -> patch it once, for every Block instance.
from phase55_ssm import Block  # noqa: E402
Block.forward = _block_forward_with_aux


def param_report(model):
    tot = sum(p.numel() for p in model.parameters())
    mlp = sum(p.numel() for b in model.blocks if getattr(b, "use_mlp", False) for p in b.mlp.parameters())
    rec = sum(p.numel() for p in model.recall.parameters()) if model.recall is not None else 0
    act = tot
    if model.is_moe:   # active params: only k of E experts fire per token
        exp = mlp - sum(p.numel() for b in model.blocks if getattr(b, "use_mlp", False) for p in b.mlp.router.parameters())
        act = tot - exp + exp * MOE["k"] / MOE["E"]
    return dict(total=tot, mlp=mlp, recall=rec, active=int(act))
