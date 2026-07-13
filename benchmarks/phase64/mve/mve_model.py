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
    def qat_ternary(s):
        n = 0
        for blk in s.blocks:
            if not getattr(blk, "use_mlp", False): continue
            for name in ("gate", "up", "down"):
                lin = getattr(blk.mlp, name, None)
                if lin is None or isinstance(lin, BitLinear158): continue
                bl = BitLinear158(lin.in_features, lin.out_features)
                with torch.no_grad(): bl.weight.copy_(lin.weight)
                setattr(blk.mlp, name, bl); n += 1
        s.is_ternary = True
        return n

    # ---- stage E1: sparse upcycling dense -> MoE. Each expert is seeded from a 128-wide slice of the trained dense
    # hidden (1024/128 = 8 distinct slices, each replicated 4x across the 32 experts) + small noise to break the
    # replica symmetry (otherwise the router sees 4 identical experts and cannot separate them).
    def upcycle_moe(s, dev_type="cuda", load_w=0.01, lam_coh=0.0, noise=0.02, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        E, hid_e, k = MOE["E"], MOE["hid_e"], MOE["k"]
        n = 0
        for blk in s.blocks:
            if not getattr(blk, "use_mlp", False): continue
            old = blk.mlp
            moe = MoEMLP(s.D, hid_e, E, k, load_w, lam_coh, dev_type)
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
