"""S2N's core: a Ling-shaped MoE stack trained end to end on its scales (M2), codes frozen, layers streamed from host RAM to the GPU one at a time.  The same
code runs cell F (random codes at flash's shapes) and the training on mini (ks_stk4's codes), so F measures the code that trains.

A frozen matrix is either bf16 values (the organs on mini, ks_stk4's BF16 container) or codes:
    p  uint8 [.., R, I/4], four 2-bit codes c per byte along the input; value = (2c - 3) * d, d fp16 [.., R, I/256].  This is bal's form: Q4_0 with
       nibbles 8 +- 1, 3 and d equal over the 8 blocks of a 256-group (the loader checks both).
Codes are unpacked inside autograd functions and again in their backward: no unpacked matrix is kept between forward and backward.  Routed experts run
in chunks of experts as batched matmuls, tokens padded to the chunk's largest count.
Trainable scales of a matrix (kind):
    row   m [R]: y = (x W0^T) * m, the output feature scaled;
    grp   g [R, I/256]: W = W0 * g over each 256-group (e2e_grp: the experts' own per-256 scales);
    glob  one scalar shared by every matrix of a class (the trivial arm);
    none  frozen.
Scales are applied in fp32 and the product rounded once.
Loss: KL of the teacher's top 32 plus its rest mass against the student's full log-softmax in fp32, differentiated chunk by chunk; padding out.
Sequences may pack several documents (Seg): RoPE positions restart per document, attention stays inside it (block-diagonal causal mask).
Form.prefill routes a document's first positions at k = 8 (the panels' prefill) and the rest at the form's k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

CHUNK_TOKENS = 512


@dataclass(frozen=True)
class Shape:
    layers: int
    d: int
    heads: int
    heads_kv: int
    head_dim: int
    rot: int
    dense_f: int
    experts: int
    expert_f: int
    shared_f: int
    vocab: int
    groups: int = 8
    groups_kept: int = 4
    k_native: int = 8
    route_scale: float = 2.5
    eps: float = 1e-6
    theta: float = 600000.0
    expert_chunk: int = 32


FLASH = Shape(32, 4096, 32, 4, 128, 64, 9216, 256, 1024, 1024, 157184, expert_chunk=16)
MINI = Shape(20, 2048, 16, 4, 128, 64, 5120, 256, 512, 512, 157184, expert_chunk=64)


@dataclass(frozen=True)
class Form:
    """Routing: top k with mass (s1t_model's).  prefill > 0: a document's first prefill positions route at k = 8, as the panels prefill their prompt
    at k = 8 and decode at k = 4."""
    k: int = 8
    mass: int = 0
    prefill: int = 0


@dataclass
class Seg:
    """Packed sequences: pos [B, T] position inside its document (RoPE and prefill), doc [B, T] document id, -1 for padding.  Attention stays
    inside a document (block-diagonal causal mask)."""
    pos: torch.Tensor
    doc: torch.Tensor


def pack_docs(docs: list[tuple], t: int) -> list[tuple[list[int], list[int]]]:
    """Greedy packing of documents (in the given order) into sequences of at most t positions; a document is never split.
    Returns, per sequence, the indices of its documents and their lengths."""
    out, cur, used = [], ([], []), 0
    for i, (toks, rec) in enumerate(docs):
        n = len(rec)
        if n > t:
            raise ValueError(f"document of {n} positions over {t}")
        if used + n > t:
            out.append(cur)
            cur, used = ([], []), 0
        cur[0].append(i)
        cur[1].append(n)
        used += n
    if cur[0]:
        out.append(cur)
    return out


def pack(codes: torch.Tensor) -> torch.Tensor:
    """codes uint8 [..., I] in 0..3 -> [..., I/4]."""
    c = codes.reshape(*codes.shape[:-1], -1, 4).to(torch.uint8)
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


_LUTS: dict = {}


def _lut(dev) -> torch.Tensor:
    """[256, 4] bf16: byte b -> 2c_j - 3 for its four 2-bit codes c_j (low bits first), exact in bf16."""
    if dev not in _LUTS:
        b = torch.arange(256)
        _LUTS[dev] = torch.stack([((b >> (2 * j)) & 3) * 2 - 3 for j in range(4)], -1).to(torch.bfloat16).to(dev)
    return _LUTS[dev]


def unpack(p: torch.Tensor, d: torch.Tensor, exi: torch.Tensor | None = None, exv: torch.Tensor | None = None) -> torch.Tensor:
    """p uint8 [..., R, I/4], d fp16 [..., R, I/256] -> bf16 [..., R, I], in place in one bf16 buffer.  2c - 3 is exact in bf16; d rounded to bf16
    (<= 0.4%) and the product rounded once: below the Q8 activations llama.cpp's panels run with.  Exceptions (blocks of 32 not in the codes' form):
    exi int64 [n, nd] block index (expert, row, block / row, block), exv bf16 [n, 32] their exact values, written over."""
    out = torch.nn.functional.embedding(p.int(), _lut(p.device))     # one gather: byte -> its four values 2c - 3, [..., I/4, 4]
    out.view(*p.shape[:-1], -1, 256).mul_(d.to(torch.bfloat16).unsqueeze(-1))
    w = out.view(*p.shape[:-1], -1)
    if exi is not None and exi.shape[0]:
        w.view(*p.shape[:-1], -1, 32)[tuple(exi.t())] = exv
    return w


def _grp(w0: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """W0 x g per 256-group, the product in fp32 and rounded once to W0's dtype (as the export writes d x g)."""
    i = w0.shape[-1]
    return (w0.float().view(*w0.shape[:-1], i // 256, 256) * s.float().unsqueeze(-1)).view(w0.shape).to(w0.dtype)


def _scale(y: torch.Tensor, s: torch.Tensor, kind: str) -> torch.Tensor:
    """y x m in fp32 (m per output feature for row, a scalar for glob), rounded once to y's dtype."""
    return (y.float() * (s.float().unsqueeze(-2) if kind == "row" else s.float())).to(y.dtype)


class Packed(torch.autograd.Function):
    """y = x @ W^T, W from codes, scaled by kind.  x [..., N, I] with W [..., R, I] (batched over experts when 3-D).  Saves codes, never W."""

    @staticmethod
    def forward(ctx, x, p, d, s, kind, exi, exv):
        w0 = unpack(p, d, exi, exv).to(x.dtype)
        ctx.kind = kind
        if kind == "grp":
            ctx.save_for_backward(x, p, d, s, exi, exv)
            return x @ _grp(w0, s).transpose(-1, -2)
        y0 = x @ w0.transpose(-1, -2)
        if kind == "none":
            ctx.save_for_backward(p, d, exi, exv)
            return y0
        ctx.save_for_backward(p, d, s, y0, exi, exv)
        return _scale(y0, s, kind)

    @staticmethod
    def backward(ctx, gy):
        kind = ctx.kind
        if kind == "grp":
            x, p, d, s, exi, exv = ctx.saved_tensors
            w0 = unpack(p, d, exi, exv).to(gy.dtype)
            gx = gy @ _grp(w0, s)
            gw = (gy.transpose(-1, -2) @ x).float()
            i = w0.shape[-1]
            gs = (gw.view(*gw.shape[:-1], i // 256, 256) * w0.float().view(*w0.shape[:-1], i // 256, 256)).sum(-1)
            return gx, None, None, gs, None, None, None
        if kind == "none":
            p, d, exi, exv = ctx.saved_tensors
            return gy @ unpack(p, d, exi, exv).to(gy.dtype), None, None, None, None, None, None
        p, d, s, y0, exi, exv = ctx.saved_tensors
        gx = _scale(gy, s, kind) @ unpack(p, d, exi, exv).to(gy.dtype)
        g = gy.float() * y0.float()
        gs = g.sum(-2) if kind == "row" else g.sum().reshape(s.shape)
        return gx, None, None, gs, None, None, None


class Mat:
    """One frozen matrix's trainable scales: kind 'row' | 'grp' | 'glob' | 'none'.  For stacked experts, p carries the expert axis first."""

    def __init__(self, kind: str, shape: tuple = (), dev=None, glob: torch.Tensor | None = None):
        self.kind, self.flat, self._glob = kind, None, None
        if kind in ("row", "grp"):
            # held 1-D: Adafactor keeps one second moment per multiplier.  A 2-D or 3-D parameter gets a factored (row x column) estimate,
            # and a rarely routed expert's row then takes steps scaled by its small row average (S2N's dry run: up to 3.09 and sign flips).
            self.shape = tuple(shape[:-1]) if kind == "row" else (*shape[:-1], shape[-1] // 256)
            self.flat = torch.ones(math.prod(self.shape), device=dev, requires_grad=True)
            # one scalar per tensor over the multipliers: the effective scale is t x m.  A per-element normalized step moves a row by about lr
            # times its own gradient's signal-to-noise ratio, so a tensor-wide direction is slow through rows; t moves it at about lr per step.
            self.t = torch.ones(1, device=dev, requires_grad=True)
        elif kind == "glob":
            self._glob = glob

    @property
    def p(self) -> torch.Tensor | None:
        """The effective scales (t x m for row and grp), as the export writes them."""
        if self.flat is not None:
            return (self.t * self.flat).view(self.shape)
        return self._glob

    def params(self) -> list[torch.Tensor]:
        return [self.flat, self.t] if self.kind in ("row", "grp") else []

    def scale(self, sl=None) -> torch.Tensor | None:
        p = self.p
        if p is None:
            return None
        return p[sl] if (sl is not None and self.kind in ("row", "grp")) else p

    def apply(self, x: torch.Tensor, src, sl=None) -> torch.Tensor:
        """src: bf16 tensor, or (p, d[, exi, exv]); sl: an expert slice for stacked tensors."""
        s = self.scale(sl)
        if isinstance(src, tuple):
            p, d = src[0], src[1]
            exi, exv = (src[2], src[3]) if len(src) == 4 else (None, None)
            if sl is not None:
                p, d = p[sl], d[sl]
                if exi is not None:
                    m = (exi[:, 0] >= sl.start) & (exi[:, 0] < sl.stop)
                    exi, exv = exi[m] - torch.tensor([sl.start, 0, 0], device=exi.device), exv[m]
            if s is None:
                s = torch.ones((), device=x.device)
            if x.dim() == 2 or p.dim() == 2:
                shp = x.shape
                y = Packed.apply(x.reshape(-1, shp[-1]), p, d, s, self.kind, exi, exv)
                return y.view(*shp[:-1], -1)
            return Packed.apply(x, p, d, s, self.kind, exi, exv)
        w = src[sl] if sl is not None else src
        if self.kind == "grp":
            w = _grp(w, s)
        y = x @ w.transpose(-1, -2)
        if self.kind in ("row", "glob"):
            y = _scale(y, s, self.kind)
        return y


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rope(n: int, sh: Shape, dev) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (sh.theta ** (torch.arange(0, sh.rot, 2, device=dev, dtype=torch.float32) / sh.rot))
    f = torch.outer(torch.arange(n, device=dev, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos(), e.sin()


def route(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, sh: Shape, form: Form) -> tuple[torch.Tensor, torch.Tensor]:
    """s1t_model.route at any shape.  x [N, D] -> ids [N, k], weights fp32 [N, k], scaled."""
    n = x.shape[0]
    p = torch.sigmoid(x.float() @ gate.float().t())
    b = p + bias.float()
    gs = b.view(n, sh.groups, -1).topk(2, dim=-1)[0].sum(-1)
    gi = gs.topk(sh.groups_kept, dim=-1)[1]
    gm = torch.zeros(n, sh.groups, device=x.device).scatter_(1, gi, 1).repeat_interleave(sh.experts // sh.groups, dim=1)
    sel = b.masked_fill(gm == 0, float("-inf")).topk(sh.k_native, dim=-1)[1]
    s8 = p.gather(1, sel).sum(-1, keepdim=True)
    if form.k == sh.k_native and form.mass == 0:
        return sel, p.gather(1, sel) / s8 * sh.route_scale
    keep = sel[:, :form.k]
    pk = p.gather(1, keep)
    if form.mass == 0:
        den = pk.sum(-1, keepdim=True)
    elif form.mass == sh.k_native:
        den = s8
    else:
        den = s8 + p.scatter(1, sel, float("-inf")).topk(form.mass - sh.k_native, dim=-1)[0].sum(-1, keepdim=True)
    return keep, pk / den * sh.route_scale


@dataclass
class Layer:
    """A layer's frozen tensors on the host (pinned; a matrix is a bf16 tensor or a (codes, d) pair) and its trainable scales on the GPU."""
    i: int
    host: dict = field(default_factory=dict)
    mats: dict = field(default_factory=dict)

    def params(self) -> list[torch.Tensor]:
        return [q for m in self.mats.values() for q in m.params()]


def to_gpu(host: dict, dev) -> dict:
    def mv(v):
        if isinstance(v, tuple):
            return tuple(t.to(dev, non_blocking=True) for t in v)
        return v.to(dev, non_blocking=True)
    return {k: mv(v) for k, v in host.items()}


def experts(flat: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor, w: dict, M: dict, sh: Shape) -> torch.Tensor:
    """Sum over slots of weight * expert(x), experts in chunks as batched matmuls.  Returns fp32 [N, D]."""
    n, kk = ids.shape
    flat_ids = ids.reshape(-1)
    order = flat_ids.argsort()
    counts = torch.bincount(flat_ids, minlength=sh.experts)
    starts = torch.cumsum(counts, 0) - counts
    tok = torch.cat([order // kk, order.new_full((1,), n)])          # sorted slot -> token; the last entry is the padding row
    wsort = torch.cat([wts.reshape(-1)[order], wts.new_zeros(1)])
    xp = torch.cat([flat, flat.new_zeros(1, flat.shape[1])])
    last = order.shape[0]
    cl = counts.tolist()
    y = torch.zeros(n + 1, flat.shape[1], device=flat.device)
    for e0 in range(0, sh.experts, sh.expert_chunk):
        e1 = min(e0 + sh.expert_chunk, sh.experts)
        cmax = max(cl[e0:e1])
        if cmax == 0:
            continue
        ar = torch.arange(cmax, device=flat.device)
        slot = torch.where(ar[None, :] < counts[e0:e1, None], starts[e0:e1, None] + ar[None, :], last)
        t = tok[slot]
        xe = xp[t]
        sl = slice(e0, e1)
        g = M["ffn_gate_exps"].apply(xe, w["ffn_gate_exps"], sl)
        u = M["ffn_up_exps"].apply(xe, w["ffn_up_exps"], sl)
        o = M["ffn_down_exps"].apply(F.silu(g) * u, w["ffn_down_exps"], sl)
        o = o * wsort[slot].to(o.dtype).unsqueeze(-1)
        y = y.index_add(0, t.reshape(-1), o.reshape(-1, o.shape[-1]).float())
    return y[:n]


def context(tokens_shape: tuple, sh: Shape, dev, dtype, seg: Seg | None = None) -> dict:
    """Per-position RoPE tables [B, T, 1, rot], the attention mask (None: one causal document per row) and flat positions."""
    b, t = tokens_shape
    pos = (torch.arange(t, device=dev).expand(b, t) if seg is None else seg.pos.to(dev))
    cos, sin = rope(t, sh, dev)
    mask = None
    if seg is not None:
        doc = seg.doc.to(dev)
        causal = torch.ones(t, t, dtype=torch.bool, device=dev).tril()
        mask = ((doc[:, :, None] == doc[:, None, :]) & causal)[:, None]
    return {"cos": cos.to(dtype)[pos][:, :, None, :], "sin": sin.to(dtype)[pos][:, :, None, :], "mask": mask, "pos": pos.reshape(-1)}


def moe(flat: torch.Tensor, w: dict, M: dict, sh: Shape, form: Form, pos: torch.Tensor) -> torch.Tensor:
    """Routed experts, fp32 [N, D].  With form.prefill, positions under it route at k = 8 (mass 0) and the rest at form."""
    if form.prefill <= 0 or (form.k == sh.k_native and form.mass == 0):
        ids, wts = route(flat, w["ffn_gate_inp"], w["exp_probs_b"], sh, form)
        return experts(flat, ids, wts, w, M, sh)
    pre = pos < form.prefill
    y = torch.zeros(flat.shape[0], flat.shape[1], device=flat.device)
    for sel, f in ((pre, Form(sh.k_native, 0)), (~pre, Form(form.k, form.mass))):
        idx = sel.nonzero().squeeze(1)
        if idx.numel():
            xs = flat.index_select(0, idx)
            ids, wts = route(xs, w["ffn_gate_inp"], w["exp_probs_b"], sh, f)
            y = y.index_copy(0, idx, experts(xs, ids, wts, w, M, sh))
    return y


def layer_forward(L: Layer, w: dict, h: torch.Tensor, cs: dict, sh: Shape, form: Form) -> torch.Tensor:
    """h [B, T, D] bf16 -> [B, T, D]; cs from context()."""
    b, t, _ = h.shape
    M = L.mats
    x = rms(h, w["attn_norm"], sh.eps)
    qkv = M["attn_qkv"].apply(x, w["attn_qkv"]).view(b, t, sh.heads + 2 * sh.heads_kv, sh.head_dim)
    q, k, v = qkv.split([sh.heads, sh.heads_kv, sh.heads_kv], dim=2)
    q, k = rms(q, w["attn_q_norm"], sh.eps), rms(k, w["attn_k_norm"], sh.eps)
    cos, sin = cs["cos"], cs["sin"]

    def rot(z):
        zr, zp = z[..., :sh.rot], z[..., sh.rot:]
        z1, z2 = zr[..., :sh.rot // 2], zr[..., sh.rot // 2:]
        return torch.cat((zr * cos + torch.cat((-z2, z1), -1) * sin, zp), -1)

    rep = sh.heads // sh.heads_kv     # K and V expanded to every head: with enable_gqa this build falls back to the math kernel, which holds T x T
    q, k, v = rot(q).transpose(1, 2), rot(k).transpose(1, 2).repeat_interleave(rep, 1), v.transpose(1, 2).repeat_interleave(rep, 1)
    if cs["mask"] is None:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=cs["mask"])
    o = o.transpose(1, 2).reshape(b, t, sh.heads * sh.head_dim)
    a = h + M["attn_output"].apply(o, w["attn_output"])
    u = rms(a, w["ffn_norm"], sh.eps)
    if L.i == 0:
        g = M["ffn_gate"].apply(u, w["ffn_gate"])
        up = M["ffn_up"].apply(u, w["ffn_up"])
        return a + M["ffn_down"].apply(F.silu(g) * up, w["ffn_down"])
    flat = u.reshape(-1, sh.d)
    sg = M["ffn_gate_shexp"].apply(flat, w["ffn_gate_shexp"])
    su = M["ffn_up_shexp"].apply(flat, w["ffn_up_shexp"])
    y = M["ffn_down_shexp"].apply(F.silu(sg) * su, w["ffn_down_shexp"]).float()
    y = y + moe(flat, w, M, sh, form, cs["pos"])
    return a + y.to(h.dtype).view(b, t, sh.d)


def logits_kl(hid: torch.Tensor, head: torch.Tensor, norm: torch.Tensor, t_ids: torch.Tensor, t_lp: torch.Tensor, eps: float, grad: bool,
              valid: torch.Tensor | None = None):
    """hid [N, D] -> (sum of truncated KL, top-1 agreements with the teacher's best, d(mean KL)/d hid or None, positions counted).
    valid [N] bool: padding positions out of the loss, the count and the gradient."""
    n = hid.shape[0]
    if valid is None:
        valid = torch.ones(n, dtype=torch.bool, device=hid.device)
    nv = max(int(valid.sum()), 1)
    total, agree, grads = torch.zeros((), device=hid.device), torch.zeros((), device=hid.device), []
    for s in range(0, n, CHUNK_TOKENS):
        vm = valid[s:s + CHUNK_TOKENS]
        with torch.set_grad_enabled(grad):
            hc = hid[s:s + CHUNK_TOKENS].detach().requires_grad_(grad)
            lq = F.log_softmax((rms(hc, norm, eps) @ head.t()).float(), -1)
            lt = t_lp[s:s + CHUNK_TOKENS]
            lq_t = lq.gather(1, t_ids[s:s + CHUNK_TOKENS])
            p = lt.exp()
            r_t = (1 - p.sum(-1)).clamp_min(1e-8)
            r_s = (1 - lq_t.exp().sum(-1)).clamp_min(1e-8)
            loss = (((p * (lt - lq_t)).sum(-1) + r_t * (r_t.log() - r_s.log())) * vm).sum()
        if grad:
            (loss / nv).backward()
            grads.append(hc.grad)
        total += loss.detach()
        agree += ((lq.detach().argmax(-1) == t_ids[s:s + CHUNK_TOKENS, 0]) & vm).sum()
        del lq
    return total, agree, (torch.cat(grads) if grad else None), nv


def forward_hidden(layers: list[Layer], embed: torch.Tensor, tokens: torch.Tensor, sh: Shape, form: Form, dev, keep: bool, seg: Seg | None = None):
    b, t = tokens.shape
    h = embed[tokens.reshape(-1)].view(b, t, sh.d).to(dev, non_blocking=True)
    cs = context((b, t), sh, dev, h.dtype, seg)
    saved = []
    with torch.no_grad():
        for L in layers:
            if keep:
                saved.append(h)
            w = to_gpu(L.host, dev)
            h = layer_forward(L, w, h, cs, sh, form)
            del w
    return h, saved, cs


def step(layers: list[Layer], embed: torch.Tensor, head: torch.Tensor, out_norm: torch.Tensor, tokens: torch.Tensor, t_ids: torch.Tensor,
         t_lp: torch.Tensor, sh: Shape, form: Form, dev, train: bool = True, weight: float = 1.0, seg: Seg | None = None) -> dict:
    """One micro-step on tokens [B, T] (host): forward layer by layer from the host, each layer's input kept on the GPU; the loss in chunks; backward
    layer by layer, each layer loaded and run again.  Gradients of weight x (mean KL over these positions) accumulate into the scales; weight = this
    micro-step's share of the step's positions, so a step's gradient is that of its mean.  t_ids, t_lp: [B*T, 32] on the GPU.  seg: packed
    documents (padding, doc = -1, out of the loss)."""
    b, t = tokens.shape
    h, saved, cs = forward_hidden(layers, embed, tokens, sh, form, dev, train, seg)
    valid = None if seg is None else (seg.doc.reshape(-1) >= 0).to(dev)
    tot, agr, g, nv = logits_kl(h.reshape(-1, sh.d), head, out_norm, t_ids, t_lp, sh.eps, train, valid)
    if train:
        g = (g * weight).view(b, t, sh.d).to(h.dtype)
        for L, h_in in zip(reversed(layers), reversed(saved)):
            w = to_gpu(L.host, dev)
            x = h_in.detach().requires_grad_(True)
            with torch.enable_grad():
                y = layer_forward(L, w, x, cs, sh, form)
            y.backward(g)
            g = x.grad
            del w, y, x
    return {"kl": float(tot), "agree": int(agr), "n": nv}
