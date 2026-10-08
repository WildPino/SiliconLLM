"""Compact trainable Falcon transfer target; native C parity is a later gate.

F32 control organs, selected ternary SwiGLU banks and AQ63 integer-dot forward.
The source SSD implementation supplies a differentiable recurrent training path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import FalconH1Config
from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1Mixer, FalconH1RMSNorm

D, L, E, K, H, V = 512, 12, 72, 8, 128, 65537
SWA_SITES = (5, 11)


def target_config(source):
    c = FalconH1Config.from_dict(source)
    c.hidden_size, c.num_hidden_layers = D, L
    c.mamba_d_ssm, c.mamba_n_heads, c.mamba_d_head = 768, 48, 16
    c.mamba_chunk_size = 16
    # Source config derives this read-only property from the current layer count.
    # Target core types are selected explicitly by Block, not by its source cache.
    assert len(c.layer_types) == L
    c.ssm_in_multiplier = 1.0
    c.ssm_multipliers = [1.0] * 5
    return c


def aq63(x):
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 63
    q = torch.round(x.detach() * (1 / scale)).clamp(-63, 63)
    return q, scale


def ternary_linear(x, master, row_scale):
    """Integer-dot forward; STE for both master weights and learned row scales."""
    q, a = aq63(x)
    code = torch.round((master / row_scale[:, None]).detach()).clamp(-1, 1)
    exact = F.linear(q, code) * row_scale * a
    if not torch.is_grad_enabled():
        return exact
    x_st = x + (q * a - x).detach()
    w_st = code * row_scale[:, None] + (master - master.detach())
    proxy = F.linear(x_st, w_st)
    return proxy + (exact - proxy).detach()


@torch.no_grad()
def row_scale(w):
    s = w.abs().mean(-1).clamp_min(1e-8)
    for _ in range(8):
        code = torch.round(w / s[..., None]).clamp(-1, 1)
        s = ((w * code).sum(-1) / code.square().sum(-1).clamp_min(1)).clamp_min(1e-8)
    return s


class Banks(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(E, H, D))
        self.up = nn.Parameter(torch.empty(E, H, D))
        self.down = nn.Parameter(torch.empty(E, D, H))
        self.gate_scale = nn.Parameter(torch.ones(E, H))
        self.up_scale = nn.Parameter(torch.ones(E, H))
        self.down_scale = nn.Parameter(torch.ones(E, D))
        self.router = nn.Linear(D, E)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, D)
        scores = self.router(x)
        # Stable descending order makes equal-score ties prefer the lower bank ID.
        ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :K]
        mass = torch.softmax(torch.gather(scores, 1, ids), dim=-1)
        y = torch.zeros_like(x)
        for e in range(E):
            positions, slots = (ids == e).nonzero(as_tuple=True)
            if positions.numel() == 0:
                continue
            inputs = x[positions]
            g = ternary_linear(inputs, self.gate[e], self.gate_scale[e])
            u = ternary_linear(inputs, self.up[e], self.up_scale[e])
            z = F.silu(g) * u
            out = ternary_linear(z, self.down[e], self.down_scale[e])
            y = y.index_add(0, positions, out * mass[positions, slots, None])
        return y.reshape(shape)


class SWA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = [nn.Linear(D, D, bias=False) for _ in range(4)]
        self.register_buffer('inv_freq', 1 / (1e11 ** (torch.arange(0, 128, 2).float() / 128)), persistent=False)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [p(x).reshape(b, t, 4, 128).transpose(1, 2) for p in (self.q, self.k, self.v)]
        f = torch.arange(t, device=x.device).float()[:, None] * self.inv_freq
        f = torch.cat((f, f), -1)[None, None]
        def rotate(a):
            return a * f.cos() + torch.cat((-a[..., 64:], a[..., :64]), -1) * f.sin()
        q, k = rotate(q), rotate(k)
        i = torch.arange(t, device=x.device)
        allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < 128)
        # Source default scale remains 1/sqrt(128); source key multiplier is in K.
        score = (q @ k.transpose(-1, -2)) / math.sqrt(128)
        score = score.masked_fill(~allowed, float('-inf'))
        y = torch.softmax(score, -1) @ v
        return self.o(y.transpose(1, 2).reshape(b, t, D))


class Block(nn.Module):
    def __init__(self, c, site):
        super().__init__()
        self.input_norm = FalconH1RMSNorm(D, c.rms_norm_eps)
        self.ff_norm = FalconH1RMSNorm(D, c.rms_norm_eps)
        if site in SWA_SITES:
            self.core = SWA()
        else:
            self.core = FalconH1Mixer(c, site)
            self.core.register_buffer('mup_vector', torch.ones(1, 1, 2096), persistent=False)
        self.banks = Banks()

    def forward(self, x):
        y = self.core(self.input_norm(x))
        assert y.shape == x.shape, (y.shape, x.shape)
        x = x + y
        return x + self.banks(self.ff_norm(x))


class Target(nn.Module):
    def __init__(self, source):
        super().__init__()
        c = target_config(source)
        self.embed = nn.Embedding(V, D)
        self.layers = nn.ModuleList([Block(c, i) for i in range(L)])
        self.final_norm = FalconH1RMSNorm(D, c.rms_norm_eps)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, ids, positions):
        x = self.embed(ids)
        for block in self.layers:
            x = checkpoint(block, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else block(x)
        x = self.final_norm(x)
        return self.head(x[:, positions])


@torch.no_grad()
def basis(source):
    """Equal trace weight for embedding and untied readout Grams."""
    grams = []
    for w in (source.model.embed_tokens.weight, source.lm_head.weight):
        g = torch.zeros(2048, 2048, device=w.device, dtype=torch.float32)
        for row in range(0, V, 4096):
            a = w[row:row+4096].float()
            g.add_(a.T @ a)
        grams.append(g / g.trace())
    g = grams[0] + grams[1]
    values, vectors = torch.linalg.eigh(g)
    p = vectors[:, -D:].flip(1).contiguous()
    pivots = p.abs().argmax(0)
    p *= torch.sign(p[pivots, torch.arange(D, device=p.device)])[None]
    error = (p.T @ p - torch.eye(D, device=p.device)).abs().max().item()
    assert error <= 1e-4 and torch.isfinite(p).all(), error
    return p, dict(orthogonality_max_abs=error,
                   balanced_trace_fraction=(values[-D:].sum() / values.sum()).item(),
                   normalized_source_trace=[g.trace().item() for g in grams])


@torch.no_grad()
def init_component(name, module, source, p):
    c = source.config
    def copy(a, b):
        assert a.shape == b.shape, (a.shape, b.shape)
        a.copy_(b)
    def norm(a, b):
        copy(a.weight, (p.square() * b.weight.float()[:, None]).sum(0) * 2)
    if name in ('embed', 'head'):
        w = source.model.embed_tokens.weight if name == 'embed' else source.lm_head.weight
        multiplier = c.embedding_multiplier if name == 'embed' else c.lm_head_multiplier
        for r in range(0, V, 4096):
            copy(module.weight[r:r+4096], (w[r:r+4096].float() @ p) * multiplier)
        return
    if name == 'final_norm':
        norm(module, source.model.final_layernorm)
        return
    site = int(name.split('.')[1])
    old = source.model.layers[2*site+1]
    norm(module.input_norm, old.input_layernorm)
    norm(module.ff_norm, old.pre_ff_layernorm)
    if site in SWA_SITES:
        qi = (torch.tensor([0, 2, 4, 6], device=p.device)[:, None] * 128 + torch.arange(128, device=p.device)).flatten()
        ki = (torch.tensor([0, 0, 1, 1], device=p.device)[:, None] * 128 + torch.arange(128, device=p.device)).flatten()
        for new, oldproj, idx, mul in (
            (module.core.q, old.self_attn.q_proj, qi, c.attention_in_multiplier),
            (module.core.k, old.self_attn.k_proj, ki, c.attention_in_multiplier*c.key_multiplier),
            (module.core.v, old.self_attn.v_proj, ki, c.attention_in_multiplier)):
            copy(new.weight, (oldproj.weight[idx].float() @ p) * mul)
        copy(module.core.o.weight, (p.T @ old.self_attn.o_proj.weight[:, qi].float()) * c.attention_out_multiplier)
    else:
        idx = (torch.arange(48, device=p.device)[:, None]*64 + torch.arange(0, 64, 4, device=p.device)).flatten()
        rows = torch.cat((idx, 3072+idx, torch.arange(6144, 6704, device=p.device)))
        mul = old.mamba.mup_vector.flatten()[rows].float() * c.ssm_in_multiplier
        copy(module.core.in_proj.weight, (old.mamba.in_proj.weight[rows].float() @ p) * mul[:, None])
        ci = torch.cat((idx, torch.arange(3072, 3584, device=p.device)))
        copy(module.core.conv1d.weight, old.mamba.conv1d.weight[ci].float())
        copy(module.core.conv1d.bias, old.mamba.conv1d.bias[ci].float())
        copy(module.core.out_proj.weight, (p.T @ old.mamba.out_proj.weight[:, idx].float()) * c.ssm_out_multiplier)
        copy(module.core.norm.weight, old.mamba.norm.weight[idx].float())
        for key in ('A_log', 'dt_bias', 'D'):
            copy(getattr(module.core, key), getattr(old.mamba, key).float())
    for half in range(2):
        mlp = source.model.layers[2*site+half].feed_forward
        for e0 in range(36):
            e = half*36+e0
            rows = slice(e0*H, (e0+1)*H)
            gate = (mlp.gate_proj.weight[rows].float() @ p) * c.mlp_multipliers[0]
            up = mlp.up_proj.weight[rows].float() @ p
            down = (p.T @ mlp.down_proj.weight[:, rows].float()) * c.mlp_multipliers[1]
            for label, w in (('gate', gate), ('up', up), ('down', down)):
                copy(getattr(module.banks, label)[e], w)
                copy(getattr(module.banks, label+'_scale')[e], row_scale(w))
            key = gate.mean(0)
            copy(module.banks.router.weight[e], key / key.norm().clamp_min(1e-8))
    module.banks.router.bias.zero_()
