"""C50-T2: a Ling-2.0 (bailingmoe2) routed-expert block in numpy, from the GGUF's own tensors.

It follows the pinned llama.cpp's build_moe_ffn for this architecture: sigmoid gating on the F32 router, the
expert-probability bias added for selection only, group-limited selection (the top-2 sum per group picks
expert_group_used_count of expert_group_count groups), top-k of the masked scores, weights = the unbiased sigmoid
probabilities of the selected experts, normalised to sum 1 (sum clamped at 6.1e-5), times expert_weights_scale;
each expert is down(silu(gate x) * up x).  The shared expert is not part of it (ffn_moe_out in llama.cpp).

Experts are dequantized one at a time with the pinned gguf-py, and every row that selected an expert is computed
with it at once, so one call over several input sets reads each expert once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4/gguf-py")
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ARCH = "bailingmoe2"


def _field(r: gguf.GGUFReader, key: str):
    f = r.fields[key]
    v = f.parts[f.data[0]]
    return v.tolist()[0] if hasattr(v, "tolist") else v


class Model:
    def __init__(self, path: str | Path):
        self.r = gguf.GGUFReader(str(path))
        g = lambda k: _field(self.r, f"{ARCH}.{k}")
        self.n_layer, self.d = g("block_count"), g("embedding_length")
        self.n_expert, self.k = g("expert_count"), g("expert_used_count")
        self.n_group, self.n_group_used = g("expert_group_count"), g("expert_group_used_count")
        self.gating, self.norm_w, self.scale = g("expert_gating_func"), bool(g("expert_weights_norm")), g("expert_weights_scale")
        self.dense_lead = g("leading_dense_block_count")
        assert self.gating == 2, "only sigmoid gating is implemented"
        self.t = {t.name: t for t in self.r.tensors}

    def layer(self, il: int) -> "Layer":
        assert self.dense_lead <= il < self.n_layer, il
        return Layer(self, il)


class Layer:
    def __init__(self, m: Model, il: int):
        self.m, self.il = m, il
        t = lambda n: m.t[f"blk.{il}.{n}"]
        router = t("ffn_gate_inp.weight")
        assert router.tensor_type == gguf.GGMLQuantizationType.F32
        self.router = np.asarray(router.data, dtype=np.float32).reshape(m.n_expert, m.d)
        self.bias = np.asarray(t("exp_probs_b.bias").data, dtype=np.float32).reshape(m.n_expert)
        self.gate, self.up, self.down = t("ffn_gate_exps.weight"), t("ffn_up_exps.weight"), t("ffn_down_exps.weight")

    def expert(self, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(gate [ff, d], up [ff, d], down [d, ff]) of expert e, float32."""
        return tuple(dequantize(np.asarray(w.data[e]), w.tensor_type).astype(np.float32) for w in (self.gate, self.up, self.down))

    def route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x [n, d] float32 -> (selected experts [n, k] int, weights [n, k] float32)."""
        m = self.m
        probs = 1.0 / (1.0 + np.exp(-(x @ self.router.T)))
        sel = probs + self.bias
        if m.n_group > 1:
            gs = sel.reshape(len(x), m.n_group, -1)
            top2 = -np.sort(-gs, axis=2)[:, :, :2].sum(axis=2)
            keep = np.argsort(-top2, axis=1, kind="stable")[:, :m.n_group_used]
            mask = np.full(gs.shape[:2], -np.inf, dtype=np.float32)
            np.put_along_axis(mask, keep, 0.0, axis=1)
            sel = (gs + mask[:, :, None]).reshape(len(x), -1)
        top = np.argsort(-sel, axis=1, kind="stable")[:, :m.k]
        w = np.take_along_axis(probs, top, axis=1)
        if m.norm_w:
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 6.103515625e-5)
        if m.scale not in (0.0, 1.0):
            w = w * m.scale
        return top, w.astype(np.float32)

    def forward(self, x: np.ndarray, top: np.ndarray | None = None, w: np.ndarray | None = None, device=None) -> np.ndarray:
        """Routed-expert output [n, d] for x [n, d]; routes x itself unless top and w are given.
        device: a torch device for the expert products (float32, TF32 off); None computes in numpy."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if top is None:
            top, w = self.route(x)
        if device is not None:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            xt, out = torch.from_numpy(x).to(device), torch.zeros(x.shape, dtype=torch.float32, device=device)
            wt = torch.from_numpy(np.ascontiguousarray(w)).to(device)
        else:
            out = np.zeros_like(x)
        for e in np.unique(top):
            rows, slot = np.nonzero(top == e)
            g, u, dn = self.expert(int(e))
            if device is not None:
                g, u, dn = (torch.from_numpy(m).to(device) for m in (g, u, dn))
                r, sl = torch.from_numpy(rows).to(device), torch.from_numpy(slot).to(device)
                xe = xt[r]
                a = xe @ g.T
                h = torch.nn.functional.silu(a) * (xe @ u.T)
                out.index_add_(0, r, (h @ dn.T) * wt[r, sl][:, None])
                continue
            xe = x[rows]
            a = xe @ g.T
            h = (a / (1.0 + np.exp(-a))) * (xe @ u.T)
            out[rows] += (h @ dn.T) * w[rows, slot][:, None]
        return out if device is None else out.cpu().numpy()
