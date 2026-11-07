"""S2L's dry checks, no model.  Asym: a planted grid comes back, the estimators match a numerical reference on a row of each sign, the Q4_0 blocks read back bit for bit, a 1% scale error fails.

Usage: python -I benchmarks/stage2/s2y_dry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
from s1t_export import bal_error, gguf  # noqa: E402
from s2s_train import Asym, asym_init  # noqa: E402
from s2y_export import asym_blocks  # noqa: E402


def main() -> int:
    g = torch.Generator().manual_seed(1)
    rows, cols = 64, 1024
    level = torch.tensor([-1.0, 0.0, 1.0, 2.0])[torch.randint(0, 4, (1, rows, cols), generator=g)]
    sign = torch.where(torch.arange(rows)[None, :, None] % 2 == 0, 1.0, -1.0)
    scale = 0.01 + 0.04 * torch.rand(1, rows, 1, generator=g)
    w = level * scale * sign + 0.05 * scale * torch.randn(1, rows, cols, generator=g)
    s, sg = asym_init(w)
    grid_ok = bool((sg == sign).all()) and float(((s - scale).abs() / scale).max()) < 0.02
    y = Asym.apply(w, s, sg)
    back = float((y.float() / (s * sg)).round().ne(level).float().mean())
    print("planted: sign and scale", grid_ok, "levels wrong", back)
    p = w.clone().requires_grad_(True)
    q = s.clone().requires_grad_(True)
    Asym.apply(p, q, sg).float().sum().backward()
    x = w * sg / s
    inside = ((x >= -1) & (x <= 2)).float()
    ref = (sg * (x.clamp(-1, 2).round() - x * inside)).sum(-1, keepdim=True)
    est_ok = torch.equal(p.grad, inside) and float((q.grad - ref).abs().max()) < 1e-3
    print("estimators", est_ok)
    e = {"w": y[0], "s": s.to(torch.bfloat16)[0], "g": sg[0]}
    sb = e["s"].float() * e["g"]
    e["w"] = ((y[0].float() / (s[0] * sg[0])).round() * sb).to(torch.bfloat16)
    blocks, low = asym_blocks(e)
    got = gguf.quants.dequantize(blocks.reshape(rows, -1), gguf.GGMLQuantizationType.Q4_0).reshape(rows, cols)
    want = e["w"].float().numpy()
    exact = bool(np.array_equal(got, want))
    planted = bal_error(got * 1.01, want)
    print("Q4_0 exact", exact, "under normal", low, "planted 1%", planted)
    ok = grid_ok and back == 0 and est_ok and exact and planted > 0.0039
    print("DRY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
