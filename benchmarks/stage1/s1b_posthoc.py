"""Stage 1 (b), arm tern_ph: the routed experts of every MoE layer ternary with the initial scales of s1t_train.py
--ternary (TWN, outlier groups at their max), no training.  Writes layer_<i>.pt files for s1t_export.py patch --ternary.

Usage: python -I benchmarks/stage1/s1b_posthoc.py --out DIR [--group 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1t_model as M  # noqa: E402
from s1t_train import EXP, Tern, init_scale  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group", type=int, default=256)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    W = M.Weights()
    for i in range(1, M.N_LAYER):
        w = W.layer(i)
        st = {"mlp.gate.weight": w["mlp.gate.weight"].float()}
        with torch.no_grad():
            for p in EXP:
                x = w[f"mlp.experts.{p}"].float()
                s, big = init_scale(x, a.group)
                st[f"mlp.experts.{p}"] = Tern.apply(x, s)
                st[f"ternary_scale.{p}"] = s.to(torch.bfloat16)
        torch.save(st, a.out / f"layer_{i}.pt")
        print("layer", i, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
