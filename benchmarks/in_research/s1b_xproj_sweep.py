#!/usr/bin/env python3
# Inventor / S1b - follow-up to s1_spectral_probe: x_proj emerged as the LOW-RANK ORGAN (rank-halving
# post-hoc = +0.0001 BPB, while in/out_proj cliff). How deep does x_proj's rank go before quality moves?
# Architecture context: x_proj (Dn -> dt_rank + 2N) is the input to the ALREADY-rank-16 dt path plus the
# B/C selective parameters - the selective-SSM control pathway. Its trained PR-rank is 53 of 208.
#
# Read-only, CPU. Output: prints table; tee to docs/in_research/s1b_xproj_sweep_out.txt
import os, sys, math, time
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase57"))
from phase55_ssm import ArchA, load_meta, IDS, META      # noqa: E402
from phase57_sparse import sparsify_mlp                  # noqa: E402

CKPT = os.path.join(ROOT, "results", "phase57", "sp58_base.pt")
EVAL_TOK, SEQ = 50000, 512


def main():
    torch.manual_seed(0); np.random.seed(0)
    sd = torch.load(CKPT, map_location="cpu")
    msd, cfg = sd["model"], sd.get("cfg", {})
    L = cfg.get("L", 6)
    V, exp_len, _ = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    val = ids[int(len(ids) * 0.9):]
    el_t = torch.tensor(exp_len)
    AC = dict(D=cfg.get("D", 256), N=cfg.get("N", 96), H=cfg.get("H", 8), L=L,
              swa_layer=cfg.get("swa_layer", 5), use_mlp=True, mlp_mult=cfg.get("mlp_mult", 4),
              dt_rank=cfg.get("dt_rank", 16))

    def build(state):
        m = ArchA(V, **AC)
        sparsify_mlp(m, AC["D"], AC["mlp_mult"] * AC["D"], act="drelu", gated=True, topk=0.0, ternary=True)
        m.load_state_dict(state, strict=False)
        m.eval(); m.use_ckpt = False
        return m

    @torch.no_grad()
    def ev(model):
        bits = 0.0; nb = 0; tops = []; pos = 0
        while pos + SEQ + 1 <= min(EVAL_TOK, len(val) - 1):
            x = torch.from_numpy(val[pos:pos + SEQ][None, :]); y = torch.from_numpy(val[pos + 1:pos + 1 + SEQ][None, :])
            lg = model(x)
            bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
            nb += int(el_t[y.reshape(-1)].sum().item())
            tops.append(lg.reshape(-1, V).argmax(-1).numpy()); pos += SEQ
        return bits / max(nb, 1), np.concatenate(tops)

    def trunc_xproj(r):
        out = {k: v.clone() for k, v in msd.items()}
        for l in range(L):
            k = f"blocks.{l}.mix.x_proj.weight"
            if k not in out: continue
            W = out[k].float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            out[k] = (U[:, :r] * S[:r]) @ Vh[:r]
        return out

    shp = msd["blocks.0.mix.x_proj.weight"].shape       # (dt_rank+2N, Dn)
    m0, n0 = shp; full = min(m0, n0)
    print("=" * 90)
    print(f"S1b - x_proj-only rank sweep  (shape {tuple(shp)}, full rank {full}, trained PR-rank ~53)")
    print("=" * 90)
    t0 = time.time()
    ref = build(msd); rbpb, rtop = ev(ref)
    print(f"  r=full  BPB={rbpb:.4f}  (ref, {EVAL_TOK/1000:.0f}K tok)  [{time.time()-t0:.0f}s]")
    print(f"  {'r':>6}{'xproj bytes':>12}{'all-proj bytes':>15}{'BPB':>9}{'dBPB':>9}{'top1':>8}")
    dense_x = m0 * n0
    proj_keys = [f"blocks.{l}.mix.{n}.weight" for l in range(L) for n in ("in_proj", "x_proj", "out_proj")]
    nssm = sum(1 for l in range(L) if f"blocks.{l}.mix.x_proj.weight" in msd)   # SWA layer has no projections
    dense_all = sum(msd[k].numel() for k in proj_keys if k in msd) / max(nssm, 1)
    for r in (104, 78, 52, 26, 16):
        st = trunc_xproj(r)
        bpb, top = ev(build(st))
        bx = r * (m0 + n0) / dense_x
        ball = (dense_all - dense_x + r * (m0 + n0)) / dense_all
        print(f"  {r:>6}{bx*100:>11.1f}%{ball*100:>14.1f}%{bpb:>9.4f}{bpb-rbpb:>+9.4f}{100*(top==rtop).mean():>7.2f}%")
    print("\nreading: the deepest r with dBPB <~ +0.002 is the free post-hoc rank; from-scratch structured")
    print("training (the real S1 arm) should do at least as well. dt path precedent: rank 16 by design.")


if __name__ == "__main__":
    main()
