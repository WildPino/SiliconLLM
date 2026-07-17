#!/usr/bin/env python3
# Inventor / S1 - do the fp32 SSM projections (the P61 "precision-hungry organs", 52.7% of engine time,
# largest resident byte class at scale) carry LOW-RANK structure - i.e. is a structured re-parameterization
# (low-rank now; butterfly/Hadamard-sandwich as the trained-from-scratch version) plausible?
#
# Two parts, both read-only, CPU, zero training:
#   1. SPECTRA: singular-value spectra of in_proj / x_proj / out_proj (dt_proj is ALREADY a rank-16
#      bottleneck by architecture - the existing precedent that structure is tolerated in this pathway).
#      Reported: effective rank (participation ratio), energy captured at rank r, vs a random-init control.
#   2. FUNCTION: post-hoc SVD truncation of the projections at keep-fraction rho, then REAL val BPB
#      (phase57 protocol, TinyStories val) + top-1 agreement vs the untouched model on the same slice.
#      Honest scope: post-hoc truncation is the PESSIMISTIC bound (P61's lesson: from-scratch beats
#      post-hoc); a flat curve here = strong evidence FOR S1; a cliff does NOT kill the from-scratch
#      version (and does not touch butterfly, which can represent full-rank maps).
#
# Model: results/phase57/sp58_base.pt (the E1 anchor, known BPB 0.8799 @200K protocol).
# Output: prints tables; run with  > docs/in_research/s1_spectral_probe_out.txt
import os, sys, math, time, argparse
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase57"))
from phase55_ssm import ArchA, load_meta, IDS, META      # noqa: E402  (read-only imports)
from phase57_sparse import sparsify_mlp                  # noqa: E402

CKPT = os.path.join(ROOT, "results", "phase57", "sp58_base.pt")
PROJ_KEYS = ("in_proj", "x_proj", "out_proj")            # dt_proj excluded: (Dn,16) = already rank-16 by design


def pr_rank(svals):
    s2 = svals ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def energy_at(svals, r):
    s2 = svals ** 2
    return float(s2[:r].sum() / s2.sum())


def spectra(msd, L):
    print("\n-- part 1: spectra (trained vs random-init control) --")
    print(f"  {'matrix':<10}{'shape':>12}{'PRrank':>8}{'PR/min':>8}{'E@r/4':>8}{'E@r/2':>8}  (control PRrank in parens)")
    for name in PROJ_KEYS:
        prs, e4, e2, ctls = [], [], [], []
        for l in range(L):
            k = f"blocks.{l}.mix.{name}.weight"
            if k not in msd: continue
            W = msd[k].float().numpy()
            sv = np.linalg.svd(W, compute_uv=False)
            m = min(W.shape)
            prs.append(pr_rank(sv)); e4.append(energy_at(sv, m // 4)); e2.append(energy_at(sv, m // 2))
            Wr = np.random.default_rng(l).standard_normal(W.shape).astype(np.float32)
            ctls.append(pr_rank(np.linalg.svd(Wr, compute_uv=False)))
        m_shape = msd[f"blocks.0.mix.{name}.weight"].shape
        mn = min(m_shape)
        print(f"  {name:<10}{str(tuple(m_shape)):>12}{np.mean(prs):>8.1f}{np.mean(prs)/mn:>8.2f}"
              f"{np.mean(e4)*100:>7.1f}%{np.mean(e2)*100:>7.1f}%   (ctl {np.mean(ctls):.1f})")
    print("  reading: PR/min ~1 and E@r/2 ~ r/2-share = full-rank isotropic (no low-rank shortcut);")
    print("  PR/min << ctl/min = the trained maps concentrate -> low-rank/structured is plausible.")


def truncate_state(msd, rho, only=None):
    """Return a copy of the state dict with SVD rank-truncation at keep-fraction rho on the projections."""
    out = {k: v.clone() for k, v in msd.items()}
    for k in list(out.keys()):
        if not k.endswith(".weight"): continue
        if not any(f".mix.{n}." in k for n in (only or PROJ_KEYS)): continue
        W = out[k].float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = max(1, int(round(len(S) * rho)))
        out[k] = (U[:, :r] * S[:r]) @ Vh[:r]
    return out


def byte_fraction(msd, L, rho):
    """resident-byte cost of factored rank-r storage vs dense, over the truncated classes."""
    dense = fact = 0
    for l in range(L):
        for name in PROJ_KEYS:
            k = f"blocks.{l}.mix.{name}.weight"
            if k not in msd: continue
            m, n = msd[k].shape
            r = max(1, int(round(min(m, n) * rho)))
            dense += m * n; fact += r * (m + n)
    return fact / dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-tok", type=int, default=100000)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    sd = torch.load(CKPT, map_location="cpu")
    msd, cfg = (sd["model"], sd.get("cfg", {})) if "model" in sd else (sd, {})
    L = cfg.get("L", 6)
    print("=" * 100)
    print("S1 - spectral + functional probe of the fp32 SSM projections (sp58_base, the E1 anchor)")
    print("=" * 100)
    print(f"[sp58_base.pt] cfg={ {k: cfg.get(k) for k in ('D','N','L','mlp_mult','bpb')} }")
    print("architecture precedent: dt path is ALREADY low-rank (x_proj -> dt_rank=16 -> dt_proj); S1 asks")
    print("whether the OTHER projections tolerate structure too.")

    spectra(msd, L)

    # ---- part 2: functional truncation sweep ----
    V, exp_len, _ = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    val = ids[int(len(ids) * 0.9):]
    el_t = torch.tensor(exp_len)
    AC = dict(D=cfg.get("D", 256), N=cfg.get("N", 96), H=cfg.get("H", 8), L=L,
              swa_layer=cfg.get("swa_layer", 5), use_mlp=True, mlp_mult=cfg.get("mlp_mult", 4),
              dt_rank=cfg.get("dt_rank", 16))

    def build_and_load(state):
        m = ArchA(V, **AC)
        sparsify_mlp(m, AC["D"], AC["mlp_mult"] * AC["D"], act="drelu", gated=True, topk=0.0, ternary=True)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}"
                  + (f" e.g. {missing[:2]}{unexpected[:2]}" if (missing or unexpected) else ""))
        m.eval(); m.use_ckpt = False
        return m

    @torch.no_grad()
    def eval_bpb_and_logits(model, cap, W):
        bits = 0.0; nb = 0; tops = []
        pos = 0
        while pos + W + 1 <= min(cap, len(val) - 1):
            x = torch.from_numpy(val[pos:pos + W][None, :])
            y = torch.from_numpy(val[pos + 1:pos + 1 + W][None, :])
            logits = model(x)
            ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            bits += ce.item() / math.log(2)
            nb += int(el_t[y.reshape(-1)].sum().item())
            tops.append(logits.reshape(-1, V).argmax(-1).numpy())
            pos += W
        return bits / max(nb, 1), np.concatenate(tops)

    print(f"\n-- part 2: functional SVD truncation (REAL val BPB, {a.eval_tok/1000:.0f}K tokens, seq {a.seq}) --")
    t0 = time.time()
    ref_model = build_and_load(msd)
    ref_bpb, ref_top = eval_bpb_and_logits(ref_model, a.eval_tok, a.seq)
    print(f"  rho=1.00  BPB={ref_bpb:.4f}  (reference; anchor 0.8799 @200K protocol)  [{time.time()-t0:.0f}s]")

    print(f"  {'rho':>6}{'proj bytes':>11}{'BPB':>9}{'dBPB':>9}{'top1 vs ref':>12}")
    for rho in (0.75, 0.5, 0.375, 0.25):
        st = truncate_state(msd, rho)
        m = build_and_load(st)
        bpb, top = eval_bpb_and_logits(m, a.eval_tok, a.seq)
        bf = byte_fraction(msd, L, rho)
        ag = (top == ref_top).mean()
        print(f"  {rho:>6.3f}{bf*100:>10.1f}%{bpb:>9.4f}{bpb-ref_bpb:>+9.4f}{ag*100:>11.2f}%")

    print(f"\n  per-class sensitivity at rho=0.5 (which projection is the hungry one?):")
    for name in PROJ_KEYS:
        st = truncate_state(msd, 0.5, only=(name,))
        m = build_and_load(st)
        bpb, top = eval_bpb_and_logits(m, min(a.eval_tok, 50000), a.seq)
        rb, rt = eval_bpb_and_logits(ref_model, min(a.eval_tok, 50000), a.seq)
        print(f"    {name:<10} BPB={bpb:.4f}  dBPB={bpb-rb:+.4f}  top1={100*(top==rt).mean():.2f}%")

    print("\nreading: dBPB <= ~+0.005 at rho<=0.5 -> post-hoc HALVING of projection bytes is nearly free ->")
    print("S1 (trained-from-scratch structured) graduates to a pre-registered probe with strong prior.")
    print("A cliff -> only the from-scratch arm remains (P61 lesson: post-hoc is the pessimistic bound).")


if __name__ == "__main__":
    main()
