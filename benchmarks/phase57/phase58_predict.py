#!/usr/bin/env python3
# Phase 58 / Probe (Finding-7 / K3) - PREDICTABILITY of the active-set, MEASURED on a frozen checkpoint, ZERO training.
#   magic: "P58A" 0x50353841
#
#   Question (58.A): on the gated dReLU-ternary MLP of probe-2 (results/phase57/sp_drelu.pt), is the per-token ACTIVE
#   SET of MLP units (gate>0) PREDICTABLE cheaply, at a BLOCK granularity useful for a block-sparse kernel? If yes, a
#   predictor can pre-select which gate/up/down blocks to stream -> skip the rest (the Finding-7 bandwidth win). This
#   probe only MEASURES the property; it trains nothing (58.B trains the regularizer, only after the Architect reads
#   this). One variable per readout. Held-out everywhere W_p is fit (no leakage).
#
#   For each MLP block, over many val tokens, capture (a) the block INPUT x_t = norm2(residual) feeding the MLP, and
#   (b) the gate pre-activation g (support = g>0, since relu(g)>0 <=> g>0). Group the `hid` units into blocks of
#   32/64/128; a block is ACTIVE at token t if ANY of its units has gate>0. Per (layer x block-size) report:
#     (1) PERSISTENCE = mean overlap(active[t], active[t-1]) = |A_t & A_{t-1}| / |A_t|  (primary; amortizes block-decode)
#     (2) RECALL in-place  = top-k of W_p . x_t   vs  true support, recall@k with k=|active set|.  W_p ridge-fit on a
#         FIT split, recall measured on a HELD-OUT split.   (this is "the real case": predict from the current input)
#     (3) RECALL prefetch  = top-k of W_p . x_{t-1} (cross-position) vs support_t.  Labelled PREFETCH, not the real case.
#   Also reports ACTIVE% (mean active-set fraction) so the Architect can see when block-grouping makes the set ~all-on
#   (then recall is trivially high but the saving is gone) vs when it stays sparse (the useful regime).
#
#   Honest scope: measures the PROPERTY (predictable? at what granularity?), NOT a realized speedup. No kernel.
#
# Smoke (CPU, no checkpoint needed): python benchmarks/phase57/phase58_predict.py --smoke
# Real  (where sp_drelu.pt lives)  : python benchmarks/phase57/phase58_predict.py --ckpt results/phase57/sp_drelu.pt
import argparse, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55"))
sys.path.insert(0, HERE)
from phase55_ssm import ArchA, load_meta, IDS, META
from phase57_ternary import BitLinear158                       # noqa: F401  (registers ternary linear used by checkpoint)
from phase57_sparse import sparsify_mlp                        # rebuild the SparseMLP blocks exactly as trained

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def ridge_fit(F_fit, Y_fit, lam):
    # closed-form ridge with bias column: W = (FtF + lam I)^-1 Ft Y ; F has a ones column appended
    n, d = F_fit.shape
    Fb = np.concatenate([F_fit, np.ones((n, 1), dtype=np.float64)], axis=1)
    A = Fb.T @ Fb
    A[np.arange(d + 1), np.arange(d + 1)] += lam
    W = np.linalg.solve(A, Fb.T @ Y_fit)                       # (d+1, nb)
    return W


def ridge_apply(F_eval, W):
    n = F_eval.shape[0]
    Fb = np.concatenate([F_eval, np.ones((n, 1), dtype=np.float64)], axis=1)
    return Fb @ W


def recall_at_k(scores, active):
    # scores,(active bool): (M, nb). per row k = #active; recall = |topk & active| / |active|. rows with 0 active skipped.
    k = active.sum(1)                                          # (M,)
    sel = k > 0
    if sel.sum() == 0:
        return float("nan")
    scores, active, k = scores[sel], active[sel], k[sel]
    order = np.argsort(-scores, axis=1)                        # cols sorted by score desc
    rank = np.argsort(order, axis=1)                           # rank[i,c] = position of col c
    predicted = rank < k[:, None]                             # top-k columns per row
    hit = (predicted & active).sum(1)
    return float(np.mean(hit / k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="results/phase57/sp_drelu.pt (real run); omit for --smoke")
    ap.add_argument("--smoke", action="store_true", help="no checkpoint: fresh untrained model, tiny capture, prove the table")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nbatch", type=int, default=24, help="val windows to capture (more = better ridge fit)")
    ap.add_argument("--fit-frac", type=float, default=0.6, help="fraction of windows used to FIT W_p; rest is HELD-OUT")
    ap.add_argument("--block-sizes", type=str, default="32,64,128")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()
    if a.smoke:
        a.seq, a.batch, a.nbatch = 64, 4, 8
    dev = a.device
    block_sizes = [int(s) for s in a.block_sizes.split(",")]
    torch.manual_seed(0); np.random.seed(0)

    V, exp_len, id2bytes = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); val = ids[ntr:]

    # ---- rebuild model exactly as probe-2 trained it, then load the frozen weights ----
    D = 256; AC = dict(D=D, N=96, H=8, L=6, swa_layer=5, use_mlp=True, mlp_mult=4, dt_rank=16); hid = 4 * D
    cfg_act, cfg_gated, cfg_topk = "drelu", True, 0.0
    model = ArchA(V, **AC).to(dev); model.use_ckpt = False
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location=dev)
        cfg = sd.get("cfg", {}) if isinstance(sd, dict) else {}
        cfg_act = cfg.get("act", cfg_act); cfg_gated = cfg.get("gated", cfg_gated); cfg_topk = cfg.get("topk", cfg_topk)
        hid = cfg.get("mlp_mult", 4) * D
    nsw = sparsify_mlp(model, D, hid, cfg_act, cfg_gated, cfg_topk, ternary=True)
    if a.ckpt:
        msd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        miss, unexp = model.load_state_dict(msd, strict=False)
        print(f"  loaded {a.ckpt} | act={cfg_act} gated={cfg_gated} topk={cfg_topk} hid={hid} | missing={len(miss)} unexpected={len(unexp)}")
    else:
        print(f"  [SMOKE] no checkpoint: FRESH UNTRAINED model (pipeline test only; numbers are noise) act={cfg_act} hid={hid}")
    model.eval()
    if not cfg_gated:
        print("  NOTE: checkpoint MLP is not gated -> 'gate support' = post-activation support.");
    mlp_blocks = [blk.mlp for blk in model.blocks if getattr(blk, "use_mlp", False)]
    nblk = len(mlp_blocks)
    print(f"  sparsified {nsw} MLP blocks | capturing {a.nbatch} windows x {a.batch} x {a.seq} = {a.nbatch*a.batch*a.seq} tokens/block\n")

    # ---- capture x_t (mlp input = norm2 residual) and gate support via hooks ----
    cap_x = [[] for _ in range(nblk)]; cap_g = [[] for _ in range(nblk)]
    def mk_pre(i):
        def h(mod, inp): cap_x[i].append(inp[0].detach().to(torch.float32).cpu().numpy())   # (B,T,D)
        return h
    def mk_gate(i):
        def h(mod, inp, out): cap_g[i].append((out.detach() > 0).cpu().numpy())              # (B,T,hid) gate>0
        return h
    handles = []
    for i, m in enumerate(mlp_blocks):
        handles.append(m.register_forward_pre_hook(mk_pre(i)))
        gate_mod = m.gate if getattr(m, "gated", False) and hasattr(m, "gate") else m.up     # plain: support from up->act
        handles.append(gate_mod.register_forward_hook(mk_gate(i)))

    t0 = time.time()
    with torch.no_grad():
        for b in range(a.nbatch):
            pos = b * a.batch * a.seq
            need = a.batch * a.seq + 1
            if pos + need > len(val): break
            block = val[pos:pos + a.batch * a.seq].reshape(a.batch, a.seq)
            model(torch.from_numpy(block).to(dev))
    for h in handles: h.remove()
    nwin = len(cap_x[0])
    nfit = max(1, int(round(nwin * a.fit_frac)))
    print(f"  captured {nwin} windows in {time.time()-t0:.1f}s -> fit={nfit} held-out={nwin-nfit}\n")

    # ---- per (layer x block-size): persistence + recall (in-place, prefetch) ----
    hdr = f"  {'layer':>5} {'BS':>4} {'nblk':>5} {'active%':>8} {'persist%':>9} {'recall_inplace%':>16} {'recall_prefetch%':>17}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for i in range(nblk):
        Xall = np.concatenate(cap_x[i], axis=0)               # (nwin*B, T, D)
        Gall = np.concatenate(cap_g[i], axis=0)               # (nwin*B, T, hid) bool
        # split by WINDOW-rows to keep temporal contiguity within each row; fit on first rows, eval on the rest
        Brow = Xall.shape[0]; nrf = max(1, int(round(Brow * a.fit_frac)))
        for BS in block_sizes:
            nb = hid // BS
            # block-active: any gate>0 within each BS-wide group  -> (rows, T, nb)
            A = Gall[:, :, :nb * BS].reshape(Brow, Xall.shape[1], nb, BS).any(axis=3)
            active_frac = float(A.mean())
            # persistence within each row along T: |A_t & A_tm1| / |A_t|
            At, Atm = A[:, 1:, :], A[:, :-1, :]
            inter = (At & Atm).sum(-1).astype(np.float64); denom = At.sum(-1).astype(np.float64)
            msk = denom > 0
            persist = float((inter[msk] / denom[msk]).mean()) if msk.any() else float("nan")
            # ---- in-place: features x_t -> support_t ----
            Xf = Xall[:nrf].reshape(-1, D).astype(np.float64); Yf = A[:nrf].reshape(-1, nb).astype(np.float64)
            Xe = Xall[nrf:].reshape(-1, D).astype(np.float64); Ye = A[nrf:].reshape(-1, nb)
            if Xe.shape[0] == 0:
                Xe, Ye = Xf, A[:nrf].reshape(-1, nb)          # smoke fallback: not enough rows to hold out
            W = ridge_fit(Xf, Yf, a.ridge)
            rec_in = recall_at_k(ridge_apply(Xe, W), Ye)
            # ---- prefetch: features x_{t-1} -> support_t (cross-position) ----
            Xf2 = Xall[:nrf, :-1, :].reshape(-1, D).astype(np.float64); Yf2 = A[:nrf, 1:, :].reshape(-1, nb).astype(np.float64)
            Xe2 = Xall[nrf:, :-1, :].reshape(-1, D).astype(np.float64); Ye2 = A[nrf:, 1:, :].reshape(-1, nb)
            if Xe2.shape[0] == 0:
                Xe2, Ye2 = Xf2, A[:nrf, 1:, :].reshape(-1, nb)
            W2 = ridge_fit(Xf2, Yf2, a.ridge)
            rec_pf = recall_at_k(ridge_apply(Xe2, W2), Ye2)
            print(f"  {i:>5} {BS:>4} {nb:>5} {active_frac*100:>7.1f}% {persist*100:>8.1f}% {rec_in*100:>15.1f}% {rec_pf*100:>16.1f}%")
    print("\n  reading: PERSIST = free carry-over of the active set step-to-step (block-decode amortization).")
    print("           RECALL_INPLACE = the real case (predict support from current input); PREFETCH = ahead-of-time (label only).")
    print("           ACTIVE% near 100 = block grouping ate the sparsity (recall trivial, no saving); the useful regime is ACTIVE% well below 100.")
    print("STOP. table above -> the Architect reads it. No training here, no commit.")


if __name__ == "__main__":
    main()
