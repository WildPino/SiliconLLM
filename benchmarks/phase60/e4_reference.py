#!/usr/bin/env python3
# Phase 60 / E4 - PyTorch reference dumps for the MoE ladder. Rebuilds moe_gran (build_model from phase59), fp32,
#   no autocast. Manual block forward (== moe_forward) capturing per-block residuals AND per-MoE-layer dispatch.
#   Writes (results/phase60/): golden_moe_trace.bin (G1), golden_moe_val.bin (top-1/BPB), golden_moe_dispatch.bin (top-8 ids+weights).
import os, sys, struct, math, argparse, numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55")); sys.path.insert(0, os.path.join(HERE, "..", "phase57"))
from phase55_ssm import load_meta, IDS, META
from phase59_moe import build_model
ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); OUT = os.path.join(ROOT, "results", "phase60"); os.makedirs(OUT, exist_ok=True)

def build():
    sd = torch.load(os.path.join(ROOT, "results", "phase57", "moe_gran.pt"), map_location="cpu"); cfg = sd["cfg"]
    AC = dict(D=cfg["D"], N=cfg["N"], H=cfg["H"], L=cfg["L"], swa_layer=cfg["swa_layer"], use_mlp=True, mlp_mult=cfg["mlp_mult"], dt_rank=cfg["dt_rank"])
    model, _ = build_model(cfg["V"], AC, "moe-gran", cfg["load_balance_w"], cfg["lam_coh"], "cpu", "cpu")
    model.load_state_dict(sd["model"]); model.eval()
    return model, cfg

def fwd_capture(model, idx, cap_res=False, cap_disp=False):
    # returns logits, (residuals list per NLAYER if cap_res), (dispatch list per MoE layer if cap_disp)
    x = model.emb(idx); res = [x.detach().clone()] if cap_res else None; disp = [] if cap_disp else None
    for b in model.blocks:
        x = x + b.mix(b.norm(x))
        if getattr(b, "use_mlp", False):
            xn = b.norm2(x); m = b.mlp
            if cap_disp:
                probs = torch.softmax(m.router(xn.reshape(-1, m.D).float()), dim=-1)
                topv, topi = probs.topk(m.k, dim=-1); topw = topv / topv.sum(-1, keepdim=True)
                disp.append((topi.detach().cpu().numpy(), topw.detach().cpu().numpy()))
            out, _ = m(xn); x = x + out
        if cap_res: res.append(x.detach().clone())
    xn = model.norm_f(x); logits = model.head(xn)
    if cap_res: res.append(xn.detach().clone())
    return logits.detach(), res, disp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seq", type=int, default=512); ap.add_argument("--eval-tok", type=int, default=200000)
    ap.add_argument("--g2-windows", type=int, default=40); ap.add_argument("--trace-len", type=int, default=64); a = ap.parse_args()
    torch.manual_seed(0); model, cfg = build(); V, D, L = cfg["V"], cfg["D"], cfg["L"]
    Vt, exp_len, id2b = load_meta(META); ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); val = ids[int(n*0.9):]; el = torch.tensor(exp_len)

    # G1 trace + dispatch (64 tok)
    T = a.trace_len; seq = torch.from_numpy(val[:T][None, :]).long()
    with torch.no_grad(): logits, res, disp = fwd_capture(model, seq, cap_res=True, cap_disp=True)
    NL = L + 2
    with open(os.path.join(OUT, "golden_moe_trace.bin"), "wb") as f:
        f.write(struct.pack("<4I", 0x4D543031, T, D, NL)); f.write(np.asarray(val[:T], dtype="<u2").tobytes())
        for r in res: f.write(r[0].contiguous().numpy().astype("<f4").tobytes())
        f.write(struct.pack("<I", V)); f.write(logits[0].contiguous().numpy().astype("<f4").tobytes())
    print(f"G1 golden_moe_trace.bin: T={T} NL={NL}")
    # dispatch: per MoE layer (len(disp)) x T positions x k ids(u16)+weights(f32)
    E, K = cfg["E"], cfg["topk"]; nmoe = len(disp)
    with open(os.path.join(OUT, "golden_moe_dispatch.bin"), "wb") as f:
        f.write(struct.pack("<5I", 0x4D443031, nmoe, T, K, E))
        for (topi, topw) in disp:
            f.write(topi.astype("<u2").tobytes()); f.write(topw.astype("<f4").tobytes())
    print(f"G2 golden_moe_dispatch.bin: {nmoe} MoE layers x {T} pos x top-{K}")

    # top-1 argmax over windows + BPB
    W = a.seq; nwin = a.g2_windows; argmax = np.zeros((nwin, W), dtype="<u2"); bits = 0.0; nb = 0
    with torch.no_grad():
        for w in range(nwin):
            pos = w*W; x = torch.from_numpy(val[pos:pos+W][None, :]).long(); y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).long()
            lg, _, _ = fwd_capture(model, x); lg = lg[0]; argmax[w] = lg.argmax(-1).numpy().astype("<u2")
            bits += F.cross_entropy(lg, y[0], reduction="sum").item()/math.log(2); nb += int(el[y[0]].sum().item())
    with open(os.path.join(OUT, "golden_moe_val.bin"), "wb") as f:
        f.write(struct.pack("<3I", 0x4D563031, W, nwin)); f.write(argmax.tobytes())
    print(f"golden_moe_val.bin: W={W} nwin={nwin} | pytorch BPB(slice)={bits/max(nb,1):.4f}")
    # full BPB
    bits = 0.0; nb = 0
    with torch.no_grad():
        lim = min(a.eval_tok, len(val)-1); pos = 0
        while pos+W+1 <= lim:
            x = torch.from_numpy(val[pos:pos+W][None, :]).long(); y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).long()
            lg, _, _ = fwd_capture(model, x)
            bits += F.cross_entropy(lg[0], y[0], reduction="sum").item()/math.log(2); nb += int(el[y[0]].sum().item()); pos += W
    print(f"G-full pytorch BPB(eval_tok={a.eval_tok})={bits/max(nb,1):.6f}  (ckpt 0.858854)")
    print("STOP (E4 reference dumps written).")

if __name__ == "__main__": main()
