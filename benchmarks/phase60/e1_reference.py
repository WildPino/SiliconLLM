#!/usr/bin/env python3
# Phase 60 / E1 - PyTorch REFERENCE dumps for the 5 parity gates. Rebuilds the exact inference model
#   (ArchA + SparseMLP gated-dReLU-ternary; the sp58_base pred head is dropped via strict=False since reg was off,
#   so SparseMLP.forward IS the inference path). Everything fp32, NO autocast (matches the val_bpb that produced 0.8799).
#
#   Writes (results/phase60/):
#     golden_trace.bin  (G1): fixed 64-token seq + per-layer residual-stream snapshots (emb, after each block, norm_f) + logits.
#     golden_val.bin    (G2/G3): W, nwin, then pytorch top-1 argmax per position over the first nwin windows; prints pytorch BPB.
#     golden_gen.bin    (G5): fixed seeds + greedy (rep1.2/win128/no-top-p) continuations, token streams.
#   The C engine reads val tokens itself from ids.u16 (same split), so only argmax/streams are dumped here.
#
# Run: .venv/Scripts/python.exe benchmarks/phase60/e1_reference.py
import os, sys, struct, math, argparse, numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55"))
sys.path.insert(0, os.path.join(HERE, "..", "phase57"))
from phase55_ssm import ArchA, load_meta, IDS, META
from phase57_sparse import sparsify_mlp

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "phase60"); os.makedirs(OUT, exist_ok=True)

def build_model():
    sd = torch.load(os.path.join(ROOT, "results", "phase57", "sp58_base.pt"), map_location="cpu")
    cfg = sd["cfg"]; V = cfg["V"]; D = cfg["D"]; hid = cfg["mlp_mult"] * D
    AC = dict(D=D, N=cfg["N"], H=cfg["H"], L=cfg["L"], swa_layer=cfg["swa_layer"],
              use_mlp=True, mlp_mult=cfg["mlp_mult"], dt_rank=cfg["dt_rank"])
    model = ArchA(V, **AC)
    sparsify_mlp(model, D, hid, act="drelu", gated=True, topk=0.0, ternary=True)
    msd = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    missing, unexpected = model.load_state_dict(msd, strict=False)
    unexpected = [k for k in unexpected if not k.endswith(".pred.weight") and not k.endswith(".pred.bias")]
    assert not missing and not unexpected, f"load mismatch missing={missing} unexpected={unexpected}"
    model.eval()
    return model, cfg, V, D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval-tok", type=int, default=200000)
    ap.add_argument("--g2-windows", type=int, default=40, help="windows of --seq to dump argmax for (>=20 => >=10K tok)")
    ap.add_argument("--trace-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=200)
    ap.add_argument("--gen-seeds", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(0)
    model, cfg, V, D = build_model()
    Vt, exp_len, id2bytes = load_meta(META)
    ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    n = len(ids); ntr = int(n * 0.9); val = ids[ntr:]
    el = torch.tensor(exp_len)

    # ---------------- G1 golden-trace ----------------
    T = a.trace_len
    seq = torch.from_numpy(val[:T][None, :]).long()
    caps = {}
    hooks = []
    hooks.append(model.emb.register_forward_hook(lambda m, i, o: caps.__setitem__("emb", o.detach())))
    for l, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(f"blk{l}", o.detach())))
    hooks.append(model.norm_f.register_forward_hook(lambda m, i, o: caps.__setitem__("norm_f", o.detach())))
    with torch.no_grad():
        logits = model(seq).detach()
    for h in hooks: h.remove()
    L = cfg["L"]
    order = ["emb"] + [f"blk{l}" for l in range(L)] + ["norm_f"]
    nlayer = len(order)
    with open(os.path.join(OUT, "golden_trace.bin"), "wb") as f:
        f.write(struct.pack("<4I", 0x47543031, T, D, nlayer))       # 'GT01'
        f.write(np.asarray(val[:T], dtype="<u2").tobytes())          # the fixed input ids
        for name in order:
            f.write(caps[name][0].contiguous().numpy().astype("<f4").tobytes())   # (T,D)
        f.write(struct.pack("<I", V))
        f.write(logits[0].contiguous().numpy().astype("<f4").tobytes())            # (T,V)
    print(f"G1 golden_trace.bin: T={T} D={D} nlayer={nlayer} (emb+{L} blocks+norm_f) + logits(T,{V})")

    # ---------------- G2/G3 val argmax + BPB ----------------
    W = a.seq; nwin = a.g2_windows
    argmax = np.zeros((nwin, W), dtype="<u2")
    bits = 0.0; nb = 0
    with torch.no_grad():
        for w in range(nwin):
            pos = w * W
            x = torch.from_numpy(val[pos:pos+W][None, :]).long()
            y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).long()
            lg = model(x)[0]                                          # (W,V)
            argmax[w] = lg.argmax(-1).numpy().astype("<u2")
            ce = F.cross_entropy(lg, y[0], reduction="sum")
            bits += ce.item() / math.log(2); nb += int(el[y[0]].sum().item())
    g2_bpb = bits / max(nb, 1)
    with open(os.path.join(OUT, "golden_val.bin"), "wb") as f:
        f.write(struct.pack("<3I", 0x47563031, W, nwin))             # 'GV01'
        f.write(argmax.tobytes())
    print(f"G2/G3 golden_val.bin: W={W} nwin={nwin} ({nwin*W} tok) | pytorch BPB(slice)={g2_bpb:.4f}")

    # full-eval BPB (the 0.8799 anchor) for G3
    bits = 0.0; nb = 0
    with torch.no_grad():
        lim = min(a.eval_tok, len(val) - 1); pos = 0
        while pos + W + 1 <= lim:
            x = torch.from_numpy(val[pos:pos+W][None, :]).long()
            y = torch.from_numpy(val[pos+1:pos+1+W][None, :]).long()
            ce = F.cross_entropy(model(x)[0], y[0], reduction="sum")
            bits += ce.item() / math.log(2); nb += int(el[y[0]].sum().item()); pos += W
    print(f"G3 pytorch full BPB(eval_tok={a.eval_tok}, seq={W})={bits/max(nb,1):.6f}  (ckpt-recorded 0.8799)")

    # ---------------- G5 greedy generation ----------------
    def greedy(seed, ngen, rep=1.2, repwin=128):
        ctx = [int(t) for t in seed]
        with torch.no_grad():
            for _ in range(ngen):
                xin = torch.tensor(ctx[-W:], dtype=torch.long)[None, :]
                lg = model(xin)[0, -1].float().clone()
                idx = torch.tensor(sorted(set(ctx[-repwin:])), dtype=torch.long)
                s = lg[idx]; lg[idx] = torch.where(s > 0, s / rep, s * rep)
                ctx.append(int(lg.argmax().item()))
        return ctx
    seeds_pos = [1000, 20000, 50000]
    with open(os.path.join(OUT, "golden_gen.bin"), "wb") as f:
        f.write(struct.pack("<4I", 0x47473031, a.gen_seeds, 16, a.gen_len))   # 'GG01' nseed seedlen genlen
        for si in range(a.gen_seeds):
            seed = val[seeds_pos[si]:seeds_pos[si]+16]
            out = greedy(seed, a.gen_len)
            f.write(np.asarray(seed, dtype="<u2").tobytes())
            f.write(np.asarray(out[16:], dtype="<u2").tobytes())    # generated continuation
    print(f"G5 golden_gen.bin: {a.gen_seeds} seeds x greedy {a.gen_len} tok (rep1.2/win128, seed+gen<=512)")
    print("STOP (reference dumps written).")

if __name__ == "__main__":
    main()
