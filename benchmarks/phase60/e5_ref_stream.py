#!/usr/bin/env python3
# Phase 60 / E5 prerequisite - STREAMING-STATEFUL PyTorch reference. The E1 generation gate (G5) was bit-exact only
#   while gen<=seq, because the PyTorch generate() recomputes the whole window from scratch each step (stateless), so
#   beyond `seq` it cannot represent the C engine's O(1) rolling state. This builds an INDEPENDENT PyTorch reference
#   that carries SSM state / causal-conv state / SWA kv-ring incrementally, token by token -- exactly the C engine's
#   forward_token recurrence -- so long-context (>seq) generation has a ground truth that is NOT the C engine itself.
#
#   Two things: (1) SANITY - stateful per-token logits over a 64-tok seq must equal the batched parallel forward
#   (proves the incremental re-implementation is correct). (2) REFERENCE - greedy (rep1.2/win128) generation carrying
#   state for gen>>seq, dumped for the C engine to match (removes E1-G5's gen<=seq limitation).
#
# Run: .venv/Scripts/python.exe benchmarks/phase60/e5_ref_stream.py
import os, sys, struct, argparse, numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase55")); sys.path.insert(0, os.path.join(HERE, "..", "phase57"))
from phase55_ssm import load_meta, IDS, META
from e1_reference import build_model                 # reuses the exact sp58_base rebuild (SparseMLP, strict=False)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); OUT = os.path.join(ROOT, "results", "phase60")


class Streamer:
    """Incremental, stateful single-token forward matching the C engine forward_token exactly (fp32)."""
    def __init__(s, model, cfg):
        s.m = model; s.cfg = cfg; s.D = cfg["D"]; s.N = cfg["N"]; s.H = cfg["H"]; s.L = cfg["L"]
        s.Dn = s.D * 2; s.conv = cfg.get("conv", 4); s.win = cfg.get("win", 128); s.swa = cfg["swa_layer"]
        s.reset()
    def reset(s):
        s.h = [torch.zeros(s.Dn, s.N) for _ in range(s.L)]                 # SSM state per layer
        s.convbuf = [torch.zeros(s.Dn, s.conv) for _ in range(s.L)]        # causal-conv ring (last `conv` inputs)
        s.kring = torch.zeros(s.win, s.D); s.vring = torch.zeros(s.win, s.D); s.kvpos = 0; s.kvcnt = 0
    @torch.no_grad()
    def step(s, tok):
        m = s.m; x = m.emb(torch.tensor([tok])).squeeze(0)                 # (D,)
        for l, blk in enumerate(m.blocks):
            xn = blk.norm(x)
            if l == s.swa:
                qkv = blk.mix.qkv(xn); q, k, v = qkv.split(s.D)
                slot = s.kvpos % s.win; s.kring[slot] = k; s.vring[slot] = v
                s.kvpos += 1; s.kvcnt = min(s.kvcnt + 1, s.win)
                hd = s.D // s.H; ao = torch.zeros(s.D)
                idxs = [(s.kvpos - s.kvcnt + j) % s.win for j in range(s.kvcnt)]
                Ks = s.kring[idxs]; Vs = s.vring[idxs]                     # (kvcnt, D)
                for hh in range(s.H):
                    qh = q[hh*hd:(hh+1)*hd]; Kh = Ks[:, hh*hd:(hh+1)*hd]; Vh = Vs[:, hh*hd:(hh+1)*hd]
                    att = (Kh @ qh) / (hd ** 0.5); att = torch.softmax(att, 0)
                    ao[hh*hd:(hh+1)*hd] = att @ Vh
                x = x + blk.mix.o(ao)
            else:
                mix = blk.mix; xz = mix.in_proj(xn); xx, z = xz.chunk(2)
                cb = s.convbuf[l]; cb[:, :-1] = cb[:, 1:].clone(); cb[:, -1] = xx
                cw = mix.conv1d.weight.squeeze(1)                          # (Dn, conv)
                xx = F.silu((cw * cb).sum(-1) + mix.conv1d.bias)
                dbl = mix.x_proj(xx); dt, B, C = torch.split(dbl, [mix.dt_rank, s.N, s.N])
                dt = F.softplus(mix.dt_proj(dt))                          # (Dn,)
                A = -torch.exp(mix.A_log.float())                        # (Dn,N)
                dA = torch.exp(dt.unsqueeze(-1) * A)                     # (Dn,N)
                s.h[l] = dA * s.h[l] + (dt.unsqueeze(-1) * B.unsqueeze(0) * xx.unsqueeze(-1))
                y = (s.h[l] * C.unsqueeze(0)).sum(-1) + mix.Dskip * xx
                y = y * F.silu(z); x = x + mix.out_proj(y)
            x = x + blk.mlp(blk.norm2(x).unsqueeze(0)).squeeze(0) if hasattr(blk, "mlp") else x
        return m.head(m.norm_f(x))                                        # (V,)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sanity-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=800); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    torch.manual_seed(0)
    model, cfg, V, D = build_model(); st = Streamer(model, cfg)
    Vt, exp_len, id2b = load_meta(META); ids = np.fromfile(IDS, dtype=np.uint16).astype(np.int64)
    val = ids[int(len(ids)*0.9):]

    # (1) SANITY: stateful step-by-step vs batched parallel forward on a fixed seq
    T = a.sanity_len; seq = torch.from_numpy(val[:T][None, :]).long()
    with torch.no_grad(): par = model(seq)[0]                            # (T,V) parallel
    st.reset(); worst = 0.0
    for t in range(T):
        lg = st.step(int(val[t])); worst = max(worst, (lg - par[t]).abs().max().item())
    print(f"SANITY stateful-vs-parallel over {T} tok: max_abs_logit_diff={worst:.3e}  {'OK' if worst < 1e-3 else 'MISMATCH'}")

    # (2) REFERENCE: greedy rep1.2/win128 generation carrying state, gen>>seq -> dump token streams
    def greedy(seedpos, ngen, rep=1.2, repwin=128):
        st.reset(); ctx = [int(val[seedpos + i]) for i in range(16)]
        for i in range(16): lg = st.step(ctx[i])
        for _ in range(ngen):
            l2 = lg.clone().float()
            recent = sorted(set(ctx[-repwin:])); idx = torch.tensor(recent)
            sv = l2[idx]; l2[idx] = torch.where(sv > 0, sv / rep, sv * rep)
            tok = int(l2.argmax()); ctx.append(tok); lg = st.step(tok)
        return ctx[16:]
    seeds = [1000, 20000, 50000][:a.seeds]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "golden_stream_gen.bin"), "wb") as f:
        f.write(struct.pack("<4I", 0x53473031, a.seeds, 16, a.gen_len))   # 'SG01'
        for sp in seeds:
            f.write(np.asarray(val[sp:sp+16], dtype="<u2").tobytes())
            f.write(np.asarray(greedy(sp, a.gen_len), dtype="<u2").tobytes())
    print(f"golden_stream_gen.bin: {a.seeds} seeds x greedy {a.gen_len} tok (STATEFUL, gen>>seq={cfg.get('seq','?')}) -> long-context G5 reference")
    print("STOP (streaming-stateful reference written).")

if __name__ == "__main__":
    main()
