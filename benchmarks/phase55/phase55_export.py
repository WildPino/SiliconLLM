#!/usr/bin/env python3
# Phase 55 - export trained Arch-A weights (archA_seq512.pt) to a flat fp32 binary the C engine reads.
#   Layout (little-endian, contiguous, FIXED order the C generator reloads):
#     header: magic=0x41524341('ARCA'), V,D,N,H,L,Dn,dt_rank,conv,win,swa_layer  (11 x uint32)
#     emb (V*D)
#     for l in 0..L-1:  norm.w (D)
#         SSM:  in_proj(2Dn*D) conv_w(Dn*conv) conv_b(Dn) x_proj((dt_rank+2N)*Dn)
#               dt_proj_w(Dn*dt_rank) dt_proj_b(Dn) A_log(Dn*N) Dskip(Dn) out_proj(D*Dn)
#         SWA:  qkv(3D*D) o(D*D)
#     norm_f.w (D) ; head (V*D)
#   Linear.weight is (out,in) row-major == C matvec(W,x,y,out,in) reads W+o*in. conv1d.weight is (Dn,1,conv).
#   A_log exported RAW (C computes A=-exp(A_log), matching PyTorch -torch.exp(A_log.float())).
#
#   NOTE: gate-1 reference (PyTorch val BPB) is produced by phase55_ssm.py --load (it reprints val BPB);
#   this script only exports the weights. No commit.
#
# Run: python benchmarks/phase55/phase55_export.py
import os, struct, numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(ROOT, "results", "phase55", "archA_seq512.pt")
OUT  = os.path.join(ROOT, "results", "phase55", "archA_weights.bin")
V,D,N,H,L = 1024,192,64,6,4
Dn,DTR,CONV,WIN,SWA = D*2, 12, 4, 128, 3

def main():
    sd = torch.load(CKPT, map_location="cpu")
    msd = sd.get("model", sd) if isinstance(sd, dict) else sd
    msd = {k.replace("_orig_mod.",""):v for k,v in msd.items()}    # strip torch.compile prefix if present
    def g(k):
        if k not in msd: raise KeyError(f"missing weight {k}; keys sample: {list(msd)[:6]}")
        return msd[k].float().contiguous().cpu().numpy().astype("<f4")
    blobs = []
    blobs.append(struct.pack("<11I", 0x41524341, V,D,N,H,L,Dn,DTR,CONV,WIN,SWA))
    def put(arr, expect):
        a = np.asarray(arr).ravel()
        assert a.size == expect, f"size {a.size} != expected {expect}"
        blobs.append(a.astype("<f4").tobytes())
    put(g("emb.weight"), V*D)
    for l in range(L):
        put(g(f"blocks.{l}.norm.w"), D)
        if l == SWA:
            put(g(f"blocks.{l}.mix.qkv.weight"), 3*D*D)
            put(g(f"blocks.{l}.mix.o.weight"),   D*D)
        else:
            m=f"blocks.{l}.mix."
            put(g(m+"in_proj.weight"),  2*Dn*D)
            put(g(m+"conv1d.weight"),   Dn*CONV)        # (Dn,1,conv) -> c*conv+k
            put(g(m+"conv1d.bias"),     Dn)
            put(g(m+"x_proj.weight"),   (DTR+2*N)*Dn)
            put(g(m+"dt_proj.weight"),  Dn*DTR)
            put(g(m+"dt_proj.bias"),    Dn)
            put(g(m+"A_log"),           Dn*N)
            put(g(m+"Dskip"),           Dn)
            put(g(m+"out_proj.weight"), D*Dn)
    put(g("norm_f.w"), D)
    put(g("head.weight"), V*D)
    data = b"".join(blobs)
    with open(OUT, "wb") as f: f.write(data)
    total_floats = (len(data)-44)//4
    print(f"exported -> {OUT}  ({len(data)} bytes, {total_floats} floats = {total_floats/1e6:.3f}M params + header)")
    print("  keys used:", len(msd), "tensors in state_dict")

if __name__ == "__main__":
    main()
