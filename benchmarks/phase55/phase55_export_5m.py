#!/usr/bin/env python3
# Phase 55 - export trained 5M Arch-A weights to a flat fp32 binary the 5M C engine reads.
#   Reads the cfg dict from the checkpoint (D,N,H,L,swa_layer,dt_rank,mlp_mult,use_mlp).
#   Layout (little-endian, FIXED order the phase55_generator_5m.c reloads):
#     header: magic=0x41524342('ARCB'), V,D,N,H,L,Dn,dt_rank,conv,win,swa_layer,mlp_hid (12 x uint32)
#     emb (V*D)
#     for l in 0..L-1:
#         norm.w (D)
#         SSM: in_proj(2Dn*D) conv_w(Dn*conv) conv_b(Dn) x_proj((dt_rank+2N)*Dn)
#              dt_proj_w(Dn*dt_rank) dt_proj_b(Dn) A_log(Dn*N) Dskip(Dn) out_proj(D*Dn)
#         SWA: qkv(3D*D) o(D*D)
#         MLP (if use_mlp): norm2.w(D) fc1(mlp_hid*D) fc2(D*mlp_hid)
#     norm_f.w (D) ; head (V*D)
#   No commit.
#
# Run: python benchmarks/phase55/phase55_export_5m.py [--ckpt results/phase55/archA_5m.pt]
import os, struct, sys, argparse, numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT,"results","phase55","archA_5m.pt"))
    ap.add_argument("--out",  default=os.path.join(ROOT,"results","phase55","archA_5m_weights.bin"))
    a=ap.parse_args()
    sd=torch.load(a.ckpt, map_location="cpu")
    msd=sd.get("model",sd) if isinstance(sd,dict) else sd
    msd={k.replace("_orig_mod.",""):v for k,v in msd.items()}
    cfg=sd.get("cfg",{}) if isinstance(sd,dict) else {}
    V=cfg.get("V",1024); D=cfg["D"]; N=cfg["N"]; H=cfg["H"]; L=cfg["L"]
    swa=cfg["swa_layer"]; DTR=cfg["dt_rank"]; CONV=cfg.get("conv",4); WIN=cfg.get("win",128)
    use_mlp=cfg["use_mlp"]; mlp_mult=cfg.get("mlp_mult",4); Dn=D*cfg.get("expand",2); mlp_hid=mlp_mult*D if use_mlp else 0
    print(f"cfg: V={V} D={D} N={N} H={H} L={L} Dn={Dn} dt_rank={DTR} swa@{swa} mlp={use_mlp}x{mlp_mult} (hid={mlp_hid}) bpb={cfg.get('bpb','?')}")
    def g(k):
        if k not in msd: raise KeyError(f"missing {k}; sample keys {list(msd)[:8]}")
        return msd[k].float().contiguous().cpu().numpy().astype("<f4").ravel()
    blobs=[struct.pack("<12I", 0x41524342, V,D,N,H,L,Dn,DTR,CONV,WIN,swa,mlp_hid)]
    def put(a_,exp):
        assert a_.size==exp, f"size {a_.size}!={exp}"; blobs.append(a_.astype("<f4").tobytes())
    put(g("emb.weight"), V*D)
    for l in range(L):
        put(g(f"blocks.{l}.norm.w"), D)
        if l==swa:
            put(g(f"blocks.{l}.mix.qkv.weight"), 3*D*D); put(g(f"blocks.{l}.mix.o.weight"), D*D)
        else:
            m=f"blocks.{l}.mix."
            put(g(m+"in_proj.weight"), 2*Dn*D); put(g(m+"conv1d.weight"), Dn*CONV); put(g(m+"conv1d.bias"), Dn)
            put(g(m+"x_proj.weight"), (DTR+2*N)*Dn); put(g(m+"dt_proj.weight"), Dn*DTR); put(g(m+"dt_proj.bias"), Dn)
            put(g(m+"A_log"), Dn*N); put(g(m+"Dskip"), Dn); put(g(m+"out_proj.weight"), D*Dn)
        if use_mlp:
            put(g(f"blocks.{l}.norm2.w"), D)
            put(g(f"blocks.{l}.mlp.fc1.weight"), mlp_hid*D); put(g(f"blocks.{l}.mlp.fc2.weight"), D*mlp_hid)
    put(g("norm_f.w"), D); put(g("head.weight"), V*D)
    data=b"".join(blobs)
    with open(a.out,"wb") as f: f.write(data)
    tf=(len(data)-48)//4
    print(f"exported -> {a.out}  ({len(data)} bytes, {tf} floats = {tf/1e6:.3f}M params + header)")

if __name__=="__main__": main()
