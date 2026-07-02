#!/usr/bin/env python3
# Phase 60 / E4 - exporter: moe_gran.pt (probe-4 promoted MoE) -> versioned binary the E4 engine reads.
#   MoE MLP per layer: router (fp32 Linear D->E, +bias) + gate/up = merged BitLinear158(D, E*hid_e) (per-row ternary
#   over D) + Wd = per-expert down (E,D,hid_e) ternary (per-row over hid_e). E=32 hid_e=128 k=8. SSM/SWA/head/emb/norms
#   = standard ArchA fp32 (same as E1). Ternary parts exported BOTH dequant-fp32 (E4-ref path) AND packed int8+scale.
#
#   Format (LE), magic 0x45344D31 ('E4M1'):
#     header 16xu32: magic,V,D,N,H,L,Dn,dt_rank,conv,win,swa_layer,E,hid_e,k,has_packed,reserved
#     emb; per layer: norm.w; [SSM tensors | SWA]; norm2.w;
#        router_w(E*D) router_b(E); gate_deq(E*hid_e*D) up_deq(E*hid_e*D) Wd_deq(E*D*hid_e)
#     norm_f.w; head
#     packed (if has_packed) per layer: gate_q(i8 E*hid_e*D) gate_sc(E*hid_e) up_q up_sc Wd_q(i8 E*D*hid_e) Wd_sc(E*D)
# Run: .venv/Scripts/python.exe benchmarks/phase60/e4_export.py
import os, struct, argparse, numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAGIC = 0x45344D31

def deq_pack(w, dim):   # ternary per-row over `dim`; returns (deq fp32, q int8, scale fp32 (rows,))
    scale = w.abs().mean(dim=dim, keepdim=True).clamp_min(1e-5)
    wq = (w / scale).round().clamp(-1, 1)
    deq = wq * scale
    return deq.contiguous().numpy().astype("<f4"), wq.contiguous().numpy().astype(np.int8), scale.squeeze(dim).contiguous().numpy().astype("<f4")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results", "phase57", "moe_gran.pt"))
    ap.add_argument("--out",  default=os.path.join(ROOT, "results", "phase60", "e4_model.bin"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sd = torch.load(a.ckpt, map_location="cpu"); msd = sd["model"]; cfg = sd["cfg"]
    V,D,N,H,L = cfg["V"],cfg["D"],cfg["N"],cfg["H"],cfg["L"]
    swa,DTR,CONV,WIN = cfg["swa_layer"],cfg["dt_rank"],cfg.get("conv",4),cfg.get("win",128)
    Dn = D*2; E,hid_e,k = cfg["E"],cfg["hid_e"],cfg["topk"]
    print(f"cfg V={V} D={D} N={N} L={L} Dn={Dn} E={E} hid_e={hid_e} k={k} bpb={cfg.get('bpb')}")
    def g(kk):
        if kk not in msd: raise KeyError(f"missing {kk}")
        return msd[kk].float().contiguous()
    core=[struct.pack("<16I", MAGIC, V,D,N,H,L,Dn,DTR,CONV,WIN,swa,E,hid_e,k,1,0)]
    packed=[]
    def put(t,exp): arr=t.numpy().astype("<f4").ravel(); assert arr.size==exp,f"{arr.size}!={exp}"; core.append(arr.tobytes())
    put(g("emb.weight"), V*D)
    for l in range(L):
        put(g(f"blocks.{l}.norm.w"), D)
        if l==swa:
            put(g(f"blocks.{l}.mix.qkv.weight"),3*D*D); put(g(f"blocks.{l}.mix.o.weight"),D*D)
        else:
            m=f"blocks.{l}.mix."
            put(g(m+"in_proj.weight"),2*Dn*D); put(g(m+"conv1d.weight"),Dn*CONV); put(g(m+"conv1d.bias"),Dn)
            put(g(m+"x_proj.weight"),(DTR+2*N)*Dn); put(g(m+"dt_proj.weight"),Dn*DTR); put(g(m+"dt_proj.bias"),Dn)
            put(g(m+"A_log"),Dn*N); put(g(m+"Dskip"),Dn); put(g(m+"out_proj.weight"),D*Dn)
        put(g(f"blocks.{l}.norm2.w"), D)
        # MoE: router fp32
        put(g(f"blocks.{l}.mlp.router.weight"), E*D); put(g(f"blocks.{l}.mlp.router.bias"), E)
        # gate/up merged (E*hid_e, D) ternary per-row over D (dim=1)
        for name in ("gate","up"):
            w=g(f"blocks.{l}.mlp.{name}.weight")
            deq,q,sc=deq_pack(w, dim=1); assert deq.size==E*hid_e*D
            core.append(deq.tobytes()); packed.append(q.tobytes()); packed.append(sc.tobytes())
        # Wd (E,D,hid_e) ternary per-row over hid_e (dim=2)
        Wd=g(f"blocks.{l}.mlp.Wd"); deq,q,sc=deq_pack(Wd, dim=2); assert deq.size==E*D*hid_e
        core.append(deq.tobytes()); packed.append(q.tobytes()); packed.append(sc.tobytes())
    put(g("norm_f.w"), D); put(g("head.weight"), V*D)
    data=b"".join(core)+b"".join(packed)
    with open(a.out,"wb") as f: f.write(data)
    print(f"exported -> {a.out} ({len(data)} bytes; core {len(b''.join(core))} + packed {len(b''.join(packed))})")
    # health: ternary zero-frac of gate + Wd (layer 0)
    _,q,_=deq_pack(g("blocks.0.mlp.gate.weight"),1); _,qd,_=deq_pack(g("blocks.0.mlp.Wd"),2)
    print(f"  L0 ternary zero-frac gate={float((q==0).mean()):.2f} Wd={float((qd==0).mean()):.2f}")

if __name__=="__main__": main()
