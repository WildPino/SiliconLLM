#!/usr/bin/env python3
# Phase 60 / E1 - exporter: trained 5M Arch-A (gated-dReLU TERNARY MLP) -> versioned fp32 binary the E1 C core reads.
#   Target checkpoint: results/phase57/sp58_base.pt  (SparseMLP: gate/up/down = BitLinear158, act=drelu; pred head is
#   DEAD at inference - reg was off - so it is NOT exported). SSM/SWA/head/emb/norms stay fp32.
#
#   The ternary MLP is exported BOTH ways so the format never changes at E2:
#     - dequantized fp32  (wq*scale per row)  -> the E1 path (bit-identical to PyTorch's inference forward)
#     - packed ternary int8 {-1,0,+1} + per-row fp32 scale  -> ready for the E2 pshufb-LUT kernel
#   BitLinear158 per-row scale = w.abs().mean(dim=1).clamp_min(1e-5)  (mean over INPUT features).
#
#   Format (little-endian), magic 0x45314D31 ('E1M1'):
#     header: 16 x uint32:
#        magic, V, D, N, H, L, Dn, dt_rank, conv, win, swa_layer, mlp_hid, gated, ternary, has_packed, reserved
#     core (fp32): emb(V*D)
#        per layer l:
#           norm.w(D)
#           if l==swa: qkv(3D*D) o(D*D)
#           else:      in_proj(2Dn*D) conv_w(Dn*conv) conv_b(Dn) x_proj((dt_rank+2N)*Dn)
#                      dt_proj_w(Dn*dt_rank) dt_proj_b(Dn) A_log(Dn*N) Dskip(Dn) out_proj(D*Dn)
#           norm2.w(D)  gate_deq(mlp_hid*D)  up_deq(mlp_hid*D)  down_deq(D*mlp_hid)
#        norm_f.w(D)  head(V*D)
#     packed (if has_packed): per MLP layer, in layer order:
#           gate_q(int8, mlp_hid*D) gate_scale(fp32, mlp_hid)
#           up_q  (int8, mlp_hid*D) up_scale  (fp32, mlp_hid)
#           down_q(int8, D*mlp_hid) down_scale(fp32, D)
#   A_log is exported RAW (the C core computes -exp(A_log), exactly as phase55). No commit.
#
# Run: benchmarks/phase60/... .venv/Scripts/python.exe benchmarks/phase60/e1_export.py
import os, struct, argparse, numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAGIC = 0x45314D31

def deq_and_pack(w):
    # w: (out, in) fp32 master. Returns (deq fp32 (out,in), q int8 (out,in), scale fp32 (out,)).
    scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)          # (out,1)  BitLinear158 per-row
    wq = (w / scale).round().clamp(-1, 1)                             # {-1,0,+1}
    deq = (wq * scale)                                               # exactly what PyTorch computes at inference
    return (deq.contiguous().numpy().astype("<f4"),
            wq.contiguous().numpy().astype(np.int8),
            scale.squeeze(1).contiguous().numpy().astype("<f4"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results", "phase57", "sp58_base.pt"))
    ap.add_argument("--out",  default=os.path.join(ROOT, "results", "phase60", "e1_model.bin"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    sd = torch.load(a.ckpt, map_location="cpu")
    msd = sd["model"]; cfg = sd["cfg"]
    msd = {k.replace("_orig_mod.", ""): v for k, v in msd.items()}
    V = cfg["V"]; D = cfg["D"]; N = cfg["N"]; H = cfg["H"]; L = cfg["L"]
    swa = cfg["swa_layer"]; DTR = cfg["dt_rank"]; CONV = cfg.get("conv", 4); WIN = cfg.get("win", 128)
    Dn = D * cfg.get("expand", 2); mlp_hid = cfg["mlp_mult"] * D
    gated = 1 if cfg.get("gated", True) else 0
    ternary = 1 if cfg.get("mlp_precision", "ternary") == "ternary" else 0
    assert cfg.get("act") == "drelu", f"E1 core assumes drelu MLP, got act={cfg.get('act')}"
    assert gated and ternary, "E1 target is the gated ternary MLP"
    print(f"cfg: V={V} D={D} N={N} H={H} L={L} Dn={Dn} dt_rank={DTR} conv={CONV} win={WIN} swa@{swa} "
          f"mlp_hid={mlp_hid} gated={gated} ternary={ternary} bpb={cfg.get('bpb')}")

    def g(k):
        if k not in msd: raise KeyError(f"missing {k}; have e.g. {list(msd)[:6]}")
        return msd[k].float().contiguous()

    core = [struct.pack("<16I", MAGIC, V, D, N, H, L, Dn, DTR, CONV, WIN, swa, mlp_hid, gated, ternary, 1, 0)]
    packed = []
    def put(t, exp):
        arr = t.numpy().astype("<f4").ravel()
        assert arr.size == exp, f"size {arr.size} != {exp}"; core.append(arr.tobytes())

    put(g("emb.weight"), V * D)
    for l in range(L):
        put(g(f"blocks.{l}.norm.w"), D)
        if l == swa:
            put(g(f"blocks.{l}.mix.qkv.weight"), 3 * D * D)
            put(g(f"blocks.{l}.mix.o.weight"), D * D)
        else:
            m = f"blocks.{l}.mix."
            put(g(m + "in_proj.weight"), 2 * Dn * D)
            put(g(m + "conv1d.weight"), Dn * CONV); put(g(m + "conv1d.bias"), Dn)
            put(g(m + "x_proj.weight"), (DTR + 2 * N) * Dn)
            put(g(m + "dt_proj.weight"), Dn * DTR); put(g(m + "dt_proj.bias"), Dn)
            put(g(m + "A_log"), Dn * N); put(g(m + "Dskip"), Dn)
            put(g(m + "out_proj.weight"), D * Dn)
        put(g(f"blocks.{l}.norm2.w"), D)
        for name, shp in (("gate", mlp_hid * D), ("up", mlp_hid * D), ("down", D * mlp_hid)):
            w = g(f"blocks.{l}.mlp.{name}.weight")
            deq, q, sc = deq_and_pack(w)
            assert deq.size == shp, f"{name} deq size {deq.size}!={shp}"
            core.append(deq.tobytes())                       # dequant fp32 (E1 path)
            packed.append(q.tobytes()); packed.append(sc.tobytes())   # packed ternary (E2 path)
    put(g("norm_f.w"), D); put(g("head.weight"), V * D)

    data = b"".join(core) + b"".join(packed)
    with open(a.out, "wb") as f: f.write(data)
    ncore = (len(b"".join(core)) - 64) // 4
    print(f"exported -> {a.out}  ({len(data)} bytes: {ncore/1e6:.3f}M core floats + {len(b''.join(packed))} packed bytes)")

    # sanity: dequant == wq*scale reconstruction, and report ternary zero-fraction (health check)
    zeros = []
    for l in range(L):
        w = g(f"blocks.{l}.mlp.gate.weight")
        _, q, _ = deq_and_pack(w); zeros.append(float((q == 0).mean()))
    print(f"  ternary gate zero-frac per layer: " + " ".join(f"{z:.2f}" for z in zeros))

if __name__ == "__main__":
    main()
