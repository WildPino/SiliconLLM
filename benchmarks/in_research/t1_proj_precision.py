#!/usr/bin/env python3
# Inventor / TEARDOWN pass-2, move G1 - the HOLE in the projection precision map.
#
# The engine's dominant component is the fp32 SSM-projection GEMV block (in/x/dt/out_proj):
# 52.7% of per-token time (ENGINE_PLAN E3.5 profile), MEASURED memory-bound (P61 GEMV-vec rejection).
# On a weight-bandwidth-bound GEMV, time ~ weight bytes stored -> fp16 ~2x, int8 ~4x on THAT component.
# P61 measured only the EXTREME (1.58-bit ternary -> +0.018/0.022 BPB, FAIL: "precision-hungry organs").
# The map therefore has exactly two points: 1.58-bit (dead) and 32-bit (baseline). fp16 / bf16 / int8
# are UNTESTED, and they are pure INFERENCE-TIME weight casts on the FIXED trained weights (no retrain).
#
# This probe fills the hole: cast ONLY the four SSM projections to each precision, keep everything else
# (ternary MLP, fp32 head/embed) fixed = ONE variable = projection storage precision. Measure val BPB
# delta + top-1 agreement vs the fp32 reference (sp58_base, the E1 anchor, known BPB 0.8799).
#
# It answers the QUALITY question (the necessary gate); the SPEED follows from memory-boundedness and
# from the keystone (fp16 halves the RESIDENT footprint -> more model stays under the 16MB L3 cliff).
# Engine gate to read against: the E2 quality bar dBPB <= +0.005 (half the ternary cost itself).
#
# CPU-only, $0.  python benchmarks/in_research/t1_proj_precision.py [--eval-tok 100000] [--seq 512]
import argparse, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase55"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase57"))
from phase55_ssm import ArchA, load_meta, IDS, META      # noqa: E402
from phase57_sparse import sparsify_mlp                  # noqa: E402

CKPT = os.path.join(ROOT, "results", "phase57", "sp58_base.pt")
PROJ = ("in_proj", "x_proj", "dt_proj", "out_proj")       # the whole fp32 SSM-projection block


def cast_roundtrip(w, dtype):
    """Store-in-{fp16,bf16} then read back to fp32 = the representation error the engine would carry."""
    return w.to(dtype).to(torch.float32)


def quant_int8_perrow(w):
    """Per-output-row absmax int8 (per-channel, the standard weight-quant; matches the engine's per-row
    scale discipline). w is [out, in]; scale per row over the in dimension."""
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    q = (w / scale).round().clamp(-127, 127)
    return q * scale


def quant_int8_pertensor(w):
    scale = w.abs().amax().clamp_min(1e-8) / 127.0
    return (w / scale).round().clamp(-127, 127) * scale


def apply_precision(state, mode):
    """Return a new state dict with ONLY the SSM projection weights transformed to `mode`."""
    out = {}
    hit = 0
    for k, v in state.items():
        is_proj = k.endswith(".weight") and any(f".{p}." in k or k.endswith(f".{p}.weight") for p in PROJ)
        if is_proj and v.dim() == 2:
            if   mode == "fp16":  out[k] = cast_roundtrip(v, torch.float16)
            elif mode == "bf16":  out[k] = cast_roundtrip(v, torch.bfloat16)
            elif mode == "int8r": out[k] = quant_int8_perrow(v)
            elif mode == "int8t": out[k] = quant_int8_pertensor(v)
            else: out[k] = v
            hit += 1
        else:
            out[k] = v
    return out, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-tok", type=int, default=100000)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    msd, cfg = (sd["model"], sd.get("cfg", {})) if "model" in sd else (sd, {})
    L = cfg.get("L", 6)
    print("=" * 96)
    print("TEARDOWN G1 - projection precision map (sp58_base, the E1 fp32 anchor). ONE variable = proj precision.")
    print("=" * 96)
    print(f"[sp58_base.pt] cfg={ {k: cfg.get(k) for k in ('D','N','L','mlp_mult','bpb')} }")
    print(f"projection block quantized = {PROJ}  (the 52.7%-of-engine fp32 GEMVs; MLP stays ternary, head/embed fp32)")

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
        m.load_state_dict(state, strict=False)
        m.eval(); m.use_ckpt = False
        return m

    @torch.no_grad()
    def eval_bpb(model, cap, W):
        bits = 0.0; nb = 0; tops = []
        pos = 0
        while pos + W + 1 <= min(cap, len(val) - 1):
            x = torch.from_numpy(val[pos:pos + W][None, :])
            y = torch.from_numpy(val[pos + 1:pos + 1 + W][None, :])
            logits = model(x)
            bits += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
            nb += int(el_t[y.reshape(-1)].sum().item())
            tops.append(logits.reshape(-1, V).argmax(-1).numpy())
            pos += W
        return bits / max(nb, 1), np.concatenate(tops)

    # per-weight byte cost of the projection block (what the engine stores/streams per proj weight)
    bytes_per = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8r": 1.0, "int8t": 1.0}
    print(f"\n-- val BPB, {a.eval_tok/1000:.0f}K tokens, seq {a.seq} (within-script matched; delta is the signal) --")
    t0 = time.time()
    ref = build_and_load(msd)
    ref_bpb, ref_top = eval_bpb(ref, a.eval_tok, a.seq)
    print(f"  {'mode':>7}{'B/weight':>10}{'proj-GEMV~':>11}{'BPB':>9}{'dBPB':>9}{'top1 vs fp32':>14}   verdict")
    print(f"  {'fp32':>7}{'4.0':>10}{'1.00x':>11}{ref_bpb:>9.4f}{0.0:>+9.4f}{100.0:>13.2f}%   reference (anchor 0.8799@200K)")

    for mode in ("fp16", "bf16", "int8r", "int8t"):
        st, hit = apply_precision(msd, mode)
        m = build_and_load(st)
        bpb, top = eval_bpb(m, a.eval_tok, a.seq)
        d = bpb - ref_bpb
        speed = 4.0 / bytes_per[mode]      # memory-bound ideal on the proj-GEMV component
        ag = 100.0 * (top == ref_top).mean()
        verdict = "PASS (<=+0.005)" if d <= 0.005 else ("gray (<=+0.010)" if d <= 0.010 else "FAIL")
        print(f"  {mode:>7}{bytes_per[mode]:>10.1f}{'~'+f'{speed:.2f}x':>11}{bpb:>9.4f}{d:>+9.4f}{ag:>13.2f}%   {verdict}  [{hit} tensors]")

    print(f"\n  [{time.time()-t0:.0f}s total]")
    print("reading: any mode PASSING the +0.005 engine gate is a pure INFERENCE win on the dominant component ->")
    print("  (a) ~4/Bpw faster on the memory-bound proj-GEMV; (b) smaller RESIDENT footprint under the 16MB keystone.")
    print("P61 (ternary=1.58bit) FAILED at +0.018/+0.022; this probe maps what survives BETWEEN ternary and fp32.")
    print("Amdahl on the full engine (proj=52.7%): fp16 ~1.35x, int8 ~1.6x end-to-end IF the quality gate holds.")


if __name__ == "__main__":
    main()
