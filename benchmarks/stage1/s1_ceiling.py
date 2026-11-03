"""Stage 1 delivery: Ling-flash's desk ceiling at context 2048 with decode-only k = 4, recomputed with the kernel
rates measured in stage 1 (c) (results/s1k_verdict.json) instead of the RAM-wall assumption.  DESK-MODEL: the byte
map, the 32 GB/s wall and the fixed 2.70 ms per 20 layers are stage 0's (benchmarks/wallmap/w0_desk.py); only the
ternary organs' times change.

A ternary organ costs weights / R, with R the measured streaming rate of the format at the closest measured shape:
    routed experts       the "experts" cell (mul_mat_id, 1024 x 4096 and 4096 x 1024);
    shared expert        the "experts2d" cell (one plain product per matrix, the same shapes);
    attention, dense FFN and head: the "attn" cell (4096 x 4096).
Every other byte (router int8, KV cache, ring state, embedding row, stored Q4_K_M organs) costs bytes / 32 GB/s, as in
stage 0 (Q4_K streamed at 34 GB/s in stage 1 (c), so that stays a fair bound).  Rows with "ring-linear" keep stage 0's
scenario (a KV cache in 1 layer of 8, plus each linear layer's fp32 state read and written per token); "full q8_0 KV"
keeps every layer's attention.  The Q2_K row times the experts at Q2_K's
measured rate; its MB column still counts them at 0.25 B/weight.

Usage: python -I benchmarks/stage1/s1_ceiling.py --json benchmarks/stage1/results/s1_ceiling.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wallmap"))
import w0_desk as D  # noqa: E402

TERNARY_ORGANS = ("attention", "shared", "dense_ffn", "routed", "head")
CELL = {"routed": "experts", "shared": "experts2d", "attention": "attn", "dense_ffn": "attn", "head": "attn"}


def rates(summary: dict, build_type: str) -> dict:
    def r(cell: str) -> float:
        return summary[f"{build_type}|{cell}|stream"]["gweights_per_s"] * 1e9
    return {o: r(c) for o, c in CELL.items()}


def ceiling(m: dict, forms: dict, R: dict | None, kv=(1.0625, "ring"), ctx: int = 2048, k: int = 4) -> dict:
    k0 = int(m["hparams"]["expert_used_count"])
    m = dict(m, hparams=dict(m["hparams"], expert_used_count=k))
    b = D.bytes_of(m, {"forms": forms, "kv": kv}, ctx)
    if "routed" not in forms:  # the stored routed bytes are per token at the file's k
        b["routed"] *= k / k0
    E = int(m["hparams"]["expert_count"])
    t = {}
    for o, v in b.items():
        if R is not None and forms.get(o) == "nibble" and o in R:
            w = m["params"][o] * (k / E if o == "routed" else 1)
            t[o] = w / R[o]
        else:
            t[o] = v / D.BW
    fixed = D.T_FIXED_20 * int(m["hparams"]["block_count"]) / 20
    total = sum(t.values()) + fixed
    return {"mb_per_token": sum(b.values()) / 1e6, "ms": {o: 1e3 * x for o, x in t.items()}, "fixed_ms": 1e3 * fixed,
            "ms_per_token": 1e3 * total, "tok_s": 1 / total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(HERE.parent / "wallmap" / "results" / "w0_byte_map.json"))
    ap.add_argument("--verdict", default=str(HERE / "results" / "s1k_verdict.json"))
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    flash = [m for m in json.loads(Path(a.map).read_text(encoding="utf-8"))["models"] if m["model"] == "flash"][0]
    s = json.loads(Path(a.verdict).read_text(encoding="utf-8"))["summary"]
    all_tern = {o: "nibble" for o in TERNARY_ORGANS} | {"router": "int8"}
    experts_only = {"routed": "nibble", "router": "int8"}
    iq2, tq2 = rates(s, "ik_llama.cpp|iq2_bn"), rates(s, "llama.cpp|tq2_0")
    q2k = dict(iq2, routed=s["llama.cpp|q2_K|experts|stream"]["gweights_per_s"] * 1e9)
    full = (1.0625, 1.0)  # q8_0 KV in every layer, no ring-linear conversion
    rows = {
        "all organs ternary, at the wall, ring-linear (stage 0's scenario)": ceiling(flash, all_tern, None),
        "all organs ternary, at the wall, full q8_0 KV": ceiling(flash, all_tern, None, kv=full),
        "all organs ternary, IQ2_BN measured, ring-linear": ceiling(flash, all_tern, iq2),
        "all organs ternary, IQ2_BN measured, full q8_0 KV": ceiling(flash, all_tern, iq2, kv=full),
        "all organs ternary, TQ2_0 measured, full q8_0 KV": ceiling(flash, all_tern, tq2, kv=full),
        "experts Q2_K measured, the rest ternary IQ2_BN measured, full q8_0 KV": ceiling(flash, all_tern, q2k, kv=full),
        "experts ternary IQ2_BN measured, the rest as stored (Q4_K_M, full f16 KV)":
            ceiling(flash, experts_only, iq2, kv=(2, 1.0)),
        "k = 4 alone: Q4_K_M as stored, full f16 KV": ceiling(flash, {}, None, kv=(2, 1.0)),
    }
    out = {"note": "DESK-MODEL, Ling-flash, context 2048, decode-only k = 4", "rates_g_weights_per_s": {
        "iq2_bn": {o: r / 1e9 for o, r in iq2.items()}, "tq2_0": {o: r / 1e9 for o, r in tq2.items()},
        "q2_K_routed": q2k["routed"] / 1e9}, "rows": rows}
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, r in rows.items():
        print(f"{r['tok_s']:5.1f} tok/s  {r['ms_per_token']:5.1f} ms  routed {r['ms']['routed']:5.2f} ms  "
              f"attention {r['ms']['attention']:5.2f} ms  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
