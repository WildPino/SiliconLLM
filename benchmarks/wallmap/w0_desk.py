"""Desk model of decode speed per organ form, on top of w0_byte_map.json (DESK-MODEL: nothing here is measured).

    t_token = bytes_per_token / BW + T_FIXED * (layers / 20),    tok/s = 1 / t_token

- BW = 32 GB/s, the plan constant of ROADMAP §3 (0.88 of the E30 median DRAM read). It assumes every kernel streams at
  the wall. That holds for Q4_K/Q6_K/Q8_0/IQ4_XS at Ling-mini shapes (A1: 35-41 GB/s) and does NOT hold for IQ3_S
  (A1: 12.45 GB/s, compute-bound). The nibble-ternary LUT kernel has never been timed streaming at these shapes.
- T_FIXED = 2.70 ms per 20 layers: A1's measured non-streaming residual on a 20-layer Ling-mini token, scaled by layer
  count only. Ling-flash's layers are twice as wide, so this is a floor, not an estimate (the "wide" variant doubles it).
- Bytes per weight of each form: Q4_K_M as stored (from the file); ternary "byte" 0.5 and "nibble" 0.25 (both built in
  engine.c, `--pack byte|nibble`); "trit5" 0.2 (projected, not built); "sub1" 0.1 (QMoE's 0.8 bit/param, a GPU format,
  no CPU kernel exists); int8 1.0625 (Q8_0); bf16 2.

Kernel rate (added after the independent review of 29 October). The ternary forms are not bandwidth-bound on this
box: the operative packed ternary kernel processes 46.16 G weights/s at the goal shape T10 (E28's session, packed
arm; `--lutblk` reached 61.64 but its quality licence was revoked by E32). At 0.25 B/weight that is 11.5 GB/s, a third
of the wall. So each organ costs max(bytes / BW, weights / R(form)); the stored Q4_K/Q6_K/Q8_0/F32 forms and int8 keep
R = infinity (A1: they stream at 35-41 GB/s). sub1 and trit5 have no CPU kernel: they get the ternary rate (QMoE's
0.8 bit is its ternary model, entropy-coded). Three figures per scenario: bytes only (R = infinity), packed (46.16),
lutblk (61.64). Flash's expert matrices (1024 x 4096) are not T10's shape; the rate there is not measured.

Recurrent state of the ring scenarios: each linear layer keeps heads x head_dim^2 fp32 values (Ring-linear configs:
32 x 128^2 in flash, 16 x 128^2 in mini), read and written once per token, so 2 x 4 B each. Extra projections of the
linear layers, if any, are not counted.

Each scenario maps every organ to a form. The scenarios and why each organ sits where it does are in the protocol.

    python -I benchmarks/wallmap/w0_desk.py --map benchmarks/wallmap/results/w0_byte_map.json --ctx 2048 8192 \
        --out benchmarks/wallmap/results/w0_desk.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BW, T_FIXED_20 = 32e9, 2.70e-3
R_TERNARY = {"packed": 46.16e9, "lutblk": 61.64e9}      # E28, T10 shape, weights per second
TERNARY_FORMS = ("nibble", "byte", "trit5", "sub1")
RING_STATE = {"mini": 16 * 128 * 128, "flash": 32 * 128 * 128}   # fp32 values per linear layer
BPW = {"nibble": 0.25, "byte": 0.5, "trit5": 0.2, "sub1": 0.1, "int8": 1.0625, "bf16": 2.0}
WEIGHT_ORGANS = ("attention", "shared", "dense_ffn", "router", "routed", "head", "small")

# organ -> form; "stored" = the Q4_K_M file as is. kv: (bytes per element, fraction of layers that keep a KV cache)
SCENARIOS = {
    "today_q4km": {"forms": {}, "kv": (2, 1.0)},
    "ptq_safe": {"forms": {"router": "bf16"}, "kv": (1.0625, 1.0)},
    "ring_linear_ptq": {"forms": {"router": "bf16"}, "kv": (1.0625, "ring")},
    "all_ternary_nibble": {"forms": {o: "nibble" for o in ("attention", "shared", "dense_ffn", "routed", "head")}
                           | {"router": "int8"}, "kv": (1.0625, "ring")},
    "all_ternary_trit5": {"forms": {o: "trit5" for o in ("attention", "shared", "dense_ffn", "routed", "head")}
                          | {"router": "int8"}, "kv": (1.0625, "ring")},
    "likely_experts_2bit": {"forms": {"routed": "nibble", "router": "int8"}, "kv": (1.0625, 1.0)},
    "experts_sub1_rest_nibble": {"forms": {o: "nibble" for o in ("attention", "shared", "dense_ffn", "head")}
                                 | {"routed": "sub1", "router": "int8"}, "kv": (1.0625, "ring")},
}
RING_FULL_ATTN = {"mini": 1 / 5, "flash": 1 / 8}      # layer_group_size 5 and 8 in the Ring-linear configs


def bytes_of(m: dict, scen: dict, ctx: int) -> dict:
    hp = m["hparams"]
    E, k = int(hp["expert_count"]), int(hp["expert_used_count"])
    out = {}
    for o in WEIGHT_ORGANS:
        f = scen["forms"].get(o)
        if f is None:
            out[o] = m["weight_bytes_per_token"].get(o, 0.0)
        else:
            p = m["params"].get(o, 0)
            out[o] = p * BPW[f] * (k / E if o == "routed" else 1)
    out["embedding"] = m["weight_bytes_per_token"]["embedding"]
    eb, frac = scen["kv"]
    frac = RING_FULL_ATTN[m["model"]] if frac == "ring" else frac
    out["kv_cache"] = m["kv_bytes_per_position"] / 2 * eb * frac * ctx
    if scen["kv"][1] == "ring":
        L = int(m["hparams"]["block_count"])
        out["ring_state"] = (L - round(L * RING_FULL_ATTN[m["model"]])) * RING_STATE[m["model"]] * 4 * 2
    return out


def kernel_seconds(m: dict, scen: dict, rate: float) -> float:
    """Time of the ternary organs at a weight rate, where it exceeds their streaming time (bytes / BW)."""
    hp = m["hparams"]
    E, k = int(hp["expert_count"]), int(hp["expert_used_count"])
    extra = 0.0
    for o, f in scen["forms"].items():
        if f in TERNARY_FORMS:
            w = m["params"].get(o, 0) * (k / E if o == "routed" else 1)
            extra += max(0.0, w / rate - w * BPW[f] / BW)
    return extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--ctx", nargs="+", type=int, default=[2048, 8192])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = {m["model"]: m for m in json.loads(Path(a.map).read_text(encoding="utf-8"))["models"]}
    rec = {"constants": {"BW": BW, "T_FIXED_per_20_layers": T_FIXED_20, "bytes_per_weight": BPW,
                         "R_ternary_weights_per_s": R_TERNARY, "ring_state_fp32_per_linear_layer": RING_STATE},
           "scenarios": {k: {"forms": v["forms"], "kv": list(v["kv"])} for k, v in SCENARIOS.items()}, "rows": []}
    for ctx in a.ctx:
        print(f"== context {ctx}")
        for sname, scen in SCENARIOS.items():
            row = {"ctx": ctx, "scenario": sname}
            for name, m in models.items():
                b = bytes_of(m, scen, ctx)
                tot = sum(b.values())
                L = int(m["hparams"]["block_count"])
                fixed = T_FIXED_20 * L / 20
                width = int(m["hparams"]["embedding_length"]) / 2048
                row[name] = {"bytes": b, "total": tot, "tok_s": 1 / (tot / BW + fixed),
                             "tok_s_wide_fixed": 1 / (tot / BW + fixed * width)}
                for rn, rate in R_TERNARY.items():
                    row[name][f"tok_s_{rn}"] = 1 / (tot / BW + fixed + kernel_seconds(m, scen, rate))
            if "mini" in row and "flash" in row:
                row["size_ratio"] = row["flash"]["tok_s"] / row["mini"]["tok_s"]
            rec["rows"].append(row)
            print(f"  {sname:26s} " + "  ".join(
                f"{n} {row[n]['total'] / 1e6:7.1f} MB {row[n]['tok_s']:6.1f} tok/s ({row[n]['tok_s_wide_fixed']:5.1f}; "
                f"packed {row[n]['tok_s_packed']:5.1f}, lutblk {row[n]['tok_s_lutblk']:5.1f})"
                for n in models) + (f"  flash/mini {row['size_ratio']:.2f}" if "size_ratio" in row else ""))
    Path(a.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
