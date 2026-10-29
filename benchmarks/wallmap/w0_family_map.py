"""Per-organ parameter counts from each model's config.json (SOURCE-DERIVED), for the families the map compares.

Organs as in w0_byte_map.py. Per decoded token: every organ once, except routed experts (k of E) and the embedding
(one row). KV elements per position: full-attention layers x 2 x n_kv x head_dim, plus sliding layers capped at
their window (reported at context 2048 as elements per token). Linear-attention layers (Qwen3-Next's Gated DeltaNet,
Ring-linear) keep a state instead; their projection parameters are approximated from the config and flagged.

Desk (w0_desk.py's model, one scenario for every family): every weight ternary at 0.25 B (nibble), router int8,
F16 KV at context 2048 (sliding layers at min(window, 2048)), t = bytes / 32 GB/s + 2.70 ms x layers / 20. Linear-
attention state is left out (flagged).

Control: for Ling-mini-2.0 and Ling-flash-2.0 the counts must equal w0_byte_map.json's (from the GGUF tensors), organ
by organ, within 0.1%; otherwise the script stops.

    python -I benchmarks/wallmap/w0_family_map.py --configs E:/.../wallmap/configs \
        --byte-map benchmarks/wallmap/results/w0_byte_map.json --out benchmarks/wallmap/results/w0_family_map.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BW, T_FIXED_20, CTX = 32e9, 2.70e-3, 2048

MODELS = {   # file stem -> (family, label)
    "inclusionAI_Ling-mini-2.0": ("Ling 2.0", "mini"), "inclusionAI_Ling-flash-2.0": ("Ling 2.0", "flash"),
    "inclusionAI_Ling-1T": ("Ling 2.0", "1T"),
    "openai_gpt-oss-20b": ("gpt-oss", "20b"), "openai_gpt-oss-120b": ("gpt-oss", "120b"),
    "Qwen_Qwen3-30B-A3B": ("Qwen3", "30B-A3B"), "Qwen_Qwen3-235B-A22B": ("Qwen3", "235B-A22B"),
    "Qwen_Qwen3-Next-80B-A3B-Instruct": ("Qwen3-Next", "80B-A3B"),
}


def organs(c: dict) -> dict:
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    hd = c.get("head_dim") or d // c["num_attention_heads"]
    nq, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    E = c.get("num_experts") or c.get("num_local_experts")
    k = c.get("num_experts_per_tok") or c.get("experts_per_token")
    moe_ff = c.get("moe_intermediate_size") or c["intermediate_size"]
    dense_layers = c.get("first_k_dense_replace", 0)
    moe_layers = L - dense_layers
    sh_ff = c.get("moe_shared_expert_intermediate_size") or c.get("shared_expert_intermediate_size") or 0
    n_sh = c.get("num_shared_experts", 1 if sh_ff else 0)
    if n_sh and not sh_ff:
        sh_ff = moe_ff
    full_attn_layers, linear_layers, approx = L, 0, []
    if "full_attention_interval" in c:                       # Qwen3-Next: 1 full attention layer in every 4
        full_attn_layers = L // c["full_attention_interval"]
        linear_layers = L - full_attn_layers
    if "layer_group_size" in c:                              # Ring-linear: 1 softmax layer in every group
        full_attn_layers = L // c["layer_group_size"]
        linear_layers = L - full_attn_layers
    attn_full = d * (nq * hd + 2 * nkv * hd) + nq * hd * d
    if "full_attention_interval" in c:
        attn_full = d * (2 * nq * hd + 2 * nkv * hd) + nq * hd * d     # Qwen3-Next's output gate doubles q
    attn_lin = 0
    if linear_layers and "linear_num_value_heads" in c:
        hk, hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        attn_lin = d * (2 * hk * dk + 2 * hv * dv + 2 * hv) + hv * dv * d
        approx.append("linear-attention projections from the config (q,k,v,z,b,a and out); conv and norms omitted")
    elif linear_layers:
        attn_lin = attn_full
        approx.append("linear layers counted with the softmax layer's projections")
    gpt_oss = "num_local_experts" in c
    p = {
        "attention": full_attn_layers * attn_full + linear_layers * attn_lin,
        "dense_ffn": dense_layers * 3 * d * c["intermediate_size"],
        "shared": moe_layers * n_sh * 3 * d * sh_ff,
        "router": moe_layers * d * E,
        "routed_total": moe_layers * E * 3 * d * moe_ff,
        "head": V * d,
        "embedding_row": d,
    }
    if gpt_oss:
        approx.append("gpt-oss biases omitted (attention, router and experts carry biases: small)")
    p["routed_active"] = p["routed_total"] * k / E
    p["active"] = sum(p[o] for o in ("attention", "dense_ffn", "shared", "router", "head")) + p["routed_active"] + d
    p["core_active"] = p["active"] - p["routed_active"]
    p["total"] = sum(p[o] for o in ("attention", "dense_ffn", "shared", "router", "routed_total", "head")) + V * d
    win = c.get("sliding_window") or 0
    lt = c.get("layer_types")
    if lt:
        nfull = sum(t == "full_attention" for t in lt)
        nslide = len(lt) - nfull
    else:
        nfull, nslide = full_attn_layers, 0
    p["kv_elems_per_pos_full"] = nfull * 2 * nkv * hd
    p["kv_elems_sliding_total"] = nslide * 2 * nkv * hd * win
    p["layers"] = L
    kv = 2 * (p["kv_elems_per_pos_full"] * CTX + nslide * 2 * nkv * hd * min(win, CTX))
    nbytes = 0.25 * (p["active"] - p["router"]) + 1.0625 * p["router"] + kv
    p["desk_nibble_ctx2048"] = {"bytes": nbytes, "kv_bytes": kv,
                                "tok_s": 1 / (nbytes / BW + T_FIXED_20 * L / 20)}
    p["E"], p["k"] = E, k
    p["approximations"] = approx
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--byte-map", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for stem, (fam, lab) in MODELS.items():
        c = json.loads((Path(a.configs) / f"{stem}.json").read_text(encoding="utf-8"))
        rows[stem] = {"family": fam, "label": lab, **organs(c)}
    bm = {m["model"]: m for m in json.loads(Path(a.byte_map).read_text(encoding="utf-8"))["models"]}
    control = {}
    for name, stem in (("mini", "inclusionAI_Ling-mini-2.0"), ("flash", "inclusionAI_Ling-flash-2.0")):
        g, r = bm[name]["params"], rows[stem]
        pairs = {"attention": (g["attention"], r["attention"]), "dense_ffn": (g["dense_ffn"], r["dense_ffn"]),
                 "shared": (g["shared"], r["shared"]), "router": (g["router"], r["router"]),
                 "routed": (g["routed"], r["routed_total"]), "head": (g["head"], r["head"])}
        control[name] = {o: {"gguf": x, "config": y, "rel": abs(x - y) / x} for o, (x, y) in pairs.items()}
    bad = [(n, o) for n, v in control.items() for o, z in v.items() if z["rel"] > 1e-3]
    print("control (config vs GGUF, Ling):", "PASS" if not bad else f"FAIL {bad}")
    if bad:
        return 1
    print(f"{'model':34s} {'total':>7s} {'active':>7s} {'core':>6s} {'routed':>6s} {'attn':>6s} {'head':>6s}  (B params)")
    for stem, r in rows.items():
        print(f"{r['family'] + ' ' + r['label']:34s} {r['total'] / 1e9:7.1f} {r['active'] / 1e9:7.2f} "
              f"{r['core_active'] / 1e9:6.2f} {r['routed_active'] / 1e9:6.2f} {r['attention'] / 1e9:6.2f} "
              f"{r['head'] / 1e9:6.2f}  desk {r['desk_nibble_ctx2048']['tok_s']:5.1f} tok/s"
              + ("  *approx" if r["approximations"] else ""))
    pairs = [("Ling 2.0", "inclusionAI_Ling-mini-2.0", "inclusionAI_Ling-flash-2.0"),
             ("Ling 2.0 (1T)", "inclusionAI_Ling-mini-2.0", "inclusionAI_Ling-1T"),
             ("gpt-oss", "openai_gpt-oss-20b", "openai_gpt-oss-120b"),
             ("Qwen3", "Qwen_Qwen3-30B-A3B", "Qwen_Qwen3-235B-A22B")]
    ratios = {}
    for fam, s, b in pairs:
        ratios[fam] = {"total": rows[b]["total"] / rows[s]["total"], "active": rows[b]["active"] / rows[s]["active"],
                       "core": rows[b]["core_active"] / rows[s]["core_active"]}
        ratios[fam]["desk_speed"] = rows[b]["desk_nibble_ctx2048"]["tok_s"] / rows[s]["desk_nibble_ctx2048"]["tok_s"]
        print(f"  {fam:14s} large/small: total x{ratios[fam]['total']:.1f}, active x{ratios[fam]['active']:.2f}, "
              f"core x{ratios[fam]['core']:.2f}; desk speed large/small (all nibble) {ratios[fam]['desk_speed']:.2f}")
    Path(a.out).write_text(json.dumps({"rows": rows, "control": control, "ratios": ratios}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
