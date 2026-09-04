#!/usr/bin/env python
"""
donor_speed_budget.py -- what a donor must cost per token for engine.c to hit 50 / 100 tok/s.

This is ARITHMETIC OVER MEASURED RATES, not a measurement. It composes:

  (a) donor shapes, read from the config.json actually on disk in the local HF cache
      (never from memory -- see `--show-sources`), and

  (b) the engine rate anchors registered in docs/PHASE64_BUDGET.md, which come from
      the 64.0 tables (HEAD be5f448) and the 64.1b microbenches (HEAD f4a53cf) on the
      3600X reference machine.

Every number it prints is (donor bytes) / (a measured rate), plus the inversion of that
division. It is a BUDGET, and a budget's job is to say what is out of reach before anyone
spends a week measuring how far out of reach it is.

WHAT THIS IS NOT
  - Not a measurement of engine.c on a donor. engine.c has never executed a transformer:
    it is an SSM engine, and attention is an organ it does not have. Every attention row
    below is priced by ANALOGY to the proj-GEMV path, and that analogy is the single
    largest source of error here. It is flagged in the output.
  - Not a quality statement. It says nothing about BPB. The companion question -- what a
    given activation fraction COSTS in BPB -- is what the donor-adaptation probes measure,
    and §"CROSS-REFERENCE" ties the two together.
  - Not a claim that any of these donors runs. It is the precondition check.

Usage:
    python donor_speed_budget.py                    # the ledger
    python donor_speed_budget.py --ctx 8192         # KV traffic at another context
    python donor_speed_budget.py --show-sources     # provenance of every constant
    python donor_speed_budget.py --json out.json
"""

import argparse
import glob
import json
import os
import sys

# ---------------------------------------------------------------------------
# ENGINE RATE ANCHORS -- every one of these is MEASURED, and cited to the file
# and the commit that registered it. Nothing here is estimated or remembered.
# ---------------------------------------------------------------------------

RATES = {
    # (name, GB/s, provenance)
    "moe_lut_integrated": (
        4.2,
        "PHASE64_BUDGET.md §1: MoE LUT effective rate = 2304 KB / 543 us. "
        "The engine-integrated expert path AS IT EXISTS TODAY, t6. Overhead-bound, "
        "not bandwidth-bound (64.0b, HEAD be5f448).",
    ),
    "moe_lut_kernel_pure": (
        17.0,
        "PHASE64_BUDGET.md §1b(b): 512 MB pool, 10922 x 48 KB experts, i.i.d. gather, "
        "t6 = 17.0 GB/s (2.88 us/expert). The kernel with the ~8.4 us/expert of "
        "dispatch/gather/dequant/combine overhead REMOVED (64.1b, HEAD f4a53cf). "
        "This is a CEILING the integrated engine is 3.9x away from.",
    ),
    "proj_gemv_streamed": (
        37.0,
        "PHASE64_BUDGET.md §1b(a): t6 asymptote 34-36 GB/s, fully-streamed floor "
        "declared [34-40]. Midpoint used; both edges reported in the brackets.",
    ),
    "proj_gemv_resident": (
        185.0,
        "PHASE64_BUDGET.md §1b(a): t6 row-partitioned, matrix resident in aggregate "
        "L3 (2x16 MB) -- 187.0 at 4 MB, 185.0 at 8 MB. Applies ONLY while resident, "
        "and 64.1b explicitly warns streamed experts pollute L3 and push the real "
        "rate toward the streaming tail.",
    ),
    "dram_aggregate": (
        42.0,
        "PHASE64_BUDGET.md §1: DRAM cold-stream aggregate ceiling 40-44 GB/s, "
        "saturated at 3 threads. The hard wall for anything that must cross DRAM.",
    ),
}

# Weight encodings. 0.5 B/weight is what the engine actually emits today.
ENCODINGS = {
    "ternary_4bit": (
        0.500,
        "base-3 g=2 codes, 4 bits/weight -- what engine.c emits TODAY (ENGINE_PLAN.md "
        "E4 packing note). ~2.4x the 1.58-bit information-theoretic ideal.",
    ),
    "ternary_1p6bit": (
        0.200,
        "5 trits/byte = 1.6 bits/weight. QUEUED SINCE E4 AND NEVER BUILT "
        "(ENGINE_PLAN.md: 'worthless at sandbox, real at scale-up'). 2.5x fewer bytes.",
    ),
    "fp16": (2.0, "donor's native weights, unconverted."),
}

TARGETS = [(50, "good"), (100, "excellent")]


# ---------------------------------------------------------------------------
# DONOR SHAPES -- read from config.json on disk, never from memory.
# ---------------------------------------------------------------------------

def hf_cache_configs():
    """Every Qwen/Smol config.json actually present in the local HF cache."""
    root = os.path.expanduser("~/.cache/huggingface/hub")
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "models--*"))):
        hits = glob.glob(os.path.join(d, "snapshots", "*", "config.json"))
        if not hits:
            continue
        name = os.path.basename(d).replace("models--", "").replace("--", "/")
        try:
            with open(hits[0], "r", encoding="utf-8") as fh:
                out[name] = (json.load(fh), hits[0])
        except Exception:
            pass
    return out


def shape_from_config(cfg):
    """Normalise a HF config into the fields the budget needs, and FLAG what it cannot model.

    Four donor classes appear in this cache and the accounting below handles two of them
    exactly. The other two are flagged rather than approximated, because a plausible wrong
    row is the failure mode this project actually loses to.

      dense       plain decoder (Qwen2.5, Mistral, SmolLM2, OLMo-2, starcoder2, Phi-3) -- exact
      moe         routed experts, no shared expert, every layer MoE (Mixtral, OLMoE,
                  gpt-oss, Qwen3-30B-A3B) -- exact. NOTE the expert width lives in
                  `moe_intermediate_size` when present and in `intermediate_size` otherwise;
                  reading the wrong one silently undercounts the model by the expert count.
      flagged     shared experts / first-k-dense / hybrid SSM-attention stacks -- NOT modelled
    """
    D = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim", D // nh)
    V = cfg["vocab_size"]
    tied = bool(cfg.get("tie_word_embeddings", False))
    F = cfg.get("intermediate_size")
    # Gated (SwiGLU) FFNs hold THREE matrices per layer (gate, up, down); GPT/starcoder-style
    # FFNs hold TWO (c_fc, c_proj). Assuming three everywhere overcounted starcoder2-3b by
    # 44%, and the name self-check below is what caught it.
    act = str(cfg.get("hidden_act", cfg.get("hidden_activation", "silu"))).lower()
    n_ffn_mats = 3 if ("silu" in act or "swish" in act or "swiglu" in act) else 2

    n_exp = cfg.get("num_experts", cfg.get("num_local_experts", cfg.get("n_routed_experts")))
    topk = cfg.get("num_experts_per_tok", cfg.get("experts_per_token"))
    # the expert width: explicit field wins, else `intermediate_size` IS the expert width
    f_moe = cfg.get("moe_intermediate_size") or (F if n_exp and topk else None)

    flags = []
    if n_exp and topk:
        flags.append("moe")
    if any(k in cfg for k in ("n_shared_experts", "shared_expert_intermediate_size",
                              "shared_intermediate_size")):
        flags.append("shared-experts:NOT-COUNTED")
    if cfg.get("first_k_dense_replace"):
        flags.append("first-k-dense:NOT-COUNTED")
    # SSM/hybrid: real recurrent layers, not merely a per-layer attention *pattern*.
    # `layer_types` alone is NOT evidence of SSM -- gpt-oss uses it to alternate full and
    # sliding attention. Require an actual mamba/linear-attention parameter block.
    lt = cfg.get("layer_types") or []
    ssm_layer_types = [t for t in lt if not isinstance(t, str) or
                       ("attention" not in t and "attn" not in t)]
    if any(k.startswith("mamba") or k.startswith("linear_") or k.startswith("ssm_")
           for k in cfg) or "hybrid_override_pattern" in cfg or        "hybrid_layer_ids" in cfg or ssm_layer_types:
        flags.append("hybrid-ssm:ATTN-AND-KV-WRONG")

    # Sliding-window attention shrinks the KV term. Only real when ENABLED: Qwen2.5 ships
    # `sliding_window: 32768` together with `use_sliding_window: false`, and treating that
    # as a window would wrongly exclude the donor this whole programme is built on.
    swa = cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None
    n_full = None
    if lt and any(isinstance(t, str) and "sliding" in t for t in lt):
        n_full = sum(1 for t in lt if isinstance(t, str) and "sliding" not in t)

    return dict(D=D, L=L, nh=nh, nkv=nkv, hd=hd, V=V, tied=tied, n_ffn_mats=n_ffn_mats,
                n_exp=n_exp, topk=topk, f_moe=f_moe, F=F, flags=flags, swa=swa,
                n_full_attn_layers=n_full,
                modelled=not any(":" in f for f in flags))


def account(s):
    """Per-token ACTIVE weight counts, decomposed by which engine path they'd use.

    'Active' = touched to produce ONE token at decode time. For a dense FFN that is
    all of it; for an MoE donor it is top-k experts only. Embeddings are excluded
    (one row lookup, not a matmul); the output head is counted separately because
    it is one big dense GEMV that no sparsity scheme in this programme touches.
    """
    D, L, nkv, hd, nh = s["D"], s["L"], s["nkv"], s["hd"], s["nh"]
    q_out = nh * hd
    kv_out = nkv * hd

    # attention projections: q, k, v, o -- dense, every token, unskippable
    attn_per_layer = D * q_out + 2 * (D * kv_out) + q_out * D
    attn = attn_per_layer * L

    # feed-forward: gated = 3 * D * f (gate, up, down); non-gated = 2 * D * f
    m = s["n_ffn_mats"]
    if s["n_exp"] and s["topk"] and s["f_moe"]:
        f = s["f_moe"]
        ffn_active = s["topk"] * m * D * f * L
        ffn_total = s["n_exp"] * m * D * f * L
        router = D * s["n_exp"] * L
    else:
        f = s["F"]
        ffn_active = m * D * f * L
        ffn_total = ffn_active
        router = 0

    head = D * s["V"]                       # tied or not, the GEMV still happens
    embed = D * s["V"]                      # lookup only, NOT per-token traffic

    total_params = attn + ffn_total + router + embed + (0 if s["tied"] else head)
    active = attn + ffn_active + router + head
    return dict(attn=attn, ffn_active=ffn_active, ffn_total=ffn_total,
                router=router, head=head, embed=embed,
                total_params=total_params, active=active)


def kv_bytes_per_token(s, ctx, dtype_bytes=2.0):
    """KV-cache bytes READ to generate one token at context length ctx.

    Every generated token re-reads the whole K and V cache at every layer. This term
    is independent of the weights, scales linearly with ctx, and is invisible in any
    benchmark run at ctx=1. It is the reason a tok/s number without a stated context
    length is not a number.
    """
    per_layer_ctx_full = ctx
    if s.get("swa"):
        # windowed layers only ever hold min(ctx, window) entries
        win = min(ctx, s["swa"])
        n_full = s.get("n_full_attn_layers")
        if n_full is None:
            eff = s["L"] * win                      # every layer windowed
        else:
            eff = n_full * ctx + (s["L"] - n_full) * win
    else:
        eff = s["L"] * per_layer_ctx_full
    return 2.0 * s["nkv"] * s["hd"] * eff * dtype_bytes


# ---------------------------------------------------------------------------

def name_implied_params(name):
    """Parameter count the PUBLISHER put in the model's own name, e.g. '...-8x7B' -> 56e9.

    This is the script's planted control. The parameter accounting in account() is easy to
    get plausibly wrong -- reading the wrong expert-width field silently undercounts an MoE
    donor by its expert count, which is exactly the bug this file shipped with on its first
    run. A name is an INDEPENDENT statement of the size, written by someone who was not
    doing this arithmetic. If the computed total disagrees with it by more than the
    tolerance, the accounting is wrong and the row is not to be trusted.
    """
    import re
    base = name.split("/")[-1]
    m = re.search(r"(?<![\d.])(\d+)x(\d+(?:\.\d+)?)[bB](?![\w])", base)
    if m:                                    # '8x7B' = 8 experts of 7B
        return float(m.group(1)) * float(m.group(2)) * 1e9
    hits = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)[bB](?![\w])", base)
    if not hits:
        return None
    # 'OLMoE-1B-7B' names active THEN total; the size of the model is the larger
    return max(float(h) for h in hits) * 1e9


def fmt_g(x):
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return "%.2f%s" % (x / div, unit)
    return "%.0f" % x


def fmt_bytes(b):
    for unit, div in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if abs(b) >= div:
            return "%.2f %s" % (b / div, unit)
    return "%.0f B" % b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096,
                    help="context length for the KV-cache term (default 4096)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--show-sources", action="store_true")
    args = ap.parse_args()

    if args.show_sources:
        print("ENGINE RATE ANCHORS (all measured on the 3600X reference machine)")
        for k, (v, src) in RATES.items():
            print("\n  %-22s %6.1f GB/s\n      %s" % (k, v, src))
        print("\nWEIGHT ENCODINGS")
        for k, (v, src) in ENCODINGS.items():
            print("\n  %-22s %5.3f B/weight\n      %s" % (k, v, src))
        return 0

    cache = hf_cache_configs()
    donors = []
    for name, (cfg, path) in cache.items():
        need = ("hidden_size", "num_hidden_layers", "num_attention_heads",
                "intermediate_size", "vocab_size")
        if any(k not in cfg for k in need):
            continue   # not a plain decoder config (whisper/vision/etc.)
        s = shape_from_config(cfg)
        donors.append((name, s, account(s), path))
    donors.sort(key=lambda r: r[2]["total_params"])

    print("=" * 100)
    print("SELF-CHECK -- computed parameter count vs the size in the publisher's own name")
    print("=" * 100)
    print("An independent statement of each donor's size, not derived from this arithmetic.")
    print("Tolerance 12%% (names round, and embeddings/norms/biases differ by family).")
    print()
    # Documented exceptions: cases where the NAME is wrong, not the arithmetic.
    NAME_WRONG = {
        "mistralai/Mixtral-8x7B-v0.1":
            "'8x7B' would be 56B, but Mixtral shares its attention across experts and is "
            "46.7B in fact; the computed total is the correct one and the name is marketing.",
    }
    n_ok = n_bad = n_skip = 0
    bad = []
    for name, sh, a, _ in donors:
        implied = name_implied_params(name)
        if implied is None:
            n_skip += 1
            continue
        err = abs(a["total_params"] - implied) / implied
        if err <= 0.12:
            n_ok += 1
        elif name in NAME_WRONG:
            n_ok += 1
            bad.append("    EXCEPTION %-27s computed %8s  vs name %8s\n        %s" % (
                name, fmt_g(a["total_params"]), fmt_g(implied), NAME_WRONG[name]))
        else:
            n_bad += 1
            sh["flags"].append("self-check-FAIL:%+.0f%%" % (
                100.0 * (a["total_params"] / implied - 1.0)))
            sh["modelled"] = False          # the control ACTS: this row leaves every table
            bad.append("    FAIL %-32s computed %8s  vs name %8s  (%+.0f%%)" % (
                name, fmt_g(a["total_params"]), fmt_g(implied),
                100.0 * (a["total_params"] / implied - 1.0)))
    print("    %d agree, %d disagree, %d names carry no size" % (n_ok, n_bad, n_skip))
    for line in bad:
        print(line)
    if n_bad:
        print("    ^ EXCLUDED from every table below (flag 'self-check-FAIL').")
    print()

    bw = ENCODINGS["ternary_4bit"][0]
    bw_dense = ENCODINGS["ternary_1p6bit"][0]

    print("=" * 100)
    print("DONOR SPEED BUDGET -- engine.c on the 3600X reference, t6, ctx=%d" % args.ctx)
    print("=" * 100)
    print("""
THE CURRENCY. At a fixed weight encoding, bandwidth and kernel throughput collapse into
one number: ACTIVE WEIGHTS TOUCHED PER TOKEN. The engine's measured delivery, from
PHASE64_BUDGET.md, converted at 0.5 B/weight:

    engine-integrated expert path   4.2 GB/s  =   8.4 G-weights/s   <- what it does TODAY
    kernel-pure expert ceiling     17.0 GB/s  =  34.0 G-weights/s   <- with the ~8.4 us/expert
                                                                       overhead engineered away
    DRAM aggregate ceiling         42.0 GB/s  =  84.0 G-weights/s   <- physics, unreachable

THE BUDGET. One token at 50 tok/s is 20.0 ms; at 100 tok/s it is 10.0 ms. So:""")
    for tok_s, label in TARGETS:
        for rn in ("moe_lut_integrated", "moe_lut_kernel_pure"):
            gws = RATES[rn][0] / bw
            budget = gws * 1e9 / tok_s
            print("    %3d tok/s (%-9s) at %-20s -> %8s active weights/token"
                  % (tok_s, label, rn, fmt_g(budget)))
    print()

    print("=" * 100)
    print("WHAT THE DONORS ACTUALLY COST")
    print("=" * 100)
    hdr = ("%-30s %8s %8s %6s %8s %8s %8s  %s" %
           ("donor", "total", "active", "act%", "attn", "ffn", "head", "flags"))
    print(hdr)
    print("-" * 110)
    for name, s, a, _ in donors:
        print("%-30s %8s %8s %5.1f%% %8s %8s %8s  %s" % (
            name, fmt_g(a["total_params"]), fmt_g(a["active"]),
            100.0 * a["active"] / a["total_params"],
            fmt_g(a["attn"]), fmt_g(a["ffn_active"]), fmt_g(a["head"]),
            ",".join(s["flags"]) if s["flags"] else "dense"))
    print("""
A flag containing ':' means THIS SCRIPT CANNOT MODEL THAT DONOR and its row is arithmetic
over an incomplete parameter count. Such rows are excluded from every table below and may
not be quoted. They are printed only so the gap is visible instead of silent.""")

    print()
    print("=" * 100)
    print("PREDICTED tok/s  (weights only, then + KV cache at ctx=%d)" % args.ctx)
    print("=" * 100)
    print("kernel-pure = the 17.0 GB/s ceiling, i.e. AFTER the expert-path overhead is fixed.")
    print("integrated  = the 4.2 GB/s the engine delivers today.")
    print()
    hdr2 = ("%-28s %11s %11s %11s %11s %11s" %
            ("donor", "w-bytes/tok", "integr.", "kern-pure", "KV/tok", "kern+KV"))
    print(hdr2)
    print("-" * len(hdr2))
    rows = []
    for name, s, a, path in donors:
        if not s["modelled"]:
            continue
        wb = a["active"] * bw
        kvb = kv_bytes_per_token(s, args.ctx)
        t_int = wb / (RATES["moe_lut_integrated"][0] * 1e9)
        t_ker = wb / (RATES["moe_lut_kernel_pure"][0] * 1e9)
        t_kv = kvb / (RATES["dram_aggregate"][0] * 1e9)
        rows.append(dict(donor=name, active=a["active"], total=a["total_params"],
                         weight_bytes_per_token=wb, kv_bytes_per_token=kvb,
                         tok_s_integrated=1.0 / t_int, tok_s_kernel_pure=1.0 / t_ker,
                         tok_s_kernel_pure_with_kv=1.0 / (t_ker + t_kv),
                         config_path=path, shape=s, account=a))
        print("%-28s %11s %11.1f %11.1f %11s %11.1f" % (
            name, fmt_bytes(wb), 1.0 / t_int, 1.0 / t_ker,
            fmt_bytes(kvb), 1.0 / (t_ker + t_kv)))

    print()
    print("=" * 100)
    print("THE INVERSION -- what fraction of each donor may be active to hit the target")
    print("=" * 100)
    print("Read as: 'of this donor's own active weights, what fraction must survive'.")
    print("A value >100%% means the donor already fits and has headroom.")
    print()
    hdr3 = "%-28s %14s %14s %14s %14s" % (
        "donor", "50 t/s integr", "50 t/s kern", "100 t/s kern", "attn-only cap")
    print(hdr3)
    print("-" * len(hdr3))
    for r in rows:
        s, a = r["shape"], r["account"]
        out = [r["donor"]]
        for tok_s, rn in ((50, "moe_lut_integrated"), (50, "moe_lut_kernel_pure"),
                          (100, "moe_lut_kernel_pure")):
            budget_w = (RATES[rn][0] / bw) * 1e9 / tok_s
            out.append("%13.1f%%" % (100.0 * budget_w / a["active"]))
        # the floor nothing in this programme attacks: attention projections + head,
        # dense and unskippable, priced on the proj path at its streamed floor
        hard = (a["attn"] + a["head"]) * bw
        t_hard = hard / (RATES["proj_gemv_streamed"][0] * 1e9)
        out.append("%13.1f" % (1.0 / t_hard))
        print("%-28s %14s %14s %14s %14s" % tuple(out))

    print("""
'attn-only cap' is the tok/s ceiling imposed by the attention projections plus the output
head ALONE -- dense, touched every token, and untouched by every sparsity, carving, MoE and
reconstruction result this programme owns. Nothing can exceed it. It is priced at the
proj-GEMV streamed floor (37 GB/s) with ternary 4-bit weights and NO KV cache.""")

    print()
    print("=" * 100)
    print("THE 1.6-BIT PACK -- queued since E4, never built")
    print("=" * 100)
    print("Same table, weights re-encoded at 0.200 B/weight instead of 0.500 (2.5x fewer bytes).")
    print("This is the cheapest unclaimed multiplier in the project and it needs no research.")
    print()
    print("%-28s %13s %13s %13s" % ("donor", "kern 4-bit", "kern 1.6-bit", "attn-cap 1.6b"))
    print("-" * 70)
    for r in rows:
        a = r["account"]
        t4 = a["active"] * bw / (RATES["moe_lut_kernel_pure"][0] * 1e9)
        t16 = a["active"] * bw_dense / (RATES["moe_lut_kernel_pure"][0] * 1e9)
        hard16 = (a["attn"] + a["head"]) * bw_dense / (RATES["proj_gemv_streamed"][0] * 1e9)
        print("%-28s %13.1f %13.1f %13.1f" % (r["donor"], 1.0 / t4, 1.0 / t16, 1.0 / hard16))
    print("""
CAVEAT, and it is a real one: the 17.0 GB/s expert ceiling was measured at 0.5 B/weight and
is described in 64.1b as gather/kernel-bound rather than stream-bound. Dividing its BYTE rate
by a smaller bytes-per-weight assumes the kernel's WEIGHT rate is bandwidth-limited, which is
exactly what 64.1b says it is not. The 1.6-bit column above is therefore an UPPER BOUND on
what denser packing buys on the expert path. It is unambiguous only where the path really is
stream-bound -- the attn-cap column, and the DRAM footprint.""")

    print()
    print("=" * 100)
    print("SPLIT-PATH MODEL -- FFN on the expert path, attn+head on the proj path")
    print("=" * 100)
    print("""The single-rate tables above charge EVERY weight at the expert-path rate. That is the
conservative reading and it is not the faithful one: attention projections and the output
head are dense contiguous GEMVs, which is what the proj path measures (37 GB/s streamed
floor), while only the routed FFN pays the gather rate. This table splits them. It is the
model to quote, and it is more optimistic than the tables above by roughly 1.3-1.5x.""")
    print()
    print("%-30s %9s %9s %9s %9s  %s" % (
        "donor", "ffn ms", "attn+hd", "KV ms", "tok/s", "FFN-carve ceiling"))
    print("-" * 100)
    for r in rows:
        a = r["account"]
        t_ffn = a["ffn_active"] * bw / (RATES["moe_lut_kernel_pure"][0] * 1e9)
        t_ah = (a["attn"] + a["head"] + a["router"]) * bw / (RATES["proj_gemv_streamed"][0] * 1e9)
        t_kv = r["kv_bytes_per_token"] / (RATES["dram_aggregate"][0] * 1e9)
        tot = t_ffn + t_ah + t_kv
        # the most FFN sparsity can EVER buy: the limit as ffn_active -> 0
        ceiling = tot / (t_ah + t_kv)
        r["split_path_tok_s"] = 1.0 / tot
        r["ffn_carve_ceiling_x"] = ceiling
        print("%-30s %9.2f %9.2f %9.2f %9.1f  %8.2fx" % (
            r["donor"], t_ffn * 1e3, t_ah * 1e3, t_kv * 1e3, 1.0 / tot, ceiling))
    print("""
'FFN-carve ceiling' is the speedup you would get by deleting the ENTIRE feed-forward stack.
It is the hard limit on everything the FFN-sparsity, co-activation-carve, MoE and
reconstruction probes in this programme can ever deliver, on this donor, at this context.
Where that number is small, further FFN work cannot reach the target no matter how good it
gets, and the remaining time is in attention, the head, and the KV cache -- none of which
this programme has attacked.""")

    print()
    print("=" * 100)
    print("THE GOAL, SOLVED -- a 10B donor at 50 and 100 tok/s")
    print("=" * 100)
    print("""Required: active weights/token <= (rate / bytes-per-weight) / target.
Shown as the ACTIVATION FRACTION a 10B-parameter donor may run at. The four cells are the
two engine states (today / overhead fixed) crossed with the two packings (built / queued).""")
    print()
    print("%-34s %16s %16s" % ("10.0G-param donor", "50 tok/s", "100 tok/s"))
    print("-" * 68)
    TOTAL = 10.0e9
    for enc_name, enc_b in (("ternary_4bit", bw), ("ternary_1p6bit", bw_dense)):
        for rn in ("moe_lut_integrated", "moe_lut_kernel_pure"):
            cells = []
            for tok_s, _ in TARGETS:
                budget_w = (RATES[rn][0] / enc_b) * 1e9 / tok_s
                cells.append("%9s (%4.1f%%)" % (fmt_g(budget_w), 100.0 * budget_w / TOTAL))
            print("%-34s %16s %16s" % (enc_name + " / " + rn.replace("moe_lut_", ""),
                                       cells[0], cells[1]))
    print("""
Read the bottom-right cell: with the 1.6-bit pack built AND the expert-path overhead fixed,
a 10B donor reaches 100 tok/s at 8.5% activation and 50 tok/s at 17%. Both are inside what
already-MoE donors ship with -- Qwen3-30B-A3B runs at 10.0% activation by construction, and
Qwen3-Next-80B-A3B at 3.5%. Read the top-left cell: with the engine as it stands today and
the packing it emits today, a 10B donor needs 1.7% activation for 50 tok/s, which no donor
on this disk offers and no carve in this programme has survived.

THE TWO MULTIPLIERS ARE ENGINEERING, NOT RESEARCH. 2.5x from the pack (queued at E4, never
built) and 4.0x from the expert-path overhead (~8.4 us/expert, decomposed and named in 64.1b)
compound to 10x. Nothing below is worth measuring until they are claimed or refuted, because
they move every row of this ledger by a factor larger than any quality lever measured so far.""")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dict(ctx=args.ctx, rates=RATES, encodings=ENCODINGS,
                           bytes_per_weight=bw, rows=rows), fh, indent=1, default=str)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
