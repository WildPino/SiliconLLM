"""Stage 2 part 1, test 4: which training that sees the whole model fits Ling-flash on our machines, and in how many
hours.  Desk arithmetic only: no model is loaded.  Shapes and parameter counts come from the GGUF headers
(benchmarks/wallmap/results/w0_byte_map.json); the machines' sizes are the recorded ones; the slow rate is stage 1 (b)'s
measured block-wise training of Ling-mini's experts (benchmarks/stage1/results/s1b_train.json), the fast rate a stated
fraction of the GPU's fp16 peak.  Every number the page uses is in the JSON this writes.

With --rate (s2f_rate.py's measured seconds per token and layer of the routed experts at Ling-flash's shape, idle
box), the measured hours: the routed experts at the measured rate, the other matrices (attention, shared expert, dense
FFN, head) at the fast rate.

Usage: python -I benchmarks/stage2/s2f_feasibility.py --json benchmarks/stage2/results/s2f_feasibility.json
           [--rate benchmarks/stage2/results/s2f_rate.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GB = 1e9

# Machines (decimal GB).  T4: 16 GB nominal, ~15 usable after the CUDA context; Kaggle's session container has a 30 GiB
# memory.max and 20 GB of /kaggle/working (recorded 2026-10-23).  Local: RTX 3060 12 GB, 80 GiB of RAM.
KAGGLE = {"gpu": 2 * 15.0, "gpu_each": 15.0, "host": 30 * 2**30 / GB, "disk": 20.0}
LOCAL = {"gpu": 12.0, "host": 80 * 2**30 / GB}
# Rates.  Fast: 8 TFLOP/s, about 30% of the 3060's fp16 tensor peak with fp32 accumulation (25.4); the T4 (65 peak,
# 70 W) is taken as the same.  Slow: stage 1 (b)'s measured seconds per token and layer, scaled by the FLOPs.
FAST_FLOPS = 8e12
CTX = 0.4                   # GB of CUDA context per GPU
TOKENS = (1_500_000, 16_777_216)   # stage 1 (b)'s data; EfficientQAT's E2E-QP data (4,096 samples x 4,096)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--rate", type=Path)
    a = ap.parse_args()
    models = {m["model"]: m for m in json.loads((ROOT / "benchmarks/wallmap/results/w0_byte_map.json")
                                                .read_text(encoding="utf-8"))["models"]}
    fl, mi = models["flash"], models["mini"]
    h = fl["hparams"]
    d, f, E, k = h["embedding_length"], h["expert_feed_forward_length"], h["expert_count"], h["expert_used_count"]
    moe_layers = h["block_count"] - h["leading_dense_block_count"]
    p = fl["params"]
    mats = p["routed"] + p["attention"] + p["shared"] + p["dense_ffn"] + p["head"]
    tern = mats * 2 / 8 / GB                                  # 2 bits per weight; scales are < 0.1 GB
    rows = moe_layers * E * (2 * f + d)                       # one scale per row, routed experts
    rows += h["block_count"] * (d + 2 * d // h["attention.head_count"] * h["attention.head_count_kv"] + d)
    rows += moe_layers * (2 * f + d) + 2 * h["feed_forward_length"] + d + h["vocab_size"]
    active = fl["params_active_per_token"]

    # Slow rate: stage 1 (b), layers 1-19 of Ling-mini, 1.5 M tokens, one lr, teacher and student streams included.
    s1b = json.loads((ROOT / "benchmarks/stage1/results/s1b_train.json").read_text(encoding="utf-8"))
    s1b_sec = sum(x["seconds"] for x in s1b["layers"].values())
    s1b_tl = s1b_sec / (len(s1b["layers"]) * s1b["train_tokens"])            # s per token-layer
    hm = mi["hparams"]
    mini_exp_flop = 3 * 2 * hm["expert_used_count"] * 3 * hm["embedding_length"] * hm["expert_feed_forward_length"]
    slow_flops = mini_exp_flop / s1b_tl                                       # the trainer's effective FLOP/s
    flash_exp_flop = 3 * 2 * k * 3 * d * f                                    # fwd + bwd of one layer's experts
    flash_fwd_layer = 2 * active / h["block_count"]                           # the frozen streams' forward

    def hours(flop_per_token, n, rate):
        return flop_per_token * n / rate / 3600

    out = {"flash": {"moe_layers": moe_layers, "d": d, "f": f, "experts": E, "k": k, "params_total": fl["params_total"],
                     "params_matrices": mats, "active": active, "ternary_gb": round(tern, 2),
                     "bf16_gb": round(2 * fl["params_total"] / GB, 1), "scales": rows},
           "machines": {"kaggle_t4x2": KAGGLE, "local": LOCAL},
           "rates": {"fast_flops": FAST_FLOPS, "slow_flops": round(slow_flops), "s1b_seconds": s1b_sec,
                     "s1b_ms_per_token_layer": round(1e3 * s1b_tl, 3)},
           "methods": {}}

    # M1: block-wise, sequential (stage 1 (b)), routed experts trained in chunks of C experts of one layer.
    C = 64
    chunk = C * 3 * d * f
    m1_gpu = chunk * 8 / GB + CTX + 1.0          # fp32 latent + fp32 grad, Adafactor factored; 1 GB of activations
    m1 = {"gpu_gb": round(m1_gpu, 1), "chunk_experts": C}
    for n in TOKENS:
        streams = 4 * n * d * 2 / GB             # teacher in/out, student in, the other chunks' output (bf16)
        m1[f"host_gb_{n}"] = round(streams, 1)
        per_tok = flash_exp_flop + 2 * flash_fwd_layer
        m1[f"hours_fast_{n}"] = round(moe_layers * hours(per_tok, n, FAST_FLOPS), 1)
        m1[f"hours_slow_{n}"] = round(moe_layers * hours(flash_exp_flop, n, slow_flops), 1)
    m1["teacher_read_gb"] = out["flash"]["bf16_gb"]
    out["methods"]["M1_blockwise_sequential"] = m1

    # M2: end-to-end on the logits, ternary codes frozen, the per-row scales trained (EfficientQAT's E2E-QP).
    dequant = p["routed"] / moe_layers / 16 * 2 / GB   # a 16th of one layer's experts unpacked to fp16 at a time
    logits = 512 * h["vocab_size"] * 2 * 2 / GB     # a 512-token chunk of logits and their gradient, fp16
    m2_gpu = tern + rows * 16 / GB + 2 * (CTX + dequant) + logits + 0.5
    m2 = {"gpu_gb": round(m2_gpu, 1), "fits_t4x2": m2_gpu <= KAGGLE["gpu"], "margin_t4x2_gb": round(KAGGLE["gpu"] - m2_gpu, 1),
          "fits_3060_resident": m2_gpu <= LOCAL["gpu"], "host_local_gb": round(tern + 2, 1),
          "stream_gb_per_pass_3060": round(tern, 1)}
    for n in TOKENS:
        m2[f"hours_fast_{n}"] = round(hours(3 * 2 * active, n, FAST_FLOPS), 1)
        m2[f"hours_slow_{n}"] = round(hours(3 * 2 * active, n, slow_flops), 1)
    out["methods"]["M2_e2e_scales"] = m2

    # M3: end-to-end on the logits, latent weights of one window of experts at a time, the rest frozen ternary.
    W = 16
    windows = moe_layers * E // W
    m3 = {"window_experts": W, "windows": windows,
          "gpu_gb": round(m2_gpu + W * 3 * d * f * 8 / GB, 1),
          "hours_fast_1500000": round(windows * m2["hours_fast_1500000"], 0)}
    out["methods"]["M3_e2e_latent_windows"] = m3

    # M4: full QAT end-to-end, every latent weight trained (BitDistill's setting): bf16 latent + bf16 grad at least.
    m4 = {"state_gb_min": round(4 * fl["params_total"] / GB, 0),
          "fits_host_local": 4 * fl["params_total"] / GB <= LOCAL["host"],
          "disk_traffic_gb_per_step": round(6 * fl["params_total"] / GB, 0)}
    out["methods"]["M4_full_qat"] = m4

    # M5: block-wise with the loss read K frozen blocks downstream.
    K = 4
    m5 = {"downstream_blocks": K, "gpu_gb": round(m1_gpu + K * tern / moe_layers, 1)}
    for n in TOKENS:
        per_tok = flash_exp_flop + 2 * flash_fwd_layer + 3 * K * flash_fwd_layer
        m5[f"hours_fast_{n}"] = round(moe_layers * hours(per_tok, n, FAST_FLOPS), 1)
    out["methods"]["M5_blockwise_downstream"] = m5

    if a.rate:
        rt = json.loads(a.rate.read_text(encoding="utf-8"))
        fr = rt["shapes"]["flash"]
        assert fr["valid"], "the rate's flash shape spilled out of the GPU's memory"
        other = active - k * 3 * d * f * moe_layers                 # active weights outside the routed experts
        out["rates"]["measured"] = {"file": str(a.rate), "gpu": rt["config"]["gpu"], **{
            x: fr[f"{x}_ms_per_token_layer"] for x in ("block", "frozen", "scales")}, "other_active": other}
        for n in TOKENS:
            fwd_other = hours(2 * other, n, FAST_FLOPS)
            m1[f"hours_measured_{n}"] = round(moe_layers * n * fr["block_ms_per_token_layer"] / 3.6e6
                                              + 2 * fwd_other, 1)
            m2[f"hours_measured_{n}"] = round(moe_layers * n * fr["scales_ms_per_token_layer"] / 3.6e6
                                              + 3 * fwd_other, 1)
            m5[f"hours_measured_{n}"] = round(moe_layers * n * (fr["block_ms_per_token_layer"]
                                                                + K * fr["frozen_ms_per_token_layer"]) / 3.6e6
                                              + (2 + 3 * K / moe_layers) * fwd_other, 1)
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
