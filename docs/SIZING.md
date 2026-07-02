# Scale-up sizing

How the two-pool memory model prices out. **This page separates what is *measured* (at 5M
sandbox scale, single-thread) from what is *projected* (scale-up, pending a target config).
No projected tok/s is asserted without the arithmetic and its assumptions.**

## The two pools (from probe-3 + probe-4)

| pool | what lives here | size target | bandwidth | reused? |
|---|---|---|---|---|
| **Resident** | SSM backbone + router + head (+ norms/emb) | **≤ 16 MB** (L3-per-CCX) | ~100 GB/s (compute-bound) | every token |
| **Streamed** | the selected MoE experts | (grows with total params) | ~28 GB/s (DRAM floor) | per token, bulk-sequential |

The keystone budget (probe-3): the resident slice must stay **≤ 16 MB**, ≈ **24–32M ternary
active params/token**. The streamed pool is *not* residency-limited — it is priced by bytes
moved at the DRAM floor, and is ρ-safe only because experts are **contiguous KB chunks** loaded
in bulk per block (not a strided gather).

## Measured anchor (engine E4, `moe_gran.pt`, 5M sandbox)

- expert = 98,304 ternary weights + per-row scales ≈ **48 KB** at the current **4-bit** g=2 codes
- streamed bytes/token = top-8 × ~48 KB × 6 layers = **2400 KB/token** (counted in-engine)
- priced at the 28 GB/s floor = **~88 µs/token** for the streamed pool alone
- a dense **trit-pack (1.6 bit/weight)** would cut this ~2.4–2.5× (queued engine optimization)

## The formula (to project to a real scale)

```
streamed_bytes_per_token ≈ n_layers · top_k · params_per_expert · bytes_per_weight · (1 − skip)
streamed_time_per_token  ≈ streamed_bytes_per_token / eff_dram_bw        (floor ≈ 28 GB/s)
resident_bytes           ≈ (backbone + router + head) · bytes_per_weight  (must be ≤ 16 MB)
tok/s (streamed-bound)   ≈ 1 / max(streamed_time, resident_compute_time)
```

with `bytes_per_weight` = 0.5 (4-bit codes, today) or 0.2 (1.6-bit trit-pack, queued).

## Projection — PENDING the Architect's target config

To fill the 1B / 3B rows we need a chosen scale-up architecture: **n_layers, n_experts, top_k,
d_model, d_expert**. Those pick both the resident footprint (must fit 16 MB) and the streamed
bytes/token (the formula above). The framework is ready; the row values are not asserted here.

| model | resident (MB) | streamed KB/tok (4-bit / trit) | streamed µs/tok @28 GB/s | tok/s (est.) |
|---|---|---|---|---|
| 5M MoE (measured) | fits (cache-resident) | 2400 / ~1000 | ~88 | 702 (whole engine, cache-resident) |
| ~1B | *pending config* | *pending* | *pending* | *pending* |
| ~3B | *pending config* | *pending* | *pending* | *pending* |

## Explicit unknowns (do not hide these)

- **DDR4 multi-thread contention is NEVER measured.** Every number in this project is
  **single-thread**. The streamed pool at scale shares one DDR4 controller across cores; the
  effective per-core bandwidth under contention is unknown and could be well below the
  single-thread 28 GB/s floor. This is the largest open risk in the sizing.
- The 28 GB/s floor is this specific box's (dual-channel DDR4-3600-class); it is per-exemplar.
- Activation-quant cost (E2's per-token int8) is validated at 5M/D256; outlier channels may
  emerge with model size (per-group scales are the known escalation).
- Ternary quality cost (+0.028 BPB at 5M) is expected to *shrink* with scale, but that is a
  literature trend, not measured here at scale.

See [`SCALEUP_ARCHITECTURE.md`](SCALEUP_ARCHITECTURE.md) for the design and [`ENGINE_PLAN.md`](ENGINE_PLAN.md)
for the measured engine numbers this page builds on.
