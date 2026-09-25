# NES-03: int8 router shortlist result

**Decision:** the opt-in int8 sketch + 32-candidate fp32 rescore preserves
the exact router's top-8 on the tested E128/E1280 inputs and cuts E1280 CPU
router time. The parallel configuration formally passes the frozen pilot
cost/quality gates; its E128 nonregression margin is only **0.12 µs/token**
against an older baseline, so it is a provisional cost candidate. It does
not repair E128's generation failure or transfer pretrained knowledge.

## Question, identity and work

The [frozen protocol](NES_03_INT8_ROUTER_SHORTLIST_PROTOCOL_20260925.md)
asks whether router traffic from NES-02 can be compressed without losing
the exact top-8. This cell keeps top-8, D256/L6/h128, experts, backbone,
validation split and output precision fixed. It changes only the router:
per-row int8 weights and per-token int8 input, an AVX2 integer dot over all
E rows, a 32-element min-heap, then exact fp32 scoring of those candidates.
The current C softmax over all E cancels when its selected top-8 weights are
renormalized; the new path normalizes only the final eight logits. The
default fp32 router remains available. `--router-parallel` additionally
spreads independent int8 row scores across six threads for E≥512.

Inputs are the same pinned E128 checkpoint/export, synthetic E1280 stress
export and phase55 token IDs as in the protocol. The offline probe captured
16×512 held-out positions beginning at validation offset 8192, six router
inputs per position, from the E128 PyTorch reference on local RTX 3060.
The input array is 53,480,290 B and SHA-256 is recorded in the
[probe JSON](nes03_int8_router_probe_20260925.json), copied byte for byte
from `results/native_expert_scaling/nes03_frozen16/probe.json`. The E1280 offline
arm applied its random synthetic router to those E128 inputs; C separately
audited E1280 on its **own** validation trajectory. No training or T4 run.

Reproduce the numerical probe and C build from the repository root:

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\probe_nes03_int8_router.py --ckpt results\native_expert_scaling\nes01_e128_final.pt --e128-export results\native_expert_scaling\nes01_e128_e4.bin --e1280-export results\native_expert_scaling\nes02_e1280_synthetic_e4.bin --out-dir results\native_expert_scaling\nes03_frozen16 --windows 16 --offset 8192 --device cuda:0
clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks\phase60\engine.c -o results\native_expert_scaling\engine_nes03_int8.exe -lm
```

The [probe tool](../../../benchmarks/native_expert_scaling/probe_nes03_int8_router.py)
verifies all source hashes, PyTorch-vs-offline fp32 top-8 agreement, and the
predeclared 32/64 shortlist ladder. The
[summary tool](../../../benchmarks/native_expert_scaling/summarize_nes03_router.py)
binds logs and decisions. Raw CPU/quality logs and the 53 MB input array
are local under `results/native_expert_scaling/`; their hashes are in the
[result JSON](nes03_int8_router_result_20260925.json). Capturing inputs took
about 5.0 s after load; the full probe command took about 22 s wall time.

## Numerical and quality results

In the offline 8,192-position probe, **every exact top-8 ID** is included
by the int8 32-candidate list in every E128 and E1280 layer. Its maximum
sketch rank for an exact top-8 ID is 11–14 across layers. The offline fp32
E128 scorer matches captured PyTorch top-8 IDs at ≥99.995% per layer,
exceeding the 99.9% apparatus requirement. The C audit, which uses the
actual self-consistent trajectory for each artifact, likewise records
**zero missed exact top-8 IDs** and zero changed route positions across
8,192 positions × 6 layers for **both** E128 and E1280. This is finite
held-out evidence; it cannot guarantee route fidelity for all inputs or a
trained E1280 model.

On the trained E128 C artifact, 20,480-token fp32/exact BPB is **0.829826**
with either router; top-1 agreement is **20,480/20,480**. The 16 frozen
128-token greedy continuations are byte identical to NES-01's fp32 control.
They therefore retain NES-01's **10/16** triple-8gram loops. All router
nonregression gates pass, but the model's original absolute generation gate
still fails. The result says the new router did not worsen this model; it
does not say that extra learned experts preserved useful quality.

## CPU cost and decision

All timings used the Ryzen 5 3600X, clang 21.1.8, six OpenMP threads,
byte ternary codes, LUT experts, fast SSM exponential, 3,000 validation input
tokens, four separate invocations and median of the last three. No other
CPU benchmark ran concurrently. The [raw result JSON](nes03_int8_router_result_20260925.json)
contains every repeat and log hash. The matched fp32 controls were rerun
after the int8 arms because the E128 threshold was sensitive to machine
variation; frozen NES-02 thresholds were not changed.

| Median µs/token | NES-02 fp32 | Matched fp32 | int8 serial | int8 parallel |
|---|---:|---:|---:|---:|
| E128 total | 1,148.4 | 1,208.0 | 1,264.7 | 1,205.7 |
| E1280 router+selection | 536.1 | 558.5 | 165.2 | **116.8** |
| E1280 total | 1,880.4 | 1,974.5 | 1,496.1 | **1,474.2** |

Against frozen NES-02 thresholds, int8 **serial fails** only E128 total
(1.101× baseline, allowed ≤1.05×); its E1280 router and total pass.
The **parallel** arm passes nominally: E128 total 1.04990×, E1280 router
0.21787×, E1280 total 0.78398×. Its E128 result is practically tied with
the contemporaneous fp32 control (1,205.7 vs 1,208.0 µs) and the old
baseline varies by about 5% in this cycle. The E128 cost gate is therefore
fragile. The E1280 router improvement is much larger than this drift:
parallel int8 is ~0.209× the matched fp32 router and ~0.747× matched total.

The extra int8 router sketch occupies **202,752 B** at E128 and
**2,027,520 B** at E1280, including row scales/sums. This prototype still
loads the full fp32 router and redundant fp32 expert reference weights; the
3.813 GB synthetic E1280 file is not a compact production export. The
measured rates are validation input-token rates, not accepted autoregressive
rates. At E≈100B, the extrapolated int8 router rows alone would still be
roughly 268 MB; no 100B throughput measurement exists.

**Next decision:** retain `--router-int8-shortlist 32` as an optional,
quality-checked router candidate. A packed-only format and learned large-E
quality are still required to make expert count a practical RAM dial. The
current research priority shifts to an actual pretrained-to-target transfer
cell, with donor-relative quality and an active-byte target; further router
micro-optimization alone cannot complete the goal.
