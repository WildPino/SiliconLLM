# METH-208: raw failing-layer count exceeds conditional fit variation

The [protocol](METH_208_LOAD_SAMPLING_PROTOCOL_20260930.md) and
[runner](../../../benchmarks/native_expert_scaling/meth208_load_sampling.py)
were frozen at `b1c981a`. The exact METH-207 router is replayed on
the 1,024 consumed raw fit and 256 consumed raw reserved sequences.
Eight-prompt BF16 teacher/control parity, all saved raw-cell metrics
and gates, per-sequence count sums and max-load reconstruction pass.
The small CPU helper check passes sequence-count reconstruction
and direct int64 bootstrap parity. In the actual run, the first
three replicates per layer also match direct int64 accumulation.

Two thousand and forty-eight 256-sequence bootstrap replicates use
seed 208208 and identical sequence weights across all 24 layers.
They preserve within-sequence route correlation. The router remains
frozen, so this is empirical fit-input variation conditional on a
fitted representation and biases, not a predictive distribution
that includes calibration-estimation uncertainty or unknown shift.

| Statistic across the 24 layers | Fit-bootstrap median | 95th percentile | 99th percentile | Reserved observed |
| --- | ---: | ---: | ---: | ---: |
| Worst max-load ratio | 1.281207 | 1.398592 | 1.494947 | 1.301374 |
| Number of layers failing <=1.25 | 1 | 4 | 5 | 6 |

A worst ratio at least 1.301374 occurs in 35.352% of replicates;
that maximum alone is compatible with this fit distribution.
Six or more failing layers occur in only 0.830% (17/2,048),
and the joint worst-ratio/failing-layer event in 0.781% (16/2,048).
The all-layer pass fraction is 27.832%. Thus the small-cohort gate
has appreciable fit-sampling variation, but the observed failure
count exceeds the frozen 99th-percentile criterion. Layers 5,
11, 16 and 20 individually exceed their conditional 99th ratio
percentile; these per-layer tails are descriptive, not independent
tests. No source cell is opened and no threshold is relaxed.

The [result](meth208_load_sampling_result.json) SHA-256 is
`7d069774448ed3fd92e6cc412d38ec99471644671d2b984ac16a211640d77c2d`.
The local count artifact `results/native_expert_scaling/meth208_sequence_route_counts.npz`
has SHA-256 `3f01cc69e0fc3b4e69bfcfdf16e706f44f3fd389cef564a79c42e34a5ee08c1c`,
716,571,678 physical bytes and exact readback. It retains fit/reserved
uint16 sequence route counts, bootstrap weights and every layer/replicate
ratio, allowing subsequent count diagnostics without another model replay.

Command:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth208_load_sampling.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth208_load_sampling_result.json --counts results/native_expert_scaling/meth208_sequence_route_counts.npz
```

Runtime was 236.125 s on the RTX 3060, peak allocated GPU memory
2.945 GB and ending RSS 5.027 GB. No T4, new fitting, chat replay,
new quality text or native performance measurement.

**Decision:** METH-207 stays failed. This conditional diagnostic
does not establish stable population drift or rule out bias-estimation
uncertainty, but its frozen rule favors changed calibration over simply
dismissing the six-layer failure as fit-sample noise. The next bounded
candidate should increase independent calibration support while
retaining the exact keys/projection/schedule and original thresholds,
exclude all 256 old reserved draws from fitting, and reserve an
unopened routing cohort. Require the old reserved gates as well;
no specialist training or native promotion until route screens pass.
