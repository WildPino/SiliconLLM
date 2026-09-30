# METH-206: annealing closes pooled argmax imbalance, domain gates still fail

The [protocol](METH_206_ANNEALED_BIAS_PROTOCOL_20260930.md) and
[runner](../../../benchmarks/native_expert_scaling/meth206_annealed_bias.py)
were frozen at `b98f177`. The runner reuses the unchanged METH-204
collection/evaluation apparatus with exact stored projection/keys,
starts from its biases and applies the fixed four-temperature schedule
with 100 updates per temperature. Eight-prompt BF16 parity and every
saved-router raw/chat layer baseline metric reconcile. Projection
and key arrays remain byte-identical in the exported artifact.

The pooled hard routing changes decisively as temperature decreases:

| Temperature | Range of worst pooled hot-parent share across layers | Range of pooled max-load ratio |
| --- | ---: | ---: |
| 0.05 | 17.406–73.769% | 1.034–1.792 |
| 0.01 | 12.571–65.063% | 1.003–1.078 |
| 0.002 | 11.832–24.905% | 1.001–1.008 |
| 0.0004 | 11.654–17.208% | 1.001–1.005 |

At the terminal temperature no pooled hot parent exceeds 25%,
and all pooled max-load ratios are below 1.25. This validates the
specified correction of the pooled soft/hard surrogate mismatch.
It does **not** pass the original separate-cell routing gates:

| Fit cell | Layers failing load ratio | Layers failing hot-parent share | Worst hot-parent share | Worst load ratio |
| --- | ---: | ---: | ---: | ---: |
| Raw | 14/24 | 21/24 | 45.019% | 1.992 |
| Chat | 12/24 | 15/24 | 40.698% | 1.694 |

Every layer still passes content-score advantage, un-biased argmax
agreement and >=4,000 coverage. Terminal argmax agreement ranges
43.51–67.06% raw and 42.02–69.10% chat. The fixed stop rule leaves
reserved and source-transfer cells unopened. The result isolates
the remaining raw/chat mixture effect under one shared bias bank;
no schedule or threshold was selected using held-out data.

The [result](meth206_annealed_bias_result.json), SHA-256
`464db925b3a6d2367a6112b4cd1a8d4d41225255c56c3a5a6c8ddc225cf4e6ca`,
includes all stage summaries and the [unchanged apparatus output](meth206_annealed_bias_result.apparatus.json)
hash. The local router at `results/native_expert_scaling/meth206_annealed_content_router.npz`
has SHA-256 `fe35eafe30683188713bd4630c4085488057d5790e3bdfab0d8539475bc1d343`
and exact readback, with the same 3,886,080 payload bytes as METH-204.
Runtime was 412.469 s on the RTX 3060, peak allocated GPU memory
2.945 GB and ending RSS 4.323 GB. No T4, new quality text or B
training was used.

**Decision:** preserve the annealed calibration mechanism as a pooled
load result, but stop this single-bias router before specialist
training. The next bounded candidate should condition calibration
on a causally observable chat/text mode, or fit a cell-robust single
bias; serving mode must not use source IDs or future labels. It
still needs the original reserved and source gates, native CPU
cost, useful distinct child functions and untouched model quality.
