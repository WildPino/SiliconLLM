# METH-209: tripled raw support halves failures but does not pass

The [protocol](METH_209_EXPANDED_RAW_BIAS_PROTOCOL_20260930.md) and
[runner](../../../benchmarks/native_expert_scaling/meth209_expanded_raw_bias.py)
were frozen at `0325c1a`. Raw bias calibration expands from 1,024
to 3,072 source-distinct METH-175 draws, excluding all old and new
reserved draws. Keys/projection/annealing and thresholds stay fixed;
the passing METH-207 ChatML bank remains byte-identical. Old chat
fit/reserved routing evidence is explicitly reused by bank identity,
with the whole-bank maximum-bias diagnostic recomputed.

The draw-partition and evidence-copy helper checks pass. The actual
run passes eight-prompt BF16 parity, original raw-fit and raw-reserved
metric reconciliation under the old bank, finite calibration, exact
export readback and ChatML-bank identity. The larger raw fit contains
390,144 tokens and 1,529,452 content selections per layer.

| Raw cell | Layers failing max-load ratio | Worst max-load ratio | Worst hot-parent share | Minimum coverage |
| --- | ---: | ---: | ---: | ---: |
| Expanded fit | 0/24 | 1.005557 | 16.261% | 10,797 |
| Old reserved | 3/24 | 1.303079 | 23.615% | 7,110 |

Both fit cells pass all five gates. The unchanged old raw reserve
still fails <=1.25 load ratio in zero-based layers 2, 11 and 16,
versus six failures in METH-207. Its other four gates pass everywhere:
minimum standardized score advantage 0.9220 and unbiased argmax
agreement 40.72%. Reused old reserved chat still passes. The worst
raw ratio does not improve (1.301374 to 1.303079). Under the frozen
stop rule, the 512 new pairs and source-document cells stay unopened.

The [result](meth209_expanded_raw_bias_result.json) SHA-256 is
`63e31b8f2c4528b913bb110cc506c5e0b0e5c9ff46e687cba6478787ac5e1030`.
The local router `results/native_expert_scaling/meth209_expanded_raw_router.npz`
has SHA-256 `3cb1e5f44fa5038bec76aba9f26f86d2a65c813c56e8830ca453486db3700f97`,
4,992,784 physical bytes and exact readback. Store and ideal addressed
payload are unchanged from METH-207 (4,992,000 stored array bytes;
2,783,616 ideal addressed bytes/token). No native cost was measured.

Command:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth209_expanded_raw_bias.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth209_expanded_raw_bias_result.json --router results/native_expert_scaling/meth209_expanded_raw_router.npz
```

Runtime was 548.328 s on the RTX 3060, peak allocated GPU memory
2.945 GB and ending RSS 4.722 GB. Capture/calibration budget checks
observed up to about 9.105 GB RSS; this is not a continuous peak-RSS
measurement. No T4, expert-B training or new quality text.

**Decision:** stop the shared-key/bias-only calibration sequence;
increasing support alone did not clear its declared route screen.
A changed conditional key geometry is the next expert-count proposal,
with selected-parent key bytes and CPU cost explicitly priced.
Do not retry another sample-size/temperature adjustment or promote
this bank. The distinct useful specialist-learning gate remains open.
Advance the separate compact-core path through Q8 embedding/head
attribution before assuming a correction format. Quality, native
rate and multi-family transfer still require actual evidence.
