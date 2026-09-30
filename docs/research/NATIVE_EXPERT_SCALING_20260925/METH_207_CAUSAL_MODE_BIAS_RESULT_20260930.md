# METH-207: causal mode biases pass fit, raw reserved max-load fails

The [protocol](METH_207_CAUSAL_MODE_BIAS_PROTOCOL_20260930.md) and
[runner](../../../benchmarks/native_expert_scaling/meth207_causal_mode_bias.py)
were frozen at `6851040`. The experiment retains METH-204's projection,
nine shared content keys and structural path, but calibrates separate
text/ChatML parent biases using METH-206's fixed annealing schedule.
The selected bank is determined only by the observed leading token
(`151644` selects ChatML), with that state carried across windows.
Cell names and future tokens do not select the evaluation bank.

The CPU evaluator check passes exact single-bank metric parity with
METH-204, mixed-bank selected-count accounting and persistent prefix
mode. The actual run passes eight-prompt BF16 teacher/control parity,
every saved-router fit baseline metric reconciliation, finite checks,
exact artifact readback and unchanged projection/key checks.
All 1,024 fit and 256 reserved raw sequences select text mode; all
1,024 fit and 256 reserved chat sequences select ChatML mode. This
validates prefix selection on these inputs only.

| Cell | Layers failing max-load ratio | Worst max-load ratio | Worst hot-parent child share | Coverage range |
| --- | ---: | ---: | ---: | ---: |
| Fit raw | 0/24 | 1.005450 | 17.123% | 10,001–11,412 |
| Fit chat | 0/24 | 1.006390 | 11.824% | 9,884–11,444 |
| Reserved raw | 6/24 | 1.301374 | 23.615% | 6,994–9,890 |
| Reserved chat | 0/24 | 1.222125 | 21.756% | 7,297–10,173 |

Every cell/layer passes the other four original gates: hot-parent
share <=25%, coverage >=4,000, score advantage >=0.05 and unbiased
argmax agreement >=15%. Reserved raw fails the unchanged <=1.25
max-load limit in zero-based layers 2, 5, 11, 15, 16 and 20.
Reserved score advantage is 0.920–1.457 raw and 0.928–1.461 chat;
unbiased argmax agreement is 40.64–65.99% raw and 43.00–66.57% chat.
The source-document cells stay unopened under the frozen stop rule.

This establishes the causal mode correction of the fit-domain mixture
problem, and a complete reserved-chat route pass. It does not pass
the router screen. The magnitude and six-layer scope of the remaining
raw max-load failure warrant a sample-variation diagnostic before
another fit change. Whether 256-sequence max-load variation explains
the failure is an **untested hypothesis**; these results do not relax
the frozen gate or reclassify the failed run as successful.

The [result](meth207_causal_mode_bias_result.json) SHA-256 is
`4e7910008f3bc07d80fd48a22c89d2430e3199b5b9116e651623ae7610540558`.
Its [apparatus output](meth207_causal_mode_bias_result.apparatus.json)
SHA-256 is `2cdaa78d5e24862af8a6df63651465eb7f6b522dc2c16a4df164cbdac50d2e97`.
The local router `results/native_expert_scaling/meth207_causal_mode_router.npz`
has SHA-256 `a5b367cc5902e745d8929d4c206161c698ad5610f9cfb9cae181f1eceadf3701`
and 4,992,784 physical bytes. Its 4,992,000-byte array payload
includes 2,211,840 bias bytes, doubled from METH-204. Tenfold parents
would store 22,118,400 bias bytes. Ideal addressed projection/keys/
selected-mode bias remain 2,783,616 bytes/token; native time and
actual DRAM traffic are unmeasured.

Command:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth207_causal_mode_bias.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth207_causal_mode_bias_result.json --router results/native_expert_scaling/meth207_causal_mode_router.npz
```

Runtime was 510.000 s on the RTX 3060, peak allocated GPU memory
2.945 GB and ending RSS 4.372 GB (not a peak-RSS measurement).
No T4, expert-B training, new quality text or native rate test.

**Decision:** stop before source/native/training promotion. Replay
the exact router on the already consumed raw fit/reserved sequences,
retain per-sequence route counts, and compare the reserved max-load
to a frozen draw-level 256-sequence sampling distribution from fit.
That diagnostic may choose between a sampling-aware new adjudication
protocol and a changed calibration mechanism; it cannot retrospectively
pass METH-207. Distinct useful child functions, compact-core quality,
same-artifact >=50 accepted tok/s and 10B/100B transfer remain open.
