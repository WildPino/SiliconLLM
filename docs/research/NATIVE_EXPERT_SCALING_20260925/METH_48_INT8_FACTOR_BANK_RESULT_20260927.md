# METH-48: per-expert int8 factor bank fails top-1 retention

The [frozen protocol](METH_48_INT8_FACTOR_BANK_PROTOCOL_20260927.md) bound
METH-47 update 512, the Instruct donor and the previously viewed METH-45
external inputs before this run. The [runner](../../../benchmarks/donor_adaptation/s1/meth48_int8_factor_bank.py)
exported both trained rank-8 factors in every E128 expert with separate
symmetric int8 scales per expert and factor, reloaded the stored codes and
scales exactly, and kept the donor and router unchanged. It reproduced all
12 saved original-student greedy continuation streams before scoring the
packed arm. This is a conversion diagnostic on a **reused** set, not a new
independent quality promotion.

| Fixed measure | Original → packed result | Gate |
|---|---:|---|
| Pooled document BPB | 1.355188 → 1.355183; Δ −0.0000054 | ≤+0.002, pass |
| Worst category BPB Δ | code +0.0001002 | ≤+0.002, pass |
| Prompt top-1 agreement | **2,004/2,091 = 95.839%** | ≥99%, **fail** |
| Greedy EOS / repeated 8-gram | 12/12 → 12/12 EOS; 0 → 0 loops | pass |
| Exact greedy continuations | 7/12 | reported, not gated |

Five continuations change, including a PG19 response that grows from 14
to 51 tokens. The small pooled BPB difference does not protect token ranking
or generated content. Relative squared factor reconstruction error is
7.61e-5 for `a` and 8.56e-5 for `b`; the inference path is more sensitive
than those weight-space errors imply. This does not isolate whether the
changes arise directly in factor output or through later routing decisions.

The saved [factor artifact](../../../results/native_expert_scaling/meth48_e128_factor_int8.safetensors)
is 44,073,128 bytes, SHA-256
`2f0985258e783daaf13edc311c8932c3294546c67d9fcda834a86c2f6cfa74d8`.
The exact tensor payload is 344,256 bytes per expert across 24 layers:
344,064 code bytes plus 192 scale bytes. At unchanged L24/D896/rank8,
E27,355 projects to 9,417,122,880 payload bytes and E273,547 projects
to 94,170,196,032 bytes. The latter is a **hypothetical** ~100B parameter
expert count, not a built or trained model; the donor, router/index and
runtime workspace add RAM. These are storage figures, not CPU bandwidth
or 50 tok/s results. At fixed top-4, selected factor code bytes per token
are 1,376,256 in either E case, before scales, LUT preparation, cache
behavior and other model traffic.

The [machine result](meth48_int8_factor_bank_result.json), SHA-256
`13e3e41e83080f6cafde890c7ed2c93026840abb286e658f23c988dcf7514238`,
contains per-document scores and both continuation streams. Command:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth48_int8_factor_bank.py --artifact results/native_expert_scaling/meth48_e128_factor_int8.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth48_int8_factor_bank_result.json
```

The measured run took 132.047 s in the runner, peaked at 2.437 GB GPU
allocation and 4.155 GB RSS on the local RTX 3060; no T4 was used.

**Decision:** reject per-expert-scale int8 factors for this checkpoint
under the registered ranking gate. Try a finer factor scale granularity
under a separately frozen protocol. A passing factor export would still
need a CPU LUT measurement and a bounded router; METH-48 changes neither
the unresolved semantic audit nor distinct large-E training.
