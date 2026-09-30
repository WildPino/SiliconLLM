# METH-189/190: rank-64 FFN distillation worsened the stored core

The [METH-189 frozen protocol](METH_189_Q6_FFN_CORRECTION_PROTOCOL_20260930.md)
and [trainer](../../../benchmarks/native_expert_scaling/meth189_train_q6_ffn_correction.py)
were committed at `911fa1c` before execution. The 256-update single
run completed on the local RTX 3060 in 252.063 s with peak 7.144 GB
GPU allocation and 2.574 GB ending RSS. The 72 rank-64 A/B pairs
were serialized as BF16 and read back exactly: 53,084,160 tensor bytes,
53,097,832 physical file bytes, SHA-256
`8bccbecfc4090814cf816eb7a710da431a53315de6a9061a9036c64df92e601c`.
The ideal core+router+correction addressed ledger is 534,619,136
bytes/token, under the 560 MB design allotment by 25,380,864 bytes.
This is arithmetic, not a measured native rate.

The [training report](meth189_q6_ffn_correction_train_result.json),
SHA-256 `e61d45e728f12af50454a8d4776131b898adc7524a1f6f1090a9c3f44830791c`,
records every seeded draw, raw/chat objective and gradient norm.
The objective was unstable across examples and the final chat objective
was 0.378 at update 256 versus 0.009 at update 1. These are different
examples, so the comparison alone does not establish divergence; the
fixed source test below does establish quality regression.

The [METH-190 scorer](../../../benchmarks/native_expert_scaling/meth190_q6_ffn_correction_development.py)
was committed at `fea9d5d` before execution. It verified per-source
NLL and top-1 parity for both BF16+E1280 and uncorrected Q6+E1280
against METH-187, then loaded the exact BF16 correction checkpoint.
It used only the already viewed METH-121 sources.

| Same 24-source measure | BF16+E1280 | Q6+E1280 | Q6+E1280+rank64 |
|---|---:|---:|---:|
| Pooled BPB | 1.184887 | 1.187487 | 1.202344 |
| Donor prompt top-1 | 94.504% | 89.186% | 73.075% |

The corrected checkpoint is +0.017456 BPB and −21.429 points top-1
relative to BF16+E1280. It is +0.014857 BPB and −16.110 points versus
the uncorrected compact core. **All four frozen METH-187 pooled/category
development gates fail.** The [raw result](meth190_q6_ffn_correction_development_result.json),
SHA-256 `b28863a2dbbc0ac1754a1b680f601ab0cd892f106e24e9682354cbfac86d9048`,
includes every source/category value. Scoring took 44.515 s with
6.995 GB peak GPU allocation. No new source or T4 was used.

Do not run a fresh quality or full native-rate test on this failed
checkpoint. The evidence isolates an unstable 256-step, learning-rate
3e-4 joint A/B distillation of the FFNs; it does not rule out a
better constrained optimizer or a different correction geometry.
Any retry must fix a training-only selection and stopping rule before
examining the viewed development sources, and must charge its actual
stored bytes. There is no evidence yet for a quality-valid 50 tok/s
artifact or 10B/100B transfer.
