# METH-129: the native two-pass head preserves chosen tokens and improves CPU decode

**Decision.** Retain the R8-proposal plus exact-FP32-row primitive for the
next compact-core experiment. On the bound native FP32-core/E1280 assembly,
64/64 prompted choices and all 64 greedy continuation tokens match the
full FP32 tied head. In the final matched run, six-thread greedy decode
rises from **17.141 to 20.791 tokens/s (+21.3%)**, exceeding the frozen
5% relative gate. An earlier full run gave 17.022 to 21.123 tok/s
(+24.1%); both runs clear the gate.
The full-model >=50 accepted-token/s gate remains open. These are
native-reference parity and cost results, not independent BF16 donor
quality or full-vocabulary probability parity.

The [protocol](METH_129_NATIVE_EXACT_HEAD_PROTOCOL_20260928.md) was
committed at `16f4b42` before implementation. Its core SHA had an extra
`f1` transcription; the SHA check stopped the apparatus before any timed
run. The protocol and runner now use the 64-character hash from the
stored METH-127 exporter sidecar and independently rehashed core,
`6b2be143303510f15785783542649026b719488407f48b350de67f429e206029`.
No source, K, prompt, threshold or measurement changed in this correction.

The [sidecar exporter](../../../benchmarks/native_expert_scaling/meth129_export_head_sidecar.py)
verifies the METH-59 source SHA and donor/parent metadata, writes exactly
the stored int8 codes and FP32 row scales, then checks their bytewise
readback. The sidecar is 136,742,416 bytes including a 16-byte header;
SHA-256 `5df1038b7224cdc44b9c62686780e08162c9e5da1ab8c998d16d999ade74dd43`.
The [C runtime](../../../benchmarks/donor_adaptation/engine/donor_engine.c)
scans all 151,936 R8 rows, keeps K=64 by approximate score and ID,
recomputes only those rows with the original tied FP32 matrix and the
same accumulation kernel as the full head, then chooses by exact score
with lowest-ID tie breaking. `--head-choice` emits chosen IDs; the
shortlisted mode rejects `--logits` and `--bpb` because remaining
vocabulary scores are approximate. Greedy generation uses the rescored
ID rather than an argmax across mixed approximate and exact logits.

The [gate runner](../../../benchmarks/native_expert_scaling/meth129_native_head_gate.py)
checks the core, bank, sidecar and input hashes, verifies that a
one-byte magic corruption fails load, compares 64 native chosen IDs
with argmax on METH-127's full-head logits, then runs four sequential
64-token greedy cells. The [machine result](meth129_native_head_result.json)
contains executable/output hashes, exact rates, per-organ timing and
peak RSS; the [export result](meth129_head_export.json) records source
binding and readback. The 64 choices have zero mismatches. Both
two-pass generations match the METH-127 full-head stream byte for byte;
there is no EOS in the 64 new tokens.

| CPU threads | Full FP32 head | Two-pass head | Relative increase | Full head ms/token | Two-pass head ms/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.092 tok/s | 13.970 tok/s | 15.5% | 20.763 | 8.971 |
| 6 | 17.141 tok/s | **20.791 tok/s** | **21.3%** | 14.491 | 4.179 |

On the six-thread two-pass cell, FFN still takes 38.637 ms/token and
dominates the 48.098 ms/token decode wall. Nominal head addresses
fall from 544,538,624 bytes for the full FP32 scan to 136,971,776
bytes for the R8 code/scale scan plus 64 FP32 rows. This is an
arithmetic address count, not a hardware DRAM counter. The original
FP32 matrix remains resident; adding the sidecar raises measured peak
RSS from 2,480,066,560 to 2,616,807,424 bytes in the six-thread
cells. The final four-cell gate plus hash checks took 33.609 seconds,
within the registered local budget; no T4 was used.

This supports a head primitive for greedy selection on this finite
native reference, with an extra 136.7 MB resident sidecar. It does not
establish likelihood, task quality, grounded generation on fresh
sources, or token parity with the BF16 donor. The decisive next step is
to combine this head with a quality-valid compact body and the exact
E1280 bank, test new disjoint documents and semantics, then measure the
same native artifact end to end. The RAM-scaled expert-count ladder and
CPU router/LUT quality and cost at 10B/100B analogues remain open.
