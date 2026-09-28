# METH-127: full C E1280 composition works; FP32 core misses 50 tokens/s

**Decision.** The existing Qwen C runtime now executes the donor core,
attention/RoPE/KV cache, dense FFN, hierarchical E1280 route, exact
METH-126 centered factors and logits together. On the pinned FP32 reference
core, its composed logits match an independent PyTorch reconstruction of the
**same stored core and factor bank** at all eight bound prompt positions
(top-1 8/8; worst relative L2 `9.55e-6`). A longer 64-position check
also matches top-1 at 64/64, with worst relative L2 `3.69e-4`.
Sixty-four actual greedy decode
steps with no EOS run at **16.818 tokens/s** on six Ryzen 5 3600X threads,
below the required 50 tokens/s. This is an end-to-end *reference* path;
the quality-valid METH-123 candidate uses a BF16 donor core, so its quality
cannot be transferred to this FP32-arithmetic native assembly without a
separate full quality audit. The intended compact native `engine.c` result
remains open.

The [protocol](METH_127_FULL_C_REFERENCE_PROTOCOL_20260928.md) was committed
at `97d2471` before export and timing. The BF16 donor source is
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The [exporter](../../../benchmarks/donor_adaptation/engine/qwen_export.py)
widens stored BF16 values to FP32 without folding norms. It needed a
compatibility repair: the installed Transformers exposes default RoPE's
theta under `rope_parameters`, not `rope_theta`; nondefault RoPE is now
rejected. The exported 1,976,131,124-byte core has SHA-256
`6b2be143303510f15785783542649026b719488407f48b350de67f429e206029`;
its [sidecar](meth127_core_export.json) records the layout and export flags.
The E1280 shared-A bank is 496,779,304 bytes, SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`.
The pair occupies 2,472,910,428 file bytes before runtime workspace.

The first eight tokens of METH-121's first external prompt were fixed at
`[151644, 8948, 198, 2610, 525, 1207, 16948, 11]`, little-endian int32
SHA-256 `b9ffb0827e01fec4ef13027d46e04a61421d2ef62c571cc3827aa049bd3013ed`.
The [full parity runner](../../../benchmarks/native_expert_scaling/meth127_full_c_parity.py)
verifies source/core/bank/ID hashes and compares the C outputs to PyTorch:

| Eight-position parity | Dense FP32 | FP32 core + BF16-value E1280 factors |
|---|---:|---:|
| Top-1 agreement | 8/8 | 8/8 |
| Worst relative logit L2 | `4.28e-6` | `9.55e-6` |
| Maximum absolute logit difference | `6.25e-5` | `1.04e-4` |

The [machine result](meth127_full_c_parity_result.json) holds every position.
The additional [64-position result](meth127_full_c_parity64_result.json)
binds the first 64 tokens of the same prompt, int32 SHA-256
`990924bdcf23204fc28ac091e960834dc775efc09cd6ce9a38f73f307ee13db7`.
The dense arm matches 64/64 top-1 with worst relative L2 `1.05e-5`;
the composed arm matches 64/64 with worst relative L2 `3.69e-4`.
The larger composed error at positions around 39–50 is evidence that
full-trajectory numeric differences accumulate; it is not a 64-token
quality gate or bitwise equivalence claim.
The PyTorch composed reference reads the **M126 bank itself**, computes the
route from BF16-rounded post-attention input, executes BF16 A/B/gate
arithmetic and adds the residual to the FP32 dense FFN output. Its run used
2.504 GB peak RTX 3060 allocation and 4.248 GB process RSS in 13.0 s.
Native C's dense and composed outputs are finite, and the bank changes the
logits (maximum absolute change 1.860). A one-byte corrupted bank magic
returns exit 1 with a format mismatch, as retained in the
[negative log](meth127_negative_raw.log).

The same executable's 64-token greedy `--generate` mode measured actual
decode, not a synthetic token stream. Neither arm emitted EOS within 64
tokens, and each arm's generated ID stream is identical at one and six
threads. Prompt prefill is excluded from these decode rates:

| Native greedy decode | Dense | Centered E1280 |
|---|---:|---:|
| One thread | 13.203 tok/s | 12.296 tok/s |
| Six threads | 18.244 tok/s | **16.818 tok/s** |
| Peak process RSS, six threads | 1,983,250,432 B | 2,480,058,368 B |

The raw generation logs are
[dense/1](meth127_dense_t1_generate_raw.log),
[E1280/1](meth127_centered_t1_generate_raw.log),
[dense/6](meth127_dense_t6_generate_raw.log), and
[E1280/6](meth127_centered_t6_generate_raw.log).
The matched [six-thread profile](meth127_centered_t6_raw.log) on 64 decode
steps charges 39.397 ms/token to the FFN, 14.767 to the tied head,
2.801 to QKV, 2.136 to O projection and 4.161 ms/token **inside FFN**
to routing plus selected-factor residual. Its wall time is
58.595 ms/token; the [dense control](meth127_dense_t6_raw.log) runs
18.23 tok/s and charges 35.417 ms/token to FFN. The
[one-thread E1280 profile](meth127_centered_t1_raw.log) and
[dense profile](meth127_dense_t1_raw.log) are retained. Profiling and
generation were separate sequential runs, so their exact times differ.
The full FP32 core's approximately 1.98 GB active stream alone makes the
20 ms/token target implausible at the observed memory rate. The measured
expert addition is about 4 ms/token; reducing router cost alone cannot
close the 39 ms/token wall-time gap to 50 tok/s.

Build and export:

```powershell
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp -std=c11 -Wall -Wextra benchmarks/donor_adaptation/engine/donor_engine.c -o benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth127_donor_engine.exe -lm -lpsapi
.venv/Scripts/python.exe benchmarks/donor_adaptation/engine/qwen_export.py --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --quant fp32 --fold none --load-dtype bfloat16 --out benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth127_qwen05b_instruct_fp32.bin
```

The C flag is `--factor-bank <meth126_shared_a_factor_bank.bin>` alongside
`--weights`, and the same pair is used for `--logits`, `--bench` and
`--generate`. This reference preserves the full Qwen architecture and FP32
active traffic. A final conversion still needs a quality-valid compact
core, LUT factor arithmetic if it is beneficial, native integration in the
target engine, quality and >=50 accepted tok/s on one artifact, plus
transfer across donor families and 10B/100B scale.
