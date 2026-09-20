# STRAT-01 GigaChat 3.1 — pilot MTP GGUF/CPU (18 settembre 2026)

## Status and scope of evidence

**REFERENCE_PILOT, not a final gate.** A pretrained GigaChat 3.1
Q4_K_M target and its BF16 MTP layer from the pinned revision performed
speculative decoding in the same CPU `llama.cpp` process. The draft was
loaded from a *separate* GGUF, invoked, and produced accepted tokens. There
is not yet evidence of Q4 target quality relative to BF16, MTP logit
parity relative to vLLM, clean-box rate with an optimized CPU build, or a port to
our `benchmarks/phase60/engine.c`. The ≥50 tok/s requirement **has not passed**.

A subsequent native x86 build enabled an initial paired comparison,
described below: the BF16 draft was slower than the control, and the greedy
sequence is not identical. This remains a **pilot n=1**, not a statistical
verdict or quality gate; the divergence must be explained before
acceptance is optimized.

This addendum supersedes only the earlier historical statements “no draft
GGUF written” and “direct-Q MTP graph still to be implemented” in the
[screen metadata](STRAT_01_GIGACHAT31_10B_METADATA_SCREEN_20260918.md)
and the [roadmap status](../../STRATEGIC_10B_20260916/STATUS_20260918.md).
It does not turn their theoretical bounds into measurements.

## Artifacts and provenance

- Producer Q4_K_M target: GGUF revision `97045b260251cfa86f5ad25638fa2dd074153446`, local file `benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf`, **6.474.702.976 B**, SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` verified locally. It has 26 blocks, 414 tensors, and does not include MTP.
- MTP source: BF16 checkpoint `ai-sage/GigaChat3.1-10B-A1.8B-bf16`, revision `189fff27a1dee68473960c3d5bca53e0e07a3191`; local 210-tensor sidecar, **1.614.456.643 B**, SHA-256 `c340ed41c1355441207357723b984fc336082e7c0e8a00deeb6f4fce15c46119`.
- Minimal HF bundle for the converter: `strat01_gigachat31_mtp_hf_bundle_189fff27_v1` under `benchmarks/donor_adaptation/density/results/`, with the sidecar hard-linked after SHA verification, root aliases for embedding/head, and distinct `model.norm.weight`. Index has 213 keys. The four JSON files are pinned to the revision and, in the builder's current version, also to the source SHA-256 values.
- Upstream `config.json` contains integer `routed_scaling_factor: 1`; the installed Transformers version requires a float. In bundle v1, the field was normalized to `1.0`, the same numeric value; the local manifest records source SHA-256 `6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503` and local SHA-256 `0c9f747879dcb35bac28ac4e864b849bd9995e0b8552405681878cb17d558dbe`. The builder now repeats the normalization and fails if the pinned hashes/field do not match. A live check of the four pinned JSON files passes: the config materialized by the builder has SHA `0651934efaaf8ce56b593aaeb50d80be84742a9ed40b3cf71c9c1da577af888d`, differing from v1 **only by a final LF** added by the first local edit; parsed JSON matches. Bundle v1 was materialized *before* this builder change: GGUF export from a fresh bundle has not been repeated.
- Local BF16 draft GGUF: `benchmarks/donor_adaptation/density/results/strat01_gigachat31_mtp_bf16_189fff27.gguf`, **1.620.709.600 B**, SHA-256 `20d80a73f833e3187adb8adda3e4f1a8d13e0500a796e69025e58473abb0f590`. It contains 23 tensors: 20 physical operations from layer `blk.26`, root `token_embd`, `output`, and `output_norm`; metadata `deepseek2.block_count=27`, `nextn_predict_layers=1`.

The patch `benchmarks/donor_adaptation/density/strat01_gigachat_llama_mtp.patch`
(SHA-256 `8676b2d32c0f7ada537ddb91c43f4997aa02a789378062bb2354419e8f2b522b`)
applies exclusively to `llama.cpp` commit
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. It enables DeepSeekV3
`--mtp` export, direct Q in the MTP graph, and the actual standalone-draft
path/parameters in the speculative loader. Without the third correction,
the previous CLI loaded the target even when a separate draft was specified.
The external temporary checkout also contains a **Windows LLVM-MinGW
build-only** adaptation (`CreateFileW`, `_WIN32_WINNT=0x0A00`), not included
in the research patch. CMake reported `GGML_SYSTEM_ARCH: UNKNOWN` and
`GGML_CPU_GENERIC`: do not use its rate as a CPU bound.

## Tests performed

The patched converter passed `--dry-run` and the real BF16 export; the
independent GGUF reader read the 23 names and `nextn_predict_layers=1`.
Target and draft match in common numeric metadata except for block count
(26 versus 27); the draft adds NextN. **25 offline tests** across the four
extractor, GGUF-contract, HF-builder, and patch suites pass.

| Attempt | Outcome | Interpretation |
|---|---|---|
| smoke, context 256 | 1.096-token chat prompt rejected; CLI remained interactive, targeted PID stopped | setup VOID; no decode |
| smoke, context 2048, 2 tokens | `Hello!` generation; standard log does not prove MTP was used | partial target check, not acceptance |
| verbose smoke, context 2048, 4 tokens | separate draft loaded, `draft-mtp` initialized and invoked, 1 proposal accepted out of 1 | **FIRES**, sample too small to estimate acceptance |
| first 64-token request | CLI rejects argument `una` due to PowerShell quoting; no model loaded | setup VOID, not negative evidence |
| corrected 64-token request | 23 proposals accepted out of 39 (58,974%), 64 output tokens, no runtime error | valid observation on **one prompt**, not generalizable |
| native x86 paired, greedy 64 tokens | baseline 12,78 tok/s; BF16 MTP 11,01 tok/s; 25/38 proposals accepted (65,789%) | one run per arm, **-13,85%** decode rate; outputs diverge, so this is not quality parity |
| repeated greedy baseline, 64 tokens | same 64 generated tokens as first baseline; slightly different timings | rules out gross non-repeatability of the control alone, does not establish the cause of MTP divergence |
| paired diagnostic, 48 tokens, default verbosity | texts diverge, but sampler-added lines are suppressed | diagnostic VOID: `llama-cli` defaults to error verbosity; do not repeat without `-lv 3` |
| paired diagnostic, 48 tokens, `-lv 3` | first difference at zero-based output token index 25, ID 4734 versus 4164 | tracks the divergence point; instrumented logging, timings not comparable |
| MTP diagnostic, 32 tokens, proposal suppressed at `n_gen=25` | same 25-ID prefix, then target again selects 4164 in single batch | the two-token batch at the divergence step alone is not a sufficient cause; accumulated state still differs |
| MTP diagnostic, 32 tokens, proposal allowed only from `n_gen=25` | first 25 IDs and top-2/margin summaries match baseline; at step 25 target selects 4164 in a two-token batch | differing batch shape at the divergence step **is sufficient** to invert top-1 in this control; does not rule out an accumulated effect in the other control |

Latest run: local log `strat01_mtp_accept64_seed42_20260918_v2.{out,err}.log`
under `benchmarks/donor_adaptation/density/results/`; Italian prompt
about CPU weight reading, seed 42, `n_predict=64`, `k=1`, default
temperature 0,8, 4 threads, thread polling disabled, context 2048,
CPU-only. The CLI added the model chat template: **1.149 prefill tokens**.
The log reports `draft acceptance = 0.58974 (23 accepted / 39
generated)`, prefill 321.736,52 ms and decode 40.541,17 ms / 64 tokens, or
**1,55 output tok/s**. This rate includes the draft and generic build; it is
not paired with a no-draft control, is not clean-box under the E63 protocol,
and is not a measurement of `engine.c`. A single sequence produces 39
correlated proposals: do not infer a binomial interval or workload rate
from 23/39.

### Addendum: native paired check and greedy divergence

In the temporary checkout of the same `llama.cpp` commit, CMake under
LLVM-MinGW had initially inferred `GGML_SYSTEM_ARCH: UNKNOWN` and compiled
`GGML_CPU_GENERIC`. An adaptation **local to the checkout**, not part of the
research patch, sets `CMAKE_SYSTEM_PROCESSOR=AMD64` only if missing on
Windows; the new `build-cpu-native` directory confirms x86 and
`-march=native`. This is an optimized reference build, **not** `engine.c`.

Both arms used the same Q4_K_M target, Italian prompt,
chat template with 1.149 prefill tokens, seed 42, `--temp 0`, 64 output
tokens, 4 threads, `--poll 0`, context 2048, and CPU-only. MTP adds the
separate BF16 GGUF with `draft-mtp`, `n_max=n_min=1`. Local logs:
`strat01_native_pair_base64_t0_seed42_20260918.{out,err}.log` e
`strat01_native_pair_mtp64_t0_seed42_20260918.{out,err}.log` in
`benchmarks/donor_adaptation/density/results/`.

| Arm | Prefill | Decode (64 output) | Draft |
|---|---:|---:|---:|
| baseline | 18.739,00 ms / 1.149 | 4.931,37 ms; **12,78 tok/s** | — |
| MTP BF16 | 20.782,21 ms / 1.149 | 5.724,12 ms; **11,01 tok/s** | 25/38 accepted |

The decode-rate delta is approximately `11,01/12,78 - 1 = -13,85%`, without
interleaving, clean-box control, or confidence interval. The baseline,
rerun with the same invocation (`...base64_t0_seed42_repeat_20260918`),
produced the **same 64-token sequence**; the MTP run diverges in the
first paragraph, beginning “Per comprendere come la CPU ...” versus
“Per comprendere il processo di generazione ...”. This is a visual text
comparison, not a token-ID diff; the two outputs share the heading.

Static audit of the pinned runtime: at `temp<=0`, the sampler leaves only
the maximum logit; `common_sampler_sample_and_accept_n` resamples each
proposal against target logits and accepts only a matching ID. The server
evaluates the current token **plus** the MTP proposal in one batch, versus
one token in the baseline. A numerical top-1 change caused by the different
batch shape is a compatible *hypothesis*; it is not yet demonstrated and
does not rule out a position/KV/verification alignment bug. Baseline
repeatability does not distinguish these causes. The experimental priority
is a controlled logit trace around the **first** divergence, comparing
context, batch shape, KV, and logit index; do not attribute unchanged
quality to MTP or use this rate to predict a model carved into the C engine.

An initial diagnostic in the temporary checkout therefore added logging
of the raw target top-2 **before** the sampling chain and of the chosen ID.
The logs `strat01_diag_{base,mtp}48_lv3_20260918.{out,err}.log` are local;
`-lv 3` is essential because the CLI otherwise suppresses INFO. The first
**25 output IDs** match; at zero-based index **25**, baseline emits
**4734**, MTP **4164**. At that step, 4734 is the baseline's raw top-1
and 4164 its top-2 (margin 0,357017517); in the MTP arm,
4164 is top-1 and 4734 top-2 (margin 0,0479545593). The draft proposal at
that step was 4734, and the verifier correctly rejected it because the
target in its batch selected 4164. The logs also show a difference in
target margins at the first verification after the first token: both select
ID 50503, but the margin is 3,49245834 in the baseline versus 3,35231781
in the MTP batch. This is consistent with batch-shape-dependent arithmetic
**before any proposal is accepted**, but does not prove batching alone
explains the divergence at step 25: numerical differences, position, and
KV state remain to be distinguished. Instrumented-run timings are not
performance benchmarks.

Next causal check: a modification **only in the temporary checkout** makes
`get_n_draft_max()=0` when `stats.n_gen==25`, leaving prior speculation
unchanged. The log `strat01_diag_mtp32_single_at25_20260918.err.log`
confirms `STRAT01_SINGLE_ONLY n_gen=25`; at that step, the target samples
from `idx=0` and **still selects 4164**, with top-2 4734 and margin
0,09333992. The single-batch baseline had selected 4734. Thus, batch shape
**at step 25 alone** is insufficient to explain the divergence. After the
preceding verifications, target state already differs enough to invert
top-1: this could be numerical drift accumulated across multi-token
decoding or a position/KV/rollback handling error; this test does not
distinguish them. Automatic ID reconstruction from the forced run's
`STRAT01_VERIFY` log does not include the single step, so for that step the
`STRAT01_SAMPLE` line and emitted text are authoritative, not a count derived
from the VERIFY-only sequence. This run is not a rate measurement either.

Mirror check: `get_n_draft_max()` returns zero while
`stats.n_gen<25`, then permits a proposal from step 25. Local log
`strat01_diag_mtp32_only_at25_20260918.{out,err}.log`. For the first 25
outputs, the IDs **and all raw top-1/top-2/margin summaries** match the
instrumented baseline exactly. The log records
`STRAT01_ONLY_AT25 n_gen=25`; the draft proposes ID 4734, but the target
in a two-token batch selects **4164** and rejects it. At that step, the
raw target margin between top-1 4164 and top-2 4734 is **0,254646301**;
in the single-batch baseline, top-1 is 4734, top-2 4164, and the margin is
**0,357017517**. This isolates a **sufficient effect of batch shape at the
divergence step** after a prefix that matches in the observed summaries.
Together with the preceding test, where after speculative history a
single-token batch selected 4164, this also indicates a persistent effect
of decoding history. We did not compare the full logit vector, KV, or
hidden states: the kernel/state-level cause remains unidentified. None of
these instrumented runs is a rate benchmark.

**Traffic estimate, not a measurement:** if `p=23/39` persisted, a
speculative `k=1` cycle would asymptotically produce `1+p=1,58974` output tokens.
The preceding theoretical ledger gives **814.039.040 B** of base W4 weights and
**256.311.296 large active MTP weights**; keeping the draft in BF16 makes
the latter **512.622.592 B**. Without reuse, payload alone would be
`(814.039.040+512.622.592)/(1+23/39) = 834.512.962 B/output token`,
or **41,73 GB/s ideal** at 50 output tok/s, versus the local E30
reference of 36,30 GB/s. Quantizing *the draft as well* to W4 would reduce
the same arithmetic to **29,63 GB/s ideal**; neither post-PTQ acceptance
nor actual reuse has been measured. The comparison excludes Q4 scales,
compute, cache/KV, verification, and a separate output head and is not an
impossibility verdict; it indicates which trade-off to measure next.

**Priority relative to the final gate:** with `k=1`, observed acceptance
`25/38`, and the deliberately favorable assumption that target verification
costs as much as baseline decoding and the draft costs nothing, the rate
projected from the single native baseline would be
`12,78 × (1+25/38) = 21,19 tok/s`; even `p=1` would yield
`12,78 × 2 = 25,56 tok/s`. Reaching 50 with `p=25/38` under
*those same assumptions* would require a target cycle equivalent to at least
`50/(1+25/38) = 30,16` iterations/s, or about **2,36×** the reference
12,78. These are **single-prompt** scenarios, not a universal bound:
batch-2 verification cost, weight reuse, draft quantization, and a new
engine can change the ratio. But they explain why it is not reasonable
to present BF16 MTP alone as a path already close to 50. The next
investment in this branch requires a *quality + target/draft kernel* pair
plausibly within the envelope, not another sweep of the same CLI.

**Exact greedy parity is not the roadmap's quality gate**: the verifier
selects the target's top-1 *in the batch actually used*, which here is not
numerically identical to the batch-1 target. Therefore, do not call
speculation lossless relative to the batch-1 baseline; conversely, divergence
in one sequence does not prove a capability deficit. Any candidate must be
judged by BPB, tasks/rollout, and rate on the **same artifact and execution
path**, against the fixed teacher.

## Next gates, in order

1. To attribute the mechanism of greedy divergence at token 26, both the
   isolated two-token batch and the preceding speculative history with a single
   batch at step 25 can invert top-1. Compare full logits
   /hidden states and KV along the prefix, positions, and rollback,
   against a target using the same prefix in single-token batches. This
   attribution is diagnostic and does not replace the BPB/task gate. Do not
   transfer either 1,55 tok/s generic or 11,01 tok/s native to the 50 gate.
2. If the branch first clears a plausible CPU budget, repeat acceptance
   on preregistered prompts and a corpus not selected after seeing the result,
   accounting for proposals, acceptances, and output tokens. Measure
   logit/hidden-state parity against a BF16 reference to attribute
   mismatches, and BPB/tasks on the actual path to judge quality.
   Draft W4 quantization requires its own gate; it is not an assumed gain.
   Rate comparisons must be interleaved and clean-box.
3. Only if quality and byte/time budget remain plausible, design the
   MLA+MoE+MTP port in `engine.c`; the direct-Q operator and speculative
   decoder in `llama.cpp` are a reference, not our engine.
