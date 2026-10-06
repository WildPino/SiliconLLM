# PQT-SRC-004: synthetic full Switch execution qualification

Prospective apparatus qualification, 6 October 2026. No original learned tensor
values or natural corpus may enter this operation. SRC003 admits a byte reader;
SRC004 must qualify the execution path needed for later original float contexts.
Research remains incomplete even if these synthetic controls pass.

## Reference, runtime and operators

Use the unmodified official Switch implementation in Transformers4.57.6.
Pinned model/config source SHA256 values from consumed METH324 are respectively
5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83 and
86a5e37411e3fa8a8d3c13f3c07f9ee7e74a6033f0da57d681cde80de8071d51.
Its attention uses unscaled QK scores and shared relative/additive masking bias
([official release source](https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/switch_transformers/modeling_switch_transformers.py)).
Keep official FFN/router/norm/block/model modules and tied embedding/head weights.
Replace attention only with an independently written eval F32 SDPA operation,
scale1, dropout0; preserve the original position-bias return/shared-mask contract.
This apparatus accepts full encoder-decoder calls with use_cache=False only.
Reject past caches, use_cache=True, requested attention weights, head masks,
pruned heads, training or non-F32 execution explicitly. Cache equivalence and
autoregressive generation are outside SRC004, not implicitly qualified.

Add pinned wheel dependencies only to the owned scientific environment under
D:/_THINGS/Progetti/SiliconLLM-progressive-runtime/.scientific-venv. Preserve all
existing distribution versions, including Torch2.6.0+cpu/NumPy2.1.3. Retrieve
PyPI metadata and compatible Windows wheels with exact size/SHA256 verification;
freeze complete plan/metadata/wheels before offline no-deps installation.
Record before/after packages, required dependency checks and installed official
source hashes. No main checkout/environment mutation. Preparation limit600s,
64 MiB combined download,32 MiB per wheel,4 MiB metadata,120s offline install.
Preserve first failures and partial environment state before any numbered repair.

Runtime completion002: the Windows-only tqdm dependency colorama0.4.6 must be
installed from an independently hash-verified wheel. Preserve initial001 setup
and its14 pinned packages; add only this missing dependency. This is a package
metadata correction before synthetic/model functions, not a model experiment.
Complete dependency checks must pass before execution qualification.

## Synthetic comparisons and gates

Seed20261006, CPU F32, one thread, eval/no-grad/no AMP/no TF32. Toy configuration:
d_model16,d_ff32,d_kv4,four heads,four experts,four encoder/four decoder layers,
two sparse layers per stack,vocab64,dropout0,router jitter0,router F32/no bias,
pad/start0,eos1. Use capacities1 and64 to cover dropped and unsaturated tokens.
Inputs deterministically generated from the seed; test batch1/3, encoder
length7/13, decoder length5/9, both full and right-padded masks. Save exact IDs,
masks, toy checkpoint and case descriptions before reference function calls.

Compare three routes on identical weights/inputs: official eager fully resident,
resident SDPA, and SDPA with only experts meta until requested by the unchanged
official sparse forward. Core weights remain resident; load and verify an expert
pair immediately before its FFN call, then release the pair to meta immediately
after. Do not retain hidden resident expert copies in the streamed model.
For every case compare whole logits, encoder/decoder hidden states, every sparse
bank's router mask/probabilities/logits and dispatched FFN inputs/outputs. Gates:
all finite, relative RMS<=1e-5 (denominator max(reference RMS,1e-12)), exact
logit argmax and all routing masks/indices, no shape/namespace discrepancy.
Resident/streamed SDPA outputs and dispatch traces must be bit-exact; eager vs
SDPA floating values may meet the declared numerical tolerance.

Assert four embedding/head aliases are the same Parameter and exactly match the
synthetic checkpoint. Every expert pair load has pinned raw hashes; loaded pair
counts match actual nonempty dispatches, at most one expert pair resident at
once, all experts meta between calls. Verify core parameter bytes unchanged and
repeat-call determinism. Separately control capacity overflow against an
independent router/FFN equation: dropped outputs zero, capacity per sequence.
Negative controls must reject unsupported attention options, changed expert
bytes/layout/dtype/namespace, nonfinite weights and unqualified lazy calls.
Retain malformed fixtures, positive predictions/intermediate traces, load/release
logs, controller/runtime/source hashes, all first failures and exact Git proof.

No fitting or scientific quality inference. Worker bound180s, peak RSS<=1 GiB,
complete local output<=64 MiB, no GPU/native timing. Live process admission
required, run sequentially; installation is its separately bounded operation.
Freeze executable sources/protocol/runtime qualification and controls before
later original-model functions. A future T4 execution must independently qualify
the deployed operator on that exact runtime/device, with fresh resource-owner
admission and separate source I/O/memory/time bounds.
