# Complete original-recipient baseline: numerical pass, cost envelope rejected

15 October2026. **BASELINE_COST_ENVELOPE_FAIL.** All frozen numerical,
identity and retained-cost checks complete; no chatbot-quality, donor-memory
transfer or useful50 admission. Do not start long recovery on the assumption
that this unchanged recipient already leaves the required CPU margin.

## Complete artifact, not component timing

The [protocol](RECIPIENT_INT8_BASELINE_PROTOCOL_20261015.md) exports the original
pinned Qwen2.5-0.5B-Instruct source with all24 blocks, dense FFNs, causal GQA,
F32 KV cache and full vocabulary head. The deployed head is explicitly untied
and INT8. Norms, biases and input embeddings remain original F32 widenings.
There is no learned bank, source-state injection, missing attention organ,
head shortlist, source FFN fallback or synthetic weight fixture.

Native execution is an `engine.c` profile reusing the existing full donor
forward path. Artifact1,040,610,356B,
SHA256 `daf94964228636d8255bd20aacc4c88c6dd1e100afd05831c479345d4bf0a280`;
executable SHA256 `aae76015210f2b40cb20c45b6f3fab0441eb09f5c5ca225ab6659cbed3593ae1`.
Active matrix codes/scales address495,785,472B per complete step. Input embedding
lookup is3,584B. These are addressed payloads, not measured DRAM. Repeated
query-head KV reads and distinct stored KV capacity are different counts.

## Numerical evidence and preserved faults

Code/protocol freeze5b52aab; isolated import repair2fa53a9; reference-buffer/
adoption/readback freezeb0116a8. The first launch failed before model loading
because runpy lacked the exporter directory; [import repair](RECIPIENT_INT8_PREFLIGHT_REPAIR1_20261015.md).
The second successfully exported/compiled/probed but its CPU reference failed
before producing logits: a nonpersistent rotary buffer remained meta. The
[reference repair](RECIPIENT_INT8_REFERENCE_REPAIR2_20261015.md) reconstructs that
parameter-free buffer on CPU from source config. Both exit1 receipts and faults
are retained. No old fault is relabeled PASS.

Repair2 adopts the same sealed export and executable, verifies every old
input pin except the two explicitly changed controller/auditor files, and
requires a new native probe to reproduce the retained old probe byte-for-byte.
No source conversion or compilation was repeated. Only then does the repaired
CPU auditor run.

The auditor checks all290 original BF16 tensor names/shapes/extents,122 exact
F32 widenings, all169 independent R8 code/scale derivations and all460 archive
extents. Its eager FP32 model uses only decoded archive tensors. All32 complete
native vocabulary rows have identical top1; worst relative L2
**5.832934e-6**, maximum absolute logit difference0.00011158. Source/artifact/
IDs/logits/config seals match before/after. Audit numerical section16.0s,
sampled peak RSS4,538,171,392B; wrapper process wall19.813s. No GPU or network.
This proves decoded-codec correspondence on this finite prefix, not preservation
of the BF16 donor's chat quality. METH59's historical quality failure remains.

## All fixed cost repetitions

Three independent processes per case; six distinct physical workers on
logical0/2/4/6/8/10, fixed ACTIVE/infinite wait and fixed kernels. Each
non-EOS emitted token receives a full native forward, including the last.
EOS scan time remains charged without inventing a forward. Loading and prefill
are reported separately. The context512/2048 cases are repeated-prose cost
fixtures, not useful-chat evaluation.

| Case | Three raw native-forward rates, /s | Median /s | Worst process ms/forward | Rate max/min |
| --- | --- | ---: | ---: | ---: |
| English explanation | 39.427 /47.387 /48.172 | 47.387 | 25.363 | 1.222 |
| Italian planning | 53.039 /58.788 /51.477 | 53.039 | 19.426 | 1.142 |
| Python coding | 38.117 /53.529 /48.719 | 48.719 | 26.235 | 1.404 |
| Context512 | 43.870 /50.158 /58.075 | 50.158 | 22.795 | 1.324 |
| Context2048 | 44.532 /44.800 /35.685 | 44.532 | 28.023 | 1.255 |

Every case fails the frozen worst-process18ms and1.10 stability allowances.
Keep all processes;58.788/s is not a selected successful baseline. No cause
for the dispersion is established. Cross-donor historical68.52/s cannot replace
these measurements. The sequence completes in239.937s after initial binding
verification; output family1,062,073,457B, largest observed child RSS
5,374,119,936B. All19 accepted/adopted commands exit0.

The separately frozen stored readback exits0 in10.406s. It verifies source/
runtime/result seals, original receipts, absent owned child identities, actual
worker masks, all930 charged forwards/933 emitted IDs, every step's time/
context/EOS accounting and all15 binary generation outputs. IDs are identical
across repetitions for each case. It independently derives the same cost FAIL.
It does not replay model arithmetic or measure physical DRAM.

## Decision and next discrimination

Close this unchanged all-I8 recipient execution envelope before investing
months in its recovery. This is not a bound on all Qwen encodings or CPU kernels
and does not reject the donor-memory method. The complete recipient still
needs materially less work or a demonstrably more efficient physical encoding.
Do not run another unchanged thread/wait-policy grid.

The separately prepared W4-FFN/integer profile has an analytical matrix payload
about357.5MB instead of495.8MB. It is uncompiled/unmeasured; smaller bytes alone
cannot establish the required improvement or source usefulness. It requires
its own decoded-wire and physical-cost test before any long recovery.

Primary-config triage also identifies a135M recipient with far lower matrix
work, but weak starting competence. A360M alternative saves fewer matrix bytes
and has substantially larger KV at2048. These are recipient alternatives,
not replacements of the final very-large donor. Preserve the original
donor-relative useful-capacity and same-artifact CPU goal; do not declare
success from a smaller fast model. Price the next complete encoding before
choosing long adaptation or acquiring a new source.

Receipts: [binding](recipient_int8_baseline_binding_20261015.json),
[all observations](recipient_int8_baseline_result_20261015.json),
[decoded audit](recipient_int8_baseline_audit_20261015.json),
[stored readback](recipient_int8_baseline_readback_20261015.json),
[closure](recipient_int8_baseline_closure_20261015.json).
Raw logs/artifacts and both historical faults remain under the three bound
`results/native_expert_scaling/recipient_int8_baseline*20261015` directories.
No T4 job or owned numerical worker remains live; goal active/incomplete.
