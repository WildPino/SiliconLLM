# STRAT-01 GigaChat 3.1 combined RMSNorm + K-B propagation protocol

**Frozen:** 2026-09-21, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-COMBINED-RMS-Q5Q8-PROPAGATION`

**Purpose:** determine whether the independently validated double-RMSNorm and K-B Q5_0×Q8_0 semantics, composed in the block-0 C graph, close the downstream attention and dense FFN projection gates in both frozen schedules.

## Nearest evidence and non-duplication

The [upstream RMSNorm propagation result](STRAT_01_GIGACHAT31_ENGINE_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC_RESULT_20260921.md) changed the two upstream RMSNorm sites and the final FFN RMSNorm. It improved `ffn_inp-0` but left up/gate at NRMSE `0.00262649240855995`/`0.00231478705864183`, above the frozen `0.002` gate.

The [K-B Q5_0×Q8_0 result](STRAT_01_GIGACHAT31_ENGINE_KB_Q5_0_Q8_0_DIAGNOSTIC_RESULT_20260921.md) then proved, without a donor graph, that pinned runtime semantics close `q_nope_absorbed_perm-0` at NRMSE `4.802399901907019e-8`. Exact-reference, upstream-double, and accepted-float `q-0` serialize to identical Q8_0 bytes.

Neither cell measured their composition through RoPE, cache writes, causal attention, V-B, output projection, residual addition, final RMSNorm, and dense up/gate projections. That downstream composition is the only changed estimand here. Tensor identities, schedules, cache layout, Q4_K×Q8_K operators, F16 conversion, RoPE, reference payloads, and gates remain fixed.

## Immutable bindings

| object | identity |
|---|---|
| accepted GGUF | 6,474,702,976 bytes; SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| pinned llama.cpp | `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` |
| upstream RMSNorm adjudication | `aebf238eb627f7eaa39f29f737d5295c22fa12a4a1fcb8bb1e88d1286d98bf80` |
| K-B Q5_0×Q8_0 adjudication | `fd2ba64075f02fa26ff9210ff06a80cd6c2e3729966311d991235edc699416eb` |
| accepted Rung-2A reference manifest | `0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f` |
| accepted Rung-2B reference manifest | `5fa1ca6317c3afb466df937f4a9f2e622c38f7d3787d84d2874b2e8afb6ca0b7` |
| combined candidate K-B Q8_0 checkpoint | 34,816 bytes; SHA-256 `5dae7c9874e8ca79fdccc9ceda45e652f729a2176de0bbebb24453ae818213f2` |
| combined candidate absorbed-Q checkpoint | 524,288 bytes; SHA-256 `cf69d34fa506ee356ab5fb49982e7f872505e47312c7863540d1e0464036d51a` |

The fixed tokens, positions, tensor inventory, `prefill8` and `cached7p1` schedules, twelve Rung-2A reference tensors, three cache checkpoints, and three dense-FFN targets are inherited from those records. No reference producer may run.

## Diagnostic implementation

Add a diagnostic-only engine command. Do not alter existing production or prior diagnostic commands. It must:

1. hash and parse the accepted GGUF and validate every inherited tensor descriptor;
2. execute both frozen schedules once, using pinned double accumulation at attention-input RMSNorm, compressed-KV RMSNorm, and final FFN RMSNorm;
3. use Q8_0 activation quantization plus Q5_0×Q8_0 dot semantics only for `blk.0.attn_k_b.weight`;
4. leave embeddings, Q/KV projections, RoPE, F16 cache conversion, causal attention, V-B, output projection, residual addition, and dense up/gate operators unchanged;
5. emit the same twelve Rung-2A tensors, cache checkpoints, and three dense-FFN tensors as the upstream diagnostic;
6. serialize the K-B Q8_0 activations and require the frozen Q8_0 and absorbed-Q checkpoint hashes above in both schedules before downstream adjudication;
7. report `donor_graph_executions=1`, `self_certifies_pass=false`, and no timing or rate claim.

The execution count is one candidate graph invocation containing both schedules, matching the prior runner convention. Apparatus-only builds and model-free self-tests do not consume the cell.

## Gates and controls

Use the existing gates unchanged:

- all Rung-2A tensors except `ffn_inp-0`: NRMSE `<= 0.002`, normalized maximum `<= 0.01`;
- `ffn_inp-0`: NRMSE `<= 0.001`, normalized maximum `<= 0.005`;
- all cache checkpoints: NRMSE `<= 0.002`, normalized maximum `<= 0.01`;
- prefill/cached token-7 continuity: NRMSE `<= 1e-7`, normalized maximum `<= 1e-6`;
- dense `ffn_norm-0`, `ffn_up-0`, and `ffn_gate-0`: NRMSE `<= 0.002`, normalized maximum `<= 0.01`.

The run is valid only if:

- both schedules reproduce the frozen K-B Q8_0 and absorbed-Q checkpoint hashes exactly;
- prefill/cached continuity passes for every emitted tensor;
- inherited cache identity and no-separate-V-cache controls pass;
- accepted-float, upstream-double, and combined source records validate by hash;
- swapped up/gate targets reject;
- a source control proves exactly two upstream double-RMSNorm sites, one final double-RMSNorm site, and one Q5_0×Q8_0 K-B site in the candidate path;
- model-free tests make omitted Q8_0 quantization, wrong head stride, and Q5_0 high-bit corruption fail;
- `errors=[]`, exactly one candidate donor graph is recorded, and no reference graph runs.

Report candidate versus reference, candidate versus accepted-float baseline, and candidate versus upstream-double baseline at every boundary. Do not change gates after observing values.

## Decision rule

- **`COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES`** if every identity/control, inherited attention/cache/continuity gate, and all three dense-FFN gates pass in both schedules.
- **`COMBINED_RMS_Q5Q8_INSUFFICIENT_FOR_PROJECTION_GATES`** if the run is valid, both frozen K-B checkpoints and inherited schedule controls pass, but either dense up or gate misses its frozen gate. Localize the new first residual; do not add another coordinate in the same cell.
- **`VOID_COMBINED_RMS_Q5Q8_PROPAGATION`** for any identity, checkpoint, source, cache, continuity, planted-control, completeness, or execution-count failure.

Exactly one non-VOID execution is allowed.

## Post-run apparatus erratum: offline adjudication of the preserved capture

**Frozen:** 2026-09-22, before implementing or running the offline adjudicator.

The first candidate process completed successfully and emitted the complete
combined payload tree, but the external runner returned
`VOID_COMBINED_RMS_Q5Q8_PROPAGATION` after the donor execution.  The preserved
records are bound as follows:

| object | SHA-256 |
|---|---|
| void adjudication | `8eca798bb38706b2d9f0a56ebebf5934f035200c5ae0e983ae4e801f56219e66` |
| void run manifest | `4964efe45dbbde5b7fd80d957b25279c8099f267c6b40e2aac573858d232db5e` |

The sole reported error is `candidate source hash mismatch`.  It occurs while
validating the already accepted upstream-double baseline, not while validating
the combined candidate: the runner recomputed the upstream baseline's expected
`engine.c` hash from the current tree, whose engine necessarily changed when
the combined diagnostic was added.  The upstream baseline report instead
contains, and must be validated against, the source hashes archived in its
hash-bound adjudication record.

A narrow offline repair is permitted.  It must:

1. require the two exact void-record hashes above, the exact error, one source
   donor execution, and a successful candidate process;
2. revalidate every existing combined payload and checkpoint against the
   current unchanged combined engine/header hashes;
3. validate the upstream-double baseline against the hashes archived in its
   already pinned adjudication, never against the current engine hash;
4. apply the original gates and controls unchanged;
5. record zero new donor or reference executions and bind the offline result to
   both preserved void records.

This is adjudication of the one already captured candidate, not a second
candidate execution.  The original void directory remains immutable.  Any
identity failure in the offline repair remains VOID; no donor rerun is allowed.

## Non-claims and stop rule

This cell does not edit production, establish Rung 2C or later-layer parity, or claim tokenizer, logits, generation, end-to-end quality, RAM, or rate. Document and index the result before deciding whether to integrate the two semantics into the production path. No `SPEED_LEDGER.md` entry is permitted.
