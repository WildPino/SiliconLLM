# Exact finite-BF activation compilation: native acceleration qualified

14 October 2026. **COMPLETE / full scalar and stored native audit PASS /
EXACT_STORED_ACCELERATION_ELIGIBLE. Goal ACTIVE / INCOMPLETE.**

Compiled the actual BF input domains of three unchanged numerical expressions:
conv SiLU outputBF, gate SiLU multiplierF64, post-bias/post-round threshold20
softplus outputBF. Shared65536-entry tables total786432bytes. Gate is never
rounded toBF; original128-column I16 readout remains. This is activation program
storage, not transferred knowledge or a semantic useful-n expert bank.

| Paired whole24 mode | Original median ms | Activation median ms | Candidate/original | Reduction | Remaining20ms median budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident, no deliberate eviction | 25.96025 | 16.81725 | .6478077060120762 | 35.2192% | 3.18275ms |
| Parameters AND shared tables evicted per arm | 26.23275 | 17.02385 | .6489540745823446 | 35.1046% | 2.97615ms |

Both modes pass frozen>=5% gain; every split/length stratum ratio<=1.05
(observed.6402..6565).64 paired positions/mode on the same four shortest/longest
FIT/DEV histories, final16 tokens, alternating arm order. All24 layers execute
sequentially, fresh own-prefix contexts, batch1, single-threaded C, affinity0.
Resident candidate min/p90/max16.0221/17.93078/20.8556ms; evicted
15.8507/18.87339/34.5284ms. Tails retained, no outlier removal. Shared desktop;
this finite screen is not a general hardware throughput guarantee.

No-op24-call median.0392ms is not subtracted. Readout-only medians after both
arms12.27115/12.55825ms resident and12.62995/12.61420ms evicted iterations use
the SAME original readout, without an additional direct-readout eviction.
Their medians are not an exact subtractable stage decomposition. The benefit
comes from a different active operator family, not the rejected integer layout.

## Numeric contract and complete qualification

Separate emitter/consumer/scalar-reference code, manualRNE reference and scalar
compiler controls verify ALL65280 finite BF codes for all3 functions (195840
table words), including signed zeros/subnormals/extremes. All256 nonfinite
probe codes follow the original runtime expression and match independently.
FE_TONEAREST and MXCSR0x1f80 control(mask0xffc0) enforced, including masked
exceptions; sticky flags/libm errno are excluded from the numeric/state contract.
Wrong bind rejects6; unbound or changed-control probe/row rejects5 before any
state/ring/seen mutation. All checks repeat in producer AND auditor.

Every192 continuation packet/intermediate and final state/ring/seen/last exact
to qualified original references. Independent integer/scales/BF verification
checks6,291,456 readout words and paired coverage/order/summary arithmetic.
Producer AND auditor replay all97 admitted native bridge trajectories:
2328 sequence-sites/580056 positions/1,187,954,688 BF output words/
686,555,136 final state words, all exact. Same-buffer/ABI/recurrence/replay helpers
and libm/compiler dependencies are declared; no independent clock authentication.
The support flushes all parameters PLUS3 tables once before EACH evicted arm;
counter128 groups verified in supplementary producer/auditor records.

Parameters remain310,855,680 bytes, plus786432 shared table bytes, total
311,642,112 array payload bytes. Exact operand/cache-control accounting is not
measured DRAM; traffic remains unresolved. Generators, FFN, attention, head,
tokenizer and transport are all outside the measured recurrent span. These
latencies are NOT final useful accepted model tokens/s. Known14/16 quality
bridge compatibility is preserved in stored traces; no fresh GPU/held-out test.
Historical GPU/transitive producer custody limitations remain explicit.

## Custody, first fault and reproduction

Numerical prototype319f5de. First preparation29496/create1791972961.48448,
exit1/20.266s/OS50,249,728: negative test supplied null output pointer arrays
to rs_block, whose outer loop builds views before the row-state guard.
Original source/receipt/12-file compiled/table family retained and inventoried.
Separate V2 adapter0487cbd fixes only test ABI arguments; C/math/gates unchanged.
V2 preparation17228/create1791973282.4027061,exit0/40.656s/OS75,857,920.
Execution wrapper7675959 hashes every input BEFORE exhaustive preflight, as
well as inherited before/after paired/replay checks; preparation904/create
1791973687.5641296,exit0/.313s/OS39,288,832. Original bindings preserved.

Producer29608/create1791973701.2330232,exit0/934.735s/OS1,270,378,496;
auditor23788/create1791974653.0257645,exit0/778.5s/OS1,164,111,872;
closure20212/create1791975543.998996,exit0/144.125s/OS51,003,392.
Every declared source/runtime/input/output pin and successful worker identity
sealed. New V2 E family15 files/1,960,331 bytes; old463MB packets/full source
traces reused rather than duplicated. No GPU/model/optimizer/RESERVED/delegation.
Per phase2700s/OS2GiB/E32MiB. Full audit completed on the unchanged protocol.

- [Protocol](NATIVE_BF_ACTIVATION_LUT_PROTOCOL_20261014.md), [first fault/V2 repair](NATIVE_BF_ACTIVATION_LUT_REPAIR_V2_20261014.md), [execution order](NATIVE_BF_ACTIVATION_LUT_EXECUTION_20261014.md), [state](NATIVE_BF_ACTIVATION_LUT_STATE_20261014.md).
- [Execution binding](native_bf_activation_lut_execution_binding_20261014.json), SHA256 d18240942552afe97351477e3f47d830b1712db37733c9bdd4f7b557bff85c47.
- [Producer](native_bf_activation_lut_result_20261014.json), SHA256128b6b1f3caf1e4a84b17c928b067b5d7dd6738838fe0511fa309de0697a8ca2.
- [Full audit](native_bf_activation_lut_adjudication_20261014.json), SHA2562e004513b5dd802a5d6427cae2d3667b61391b4d503edcda7760c24ec1427d71.
- [Closure](native_bf_activation_lut_session_closure_20261014.json), SHA256d11b89ffc12152c904cce24609d689064abc8d862551d994d9a49f0104f96f4f.

Qualified loading tool: benchmarks/native_expert_scaling/bf_activation_backend.py,
load_qualified_backend(closure_path), returns the established row_state ABI.
It verifies deployment DLL/tables/qualification metadata, retains table owners
for process lifetime and binds the exact FP contract, without reading teacher
traces for inference. Model parameter owners remain the caller's responsibility.
Load/bind/create/destroy smoke PASS. Default historic backend remains untouched;
future new model/bridge bindings must explicitly choose this qualified loader.

## Decision tied to the final goal

Use this qualified recurrent implementation when constructing the next artifact.
The median recurrent portion now leaves ONLY~3ms of the20ms budget, not a
completed useful50 pipeline. [Next actual23-generator cost screen](CONTEXTUAL_GENERATOR_COST_NEXT_20261014.md)
must price the omitted source operators before further FFN bank expansion or
promises. Actual header inventories show631,570,432 BF bytes in those generators,
1,358,954,496 FFN bytes,268,439,552 head bytes and251,658,240 attention bytes;
these are payloads, not DRAM/speed/impossibility bounds. Choose a substantive
source-dependent active-core/bytes/width/depth/conditional change from physical
cost and own-model quality. Do not keep polishing a full-width generic donor.
Same final CPU artifact quality/useful>=50, useful n, actual DRAM, multiple
families/scales and sustainable conversion costs remain incomplete.
