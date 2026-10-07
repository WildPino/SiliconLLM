# METH522: copying source neurons cannot satisfy this narrow exact-support recipe

7 October 2026. COMPLETE/ADMITTED. Exact-support B512/rho2 recipe obstructed
on development; goal ACTIVE/INCOMPLETE.
[Protocol](METH_522_REDUNDANCY_LOWER_BOUND_PROTOCOL_20261007.md),
[admission](ADMISSION_522_20261007.json), [retention](RETENTION_522_20261007.json).

## Question and evidence

Can redundant branches of at most512 original neurons preserve every observed
source support, with at most6144 logical neuron incidences per3072-neuron parent?
This tests a necessary condition independent of branch masks or their selector.
Rho2 is an exploratory budget, not a RAM ceiling imposed by the goal.

Reuse admitted500 original hidden I16 codes for ALL17540 UIDs, original support
metrics and ALL19962 occurrence joins. Development11721 UIDs alone construct
the witnesses. Consumed5819 UIDs are previously observed support diagnostics.
All127 exposed parents retained; parent0 has no observations and remains
uncompiled/unpromoted. No new source response, model, packing, fitting or C call.

## Algebra and its scope

Let J_i contain the original neurons whose saved hidden A16 codes are nonzero.
Putting every member of J_i in the selected branch preserves the original
maximum, quantizer and source output on that input. This is sufficient for
identity, not necessary: cancellation or approximation can allow omissions.

If |J_i|>512, a single512-neuron branch cannot contain that support, at any RAM.
If |J_i union J_j|>512, these two supports require different branches. For any
pairwise incompatible clique K, and requiring all3072 original atoms somewhere:

    branches >= max(|K|, ceil(3072/512)) = max(|K|, 6)
    logical incidence copies M >= 3072 + sum(i in K)|J_i| - |union(i in K)J_i|

The second inequality adds the copies required in distinct branches and each
unrepresented original atom once. ONE descending-support/ascending-UID greedy
clique is retained with its complete insertion/rejection trace. It certifies a
lower bound; it does not find a maximum clique or solve the covering problem.

M counts logical incidences. With separately stored dense I8 WI rows and WO
columns, coefficient bytes are at least2*768*M, before scales/indices/padding.
Shared backing and gathered indices have different physical storage costs.
Neither layout has measured DRAM in this experiment. An oversized support
already forbids the proposed family; its finite copy bound cannot price a RAM
remedy for that width obstruction.

## Decisive measurements

| Development result | Count |
| --- | ---: |
| Exposed original parents |127 |
| Parents with an observed support above512 |10 |
| Parents with clique copy bound above6144 |29 |
| Additional copy-obstructed parents whose supports all fit512 |19 |
| Parents passing both necessary bounds |98 |
| UIDs with support above512 |288 /11721 |
| Maximum observed support |764 |

Both prospective scientific eligibility gates FAIL. Passing parents remain
INCONCLUSIVE: neither covers, selectors nor function accuracy were constructed.
The independent audit verifies ALL63262 clique pairs, every support cardinality,
every ordered trace decision and all original report denominators.

The ten width-obstructed parent IDs are5,21,35,37,41,58,68,82,85,89. Their
maximum supports are respectively518,667,522,764,519,573,663,552,565,523.
The consumed diagnostic has132/5819 oversized supports, maximum764; it did not
choose cliques, bounds, thresholds or the decision.

Concrete copy obstructions with individually width-feasible supports:

| Parent | Dev UIDs | Largest support | Certified branches | M lower bound | M/3072 |
| --- | ---: | ---: | ---: | ---: | ---: |
|113 |133 |483 |87 |32159 |10.468424 |
|65 |116 |499 |78 |28702 |9.343099 |
|49 |123 |489 |49 |16949 |5.517253 |
|103 |121 |434 |45 |16270 |5.296224 |
|123 |102 |478 |42 |14740 |4.798177 |

The overall maximum clique size271 and copy bound195799 belong to the complete
report; oversized-support parents are included. Do not interpret this maximum
as RAM sufficient to fix their impossible width condition.

**Decision:** `EXACT_SUPPORT_B512_RHO2_OBSTRUCTED_ON_DEVELOPMENT`.
Close this exact-support recipe. This does not refute arbitrary RAM, multiple
active branches, other widths, approximation, folded functions or new features.
Repeated source knowledge remains a legitimate resource tradeoff.

## Integrity, resources and reproducibility

All5 main,6 independent Boolean audit and5 admission controls PASS. New tiny
H8/B3 supports certify a nontrivial clique/copy obstruction and a first-compatible
rejection, plus a separate oversized support. The auditor imports neither the
main clique mathematics nor its support/report helper. Witnesses were saved
before consumed reports. No first scientific fault or repair occurred.

| Instrumented stage | Terminal seconds | OS peak bytes | Envelope |
| --- | ---: | ---: | --- |
| Binding |2.125 |57769984 |60s /128MiB |
| Main |2.015 |161038336 |60s /512MiB |
| Independent audit |2.782 |317235200 |60s /512MiB |

Combined retained outputs1989756B, below16MiB. CPU10, one numerical thread;
actual loaded runtime and876-file inventory are bound. Fresh used-file hashing
is charged in each receipt. These clocks are local witness costs, not inference.
All three instrumented instances exit0 and close. Typed UTC Event1000 queries
and positive controls179810/179791 pass with zero relevant faults. Existing
foreign three SHA bytes and empty cache are preserved.

Scientific freeze8b29572, input bindinge28a567, main7919c0c, audit995c052.
The finalizer reads receipts/inventories only; it does not replay support math.
On resumption, ALL15 retained file references were freshly SHA/size-checked,
without replaying the completed main or audit.

| Record | SHA256 |
| --- | --- |
| [Binding](meth522_binding.json) |ebf00672875e55023961dec49e30e1c2476dca8ea421b167a5776ba4652ffaa4 |
| [Main](meth522_main_result.json) |6c795de6286c8034a62cf7b271f627d13d452070a37280bacd930e8e02030af3 |
| [Audit](meth522_audit_result.json) |395024cf14a413b631b074537ed7a33d4520564aaf5f0737c6247a50f2f7a5ad |
| [Admission](ADMISSION_522_20261007.json) |cd8a253fac64cef9b25f7f49105752677a491ffaddf841756be871a67490fd60 |

Entrypoints in `benchmarks/native_expert_scaling`: `meth522_prepare_binding.py`,
`meth522_bound_main.py`, `meth522_bound_audit.py`, `meth522_windows_terminal.ps1`,
`meth522_finalize.py`. Original argv, runtime, exact file extents and output SHA
identities are in the linked records. Main takes `--binding-sha`; audit also
takes `--main-sha`; finalizer additionally takes `--audit-sha` and structured
`--exit-receipts`. Use the bound isolated511 Python environment and empty cache
in a fresh output namespace, never overwrite or replay these completed outputs.

## Consequence for the method

521 shows that complete coefficients in its fixed interpolant do not preserve
the source function.522 shows why atom copying alone has a width obstruction
and can demand considerably more than twice the original logical incidence.
Neither result closes the broader transfer goal.

Select [523 source folding with explicit ReLU hinges](METH_523_SOURCE_FOLDING_HINGES_NEXT_20261007.md).
First recover and price the actual arithmetic/negative-preactivation contracts;
the proposed hybrid retains original nonlinear directions while folding stable
source contributions. No523 export, fit, numeric observation or process yet.
Compact core, useful much larger n, CPU LUT winner AND normalized mass, real
DRAM, whole fresh quality AND SAME50/s, and actual family/scale variants remain.
