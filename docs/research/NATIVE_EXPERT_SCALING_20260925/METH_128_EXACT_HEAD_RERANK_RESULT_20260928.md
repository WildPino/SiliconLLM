# METH-128: int8 shortlist retains the exact BF16 head choice on viewed E1280 states

**Decision.** The frozen development screen passes for the METH-59
per-row R8 head. On all **4,494 final hidden states** from 24 METH-121
prompts, the original BF16 head's top-1 token is in the approximate
top-16, and exact BF16 rescoring chooses the original top-1 on
4,494/4,494 positions at K=16, 64 and 256. The group-128 R8 head also
passes; the protocol prefers the smaller per-row format. This licenses a
native two-pass head experiment, **not** model-quality or speed promotion.

The [protocol](METH_128_EXACT_HEAD_RERANK_PROTOCOL_20260928.md) was
committed at `5dd3fc2` before the run. The
[runner](../../../benchmarks/donor_adaptation/s1/meth128_exact_head_rerank.py)
binds the donor source, METH-56 parent and METH-107 child checkpoint,
applies the METH-119 centered-B transform and executes the BF16
quality-gated E1280 model. The METH-121 manifest is SHA-256
`7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366`.
All 24 prompt ID hashes are checked. The candidate heads are the exact
stored code/scale tensors from METH-59 and METH-91, with hashes in the
[machine result](meth128_exact_head_rerank_result.json). The source set
was already viewed by METH-122; these numbers are diagnostic.

| Exact BF16 top-1 omitted by approximate top-K | K=1 | K=4 | K=16 | K=64 | K=256 |
|---|---:|---:|---:|---:|---:|
| Per-row R8 head | 209 | 5 | **0** | **0** | **0** |
| Group-128 R8 head | 175 | 4 | **0** | **0** | **0** |

The exact BF16 row recomputation also has **zero top-1 mismatches** at
K=16/64/256 in both arms, using the full head's tie rule: when multiple
scores have the same maximum, choose the lowest token ID. A full-vocabulary
control reproduces the original BF16 argmax. The per-row K=64 omissions
are zero in each category: 1,615 code, 1,260 prose and 1,619 technical
prompt positions. The recomputed candidate scores can differ from the
large head matmul by as much as 0.0625 due to BF16 reduction/rounding;
the correct tie rule is therefore part of the method.

An initial apparatus run used `argmax` on the **approximate ranking order**
to break equal rescored values. It reported 52/94 apparent rerank
mismatches at K=64 (per-row/group-128) despite zero candidate omissions.
That order is not the original full head's lowest-ID tie rule. The runner
was corrected to make the rule explicit, with identical weights,
prompts, K values and gate. It also scores the gathered original head
logits on each shortlist as a control. The corrected run finds zero
mismatches for both recomputed and gathered scores at K>=16.

The per-row R8 head stores 136,742,400 code/scale bytes. Keeping the
original BF16 tied head adds 272,269,312 resident bytes, or
409,011,712 bytes for both. Nominal addresses at K=64 are
136,742,400 bytes for the full R8 scan plus 114,688 BF16 row bytes,
versus 272,269,312 bytes for the full exact BF16 scan: a 135,412,224-byte
**arithmetic** active-payload reduction. This duplicates the head in RAM
and does not measure physical DRAM traffic, CPU kernel throughput or full
token rate. The group-128 R8 payload is 138,261,760 bytes.

The final scoring stage ran 5.94 seconds after source/hash checks and
device initialization, peaked at 4.171 GB allocated RTX 3060 memory;
the largest observed process RSS at the per-arm budget checks was
3.249 GB. No T4 was used. Next, implement the per-row two-pass head in
the C reference runtime and verify exact native top-1 and cost on
native full-model states, then audit a compact-body combination on new
disjoint documents, greedy generation, tasks and grounded semantics.
