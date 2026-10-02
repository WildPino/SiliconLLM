# METH-276: complete diagnostic I16/private128 archive passes composition

## Decision and reproducible artifact

The unpromoted diagnostic complete archive and its own versioned standalone
loader pass every prospective archive/component gate. This licenses only
the separately frozen consumed whole-model regression277. Quality,
same-artifact native rate,useful new n/RAM/DRAM and family transfer are open.
Fixed259 semantic267 and274/275 component-cost failures stay closed.

Freeze `0c2f27d`; environment-only repair freeze `56f05bb`.
Source `benchmarks/native_expert_scaling/meth276_diagnostic_complete_core.py`;
[protocol](METH_276_DIAGNOSTIC_COMPLETE_CORE_PROTOCOL_20261002.md).
Raw `meth276_diagnostic_complete_core_repair1_result.json` SHA
`ff4694b426725f3de2e500739ba11fb06ad280dc7a5ef04515ff5b4a592fe553`.
Artifact `results/native_expert_scaling/meth276_qwen05b_i16_private128_diagnostic_repair1.safetensors`
SHA `4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`.
Version `M276_DIAGNOSTIC_I16_INPUT_PRIVATE128_UNIQUE_BF16_V1`;
metadata explicitly `diagnostic_only=true`,`native_promotion_qualified=false`.

## Measured evidence

| Check | Result |
| --- | --- |
| Archive | 725 tensors,1,329,447,260bytes;payload1,329,221,892;header/metadata225,368 |
| Changed fields | Only72 private IDs/gate/up tensors32->128;all653 others byte exact259 |
| Source128 rows | All72 fields exact271;all old32 IDs and their BF16 gate/up rows retained |
| Extra payload | 8,262,144bytes;no new learned functions or route addresses |
| Standalone loader | All725 inventory hashes/finite fields consumed;218 archived non-FFN parameters,72 random FFN parameters removed;tied head;no donor/checkpoint fallback |
| Actual equation | All6144 loaded FP32 bases exact274 GPU reference;BF16 returned values exactly its BF16 rounding |
| Input arithmetic | All5,505,024 I16 codes and6144 inverse/scales match actual frozen CPU controls |
| Native FP32 parity | All6144 median relativeL2 4.67159035e-7,max1.04214100e-6;384 median4.71293015e-7;within1e-4/5e-4 |
| Conditional composition | All6144 parents/scores/children/alias IDs/selected A/B/isolated BF16 contribution exact259;composed forward exactly changed base plus that contribution |
| Capacity | Original30,556 unique parameter-functions/164 aliases conserved;no large-n utility claim |

The native numbers match274's previously recorded numerical summary exactly.
No C operator was retimed, changed or promoted here. The conditional term
was compared before dense-base addition: subtracting rounded BF16 outputs
would not isolate it correctly.

Successful session24677 exits0 in134.422s;end RSS5,872,050,176bytes,
peak allocated GPU2,856,219,136bytes. LocalRTX3060/six threads,no T4/download.
RSS is the budget snapshot,not an independently measured process peak.
The result records all725 shape/dtype/byte/hash fields,24 state controls,
6,144 numeric errors and76 imported project helper hashes.

## Preserved apparatus stop

First session40882 exits1 after136.938s with zero scored layers because the
deterministic CuBLAS environment was omitted in that launch. Its failure
and original archive remain; hashes and exact repair command are in the
protocol. Only the launch environment was repaired before rerunning the
same source at new output paths. No mathematical/scientific gate changed.

Original and repair archive parsed headers are logically equal and their
serialized tensor payloads have identical SHA
`c52f2df7c0c7237975fe6824d365a156cc67c2ead594579f9fd969cb3296cac5`.
Their whole-file hashes differ because header object ordering differs; no
tensor or metadata value differs. Both retained archives fit3GiB allocation.

## Next decision

Run277 on consumed121/122 full-model development controls,full-prefix/no-cache
and exact BF16 tied-head probabilities. A pass will require a newly excluded
source cohort with source-only answerability and prospective complete quality
checks; existing261/265/267 observations are consumed. The274 arithmetic is
only a reference equation until actual native complete integration qualifies
its own full quality and accepted>=50tok/s on this same artifact.
