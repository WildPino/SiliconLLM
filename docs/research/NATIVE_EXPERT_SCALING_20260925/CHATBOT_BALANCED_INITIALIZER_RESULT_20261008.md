# Complete balanced-key source initialization: result

8 October 2026. Goal ACTIVE/INCOMPLETE. This completes initialization and
coverage of the 24 conditional blocks; it does not establish chatbot quality.

## Decision and changed uncertainty

**ALL24 initialized and independently verified.** FIT-only balanced parent keys
resolve the inherited-key support failure on the acquired 200-case corpus.
Inference remains 16 parents, selected4, shared512/private128, full896 input,
query32; the support gates remain >=8 distinct FIT operands, >=2 FIT cases and
>=4 novel distinct DEV operands per leaf. No resampling or DEV optimization.

All 384 leaves pass actual saved top4 routing support. Global minima are
377 distinct FIT operands,40 FIT cases,89 novel distinct DEV operands.
Per-layer minimum novel DEV counts, layers0..23:
96,89,96,96,96,96,96,96,96,103,144,115,104,105,127,130,179,154,149,148,120,159,155,96.
These are calibration coverage counts, not independent fresh quality scores.

Construction assigns every unique FIT query to an exact quota at each of20
iterations. These construction labels are **not inference routing labels**.
The actual routing witnesses supply the above support. The greedy capacity
assignment is not a global constrained optimum or a monotonicity certificate.

## Source coefficients and composition

22 layers select source SwiGLU rows on124549 unique NEW FIT operands in total.
Their shared/private G/U/D coefficients contain151388160 promoted source BF16
elements, all independently bit exact. Each new layer has16 distinct private
coefficient hashes; distinct coefficients do not prove distinct useful functions.
Layers0/12 retain their qualified source-derived G/U/D priors. Layers0/1/12
retain projection bytes and starting centers; **every layer gets new balanced
keys**. Warm0/12 coefficient-file routing buffers must never overwrite the new
witness buffers. Warm weights are priors, not new-key-optimal row selections.

93 retained files total683677041B:24 construction traces,24 routing witnesses,
22 selection records,22 initial coefficient files and one incremental journal.
Two warm coefficient files remain referenced without duplication.
[Hydration API](../../../benchmarks/native_expert_scaling/chatbot_whole_checkpoint.py)
implements this explicit composition, but actual all24 hydration/core installation
and dense optimizer allocation have not executed at this record's creation.

## FIRST independent verification and limits

The CPU auditor independently checks NEW x-only43008B row/1792B layer offsets,
byte dictionaries, case/novelty maps, support, saved mass scopes, stop/completion
logic, all new source-copy bytes and all480 balanced construction steps.
Independent scalar lexicographic assignment, row-ordered centroid means and
costs reproduce saved F64 bytes exactly; final F32 centers also match exactly.
F32 GPU query dots satisfy the declared F64 outward rounding envelope. Maximum
observed query gap is7.033845729864652e-6; envelope maxima are up to
0.0016937212877355368. This bounds ordinary FP32 dot arithmetic with TF32 off;
it does not independently reproduce the GPU reduction order.

New projection orthogonality/eigenvalue/sign sanity is checked; covariance and
eigensolver arithmetic are not independently rerun. Actual FP32 inference
score/softmax arithmetic and source activation-energy score arithmetic are not
independently rerun. No full source/student forward, optimizer update, old
scientific namespace replay or endpoint query occurs. The independently exact
construction and copies establish their stated scopes, not transfer fidelity.

## Frozen provenance, actual resources and closure

Prospective [protocol](CHATBOT_BALANCED_INITIALIZER_PROTOCOL_20261008.md), helper,
producer, auditor and terminal helper frozen at
`d5b6ef5c00d7ceb279017927650b411bbe2049ca` before observations.
Actual producer freeze `19c6f4a9803b60691f432e73472d306376fe81b7`;
binding SHA256 `044aff4df33a5fcb9e22e2a5fc732ad539f8aa9fc003812ec17694678f0af77d`.
Actual FIRST audit freeze `42c68834691a3fa5eafce32fc99eb77bc465455d`;
binding SHA256 `40dbfda06e6d7aadc689f5992b24568c1ecd8d612a4dde5f30ef3bb1369cd7c9`.

| Measured item | Producer | FIRST auditor |
|---|---:|---:|
| Compute before final serialization |141.078s|104.922s|
| Launcher family elapsed |169.828s|130.609s|
| Worker OS peak through actual exit |1544687616B|605384704B|
| Conservative family OS peak |1579208704B|637341696B|
| GPU allocated peak |487323136B|CPU only|
| GPU reserved peak |660602880B|CPU only|

Actual exits0; all original input/runtime/resource/foreign gates PASS. Maximum
layer time9.328s versus90s. No fault/repair in this stage. Launcher receipt/stdout
tail lies outside its last memory snapshot, as declared.
[Administrative closure](chatbot_balanced_initializer_terminal_20261008.json)
closes4 exact process instances with no matched OS faults and preserved foreign
SHAs; SHA256 `5a5ad15f4338639752a818dc12b9837c59322816d1c0bb09d4e16a58f3287e7e`.

- [Producer RAW](chatbot_balanced_initializer_20261008.json):
  `61a0ee0b668302f825f90edbd596b411c6d673ff0bdccd65e48836a953e4f452`.
- [Producer terminal](chatbot_balanced_initializer_20261008.terminal.json):
  `88161c48571fff02e131c75c212aeeceb3715b3bcdc8afdde04b9d5a67f293ef`.
- [FIRST audit RAW](chatbot_balanced_initializer_audit_20261008.json):
  `1943947142b11943a356a4a7742d4231bba2abf2a59fca9e0631626ceeba5d12`.
- [FIRST audit terminal](chatbot_balanced_initializer_audit_20261008.terminal.json):
  `856a702f45edf386f0c3f9f406f575ffc3173f8ffbb7933163bfaa280f5cde74`.

## Pipeline consequence

[Next: actual whole assembly](CHATBOT_WHOLE_ASSEMBLY_NEXT_20261008.md), then one
finite final-output learner using the already qualified teacher distributions.
The old local1%/3% failures and both inherited-key failures stay CLOSED. Native
export/new C operator/catalog/canonical chat, fresh donor-relative own-history
tasks AND accepted50 on the SAME artifact, useful n/LUT mass/physical DRAM and
actual additional families/scales remain missing. Goal completion is not claimed.
