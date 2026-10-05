# METH-453 result: exact sparse native WO passes the local screen

5 October 2026. Goal ACTIVE/INCOMPLETE. Scientific sources/protocol frozen1661547;
operational single-run instructions24552aa. ONE session25101, terminal exit0.
Raw `meth453_switch_sparse_wo_result.json`,1,151,920B, SHA256
`63d0fca69ad180b5024e1152b3d2fd9e13699fcfc3efd662850e011387ba3bbf`.
No source/protocol changed after observation. Previous447/448/451 failures remain.

## What this establishes

ALL10 apparatus and ALL7 prospective local feasibility gates PASS. Retain exact
451 compact WI bytes; ORIGINAL I8 WO in column storage, skip only actual zero
A16 codes. ALL128 transposes/scales/saved bank exact, ALL336 original full heads/
states/A16 codes exact, ALL672 correct/wrong WI raw states exact451, ALL1008 sparse
WO outputs exact independent FULL I64 sums AND original native WO. Signed I32
512-product bound, I64 combination, scale/cast/fault and line-set tests qualified.
672 actual WI bases independently checked with maximum F64 relative error0.
Every computed vocabulary KL independently qualified; removed full heads exact451.

    sum_j q[r,j]*c[j] = sum_(j:c[j]!=0) q[r,j]*c[j]

Here deleting zero integer terms loses no information. WI remains approximate;
this result does not make the full transformed model exact. No WO basis mixing
is applied, preserving the original-coordinate ReLU zero structure.

## Local prediction and identity

| Control | Mean original-posterior KL | Changed argmax /336 |
| --- | ---: | ---: |
| Compact WI /native sparse WO, correct ID |.00040172282368837264|3|
| Same representation, ID+1 |.18146473108106445|40|
| Removed final expert contribution |.03270409315263313|21|

Correct median KL.0000398190725413583, p95.00201963120615134,
max.009087177982041172. Correct changed positions[30,108,217];3/336=.00892857
meets the unchanged<=.01 gate. ID intervention increases mean KL by
.1810630082573761, above>=.01. Source IDs and probabilities are held fixed;
these are consumed local prefixes, not independent generalization/useful-n trials.

| Book | Correct mean KL | Changes | Logical footprint ratio |
| --- | ---: | ---: | ---: |
|18|.00039736644573183666|1|.3474644440839282|
|19|.00035771482923330325|1|.3451960886560984|
|20|.00040155169183801715|0|.34592323707549216|
|21|.0006372130899879848|1|.3380984248014586|
|22|.0004038308878150401|0|.3424931823800254|
|23|.00021265999752405408|0|.33803179367139463|

Compared with452's rotated-source-WO WI-only diagnostic, mean KL differs by
2.5489706141428817e-8 and global changes are both3. This is descriptive similarity;
the WO input basis/A16 rounding differs and no bit identity is inferred.

## Sparsity, stored bytes and logical footprint

Correct WO mean nonzero A16 codes222.41666666666666/3072 =7.2391% active;
p95=418 (13.6068%), max=766 (24.9349%). Actual zeros are measured AFTER compact
WI and original ReLU/A16. All correct/wrong codes/scales and per-expert observed
column unions retained. All128 real functions have distinct stored fingerprints;
106 naturally selected IDs do not constitute106 independently useful functions.

Nominal stored bank472,253,184B /original605,945,856B =.7793653167585983.
Actual archive472,254,456B SHA256
`226619327842c26b30edf0522ee3a5d8c8f6f11a60d330f6ddbf4e375809a167`.
Higher WO precision uses the explicitly NEW80% storage budget. Failed both-I4
recipes'60% storage and predictive requirements have not been changed.

Declared coefficients/scales plus122,128B prospective per-function workspace
give logical mean ratio.34286786177806616, p95.3745977990482371,
max.4310546452520009 relative to original4,733,952B expert payload. All book
means<.60, p95<.65. Original WO offset phases are all0; future candidate columns
require64-byte alignment. NPZ is not that runtime layout. These are unique
logical addressed bytes/lines, NOT measured traffic, latency, cache hits or DRAM.
LUT table/build/gathers, complete core/router/head/prefill/cache/glue remain open.

## Resource and retention

104.812s total =8.187s admission+96.625s numerical; peak2,532,392,960B.
Streamed file hashing9,220,905,602B;10 archives total692,141,672B.
No GPU, optimizer updates, new coefficient roundings, engine edits or downloads.
Original source payload/engine374/389 binaries and approved daemon preserved.
Full output paths/bytes/hashes are in raw; metadata-only retention audit follows.

## Decision and whole-project reassessment

Proceed to NEW frozen C arithmetic/cost experiment, comparing direct packed WI
with exact pair-activation LUT WI, each followed by zero-only column WO. Charge
Walsh, A16 quantization, ReLU, scan/indices, LUT build/gathers, integer partials,
scales and output writes. Compare matched original dense native FFN, verify C
states/codes against retained453 before accepting any timing. Real working set
and hardware DRAM still need measurement; logical sparsity is insufficient.

This resolves a representation/execution question locally. Still required:
all-bank composition and changed routes, fresh original-relative prediction/
generation/tasks, SAME artifact>=50 accepted batch1 IDs/s, two real useful-n
increments with correct winner AND softmax mass, physical CPU LUT/routing/DRAM,
and actual other-family/~100B transfer. Stored capacity grows with RAM, but useful
capacity and active cost have not yet been proved to grow as the user intends.
