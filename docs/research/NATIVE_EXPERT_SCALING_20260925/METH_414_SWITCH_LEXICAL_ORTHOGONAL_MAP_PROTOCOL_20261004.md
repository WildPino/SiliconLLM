# M414 protocol: ONE source-weight lexical orthogonal input bridge for real384

Prospective, freeze controller/protocol before FIRST embedding value read, fit
or new map result. Retain first success/failure before a separately numbered
repair. No native/model inference/new network/new quality data/GPU/T4.
Engine opt-in413/current default and qualified Switch binaries untouched.

## Why this new variable advances useful larger n

401 real128+256 union fits RAM: ALL12 banks384 byte-distinct real WI/WO pairs,
candidate22094084608B. Both independently pretrained source artifacts have their
OWN whole quality/rate qualified. All236 core/control names differ. Neither
shape/token equality nor real source slot identities provide a combined model.
403 identity/404 rank32-activation-PCR/406 full affine-activation-ridge input
interfaces failed. These closed variants remain closed. Granite412/413 I8
kernel route now stopped before source acquisition, not further low-bit tuning.

NEW hypothesis: a common lexical coordinate basis can be inferred from the
same-token ORIGINAL source embedding weight rows, rather than per-bank paired
activation regression from18 calibration books. A single constrained orthogonal
map uses ALL32128 common rows, preserves vector lengths and has no intercept,
bank-specific fit or hyperparameter. It tests a specific WEIGHT-based alignment
not tried by403/404/406. Learned embedding spaces may NOT share a global rotation,
and even a useful lexical basis may not align deeper nonlinear/RMS-weighted
representations. Do not assume success or hidden semantic equality.

This is actual-source useful384-enabling work,1.5x256 not10x/~100B or a new family.
Only if ALL12 input eligibility gates pass may a NEW actual source-function
response/output-alignment protocol begin. It never licenses384 selector/model/
quality/rate automatically. FAIL closes THIS global lexical orthogonal bridge
before function/output maps/selector/model; do not tune centering, anchor rows,
rank/regularization, bank-specific rotations or validation-selected variants.
No broad nonlinear/weight-method impossibility claim.

## Immutable inputs and source weights

Original256 cdac1724c078ea4974b4087c59634799561e7835,
original12886c815ec05361a33a8b49fc717277da9c0a4e711.
Existing COMPLETE qualified exports335/338 and380 keep original shared embedding
F32 EXACT (original/source segmentSHA == stored segmentSHA), one physical copy
with encoder.embed_tokens.weight/decoder.embed_tokens.weight/shared.weight aliases.

| Artifact | Bytes | Complete payloadSHA | Original-F32 embeddingSHA |
| --- | ---: | --- | --- |
|256|14818015744|e0e5a940b0150b78d0080815a1fddd2a6b50f011351012ed5b4a48a88ba49056|6f49be16be1b294da85df0218ca5f834cc05eeba97d02a5de47ae88bd1e69398|
|128|7541946880|6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe|0c5ee029008c814daeae89420ff7345c474425847c018d7cef46c814fa7d4a6c|

Each embedding[32128,768],98697216B, encoding0, offset2459343360/2458578432 in
256/128 payload. Exact source/export/raw manifests bind aliases/offsets/source
SHA; fresh whole-payload SHA, manifest SHA/all records and both full embedding
segment SHA/ALL finite values before fit. End size/mtime unchanged. No original
full source rehash claimed: original-F32 identity inherited through qualified
export and exact segment/whole target hashes. No source weights modified.

401 raw d35bf7f82b064593ff8568379fc8106b100d8337c1b860395e1bc6bd83d4e160;
405 all96 paired traces raw
1fb5c222807da2ad8f8e52e8f3d2d5fc64664431edc2332070305fa46cb64788;
406 failed-map raw f1220d8c7eaa4fcc0d0749952c9eeec49cfebb937f679a4969f3599110cd6fd0.
Use qualified raw326/378 source acquisition to freshly bind same spiece.model,
tokenizer.json/special_tokens_map.json bytes, as405. ALL96 paired keys and traces
freshly SHA verified, SAME29 source/14 forced decoder IDs and bank/position ordering.
18 development books /6 validation books, all4 cases, retained/consumed cohorts.
No NEW held-out whole-quality documents consumed or source inference repeated.

## ONE map, no document/activation fitting

Read exact originalF32 weights into F64. ALL paired rows where BOTH embedding
norms are strictly positive included; zero-norm pairs omitted by this fixed
rule. No token/frequency selection, centering or learned scale. Rows normalized
to unit L2. Record32128 total and actual nonzero paired count, at least768.
All vocabulary rows includes padded/special rows; no post-outcome filtering.

Let E256/E128 be these normalized row matrices. Fit C=E256.T@E128, F64 full
SVD C=U diag(s) Vt, Q=U@Vt, the orthogonal Procrustes solution allowing determinant
+1 or-1. Direction for ROW inputs: x256@Q -> source128 coordinates.
One shared768x768 matrix, no bias, no per-bank learned maps. Independent F64
reconstruction<=1e-10, orthogonality normalized Frobenius<=1e-10, optimal
trace certificate abs(sum(Q*C)-sum(s))/sum(s)<=1e-10. Cross-covariance full
identifiability smin/smax>=1e-6 before mapping, all values finite.
F32 Q archive/readback exact, F64 normalized orthogonality<=1e-6.
Map/shared F32 parameters2359296B; each bank589824 active coefficients,
ALL12 maps7077888 coefficients if all consulted. Shared storage counted ONCE,
runtime input map not free; no folding into weights/export/router assumed.

Store QF32,QF64,singular,C in one NPZ, fresh SHA and exact map readback. Fit has
no activation/document input.18 development book activations used ONLY to define
the unchanged source128 mean-only baseline;6 validation books used to score
local alignment. Mean-only value isn't a map intercept. One global map at ALL12
normalized sparse input banks, no fit/development/validation-selection search.

## Controls, unchanged local gates and decision

- Tiny independent known signed-permutation mapping on seven4-D anchors;
  recovered knownQ within1e-12, orthogonality<=1e-12. Wrong token-anchor row
  rotation must cause >.1 relative error; injected nonorthogonal coefficient
  detected by error>.01. No tiny result used to choose actualQ.
- ALL original/source-F32 embedding segments/finite/aliases and complete target
  payload/manifests, tokenizer and96 full paired trace SHA/pairing keys exact.
- Unique lengths/stack/bank/position ordering retained405, all valid input
  states/scores finite. Same384 source applicability retained401, not recomputed
  function identity from synthetic or aliased weights.
- Independent covariance SVD reconstruction/orthogonality/optimal-trace and
  full identifiability bounds above. F32 map readback exact; all predictions finite.
- ALL12 six-book/all4-case validation mapped relative L2 quantiles and identity
  and mean-only controls. Mean-only squared error ratio exact1 within1e-12.
  F32 vs F64 mapped relativeL2<=1e-5. No validation-dependent map choice.

SAME404/406 input gates EACH bank: median relativeL2<=.25,95th<=.50,
squared error / development-mean-only validation error<=.50, F32 numerical
bridge<=1e-5. ALL12 required for eligibility, no partial promotion or deletion.
Fresh map errors only after freeze; existing405/406 results may guide the new
hypothesis but no thresholds are changed. These are calibration eligibility,
not whole-model quality or guarantees of response/output fidelity.

PASS -> separately frozen actual function-response/output-alignment screen.
FAIL -> stop THIS lexical orthogonal interface before any384 selector/artifact.
Apparatus/rank failure retained before repair; no new variant in this number.
Neither proves a384 useful model, CPU LUT/routing/realDRAM/50 accepted tokens/s,
10x/RAM-scale n/~100B or another-family transfer.

## Cost and reproducibility envelope

MAIN<=600s/max checked RSS<=2GiB, fit+evaluation<=120s, map outputs<=32MiB.
Fresh hashes require22.36GB target payload plus about200MB original-F32 embeddings,
manifest/tokenizer/trace reads; fit uses one768 SVD/unit-row covariance and12
validation applications. Record actual read/hash/output/time costs. NumPy/BLAS
runtime identity, threadpoolctl ALL BLAS one thread during fit/evaluation; no
performance/worker-profile claim. No model overlap; exact allowed user daemons
same403/412 exception. No gradient training/new source acquisition/inference/
GPU/T4/newquality cohort. Source-weight basis fit is a learned transformation,
not exact transfer or zero-cost conversion.

Physical filtered-HEAD controller/protocol/401/405/406 raws/export raws/parsers/
helpers bound before values. Fresh result/output locations, no overwrite.

```
.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth414_switch_lexical_orthogonal_map.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth414_switch_lexical_orthogonal_map_result.json
```

Retain first complete raw/report and actual hashes/resource/gates/ALL12 scores.
Existing closed activation interfaces/qualified original models remain immutable.
Final goal remains whole original-relative quality and>=50 accepted batch1 IDs/s
on the SAME new artifact, multiple families/scales and useful many choices whose
routing/LUT/physicalDRAM remain tractable as real capacity grows.
