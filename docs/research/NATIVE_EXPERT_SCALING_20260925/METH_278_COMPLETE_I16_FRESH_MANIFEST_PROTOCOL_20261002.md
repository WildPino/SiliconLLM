# METH-278: new excluded sources for complete I16/private128 quality

## Uncertainty and prospective decision

277 passes only consumed whole-model development;independent complete
prediction/generation/task/semantic quality is unknown. Reuse the original
176/213/261 model-free source/fragment selector after277 pass,adding the
24 consumed261 sources to existing exclusions. Fixed seed
`meth278-complete-i16-independent-278278`,first8 valid hash-ranked sources
per code/prose/technical category,24 total. No model scoring/fitting/source
replacement;source-only answerability must be committed before inference.

Success licenses only source-only answerability. Insufficient pool/failed
bindings preserve the stop;do not alter seed/data/fragment policy to rescue.
No inference has occurred on278 sources at this protocol/code freeze.

## Bound inputs and guards

277 result SHA
`3d997545fc690a5ebb3681e60fb0d3dc190a2a292e94056e5cf0abcb9ed88b35`;
all gates pass,and the unchanged276 archive/composition result/own loader
and76 project helper hashes must match. Actual candidate SHA
`4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`.
261 manifest SHA
`5af457561ad24fb837770735d0dbd0f23baa2f04a0525f38656285c0db7ec050`;
213 manifest SHA
`bcd8295be1b2d62dceaf103f985182bad11ac50dd739a63cfb92d11400b5af10`.
Same fixed historical tree `882bb43f9118df1e4f79f85a6072111897bede9e`;
same cached PG19 train14/parquet SHA
`15f3ecb5e314ce58bbf8eae927b51bf893b9fdabe4a04c8e246547206357a2d5`,
1243 rows. Require old tokenizer fingerprint,all prior manifest/calibration/
teacher hashes and original byte/token/fragment conditions. Original384-char
chat excerpts,unchanged request. Require3200 excluded source IDs before
selection (261's3176 plus24),all213/261 IDs excluded,24 unique new IDs.
Retain provenance,text/token hashes,source commit/parquet,seed and candidate.

New script `meth278_complete_i16_fresh_manifest.py`,manifest
`meth278_complete_i16_fresh_manifest.json`. Only source IDs are new;the source
pool/corpus is unchanged,original donor-pretraining disjointness is unknown.
Project-transfer disjointness is not a new external-corpus claim.

LocalCPU/six Torch threads,expectedabout4min,10min afterimports/12GiB RSS,
16MiB output allocation,no GPU model inference/T4/download. Freeze before
selection,refuse existing manifest/failure. Prior fixed259 semantic/component
cost stops and all native-rate/useful-n/DRAM/family requirements unchanged.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth278_complete_i16_fresh_manifest.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth278_complete_i16_fresh_manifest.json
```
