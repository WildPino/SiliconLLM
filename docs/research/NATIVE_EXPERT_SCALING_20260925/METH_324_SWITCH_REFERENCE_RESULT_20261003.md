# METH-324: official isolated reference qualified on fixed controls

**PASS, apparatus scope only.** Freeze0d182af, observations e764540. Raw
[result](meth324_switch_reference_result.json) SHA256
9420d46235cdcf2e48a1153ae250aac59deecee12d6d243e0319ed33fb91f082.
[Protocol](METH_324_SWITCH_REFERENCE_PROTOCOL_20261003.md),
driver benchmarks/native_expert_scaling/meth324_switch_reference.py.

Own4.57.6/Hub0.36.0 environment created in results/native_expert_scaling/
meth324_switch_reference/venv; own imports precede appended original .venv
runtime path. Actual Torch import from existing2.6.0+cu124 verified, all direct
base requirements satisfied; full distribution versions/metadata/RECORD hashes
in raw result. Existing5.13.1 environment not changed. Exact pure wheel SHA256:

| Package | Bytes | SHA256 |
| --- | ---: | --- |
| Transformers4.57.6 |11993498|4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550|
| Hub0.36.0 |566094|7bcc9ad17d5b3f07b57c78e79d527102d08313caa278a641993acddcb894548d|

Installed Switch model source byte-equal to official release URL, SHA256
5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83;
config86a5e37411e3fa8a8d3c13f3c07f9ee7e74a6033f0da57d681cde80de8071d51.
Raw wheel manifests, official source copies and command logs retained locally.
Total streamed download12,722,008B, setup+qualification138.406s. Worker52.813s
including imports/all META cases, endRSS1,312,391,168B; child sampled peaks in
raw record. No learned model weights or GPU work.

## Fixed gates

All five frozen gates PASS:

- Same323 direct3D capacity1 mask now EXACT expected two admitted/two dropped;
  exact selected probabilities, ORIGINAL raw logits and deterministic eval.
- Same three sparse shapes give logical relativeL2 **0.0**, dropped output
  norm **0.0**, exact raw metadata. Omitted-capacity fault on both multi-token
  inputs.8327527 detects the installed323 regression; zero-output fault1.
- Same tiny actual encoder/fullhead gives finite[1,1,32] with nonempty cache.
- Separate explicitly unsaturated capacity64 context, SAME tiny weights:
  cached/full causal prefix max relative logitL2 **2.20392735e-7** versus1e-5;
  all greedy choices equal, self-cache1..4/cross-cache3 at bothlayers correct.
- Entire321 shapes/names/unique counts/top1 router geometry match for all
  three unmodified original configs:3320/6392/6632 names,12/12/24 routers.

Capacity is PER CALL. Unsaturated cache closure does not prove saturated
full-prefix versus tokenwise equivalence; source capacity semantics must be
preserved explicitly in any scored reference/native execution. Broken5.13.1
observations remain retained and unsupported for source quality scoring.

Decision: permit separately frozen bounded actual source manifest/headers;
not a whole pretrained reference qualification, converted artifact, knowledge
preservation or CPU LUT/rate result. Exact operator/whole model qualification
against acquired source still required. Goal remains open.
