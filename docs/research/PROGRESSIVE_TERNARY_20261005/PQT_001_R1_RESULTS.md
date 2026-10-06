# PQT-001-R1: relative method improvement, inadequate absolute fidelity

5 October 2026. **PILOT COMPLETE / RESEARCH INCOMPLETE.** Independent audit passed.
Discrete output-aware refinement deserves further bounded study. Neither refined
candidate preserves the original predictions well enough for deployment: only
about 76% agreement after changing a single last-layer FFN projection.
Progressive compensation alone worsened predictive fidelity despite better layer
reconstruction. No complete-model, generation, native speed or expert-capacity
claim is established.

## Frozen comparison and results

Original Qwen2.5-0.5B donor and pinned WikiText diagnostic, 2048 calibration
activations and 1024 separate evaluation positions in eight fixed windows.
One matrix [896,4864]; four arms share identical fixed group-64 FP32 scales and
2-bit ternary storage. Scientific source `10f1f45`, dispatch `39084b7`, acct1,
private kernel `wildpino/pqt-001-r1-projection-20261005-002`, version 1,
ID 137221989. The original failed run is preserved in [first failure](PQT_001_FIRST_FAILURE.md).
R1 changes only environment binding/diagnostics and run provenance.

| Arm | Held-out normalized layer SSE | KL source→candidate | NLL increase, nats/token | Source argmax agreement | Fit seconds | Relative pilot gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct D | 0.226324 | 0.434158 | +0.452241 | 62.4023% (639/1024) | 0.0892 | Reference |
| Direct + refinement DR | 0.076498 | 0.242738 | +0.201695 | 75.8789% (777/1024) | 6.3108 | Pass |
| Progressive P | 0.119761 | 0.486770 | +0.572115 | 62.2070% (637/1024) | 1.9433 | Fail: KL and NLL |
| Progressive + refinement PR | 0.063529 | 0.211989 | +0.232268 | 76.1719% (780/1024) | 7.9066 | Pass |

DR reduces held-out layer SSE by 66.20%, PR by 71.93% relative to D. Both improve
local SSE and KL in all eight windows. P lowers layer SSE by 47.08% but worsens
KL in all eight windows. This is direct evidence against choosing the conversion
from layer reconstruction alone. PR has lower layer error/KL than DR; DR has
lower NLL increase. There is no single best arm across the declared metrics.

The prospective pilot gate required improvement over direct rounding, not
near-perfect agreement with the original model. Passing that gate authorizes a
next research screen, not promotion. A remaining 23.8–24.1% argmax disagreement
and +0.20–0.23 nats/token are material losses. No gradient optimization or
pretraining was performed. All four fitting procedures used the source
calibration activations; held-out activations were accessed after final arm
representations were frozen.

## Storage, cost and numerical verification

Every arm: 1,089,536 packed code bytes + 272,384 scale bytes = **1,361,920 bytes**.
Two NumPy file headers bring each stored projection to 1,362,176 bytes.
The payload is 15.625% of the projection's FP16 bytes (8,716,288), not of the
whole model. Actual experiment evaluation dequantized this representation on
GPU; this does not demonstrate a ternary native runtime speedup.

All-arm fitting including setup/checkpoints: 16.2817s. Remote subprocesses:
installation 16.2047s, experiment including acquisition/evaluation 93.3131s,
independent audit 10.3968s. Peak allocated GPU memory 2,607,312,384 bytes;
final process RSS 3,868,307,456 bytes. These are run-specific observations,
not local CPU timing or generalized performance estimates.

Independent process on cuda:1 decoded every packed code, reloaded original
weights/readout from the admitted safetensors, reconstructed all layer outputs
and checked storage and metric aggregates. FP64 full-vocabulary readout on the
fixed 32-position numerical-audit subset passed: maximum NLL difference
8.560e-6, maximum KL difference 2.144e-6, no argmax disagreements. Layer
reconstruction relative RMS <=2.608e-7. The primary prediction comparison
still uses all 1024 positions; the subset is an apparatus check.

Quota after retrieval: acct1 0.0497058333 h used, zero reserved, 29.9502941667 h
conservative unreserved; acct2 and acct3 each 30 h, zero used/reserved. This
cumulative acct1 usage includes the environment probe and failed first run.
No factor of two is applied to the API quota; no reset date is inferred.

## Retained evidence and reproduction

- [Frozen protocol](PQT_001_PROTOCOL.md), [R1 binding repair](PQT_001_R1_PROTOCOL.md).
- [Independent gate adjudication](PQT_001_R1_ADJUDICATION.json), [retention record](PQT_001_R1_RETENTION.json).
- Small raw evidence in `pqt_001_r1_evidence/`, including per-position records,
  independent audit, runtime, input/source manifests, all stdout/stderr and events.
- Complete retrieval: `results/progressive_ternary/PQT-001-R1/remote_001/`;
  46 files, 126,592,799 bytes. Every retrieved size/hash matches
  `pqt_001_r1_fetch.json`. Large arrays remain local outside Git; their hashes,
  tokens/source identities and conversion code permit reconstruction.
- Immutable bundle `scripts/progressive_ternary/pqt001_r1_bundle/` embeds the
  seven source/protocol files and validates their exact committed bytes.

```powershell
& 'D:/_THINGS/Progetti/SiliconLLM-progressive-runtime/.venv/Scripts/python.exe' scripts/progressive_ternary/adjudicate_pqt001.py --run-dir results/progressive_ternary/PQT-001-R1/remote_001 --out <new-adjudication.json>
```

Use a new output filename; never push an existing kernel reference or overwrite
the original evidence. The adjudicator rejects incomplete bootstrap, failed
audit, changed hashes, invalid point counts, drifting source reference, incorrect
metric aggregation or disagreement with the preregistered gate.

## Adaptive next question

A post hoc descriptive analysis, explicitly not a new validation experiment,
finds that the held-out residual mean accounts for 58.00% of D's residual energy,
46.32% of P's, 9.14% of DR's and 10.61% of PR's. See
`PQT_001_R1_DIAGNOSTIC.json` and `scripts/progressive_ternary/diagnose_residuals.py`.
No correction was fitted using those evaluation residuals. Refinement has
already removed much of the mean error; an offset alone cannot be assumed to
restore high fidelity.

Next bounded screen should compare relevant int4 compression, adapting the
existing scale parameters, and an explicitly counted calibration-only output
offset. Carry DR and PR because their likelihood ordering differs. Use different
evaluation windows, freeze absolute fidelity gates as well as relative gains,
and include a changed-state generation diagnostic before larger expert runs.
Only after that evidence should a real pretrained-expert test and native codec
cost be prioritized. The all-ternary/no-correction recipe is not currently
qualified for native-expert-scaling.
