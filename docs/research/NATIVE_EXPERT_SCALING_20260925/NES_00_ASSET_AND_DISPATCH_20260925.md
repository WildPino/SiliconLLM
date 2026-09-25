# NES-00: asset binding and E128 apparatus feasibility

**Question.** Is there a matched native E32 anchor locally, and can the
existing sparse-slot dispatch execute E128 with equivalent gradients and
reasonable local resource use? This record establishes apparatus only. It
does not evaluate learned E128 quality, routing, or C-engine rate.

## Inputs and identity

Branch `research/native-expert-scaling`, starting commit `1d65f6a`.
`phase59_moe.py` SHA-256 at final apparatus probe:
`e608eb99d921312ebbb9a4bde17a5a95a213cdc620aa57bc3d24dae281162193`.
The sparse class was moved without changing its mathematical path from
`phase64/mve/mve_model.py` into `phase57/phase59_moe.py`; the MVE imports it
from there. No donor-adaptation code or checkpoint was used.

| Local asset | SHA-256 | Status |
|---|---|---|
| `results/phase57/moe_gran.pt` | `356478b2f63ace9d6ec429056fee5c0e14e1c0f36253edbca36300eba02d4525` | Matches release manifest; E32, L6, top-8, h128, 22,516,672 parameters; saved BPB 0.8588540 |
| `results/phase55/ids.u16` | `33b8cba2a26653599f7f87a4d8e05b38be051ba850d4cbc5d09b561aae133889` | Matches release manifest; 32,723,845 BPE tokens |
| `results/phase55/meta.bin` | `d7dd5e183a3740d3e3e8968aed4868afe7d039355792754fdfc60dc9a2489a4f` | Present |
| `weights/bpe1024.bin` | `97de713392eabfa4dff6ebd9dd5502f0d2f3e93232db8f349860bb57f3148f9b` | Matches release manifest |
| `results/phase60/e4_model.bin` | `52086303c95ddab3592ae8285a9420d93f7424c3a17dbcf234ac3769fe0347e1` | Matches release manifest; E32 C export |

No trained E128 checkpoint appeared in the reviewed `results/phase57`,
`results/phase60`, or `kaggle_mve` file lists. E128 training is genuinely
missing locally. The L8/code ladder is a different configuration and cannot
use this L6/TinyStories checkpoint as its E-only control.

## Apparatus checks and raw results

Command, from repository root:

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\dispatch_probe.py --device cuda:0 --experts 128 --steps 3 --batch 4 --sequence 512 --out docs\research\NATIVE_EXPERT_SCALING_20260925\dispatch_probe_e128_fullstep_rtx3060_20260925.json
```

The [raw JSON](dispatch_probe_e128_fullstep_rtx3060_20260925.json) binds
PyTorch `2.6.0+cu124` and the code hashes. On the RTX 3060, sparse versus
compute-all at E128 differed by at most `1.05e-9` in output and
`4.62e-7` in relative parameter gradient, under the predeclared `2e-4`
apparatus tolerance. Batch 4 × sequence 512 sparse forward/backward used
3,234 MiB peak allocated and took 1.59–1.77 seconds on the two warmed
probe steps; all gradients were finite. E128 has 79,287,808 parameters.
The earlier [GTX 1660 check](dispatch_probe_e128_gtx1660_20260925.json)
also passed equivalence, but its source hash predates the final runner guards;
the RTX 3060 check is the current qualification.

A two-step training-path [smoke log](e128_smoke_20260925.log) exercised
optimizer, data split, BF16, routing telemetry and checkpoint save. Its
2048-token BPB and collapsed-looking router after only two steps are **not**
scientific scaling results. The checkpoint is an ignored apparatus artifact
under `results/native_expert_scaling/` and is not the NES-01 candidate.

Resume control: a one-step wall cap wrote optimizer/RNG state
([part 1](resume_smoke_part1_20260925.log)). The first resume attempt was
**VOID as apparatus** because `map_location=cuda:0` moved CUDA RNG byte tensors
off CPU, where `set_rng_state_all` requires them ([failed log](resume_smoke_part2_20260925.log)).
The repair converts RNG tensors to CPU, then resumed at step 1 and completed
step 2 with the same CE, aux, and gradient norm as the uninterrupted smoke
([repair log](resume_smoke_part2_repair1_20260925.log)). Its requested
`eval-tok=512` exposed a separate zero-window BPB bug; the reported 0.0000
is **invalid**. The evaluator now enforces at least one full window, and a
[re-evaluation](resume_smoke_eval_repair1_20260925.log) returned nonzero BPB.
Neither smoke BPB enters NES-01 decisions.

## Decision and limits

The local historical pair is the smallest interpretable expert-count test:
reuse the verified E32 anchor and train E128 on the same L6/TinyStories recipe.
The sparse path passes the E128 numerical apparatus check and fits RTX 3060
memory at the intended batch/context. The training cost is still real: a
4,000-step run is estimated near 7–9 hours, and each E128 expert receives
about one quarter of E32's mean training selections under equal tokens.
Proceed only under [NES-01's frozen rules](E128_EQUAL_TOKEN_PROTOCOL_20260925.md).
