# NES-01: historical L6 E32 → E128, fixed-token budget

**Frozen before the E128 training run, 25 September 2026.** This is a native
expert-count pilot, not a 10B or pretrained-quality claim. The existing E32
checkpoint is the control; E128 is trained from scratch with the same native
recipe. Training is local on the RTX 3060. No T4 job is requested.

## Decision addressed and inherited evidence

Question: under the historical 32.768M-token training budget, does quadrupling
independently learned experts improve held-out quality while keeping routing
usable and measured C-engine token cost contained? Probe-4 established a trained
E32/top-8 quality point and E4 C-engine parity/rate, but no E128 result.
Phase-64 E128 speed figures are projections. The sparse-slot implementation
already passed WS2 E32 equivalence; [NES-00](NES_00_ASSET_AND_DISPATCH_20260925.md)
checks E128 equivalence and resource feasibility. A positive result licenses a
replicated, broader-domain study; a negative result under equal tokens stops
automatic E256 escalation and directs the next question to exposure, router,
or capacity utility.

## Fixed comparison

| Coordinate | Both arms |
|---|---|
| Core | ArchA D256, N96, H8, L6, SWA layer 5, dt rank 16, V1024, 128-token SWA window |
| Experts | gated-dReLU ternary; h128, top-8 in each of 6 layers; fp32 router/control organs |
| Data | `results/phase55/ids.u16`, 90% training prefix / 10% validation suffix, BPE-1024 and `meta.bin` |
| Optimization | seed 0, AdamW lr 0.003, betas (0.9,0.95), weight decay 0.1, gradient clip 1.0, Switch aux 0.01, coherence 0, BF16 |
| Budget | 4,000 steps × batch 4 × accumulation 4 × sequence 512 = 32,768,000 training token positions |
| Evaluation | same `phase59_moe.py` held-out BPB on first 200,000 validation tokens and 24 routing batches; same validation order |

Treatment: E32 → E128 only, 22,516,672 → 79,287,808 parameters. E128 uses
active-only sparse dispatch because compute-all would add unnecessary training
work; forward and gradient equivalence at E128 is an apparatus gate. E32 was
trained with compute-all, so tiny arithmetic-order differences remain a limit
on strict causal interpretation. The loaded E32 checkpoint has E32/h128/top-8
and the exact data/export hash recorded in NES-00.

Expected mean selections per expert per layer, *not actual exposure*:
8.192M for E32 and 2.048M for E128. The runner records actual per-expert
training selections for E128; the E32 historical run has no such counter.
Equal tokens therefore do not mean equal expert exposure or equal compute.

## Decision rules, set before results

1. **Apparatus:** E128 dispatch output and parameter gradients agree with
   compute-all within 2e-4 on the pinned E128 test; no nonfinite training
   loss/gradient or OOM. A violation makes the run invalid for a scaling claim.
2. **Quality:** clear single-seed signal requires E128 held-out BPB at least
   0.005 below the E32 checkpoint's 0.858854 training-harness value, i.e.
   ≤0.853854. Within ±0.005 is inconclusive. Worse by >0.005 is a negative
   result for this fixed-token budget, not a universal expert-count failure.
   Because this is one seed per E, even a clear signal needs replication before
   claiming a stable population effect.
3. **Routing:** no dead experts and max/mean selection load ≤3 in each layer
   on the fixed 24-batch validation window. Record actual train exposure,
   normalized router entropy, top-8/9 margin, persistence, and unions at
   8/16/32 positions against the iid expectation. A quality gain with a
   collapsed router is not promoted as useful E128 capacity.
4. **Generation:** take 16 validation prefixes of length 128 at offsets
   `ntrain + 512 + 2048*i` for `i=0..15`, generate 128 greedy tokens from
   each arm, and count continuations containing any 8-gram repeated at least
   three times. E128 must have no more than E32+1 such continuations.
   Preserve decoded samples for review; this check supplements BPB.
5. **C execution:** export the trained E128 checkpoint, require the same
   E4-style C-versus-reference fidelity gate before timing, then compare E32
   and E128 with the same CPU build flags, 6 threads, 3 warm timed repeats,
   token slice, and no concurrent training. Gate for this pilot: E128 median
   total decode latency ≤1.25× E32 median. Report router/selection and expert
   times, bytes/token, packed pool bytes, and total rate. This is a small
   pilot gate; ≥50 tok/s here cannot establish target-scale performance.

Only the joint quality/routing/generation/C-cost result can promote an E128
capacity signal. If a component is not measured, mark it open; do not infer it
from another component. Do not use this TinyStories pilot as evidence for code
quality or pretrained transfer.

## Cost, stop, reproduction

The E128 RTX 3060 apparatus smoke took 6.55 seconds for its second 8,192-token
optimizer step. A 4,000-step run is therefore estimated at roughly 7.3 hours
plus validation, checkpoint writes, and variability; cap this invocation at
10 hours. Checkpoint every 200 steps to one rotating optimizer/RNG state.
Stop early only on nonfinite values, OOM, wall cap, or external interruption;
do not stop for an intermediate BPB that looks favorable or unfavorable.

From the repository root, with the verified `.venv` and no other GPU work:

```powershell
& .\.venv\Scripts\python.exe -u benchmarks\phase57\phase59_moe.py --arm moe-gran --experts 128 --sparse-moe --steps 4000 --seq 512 --batch 4 --accum 4 --bf16 --eval-tok 200000 --measure-batches 24 --device cuda:0 --checkpoint-every 200 --max-hours 10 --resume-state results\native_expert_scaling\nes01_e128_resume.pt --save results\native_expert_scaling\nes01_e128_final.pt
```

Record exact source revision/hash, checkpoint hash, raw log, execution times,
and CPU environment in the result. The same command resumes only when the
optimizer/RNG state exists and its config matches. The final trained artifact
and C export are distinct files. Any incomplete/invalid execution remains in
the ledger with its reason.
