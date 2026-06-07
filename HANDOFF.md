# SiliconLLM Handoff

## Goal

Build toward a real language model by "pulling an LLM out of the silicon, not imposing an LLM on the silicon."

The working rule is: keep the readout stupid. Gains should come from substrate geometry, memory, local plasticity, and boundary/homeostatic dynamics, while the final readout remains simple and falsifiable.

The current target is to find a stable substrate that improves teacher-forced TinyStories BPB and also survives closed-loop generation gates without attractor collapse.

## Current Progress

### Phase 42: Baseline and Bottleneck Tests

- TinyStories baseline, 64 MB corpus, linear readout: about 2.3197 BPB.
- Pooling tribunal showed `sum` was best. `max`, `range`, and `threshold` were worse, around 2.3505 BPB. Pooling was not the bottleneck.
- A second small reservoir gave a small improvement, around 2.3053 BPB. Temporal memory helped, but not enough.
- Conclusion: the bottleneck was temporal structure inside the substrate, not the final linear readout or pooling.

### Phase 43: SEE-V1, Plasticity, and Local Limits

- Multi-timescale L1 was the first clear win:
  - legacy: 2.3197 BPB
  - `ms_f0.5`: 2.2757 BPB
  - `ms_f0.7`: 2.2777 BPB
- Fast-alpha grid confirmed `fast=0.5`; differences vs 0.3 and 0.6 were under 0.001 BPB.
- Scalar `byte_gain[256]` damaged performance and saturated to clamps. It was archived as a negative result.
- Generation audit showed multiscale memory turned the old garbage loop into recognizable TinyStories-like English, but still with repetition/name attractors.
- Feature clamp/homeostasis produced SEE-V1S. Clamp `c20` was chosen as the stable baseline for generation.
- Oja plasticity on a small subset of fast-band cells improved BPB:
  - SEE-V2, 13 Oja cells, eta=1e-3: 2.2617 BPB
- Scaling Oja to 26 cells produced the Phase 43 teacher-forced champion:
  - C2.A, 26 cells, eta=1e-3: 2.2593 BPB
- However, deterministic generation checks showed C2.A was not a fully stable generator:
  - 32-sample confirmation failed top bigram and alternating-loop gates.
  - Lowering temperature made some collapses worse, proving the basins were structural, not just sampling noise.
- Byte-to-lane routing on M4 lanes, fast band only, was implemented and tested. It was flat:
  - route Round 0 reproduced SEE-V2
  - routing ended around 2.2616 BPB, not a useful new axis.
- Phase 43 is closed. Best teacher-forced artifact is `weights/phase43c2_C2A.bin`, but no Phase 43 model is a fully stable generator.

### Phase 44: Boundary-Gated Hierarchical Memory

The current direction is: the substrate learned local memory; now it must learn when not to update.

Phase 44 adds L2 hierarchical memory above the C2.A substrate while keeping the readout linear.

#### Phase 44.A

- L2 boundary gates were tested.
- Control reproduced C2.A at about 2.2592 BPB.
- Best BPB gates:
  - whitespace alpha0.95: about 2.2499 BPB
  - entropy-high: about 2.2526 BPB
  - punctuation/surprise also improved but less.
- No generation gate fully passed.
- Conclusion: L2 contains useful information, but it worsens attractor control unless homeostatically constrained.

#### Phase 44.B

- L2 homeostasis variants were tested.
- Delta-write was the major signal:
  - entropy-high H0: about 2.2526 BPB
  - entropy-high delta: about 2.2214 BPB
  - whitespace delta: about 2.2276 BPB
- Decay, cooldown, and stack variants were mostly toxic or killed the BPB win.
- Generation still failed gates.
- Conclusion: L2 wants events/changes, not snapshots. But the readout overuses L2 in closed-loop generation.

#### Phase 44.C

- Gain scaling was correctly identified as a no-op under per-dim z-normalization:
  - scaling L2 write by a scalar is canceled by per-dim mean/std normalization.
- The useful grid dropped gain and tested alpha and mix instead:
  - H0 absolute: about 2.2526 BPB
  - delta full: about 2.2214 BPB
  - delta alpha 0.995: about 2.2210 BPB
  - delta alpha 0.9975: about 2.2207 BPB, best teacher-forced
  - delta alpha 0.999: about 2.2217 BPB
  - mix25: about 2.2524 BPB
  - mix50: about 2.2517 BPB
  - mix75: about 2.2488 BPB
- Delta alpha variants failed generation badly, with low self-BPB and loops.
- H0/mix variants were closer but still failed top-bigram/alternating-loop gates.
- Conclusion: L2 signal is real, but the linear readout drinks too much from the L2 block.

### Phase 44.D: Current Active Step

Phase 44.D is the next live tribunal: Readout-L2 Homeostasis.

The intended question is not "is L2 useful?" That is already yes. The question is whether controlling the readout's dependence on L2 can preserve the BPB win without closed-loop collapse.

Planned/implemented configs:

- Controls:
  - C2.A
  - H0
  - mix50
  - delta
- L2 readout homeostasis variants:
  - D1: mix50 scale 0.50
  - D2: delta scale 0.25
  - D3: delta scale 0.50
  - D4: mix50 dropout 0.10
  - D5: delta dropout 0.10

Speedups have been implemented and build-verified:

- L2 feature cache:
  - 8 configs reduced to 3 extractions, grouped by mix/source.
  - One feature matrix plus L2 canonical cache, about 65 GB peak RAM.
  - No two trainer processes should run at the same time.
- Parallel word-gate:
  - deterministic generator runs with explicit `--rng-seed`
  - unique output files
  - fixed aggregation order
  - sequential-vs-parallel MD5 pre-check before the full sweep
- Single-trainer guard:
  - script aborts if another Phase 44 trainer is already running.
- OpenMP was deliberately deferred until a separate reproducibility-gated pass.

Run command:

```powershell
.\benchmarks\phase38-42\phase44d_readout.ps1
```

Expected control values to verify before trusting deltas:

- C2.A: about 2.2593 BPB
- H0: about 2.2526 BPB
- mix50: about 2.2517 BPB
- delta: about 2.2214 BPB

If those controls do not reproduce, stop and debug before interpreting 44.D.

## What Worked

- Multi-timescale L1 memory was the first decisive substrate improvement.
- Feature homeostasis/clamping helped stabilize generation enough to make later tests meaningful.
- Small local Oja plasticity improved BPB without making the readout smarter.
- Scaling Oja capacity improved teacher-forced BPB, but did not solve closed-loop stability.
- Deterministic RNG and reproducibility checks were essential. Earlier generation gates had RNG noise.
- Boundary-gated L2 memory clearly improves BPB.
- Delta-write L2 is the strongest teacher-forced signal found so far.
- Dropping scalar gain from 44.C was correct because z-normalization cancels scalar write gain.
- Speedups that preserve quality:
  - word-gate generation can be parallelized
  - L2 extraction can be cached by source/mix group
  - two full trainers must not run in parallel on 80 GB RAM

## What Didn't Work

- Rich pooling alternatives (`max`, `range`, `threshold`) were worse than `sum`.
- Scalar byte gain was unstable and saturated. Do not repeat scalar input-amplitude tuning.
- Temperature reduction is not a fix for the current attractors; it sometimes makes collapse worse.
- Byte-to-lane routing on M4 fast lanes was flat. Do not expand routing to more lanes unless there is a new hypothesis; local routing is not the current bottleneck.
- Reset, repetition penalty, entropy boost, and similar generation-side thermal hacks were considered toxic or insufficient. They do not solve substrate dynamics.
- L2 decay/cooldown/stack variants did not solve the problem.
- Delta L2 with slower alpha improved BPB but collapsed generation.
- Scalar L2 gain before z-normalization is mathematically a no-op.
- Running two Phase 44 trainers in parallel would exceed available RAM and likely cause swap/OOM.

## Next Steps

1. Run Phase 44.D:

   ```powershell
   .\benchmarks\phase38-42\phase44d_readout.ps1
   ```

2. First check the reproducibility pre-check:

   - If MD5 sequential-vs-parallel fails, stop immediately.
   - Do not interpret the scientific results until launch determinism is fixed.

3. Verify controls:

   - C2.A about 2.2593
   - H0 about 2.2526
   - mix50 about 2.2517
   - delta about 2.2214

4. Interpret 44.D:

   - If an L2 scale/dropout variant keeps a meaningful BPB gain and passes generation gates, promote that as the next SEE candidate.
   - If scale/dropout improves generation but kills most BPB gain, the next axis is state-dependent L2 trust rather than static scaling.
   - If delta variants still collapse even after scale/dropout, delta is probably too predictive teacher-forced and too dangerous closed-loop. Prefer mix/H0-style L2 with gating.
   - If all variants fail, Phase 45 should focus on conditional L2 readout gating or L2 logit contribution caps, not more local substrate tweaks.

5. After 44.D, add an audit table showing:

   - BPB
   - name worst
   - word-run worst
   - top bigram
   - alternating loop
   - self-BPB band
   - percent of logit norm from SEE vs L2
   - top bytes where L2 changes the prediction most

6. Only consider OpenMP after 44.D:

   - parallelize per-class loops only
   - use static scheduling
   - avoid sample-order float reductions
   - require same-run hash and metric reproducibility before using it in any tribunal

## Current Architectural Reading

The silicon has chosen this direction:

- multi-timescale memory
- feature homeostasis
- small local plasticity
- boundary/event memory
- controlled L2 influence on the readout

The current failure is not lack of signal. It is closed-loop context control. L2 knows useful things, especially deltas, but the generator falls into structural attractor basins when the readout trusts those signals too much.

The next real step toward an LLM is therefore not a bigger readout and not a Transformer-like module. It is learning when and how much hierarchical memory is allowed to influence the next byte.
