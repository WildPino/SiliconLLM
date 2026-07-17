# benchmarks/in_research — The Inventor's side-lab

Exploratory, desk-level probes. Companion to `docs/in_research/` (reports live there).

**Rules (declared):**
- Scripts here are **read-only toward the rest of the repo**: they load existing checkpoints/sources and
  write outputs ONLY under `docs/in_research/`. No file outside this folder is ever edited.
- Nothing here is a phase, a gate, or a claim: numbers are exploratory inputs for future pre-registered
  probes. Promotion path: idea → number here → owner decision → real phase with prereg discipline.
- CPU-only, no GPU, no training. Never interferes with runs in progress.

Contents:
- `s2_weight_entropy.py`  — S2: empirical entropy of the ternary weights (moe_gran.pt expert pool +
  sp58_base.pt dense MLP), vs the current 4-bit/weight engine pack and the 5-trits/byte pack.
- `s3_expert_residuals.py` — S3: cross-expert redundancy of the MoE pool (mode-centroid agreement,
  pool-wide nearest-row match, fp32-domain cosine NN) → is delta-coding against shared structure viable?
