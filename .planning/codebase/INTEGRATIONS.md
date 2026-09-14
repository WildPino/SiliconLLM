# SiliconLLM — integrations and external boundaries

## Model and data providers

- Hugging Face models are loaded by Transformers in exporters and MVE scripts. Model/revision/licence checks are explicit in `benchmarks/donor_adaptation/engine/e66_one_byte_at_7b.py` and `benchmarks/phase64/MVE_PREREG.md`.
- Current donor corpus is Qwen-family; the 7B donor is the licence-clean upgrade path recorded in the MVE documentation. Do not substitute a model without re-running the donor/config/ID controls.
- Phase 64 uses TinyStories plus a retrained production BPE/tokenizer path and a sealed Qwen2.5-Coder-1.5B teacher. Data construction is in `benchmarks/phase64/mve/mve_data.py` and `benchmarks/phase64/data/`.

## GPU / remote execution

- Kaggle is the execution boundary for T4 training and teacher scoring. `scripts/kaggle_ops.py` manages account/config switching; `KAGGLE.md` records operational lessons and current H0/H1 state.
- T4 constraints matter: fp16, `sdpa` attention, 12-hour session cap, resume/checkpoint discipline and limited memory. Kaggle run-pack staging is intentionally separate from the committed research apparatus.
- vLLM/Hugging Face backends are used for teacher-forced logit generation. The MVE explicitly distinguishes prefill scoring throughput from autoregressive generation throughput.

## Local filesystem and artifact boundary

- Large immutable donor artifacts, backups and temporary exports are usually outside Git under `D:\_ktmp\...`, with repository-side sidecars/results documenting hashes and construction.
- Gitignored model/checkpoint data is referenced by `docs/COMMUNICATION.md`, `KAGGLE.md`, probes and result JSON. A missing external artifact is a reproducibility blocker, not a reason to silently regenerate with changed inputs.
- `graphify-out/` is a local AST/content index used for navigation. It must be refreshed when the corpus changes; its generated graph is not the research verdict.

## Git as provenance

- Preregistration is pushed before apparatus/results. Use `git log -- docs/research/donor_adaptation` to reconstruct chronology.
- Current working-tree results may be newer than the canonical `INDEX.md` and `SPEED_LEDGER.md`; in particular E64 run 2 is currently an uncommitted JSON/log result.
- Every quoted measurement needs the probe/brief, result file, engine/config arm, donor revision, frozen slice and gate status.

## Integration failure modes already observed

- Kaggle mount paths vary; use recursive discovery rather than hard-coded paths.
- T4 eager attention produced NaNs; SDPA is the known stable path.
- Hugging Face gated content requires owner credentials/accepted terms for the Stack-v2 smoke.
- Windows `torchrun`/libuv and Unicode log handling need platform-specific workarounds documented in `MVE_PREREG.md` and `KAGGLE.md`.

