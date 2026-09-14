# SiliconLLM — concerns, debt and handoff risks

## Immediate research blockers

- **Clean CPU hour:** `docs/COMMUNICATION.md` asks for an idle-machine window to turn the E63 paired estimate (~49 tok/s, CI roughly 45.5–51.1) into a quotable absolute measurement (`G-E63d`). This is the most important unresolved speed number.
- **E64 promotion:** the versioned probe still says E64 is void, while the uncommitted `benchmarks/donor_adaptation/engine/results/e64_carve_on_int8_run2.json` passes the repaired controls and reports `CARVE-IS-DEARER-ON-INT8`. The run-2 evidence must be audited, incorporated into the probe/ledger, and committed before being treated as canonical.
- **E66 execution:** `docs/research/donor_adaptation/briefs/BRIEF_E66_ONE_BYTE_AT_SEVEN_BILLION.md` and `benchmarks/donor_adaptation/engine/e66_one_byte_at_7b.py` are preregistered/apparatus-ready; no result/probe is present. Do not quote a 7B int8 quality outcome.
- **H1 completion:** H1 passes `TRAINING-HELPS` on an 8-layer donor branch, but only ~390 cumulative steps were obtained against a larger intended budget. The router is still improving; experts flatten. No full-stack/free-running/10B claim follows.

## Measurement and provenance risks

- `INDEX.md` and `SPEED_LEDGER.md` are canonical but can lag uncommitted current results; compare file timestamps, `git status`, and result JSON.
- Engine flags are scientifically material. Record and assert `CONFIG`, especially `attn`, `attnr`, `fexp`, `mvacc`, `quant`, threads and sequence length.
- Large binaries/checkpoints live outside Git or under ignored paths. Hashes and sidecars are necessary but not sufficient if the source artifact disappears.
- Current working tree is dirty with E64/H1/E65/E66 logs/results and `KAGGLE.md`; preserve those changes and do not reset/clean broadly.
- The generated graph was initially stale relative to the latest commits; refresh it after research/doc changes and treat generated graph edges as navigation, not adjudication.

## Scientific risks already demonstrated

- Post-hoc ternarization/selection can destroy ranking even when BPB is near a floor; E38's perfect oracle is above chance, closing the donor post-hoc selector route for this donor.
- Speed shape results with synthetic/noise weights are not quality evidence; E39/E40 prove the 10B speed shape, while E65 prices its rank fraction on real donor weights.
- E60's precision cliff is discontinuous: 1 B/weight is nearly free in fidelity, half-byte can consume most of the dense→chance gap. Do not interpolate between formats.
- Non-monotonic ladders (E12, E20, E27/E65, E62) invalidate smooth extrapolation across scale or rank fraction.
- Selection paths amplify tiny score perturbations; dense-path tolerances cannot automatically gate top-k/carve paths.

## Engineering debt

- Provenance is uneven across historical runners; some result JSONs lack engine/config fields even when source code hard-codes them.
- Several legacy runners can resume from their own output or mutate published records; new runners should copy E66's separated-artifact pattern.
- Archive/current boundaries are broad; avoid searching `archive/` first for a live answer unless the ledger points there.
- The Phase 64 MVE has known external blockers: gated Stack-v2 content, Linux DDP differences and Kaggle session/runtime constraints.

