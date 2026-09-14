# SiliconLLM — repository structure

## Top-level map

```text
docs/
  research/donor_adaptation/   briefs, probes, audits, decisions, prior art, INDEX, SPEED_LEDGER
  research/                    broader CPU/long-context/architecture dossiers
  phases/                      phase design/history documents
benchmarks/
  donor_adaptation/             E/P/R/T/D/F runners, exporters, artifacts, logs, result JSON
  phase60/                      general engine and scale-up benchmark apparatus
  phase62/phase63/phase64/      corpus, verification, MVE and scale-up experiments
  phase55/phase57/phase60/      historical/current engine benchmark families
archive/                        superseded code, early phases and legacy reports
scripts/                        README charts, Kaggle operations and canonical evaluation
data/                           corpora and internal data manifests
assets/weights/results/logs/    large or generated runtime material
graphify-out/                   generated knowledge graph and report
.planning/codebase/             this onboarding map and research catalog
```

## Donor-adaptation layout

- `benchmarks/donor_adaptation/engine/`: central C engine, exporters, E-series runners, logs and `results/`.
- `benchmarks/donor_adaptation/density/`: D0/D0c/T1/T2/T2b/T3 density, coactivation and ternarization data/results.
- `benchmarks/donor_adaptation/ternary/`: ternarization/rank/depth/rotation scripts and logs.
- `benchmarks/donor_adaptation/r2/`: principal-angle and low-rank screening artifacts.
- `benchmarks/donor_adaptation/f1/`: gate predictor and tables.
- `benchmarks/donor_adaptation/p1/`: packing and staged adaptation apparatus.
- `benchmarks/donor_adaptation/s1/`: H0/H1 training, calibration, evaluation, bundles and results.
- `benchmarks/donor_adaptation/configs/`: donor/config manifests.

## Documentation layout

- `docs/research/donor_adaptation/briefs/` contains preregistrations and addenda.
- `docs/research/donor_adaptation/probes/` contains published experiment readings; there are probe gaps for some ledger-only E43–E57 entries and for E66 (currently prereg only).
- `docs/research/donor_adaptation/audits/` contains controller/reproduction/pre-registration audits.
- `docs/research/donor_adaptation/decisions/` contains stage decisions and the H2T/T4 healing proposal.
- `docs/research/donor_adaptation/prior_art/` contains literature/prior-art surveys; it is context, not local measurement.

## Naming conventions

- Experiment IDs are family-prefixed: `P`, `R`, `T`, `D`, `F`, `E`, `H`.
- Probes use uppercase descriptive names (`E64_CARVE_ON_INT8.md`); runners/results use lowercase IDs (`e64_carve_on_int8.py`, `e64_carve_on_int8.json`).
- `BRIEF_*.md` is preregistration; probe is post-run interpretation; `*_addendum*` changes scope before the next read.
- `*_void*`, `*_smoke*`, `*_control*`, `*_audit*` are evidence about apparatus or controls, not automatically a new scientific finding.

## High-value navigation order

1. `.planning/codebase/RESEARCH_CATALOG.md` (this map's research index).
2. `docs/research/donor_adaptation/INDEX.md` (canonical narrative).
3. `docs/research/donor_adaptation/SPEED_LEDGER.md` (speed chronology and arithmetic).
4. The cited probe, brief/addenda and result JSON.
5. Runner/exporter source and engine `CONFIG`/sidecar for provenance.

