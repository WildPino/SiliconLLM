# PQT-001-R1: environment-binding repair

Prospective repair, 5 October 2026, before any donor numerical observation.
Inherit the complete question, four arms, fixed scales, methods, data, splits,
metrics, gates, seed, budgets and independent-audit tolerances from
[PQT-001](PQT_001_PROTOCOL.md). No fitting/quantization/audit algorithm changes.
The original run remains immutable and failed before donor/data access.

## Diagnosed discrepancy and narrow repair

PQT-ENV-003 used the exact same GPU software image on a CPU-only session,
installed the same three pins, and observed:

- Selected metadata and loaded NumPy: **2.1.3**.
- NumPy distribution records: **2.1.3 and 1.26.4**.
- All other selected/loaded critical versions matched the original protocol.
- No import failures, donor/data access or scientific observations.

Raw evidence: `env_003_output/environment_repair.json`, `env_003_fetch.json`,
and exact server metadata. Initial dictionary inventory retained the last
NumPy record, creating an incorrect admission pin. The original failed run
did not save its actual dictionary; the same-image diagnostic explains the
bad binding rather than fabricating missing original runtime evidence.

R1 binds NumPy 2.1.3, checks both selected metadata and loaded numerical-library
versions, and saves runtime diagnostics before any rejection. It also saves
hardware admission separately. The image, package installation, Python, torch,
transformers, tokenizer, hub, safetensors and pyarrow bindings are unchanged.
Run ID/output namespace and bundle provenance are updated explicitly.

## Dispatch and reproduction

acct1, fresh private `wildpino/pqt-001-r1-projection-20261005-002`, T4x2, exact
same pinned image, Internet enabled for pinned input acquisition. Server timeout
5400s; install 900s, experiment 3300s, audit 900s. The remote tiny controls gate
donor access. An additional mismatch is another preserved failure; do not relax
the binding at runtime or overwrite a previous run.

Freeze repaired source commit, then generate a new bundle with:

```powershell
& 'D:/_THINGS/Progetti/SiliconLLM-progressive-runtime/.venv/Scripts/python.exe' scripts/progressive_ternary/build_pqt001.py --run-id PQT-001-R1
```

The builder rejects source/commit byte differences. Commit the fresh bundle,
perform live admission and dispatch with `kaggle_run.py push`, timeout 5400.
Record source and dispatch commits, exact kernel version/ID and terminal output
hashes. Large arrays go to `results/progressive_ternary/PQT-001-R1/remote_001/`;
small raw evidence goes to this dossier's `pqt_001_r1_evidence/` (-text).
Only a passed independent audit permits scientific interpretation.
