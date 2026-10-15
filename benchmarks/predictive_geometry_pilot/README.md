# Predictive geometry campaign

Read the [prospective protocol](../../docs/research/PREDICTIVE_GEOMETRY_20261015/PROTOCOL.md).
The three modules are independent diagnostics of a frozen pretrained donor.

- `geometry.py`: shared differentiable intervention and Fisher/PCA/random study.
- `temporal.py`: independent categorical probes of sensitivity persistence.
- `reuse.py`: training-only frame archives with causal prefix selection.
- `support.py`: Kaggle-only source/data acquisition, source capture, deadlines,
  environment and artifact custody.
- `exclusions.json`: all 176 article identities used in the previous pilot.
- `*_config.json`: immutable source revisions and the fixed fresh cohort.

No tensor library is imported by the local preparation commands. Tensor smoke
tests, model downloads, capture, derivatives and linear algebra run on Kaggle.

```text
python -B benchmarks/predictive_geometry_pilot/test_geometry.py
python -B benchmarks/predictive_geometry_pilot/test_temporal.py
python -B scripts/predictive_geometry_kaggle.py build --mode all
python -B scripts/predictive_geometry_kaggle.py launch --mode all
python -B scripts/predictive_geometry_kaggle.py status --mode all
python -B scripts/predictive_geometry_kaggle.py readback --mode all
python -B scripts/predictive_geometry_kaggle.py output --mode all
```

The initial source freeze includes unchanged acquisition utilities from the
earlier pilot as `common.py`; historical files and results are not modified.
The three accounts each receive one private kernel, with a two-hour maximum.
Maximum conservative allocation is 12 T4-hours for the whole wave. Native Kaggle
queue states are acceptable; never repeat a push to bypass waiting. The launch
attempt ledger prevents blind retries and overwrites. A single
[qualified QR replacement](../../docs/research/PREDICTIVE_GEOMETRY_20261015/REPAIR.md)
is permitted after the original geometry job's numerical failure; its shorter
cap uses only verified unused time from that same geometry allocation. The
three-study wave retains its twelve-T4-hour ceiling.

Receipts and outputs:
`E:/SiliconLLM_Artifacts/campaigns/predictive_geometry_20261015/{geometry,temporal,reuse}`.
Use those paths for custody, outside migrating result directories. The bootstrap
terminal binds the numerical artifact manifest after the numerical process ends;
the terminal itself is a separate receipt, not a file inside that manifest.

`status --mode all` includes the preserved failed geometry attempt and its
replacement. Inspect `--mode geometry-qr-repair` for the active geometry worker.

COMPLETE means execution completed. Numerical control failures, partial jobs and
rank-limited evidence cannot be silently promoted. See each report's explicit
teacher-forcing, single-position, finite-sample and routing limitations.
