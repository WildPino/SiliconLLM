# SiliconLLM — coding and research conventions

## Experimental contract

- Write/push a brief before apparatus and before the first result; amend scope in a dated addendum, never silently.
- Read planted controls before reading new cells. A failed control generally voids downstream cells; a malformed control may be re-specified only with an explicit precedent and before the next run.
- Distinguish `PASS`, `FAIL`, `VOID`, `MALFORMED`, `UNRESOLVABLE`, `OWED`, `ADMISSIBLE`, `FIRES` and descriptive outcomes. Do not promote a runner's printed verdict when the brief forbids it.
- Replication is labelled replication, not discovery. A synthetic/noise-weight speed number cannot be presented as model quality.

## Provenance

- Record donor, revision, config hash, frozen IDs/slice, engine, kernel arm, quant, fold/calibration mode, thread count, sequence length and sidecar facts.
- Prefer engine-emitted `CONFIG` over inferred settings. E66 makes quant and `attn=avx4` assertions mandatory; E64 demonstrated the failure mode when configuration is not compared.
- Keep result JSON machine-readable and probes human-readable. Put long reasoning in probes/ledger, not in `docs/COMMUNICATION.md`.

## Measurement discipline

- BPB uses a fixed frozen slice and explicit chance line; quality-only, ranking and speed estimands remain separate.
- Use paired/interleaved measurements, bootstrap intervals and a contention witness for speed. A dirty machine can produce an estimate or a void, not a clean absolute rate.
- Never extrapolate a law from two points when the programme has registered a refusal rule. E62/E65 explicitly reject exponent-fitting where non-monotonicity is observed.
- Use the denominator appropriate to the format: E60 established separate bandwidth curves for 4 B, 1 B and 0.5 B weights.

## Code style and safety

- Python scripts are procedural, self-contained runners with explicit constants, `selftest()`/assertions and JSON output.
- C code uses compact static helpers, explicit dimensions, aligned/block layouts and runtime flags. Kernel correctness is checked against reference implementations and `--kselftest` where available.
- Scripts should fail closed on missing files, wrong IDs, wrong `CONFIG`, non-finite values, mismatched sidecars or unexpected resume state.
- Do not resume from a result file that the same runner is validating; E65 run 1 and E66's comments document this hazard.

## Documentation style

- Use exact paths, experiment IDs, gate names and numeric units.
- Put corrections in the chronology and state exactly what is withdrawn, what survives and what remains owed.
- Keep `docs/COMMUNICATION.md` short and action-only; use `INDEX.md`, probes and ledger for conclusions.

