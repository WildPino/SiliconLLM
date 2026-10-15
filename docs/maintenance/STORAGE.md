# Artifact storage and document navigation

The source checkout stays at `D:/_THINGS/Progetti/SiliconLLM`. Large artifact
packages are relocated to `E:/SiliconLLM_Artifacts/workspace`, retaining the
same relative tree. Directory junctions at the original D paths preserve the
logical addresses used by existing programs, reports and frozen bindings.

## Where to work

| Material | Logical location | Physical location after verified migration |
| --- | --- | --- |
| Source code, Git database, active navigation | Original checkout | D |
| Results and runtime artifact packages | Original `results/` paths | E, through the migrated package junctions |
| Large datasets and weights | Original `data/corpora/`, `data/phase64/`, `data/phase_data/`, `weights/` | E |
| Legacy donor result/bundle packages | Original benchmark paths | E |
| Frozen native research campaign, including its large checkpoints | Original `docs/research/NATIVE_EXPERT_SCALING_20260925/` | E |
| Current handoff and new maintenance/navigation documents | Original checkout paths | D |

Keep using logical project paths. Do not create a second independently edited
copy of a result package on D. E must be mounted with its assigned drive letter
when accessing relocated artifacts. The scientific research goal remains
paused/incomplete; storage maintenance does not resume model experiments.

## Reading the project

Start with [START_HERE.md](../../START_HERE.md). It is a short reading map,
not another experimental diary. Use the searchable
[catalog](../navigation/index.html) to find historical records by title,
campaign, type and path. Original records are the scientific authority;
catalog classifications describe document discovery only.

Existing report names, paths, identifiers and contents are retained. New
research should keep human summaries short and separate them from machine
receipts and large tensors. Store large outputs in the artifact area; put
only source, concise reports and appropriate small evidence in Git.

Rebuild navigation with:

```powershell
python scripts/build_docs_catalog.py build
python scripts/build_docs_catalog.py check
```

Use the configured Python executable if `python` resolves to another runtime.
The generator has no external package dependency. It follows only the
explicitly approved E storage redirect for the frozen native campaign;
arbitrary reparse points are not traversed.

## Migration and provenance

The prospective [plan](storage_migration_plan_20261015.json) lists exact
directory packages. The migration implementation is
[migrate_artifacts.py](../../scripts/migrate_artifacts.py). Its control area is
`E:/SiliconLLM_Artifacts/migrations/20261015` and its temporary originals are
under the ignored `.codex/storage_migration/20261015/original` tree on D.

Copies are verified by SHA256 before releasing originals. Source hardlink
groups are recreated on E, nested junctions are preserved rather than followed,
and file timestamps/attributes and alternate streams are accounted for.
Each package has an explicit cutover state and per-file receipts. Source
backup deletion is restricted to verified inventoried entries; external link
targets are never recursively deleted.

The plan contains 22 packages. External corpus source clones under
`data/external/`, including their nested Git databases, remain on D. Small
baseline and fixture directories also stay on D. The main Git database,
canonical source code and development environments remain in the checkout.
The [pre-migration baseline](storage_baseline_20261015.json) records free space
and hashes of the current handoff and pre-existing unrelated modified files.

Historical scientific receipts are not rewritten to pretend they were created
on E. Their content hashes remain meaningful. New migration records describe
the old logical path and new physical placement. Physical volume/file IDs
necessarily change, and `Path.resolve()` can now return an E path. An auditor
that binds physical identity or canonical strings must interpret this recorded
relocation; it must not silently replace historical evidence. Recorded old
performance measurements retain their original storage environment.

The root handoff now contains a dated storage addendum pointing to live state
and eventual completion reports. Its exact pre-maintenance bytes are retained
in [a snapshot](snapshots/HANDOFF_PRE_STORAGE_20261015.md), with
[an update receipt](handoff_storage_update_20261015.json) binding both versions.
The relocation auditor checks earlier handoff pins against that exact snapshot
and separately checks the updated root handoff. Original scientific receipts
retain their original hashes and path meaning at the time of issuance.

Inspect progress without launching a model or starting another transfer:

```powershell
python scripts/migrate_artifacts.py status --plan docs/maintenance/storage_migration_plan_20261015.json
```

The transfer heartbeat records its owned PID, current file, verified bytes and
per-package state. A completed package is marked `BACKUP_REMOVED`; a global
`manifest.json` with `COMPLETE` is written only after all packages finish.
The [launch record](storage_launch_20261015.json) records the transfer started
on 15 October and its process identities; it is not a completion receipt.
The separate finalizer performs stored-evidence readbacks, checks historical
document pins, rebuilds navigation, and writes the maintenance result. Its
status is `E:/SiliconLLM_Artifacts/migrations/20261015/finalization_status.json`.
While it waits, `finalization_ownership.json` identifies its process and the
exact migration process creation time. On success it creates
`docs/maintenance/STORAGE_RESULT_20261015.md` and
`docs/maintenance/storage_verification_20261015.json`. On failure it records the
fault on E and does not claim a successful migration. The controller does not
commit generated completion reports; review and stage those explicit files
after it finishes. A single verified continuation is permitted only for the
original transfer's 12-hour time-cap fault, using the longer reviewed cap;
other faults require inspection of the recorded state.
No model or training job is part of either operation.

Do not manually rerun a fresh migration over an occupied destination. Use only
the implementation's explicit verified-resume path if needed. Never use a
general recursive deletion against the original logical junction paths.

## Deliberately preserved

The Git object database and its reported temporary objects are not part of
artifact relocation or history rewriting. Any Git cleanup is a separate
maintenance operation. Model outputs, failed attempts, unique trained weights,
sealed cohorts, existing foreign edits and historical hashes are retained.

This file describes the selected policy. The final migration manifest and
maintenance result determine which packages actually completed; the plan
alone is not a completion receipt.
