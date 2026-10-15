# Verified storage continuation

15 October 2026. The owner explicitly requested restarting the stopped transfer.
The migration is incomplete. Current canonical state is on E, not this dated note.

## First fault and qualified correction

The initial migration stopped at 19:49:03 UTC with `source metadata changed` for
`kaggle_rung1/_rttmp/.kaggle/uploads`. Seven packages had completed cutover and
backup removal. Their logical payload totals approximately 52.96 GB. Approximately
84.48 GB had per-file verification receipts across the first eight packages.
The interrupted eighth package had all 784 file receipts before its source-tree
metadata check failed. Its ordinary source directory remained on D.

A fresh complete source inventory of that package reproduced only two allowed
differences: ordinary directory `st_size` changed from 0 to 4096 for
`_rttmp/.kaggle/uploads` and `_ziptmp/.kaggle/uploads`. Child membership, file
identities, file sizes, timestamps, attributes, streams and link records matched
the frozen expectations. Target receipt identities and metadata also matched.

Ordinary NTFS directory length is bookkeeping rather than scientific payload.
The corrected implementation excludes only that field for ordinary directories
from source/backup metadata comparisons. Directory membership, identity, mtime
and attributes remain checked. Exact file lengths and SHA256 remain mandatory.
The immutable original inventory and historical receipts are not rewritten.
Five lightweight mocked metadata checks verified this boundary before restart.

## Active continuation

The Windows task `SiliconLLM-StorageResume-20261015` launched
`scripts/resume_storage_migration.py run`. It owns the explicit migration
`run --resume-verified` and requests system-idle sleep inhibition while waiting.
No display-awake request, recurring trigger or login credential was added.

Observed identities at launch:

- Supervisor PID 22468, parent Windows service PID 2760.
- Migration PID 30356, creation FILETIME `134365686936934048`.
- Replacement finalizer PID 27616, launched independently by Windows task
  `SiliconLLM-StorageFinalize-20261015`, parent service PID 2760.

These service-launched tasks are independent of the desktop chat's process tree.
Closing the chat does not stop them. Keep the machine powered and the volumes
attached; explicit shutdown, sign-out or task termination is outside that claim.

The first resumed finalizer rejected process adoption because Windows wall-clock
sampling placed the after-Popen timestamp approximately 1.5 ms before the OS
creation timestamp. That fault is preserved. The replacement binds the exact
raw Windows creation FILETIME and resume-launch SHA256, avoiding ordering jitter
while strengthening the PID identity check. The migration worker was not stopped
or relaunched for this correction.

The supervisor still retains its original finalizer exit status; its eventual
`resume_controller_terminal.json` can therefore report the preserved failed
adoption. The replacement finalizer owns the authoritative canonical
`finalization_status.json`; use that file and its evidence for final admission.

## Receipts and progress

Control root: `E:/SiliconLLM_Artifacts/migrations/20261015`.

- `resume_qualification.json`: exact old-input hashes and the two directory-size
  differences; no file-payload drift is accepted by this qualification.
- `resume_launch.json`: new migration PID, arguments and implementation pins.
- `resume_process_identity.json`: exact OS process creation identity.
- `finalization_ownership.json`: replacement finalizer and adopted worker.
- `resume_finalizer_replacement.json`: preserved adoption fault and replacement.
- `history/initial_failure_20261015`: byte-exact old state, status, receipts and
  finalizer outputs, including preserved original terminal files.
- `history/resume_adoption_clock_failure`: first resumed adoption fault.
- `resume_copy.stdout.log` / `resume_copy.stderr.log`: active worker output.
- `state.json` and `status.json`: current package phases and heartbeat.

`REVALIDATING_COMPLETED` means previously completed target files are being hashed
again. Current-attempt counters restart from zero during this pass; they do not
mean the old E copies were discarded. Existing incomplete-unit file receipts are
also rehashed against source and target before cutover. No duplicate copy of an
accepted file is silently substituted. No occupied unreceipted target is removed
automatically.

At the restart inspection D had 111,228,465,152 free bytes, approximately 111.23
decimal GB. Most planned payload still resides on D. Source space is released
after each whole package's verified cutover; large-package copying can therefore
continue for hours without a matching increase in D free space. Do not manually
delete source directories, migration backups, E targets or junctions.

```text
python -B scripts/migrate_artifacts.py status --plan docs/maintenance/storage_migration_plan_20261015.json
```

Only a complete global manifest plus PASS from `finalization_status.json` admits
the finished relocation. The existing independent audit and historical readbacks
remain required. No model or numerical training is part of storage continuation.
