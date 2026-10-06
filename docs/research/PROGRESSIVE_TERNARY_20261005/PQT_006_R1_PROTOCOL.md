# PQT-006-R1: process-inventory admission repair

Prospective, 5 October 2026. Inherit all arithmetic, artifacts, controls,
correctness/timing gates, compiler/flags, sampling, seeds, budgets and resource
reservation conditions from [PQT-006](PQT_006_PROTOCOL.md). No numerical change.

Source `e07a85b` compiled successfully, DLL 74,240 bytes. The first owned
qualification stopped after 2.110s at process admission, before sparse controls
or original expert access: `CalledProcessError`, PowerShell exit 1. Requesting
several process names with `Get-Process -Name ... -ErrorAction SilentlyContinue`
still produces a failing process exit status for absent names. This is an
inventory-command error, not evidence of an active competing workload or a
numerical failure. Preserve `qualification_001` and the first qualification
record unchanged; no timing occurred.

R1 enumerates processes and filters the relevant names in `Where-Object`.
Missing names therefore yield no matching row instead of an error; actual
other relevant PIDs still block qualification/timing without interruption.
Add explicit new qualification namespace `qualification_002` and allow the
timing driver to select a passed owned qualification explicitly.
The R1 summary uses a separate dossier filename. Exclude still-open external
stdout/stderr/process records from the executor's internal artifact snapshot;
bind all final bytes in post-terminal retention instead. First-run snapshots
and logs remain unchanged, including stdout captured after its internal snapshot.
Native C and independent decode/scalar fixture sources are unchanged. Recompile and
record the new DLL identity; do not overwrite the first compiler/run artifacts.
Continue requiring actual owner coordination before timing; process inventory
is not reservation authority. Goal remains active and incomplete.
