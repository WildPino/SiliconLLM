# PQT-006-R2: verified own launcher admission

Prospective, 5 October 2026. Inherit all numerical/compiler/artifact/control
and timing criteria/budgets from PQT-006/R1. Native C and independent native
decoder/fixtures remain unchanged. Preserve both prior qualification attempts.

R1 source `efafccf` compiled successfully but admission rejected a relevant
process at 2.031s, before controls/original artifacts. A bounded own-runtime
probe observed the Windows venv launcher PID 20496 (owned scientific Python
executable), actual interpreter PID 25744, parent 20496; child `sys.executable`
is the owned scientific venv path. This explains the false positive: the
launcher is still live while the child executes. It was not an independently
admitted competing workload. Probe stdout/stderr and this observation retained.

R2 records live process IDs, parent IDs and executable paths without command
lines or credentials. Exclude only the actual caller and its immediate live
parent when that parent's path exactly matches `sys.executable` and the
executable is one of the two declared owned venvs. Caller must appear in the
live inventory. All other relevant processes still prevent execution; no
interruption and no inference of timing permission from the snapshot.
Record the complete admission before deciding, including rejected inventories.

New `qualification_003`, separate R2 summary, frozen source/new DLL identity.
No previous numeric control or original-function case was executed, no
timing performed. Pass the unchanged synthetic controls and all twenty-eight
real cases before asking for the actual uncontended timing window.
The full research goal remains active and incomplete.
