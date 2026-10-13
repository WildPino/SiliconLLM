# Preparation schema fault, retained before data computation

Initial source freeze f0dd108, preparation PID3992/create1791861855.069425,
exit1 after .171999999999s, OS41836544B. Report capture uses binding_sha256
rather than a nested binding object; incorrect access c['binding'] raised
KeyError. No binding written, no output directory/arrays/source/native/GPU.
Original terminal is source_function_rank_binding_20261013.preparation_terminal.json.

Correction explicitly pins original V4 capture binding path and requires its
actual SHA to equal the captured binding_sha256. Independent auditor checks
the same link. No mathematical policy/data selection/gate change. New source
freeze and binding_v2 destination; original fault terminal never overwritten.
