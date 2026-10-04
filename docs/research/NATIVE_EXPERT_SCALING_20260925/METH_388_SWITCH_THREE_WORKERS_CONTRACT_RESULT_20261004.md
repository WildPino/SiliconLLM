# METH-388: fail-closed launch before pending freeze completed

The freeze command yielded exec session83717 while still running. The dependent
controller was launched before that session was resumed to confirmed exit0.
It failed at the first physicalHEAD binding: controller path not yet in HEAD,
git cat-file exit128/controller exit1. Raw first failure retained in
`meth388_switch_three_workers_contract_result.failure.json`. No source reversal,
compilation, artifact loading, native execution or quality/rate observation.

The freeze subsequently completed as cdd2871, exit0 fully consumed83717.
This is an orchestration apparatus failure, not a three-worker numerical or
performance result. Preserve it and frozen388 source/protocol unchanged.
NEW389 uses that same388 engine/header/math under a newly named controller/
protocol/output. Await the freeze to confirmed exit0, then preflight every
committed input before launching. No numeric/model/profile/gate change.
