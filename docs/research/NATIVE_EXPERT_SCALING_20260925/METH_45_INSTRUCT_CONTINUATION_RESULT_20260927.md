# METH-45: Instruct E128 continuation stopped at update 512

The [frozen protocol](METH_45_INSTRUCT_CONTINUATION_PROTOCOL_20260927.md)
and [external manifest](meth45_fresh_external_manifest.json) were committed
before training. The runner resumed the exact METH-44 16-update checkpoint,
including optimizer and RNG states; its starting raw BPB and 3654/3807
development top-1 matches reproduced METH-44. The [run result](meth45_instruct_continuation_result.json)
has SHA-256 `a48deede189f1ff0e91bf2a0db9876fbd246fc53c24dc70010f935ba84a0d636`.

At update 256, raw student-minus-donor BPB was **−0.025057** and the
same-corpus development chat top-1 was **3628/3807 = 95.298%**, above the
fixed 95% continuation threshold. At update 512, raw BPB improved to
**−0.028953** but chat top-1 fell to **3578/3807 = 93.985%**. The runner
stopped there by its predeclared `interim_chat_top1_failure` rule. Its
update-256 and update-512 local checkpoint SHA-256 values are respectively
`3d5835eae07384ed6709897212c9d0e31734403f049b0aa254622bafe0b57c4c`
and `c988d26fa0a122db292cd56b253317ae6c0c8ce5fff9c1aedaf004ca2b8fb6c2`.
The 512-update run took 685.485 s on the local RTX 3060, peak allocated
GPU 2.583 GB and process RSS 3.871 GB. It stayed inside all resource caps.

**Decision: reject this fixed continuation recipe.** Neither checkpoint
is promoted by selecting an earlier point after observing the failed gate.
The separately frozen 12-document/chat external manifest and PIQA gate
were **not run** because the protocol required completion at update 1024.
The [external audit runner](../../../benchmarks/donor_adaptation/s1/meth45_instruct_external_audit.py)
remains available for a future candidate with a separately justified gate.
The raw BPB gain alongside declining chat agreement suggests that stronger
or more targeted donor retention may be needed; that is a hypothesis, not
a measured cause. No large-E quality or native-rate inference follows.
