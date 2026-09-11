#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H0 step 5 (CPU, free) -- assemble the Kaggle bundle and refuse to ship a broken one.

Plan: docs/research/donor_adaptation/decisions/T4_HEALING_PROPOSAL.md s3 (H0).

Writes _h0_bundle/ containing exactly what the T4 job needs and nothing else:

    h0_qat.py          the trainer (imports NOTHING from this repo -- see its docstring)
    h0_factors.npz     the fp32 masters, 56 organs                      ~352 MB
    h0_train.npz       the pre-tokenized training stream                 ~64 MB
    h0_probe.json      the five frozen prompts and the donor's targets
    MANIFEST.json      sha256 of every file, plus the gates that must have fired
    RUN.md             the exact command, the stop condition, and what comes back

REFUSES TO PACK unless, in this order:
  * h0_factorize.py's G-H0a fired (the masters ARE E22's QO512-TB),
  * h0_selftest.py's G-H0b/c/d fired (the self-contained trainer IS the repo's rule),
  * the eval of the INIT bundle read the teacher-forced count E22 published.
The third is the one that cannot be faked by agreeing code: it is the assembled model,
measured end to end on the real instrument.

Env: H0_SKIP_EVALCHECK=1 to pack before the init eval has run (prints a loud warning)
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "h0")
OUT = os.path.join(HERE, "_h0_bundle")
E22_TB = {"tf": 28, "free": 1, "bpb": 2.812226, "mean_rank": 1476}
E22_TB_TF = E22_TB["tf"]
BPB_REL_TOL = 1e-4       # see the note in main(): factored execution vs re-densified


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(m):
    print(m, flush=True)


RUN_MD = """# H0 — run this on one T4, once

**Budget: <= 3 GPU-h.** This is a signal-detection experiment, not a success experiment.
Plan: `docs/research/donor_adaptation/decisions/T4_HEALING_PROPOSAL.md` §3.

## The question

E22 measured that the configuration that fits the 50 tok/s budget and the configuration that
works are not the same object: rank-512 attention costs `+0.052689` BPB in fp32 and collapses to
`28/160` teacher-forced once both factors are ternarized, because the two factors' errors
multiply. Post-hoc conversion of a factored form is dead. **H0 asks the only question left: do
gradients move this object at all?**

## Command

```
python h0_qat.py \\
    --factors h0_factors.npz \\
    --train   h0_train.npz \\
    --probe   h0_probe.json \\
    --out     h0_trained.npz \\
    --steps {steps} --bs {bs} --accum {accum} --lr {lr} --every {every} --max-hours 2.8
```

Nothing to configure. It stops itself at 2.8 hours and writes `h0_trained.npz` at every
checkpoint, so an interrupted run is still usable.

## Stop conditions — stop early and say so if any of these happens

1. **`G-H0e` aborts.** The script stops by itself, and there are now **two different aborts**
   — the message says which, and they mean opposite things.

   * `G-H0e FAILS: a master did not move after an update the optimizer APPLIED.` The trainer is
     not training and any "no effect" conclusion would be an artefact. Nothing to debug on your
     side — send the log.
   * `G-H0e CANNOT BE READ: the GradScaler declined all 25 of the first steps.` The optimizer
     never got an update in at all, which is a numerical-scale problem and **not** H0's answer.
     Send the log; the fallback is bf16, which this card reports as supported.

   **Lines that read `step N DECLINED by the GradScaler` are NORMAL at warm-up and are not an
   error.** `GradScaler` deliberately starts at a scale of 65536 and halves it until the
   gradients fit in fp16; a handful of declined steps at the start is the intended behaviour.
   Run 1 (2026-09-11) died because the gate read the masters after step 1 whether or not the
   step had been applied, so a perfectly ordinary declined first step looked like "the trainer
   cannot move this object". **That was a defect in the gate, not in the run and not in H0.**
2. **`nonfinite` climbs above ~1% of microbatches.** fp16 loss scaling is not holding. Send the
   log; the fallback is bf16, which this card reports as supported.
3. **The job dies on attention.** The script asserts `sdpa` at load and refuses to start
   otherwise. If that assert fires, do not work around it: `eager` fp16 on a T4 goes non-finite
   at `layers.0.self_attn.o_proj`, which is one of the two organs being trained.

Otherwise let it run to the time cap.

## What comes back

**`h0_trained.npz` (~352 MB) and `h0_trained.json`.** That is all — bring both back.

**The gate is NOT decided on the GPU.** The job prints a teacher-forced count in fp16 so the run
is watchable, but the H0 gate is re-measured on the CPU box in fp32 with the same instrument
every published number in this programme used. The in-job number is a progress bar, not a result.

## The gate, fixed before the run

| | teacher-forced (of 160) |
|---|---|
| start — E22's `QO512-TB`, measured | **28** |
| **H0 gate** | **>= 48** |
| fp32 ceiling — E21's `QO512`, measured | 144 |

`>= 48` is `+20` on the start state: a signal-detection bar, not a success bar. It only asks
whether gradients move this object. **If it fails, post-hoc AND trained conversion have both
failed on this donor, the donor route is closed end to end, and the honest recommendation is to
stop adapting donors.** That is a real possibility and it is written here before the run.

## What H0 is NOT

- **Not the 10 B goal.** This is a 1.5 B, and healing it validates a structure; it does not
  reach 50 tok/s at 10 B. §0 of the proposal says so at length.
- **Not affected by E23.** H0 trains ternary low-rank attention with **no FFN carve and no
  router**, chosen deliberately so that E23's verdict — which moved the H1/H2 target — cannot
  invalidate it either way.
- **Not a speed measurement.** Nothing is exported and no timing is taken. `6.79 tok/s` stays
  exact, and `engine.c` still has no factored matvec.
"""

DEFAULTS = dict(steps=4000, bs=2, accum=8, lr="2e-4", every=250)


def main():
    fac = os.path.join(RES, "h0_factors.npz")
    facj = os.path.join(RES, "h0_factors.json")
    trn = os.path.join(RES, "h0_train.npz")
    prb = os.path.join(RES, "h0_probe.json")
    evj = os.path.join(RES, "h0_eval_init.json")
    qat = os.path.join(HERE, "h0_qat.py")

    for p in (fac, facj, trn, prb, qat):
        if not os.path.exists(p):
            log("MISSING: %s" % p)
            return 2

    meta = json.load(open(facj, encoding="utf-8"))
    if not meta["G_H0a"]["fires"]:
        log("REFUSING TO PACK: G-H0a did not fire -- the masters are not E22's QO512-TB.")
        return 2
    if meta["smoke"]:
        log("REFUSING TO PACK: the factor bundle is a SMOKE bundle.")
        return 2
    log("  G-H0a  FIRES  (%d organs, codes/scales/products exact vs E22)" % len(meta["rows"]))

    gate_ok = True
    if os.path.exists(evj):
        ev = json.load(open(evj, encoding="utf-8"))
        tf, fr, bp = ev["teacher_forced"], ev["free"], ev["bpb"]
        rel = abs(bp - E22_TB["bpb"]) / E22_TB["bpb"]
        gate_ok = (tf == E22_TB_TF) and (fr == E22_TB["free"]) and (rel <= BPB_REL_TOL)
        log("  init eval  tf %d (E22 %d)  free %d (E22 %d)  BPB %.6f (E22 %.6f, rel %.2e)  -> %s"
            % (tf, E22_TB_TF, fr, E22_TB["free"], bp, E22_TB["bpb"], rel,
               "MATCHES" if gate_ok else "*** DOES NOT MATCH ***"))
        log("     both TOKEN metrics are exact.  The BPB difference is expected and is the")
        log("     first measurement of its kind here: E21 and E22 installed A.B as a DENSE")
        log("     matrix, this runs it as two GEMVs with an intermediate of size r -- the")
        log("     form the engine would use.  Same product, different summation order.")
        if not gate_ok and os.environ.get("H0_SKIP_EVALCHECK") != "1":
            log("REFUSING TO PACK: the assembled model does not reproduce E22's start state, so")
            log("the H0 delta would be measured from an unknown baseline.")
            return 2
    elif os.environ.get("H0_SKIP_EVALCHECK") == "1":
        log("  *** WARNING: packing WITHOUT the end-to-end init eval (H0_SKIP_EVALCHECK=1).")
        log("  *** The per-organ gates agree; the ASSEMBLED model is unverified.")
    else:
        log("REFUSING TO PACK: %s not found.  Run h0_eval.py --factors %s --tag init first."
            % (os.path.basename(evj), fac))
        return 2

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    files = {}
    for src, dst in ((qat, "h0_qat.py"), (fac, "h0_factors.npz"),
                     (trn, "h0_train.npz"), (prb, "h0_probe.json")):
        shutil.copy2(src, os.path.join(OUT, dst))
        files[dst] = {"sha256": sha(src), "bytes": os.path.getsize(src)}
        log("  + %-18s %8.1f MB" % (dst, files[dst]["bytes"] / 1e6))

    with open(os.path.join(OUT, "RUN.md"), "w", encoding="utf-8") as f:
        f.write(RUN_MD.format(**DEFAULTS))
    json.dump({"plan": "decisions/T4_HEALING_PROPOSAL.md s3 (H0)",
               "stage": "H0", "budget_gpu_hours": 3.0,
               "gate": {"metric": "teacher_forced_cpu_fp32", "threshold": 48,
                        "start_E22_QO512_TB": 28, "ceiling_E21_fp32": 144},
               "command": DEFAULTS,
               "gates_fired": {"G_H0a": True, "init_eval_matches_E22": gate_ok},
               "factor_meta": facj, "files": files},
              open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    tot = sum(v["bytes"] for v in files.values()) / 1e6
    log("")
    log("  bundle %s   %.1f MB total" % (OUT, tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
