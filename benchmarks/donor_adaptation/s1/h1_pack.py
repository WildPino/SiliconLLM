#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 step 5 (CPU, free) -- assemble the T4 bundle and REFUSE to ship a broken one.

Brief: docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md s10.

Writes _h1_bundle/ with exactly what the T4 job needs and nothing else:

    h1_qat.py         the trainer (imports NOTHING from this repo -- see its docstring)
    h0_trained3.npz   H0 run 3's q/o organs, INSTALLED AND FROZEN          ~353 MB
    labels_E256.npz   the D0c partition, E = 256                           ~0.9 MB
    h1_actstats.npz   the per-organ activation RMS the R3 rule needs       ~1.2 MB
    h1_train.npz      H0's training stream, reused verbatim                 ~64 MB
    h1_heldout.npz    the frozen eval slice -- the run's PROGRESS metric
    h1_calib.npz      the calibration slice -- G-H1e's STATIC control
    MANIFEST.json     sha256 of every file, and the gates that had to fire
    RUN.md            both commands, the stop conditions, and what comes back

REFUSES TO PACK unless, in this order:

  1. `h1_selftest.py` exits 0 -- RUN HERE, not read from a stale file.  All wiring gates plus
     T6 (G-H1e's instrument can fire) and T7 (--resume actually resumes).
  2. `applied-8L` exists, `G-H1d` fired, `G-H1a` fired on the REAL donor shape, and the number
     is the one addendum C registered.  Without it the gate has no threshold.
  3. The router smoke returned a launch decision, and the shipped `--router-lr`/`--aux` are
     READ FROM IT rather than typed here.  Addendum D.3's rule is the only thing allowed to
     choose them.
  4. h1_eval.py's NULL-CONTROL decomposition read experts 0.000000 and router 0.000000.  That
     is the one precondition that cannot be satisfied by code agreeing with itself: it is the
     assembled eval, on the real donor, showing that its three arms isolate what their names
     claim before any of them is pointed at a trained model.

Env: H1_SKIP_NULLCHECK=1 to pack before the null control has run (prints a loud warning).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "h1")
RESH0 = os.path.join(HERE, "results", "h0")
DENS = os.path.abspath(os.path.join(HERE, "..", "density"))
OUT = os.path.join(HERE, "_h1_bundle")

QO_BASE = "D:/_ktmp/h0_run3_final/h0_trained3.npz"
LABELS = os.path.join(DENS, "results", "d0c_labels", "labels_E256.npz")
APPLIED_8L = 1.096636
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
DEFAULTS = dict(steps=4000, bs=2, accum=8, lr="2e-4", every=250, hours=2.8,
                seed1=1717, seed2=2718)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(m):
    print(m, flush=True)


RUN_MD = """# H1 — two T4 sessions, and the second is not optional

**Budget: 2 x 2.8 GPU-h = {budget} GPU-h.** Brief:
`docs/research/donor_adaptation/briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md`.

## The question

H0 answered *can gradients move this object at all* — yes, on attention. H1 asks the question
that decides whether the FFN carve is a real format or only a post-hoc approximation: **is the
carve TRAINABLE, or merely APPLIED?** Every carve measured so far (E23, E37, E38, E40) was
applied to a donor that never trained under it. H1 trains 8 layers under the carve and compares
against the matched post-hoc control on the very same 8 layers.

## Commands — session 1, then session 2

```
python h1_qat.py \\
    --factors h0_trained3.npz \\
    --labels  labels_E256.npz \\
    --stats   h1_actstats.npz \\
    --train   h1_train.npz \\
    --heldout h1_heldout.npz \\
    --calib   h1_calib.npz \\
    --out     h1_trained_s1.npz \\
    --router-lr {rlr} --aux {aux} \\
    --steps {steps} --bs {bs} --accum {accum} --lr {lr} --every {every} \\
    --seed {seed1} --max-hours {hours}
```

Then, **from the bundle session 1 produced**:

```
python h1_qat.py \\
    --factors h0_trained3.npz \\
    --resume  h1_trained_s1.npz \\
    --labels  labels_E256.npz \\
    --stats   h1_actstats.npz \\
    --train   h1_train.npz \\
    --heldout h1_heldout.npz \\
    --calib   h1_calib.npz \\
    --out     h1_trained_s2.npz \\
    --router-lr {rlr} --aux {aux} \\
    --steps {steps} --bs {bs} --accum {accum} --lr {lr} --every {every} \\
    --seed {seed2} --max-hours {hours}
```

**`--factors` stays H0's bundle in BOTH sessions and `--resume` is what continues the run.**
They are different things: `--factors` installs the frozen q/o attention organs, `--resume`
continues the FFN masters and the router. Passing an H1 bundle to `--factors` would install
zero attention organs; the trainer now refuses outright rather than doing it quietly.

**The seed differs on purpose** ({seed1} then {seed2}). Batches come from a fixed rng, so a
continuation on the same seed would retrain the same draws. Adam moments are not checkpointed;
the restart is recorded in the output json.

## Why two sessions, and why the second is not padding

H0's run 1 looked flat and run 3 — same object, more steps — removed 97.9% of the damage. A
single 2.8 h session would have declared H0 dead. Brief §7.

## Stop conditions — stop early and say so if any of these happens

1. **The script refuses to start** with `--router-lr and --aux are NOT SET`. It means the flags
   above were dropped from the command. They are not optional and must not be guessed.
2. **`G-H1c` aborts** — `a master did not move after an update the optimizer APPLIED`. The
   trainer is not training; any "no effect" reading would be an artefact. Send the log.
3. **`G-H1b CANNOT BE READ`** — the GradScaler declined all {window} first steps. That is a
   numerical-scale failure of the trainer and **not** H1's answer. Send the log; the fallback is
   bf16, which this card reports as supported.
   Individual `step N DECLINED by the GradScaler` lines at warm-up are **normal**, not errors.
4. **`nonfinite` climbs above ~1% of microbatches.** fp16 loss scaling is not holding.
   (The progress line reads `BPB soft ... hard ... (gap ...)`. **The hard one is the gate.**
   A large gap is expected at the start and is not a fault.)
5. **The job dies on attention.** The script asserts `sdpa` at load. Do not work around it:
   `eager` fp16 on a T4 goes non-finite at `layers.0.self_attn.o_proj`.

Otherwise let each session run to its time cap.

## What comes back

**`h1_trained_s1.npz` + `.json`, and `h1_trained_s2.npz` + `.json`.** Bring all four (~{bundle_mb:.0f} MB
each npz).

**DO NOT PICK A CHECKPOINT.** The trainer overwrites its output every {every} steps so an
interrupted session is still usable, and it prints a held-out BPB so the run is watchable —
**but that BPB is measured on the gate's own frozen slice.** Choosing the best checkpoint by it
would be selection on the evaluation set and would void `G-H1`. Bring back the last file each
session wrote, whatever its printed number says.

**The gate is NOT decided on the GPU.** `h1_eval.py` re-measures it on the CPU box in fp32 with
the instrument every published number in this programme used.

## What the first 30 seconds should print, and what to ignore

```
   G-H1a  k=E identical, HARD gate, any router : FIRES  (max |diff| 0.000e+00)
   G-H1a  k=E identical, SOFT gate, router=0   : FIRES  (max |diff| 0.000e+00)
   carve actually masks at k=16                : FIRES
```

**Both `G-H1a` lines, every time — session 2 included.** There are two of them because at
`k = E` the gates are exactly 1 under the hard gate for *any* router, but under the soft gate
*only* at a zero router. Session 2 resumes a **trained** router, so asking the soft form alone
aborts a perfectly good run; that is exactly what happened on the CPU smoke and it is fixed
(addendum I). If either line says FAILS, stop and report it — do not re-run hoping it passes.

Torch will print `use_cache=True is incompatible with gradient checkpointing` and
`None of the inputs have requires_grad=True`. **Both are benign.** The second comes from the
*eval* pass (checkpointing under `no_grad`); training is unaffected, and `G-H1c` proves it a few
lines later by reporting that masters moved at both depth extremes, `L03.gate` and `L24.down`.

## The gate, fixed before the run

| | held-out BPB, CPU fp32, frozen 24x512 slice |
|---|---|
| donor, untouched | 0.767595 |
| H1's start line — H0 run 3 | 0.810022 |
| the 8 layers ternarised, NOT carved | 0.947851 |
| **`applied-8L` — the matched post-hoc control, HARD gate** | **1.096636** |
| **`G-H1` passes iff trained-8L, HARD gate** | **< 1.096636** |

Ordinal, no tolerance. **The gate is scored with the HARD `{{0,1}}` gate** — the one `engine.c`
runs and the one the control uses (addendum F). Training itself stays on the soft gate, because
a hard gate has no gradient to the router; the run prints **both** numbers so the gap is
visible while it is in flight. On an untrained bundle that gap is **+0.80 BPB**, so do not be
alarmed by a large soft number — watch the hard one. The bands below it are `TRAINING-HELPS` [0.947851, 1.096636),
`CARVE-IS-TRAINABLE` (0.810022, 0.947851), `CARVE-IS-FREE` <= 0.810022.

## What is already known to be against us, stated before the hours are spent

The CPU router smoke shipped `--router-lr {rlr} --aux {aux}` **with a stated shortfall**: it
beats a STATIC selection on **2 of 8 layers** (`[3, 24]`), and in total error mass it is
**2.67% worse** than STATIC. `G-H1e` is therefore **expected to fail**, and that expectation is
registered in addendum H.

Read the second sentence of that paragraph carefully, because the first version of this smoke
was wrong. It ran 11.7 h with the trained arm on the **soft** gate against a **hard** STATIC
control, and addenda D and E's whole table was retired for it (addendum G). The grid was re-run
correctly in **1.26 h** — the experts are frozen, so their quantisation is a constant and
caching it is 30x and bit-exact — over the **full** 12-setting grid, with no narrowing. The
corrected table ships the **same** setting, on the **same** two layers, and **0 of 96 cells**
changed their verdict (addendum H). The numbers above are the corrected ones. It does not block H1, because H1's primary gate is `G-H1` — the
experts — and the smoke is a frozen-expert proxy that is structurally blind to the one thing
joint training changes: which experts receive gradient.

## What H1 is NOT

- **Not the 10 B goal.** This is a 1.5 B donor and 8 of its 28 layers.
- **Not additive to 28 layers.** Measured: ternarisation costs +2.708 BPB over 28 layers and
  +0.138 over these 8. Nothing measured here may be multiplied up.
- **Not a speed measurement.** Training computes the full FFN and masks it, so it buys quality
  information and not speed. No timing is taken and none may be quoted from it.
"""


def main():
    qat = os.path.join(HERE, "h1_qat.py")
    stats = os.path.join(RES, "h1_actstats.npz")
    train = os.path.join(RESH0, "h0_train.npz")
    held = os.path.join(RES, "h1_heldout.npz")
    calib = os.path.join(RES, "h1_calib.npz")
    appj = os.path.join(RES, "h1_applied_8L.json")
    selj = os.path.join(RES, "h1_router_select.json")
    nulj = os.path.join(RES, "h1_eval_NULLCONTROL.json")

    for p in (qat, QO_BASE, LABELS, stats, train, held, calib):
        if not os.path.exists(p):
            log("MISSING: %s" % p)
            return 2

    # ---- 1. the self-test, RUN, not remembered --------------------------------------------
    log("  running h1_selftest.py ...")
    r = subprocess.run([sys.executable, os.path.join(HERE, "h1_selftest.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log("REFUSING TO PACK: the self-test FAILED.")
        log(r.stdout[-2000:])
        return 2
    log("  self-test  ALL CHECKS FIRE  (T1-T8, including --resume and G-H1a's two forms)")

    # ---- 2. the threshold ------------------------------------------------------------------
    if not os.path.exists(appj):
        log("REFUSING TO PACK: %s not found -- the gate has no threshold.  Run h1_applied.py."
            % os.path.basename(appj))
        return 2
    ap = json.load(open(appj, encoding="utf-8"))
    if not ap["G_H1d"]["fires"]:
        log("REFUSING TO PACK: G-H1d did not fire -- the control reads the 8-layer restriction "
            "rather than the carve, and applied-8L is unusable as a threshold.")
        return 2
    if not ap["G_H1a_real_shape"]["bit_identical"]:
        log("REFUSING TO PACK: G-H1a did not fire on the real donor shape.")
        return 2
    if abs(ap["bpb"]["applied-8L"] - APPLIED_8L) > 5e-7:
        log("REFUSING TO PACK: applied-8L on disk is %.6f, addendum C registered %.6f."
            % (ap["bpb"]["applied-8L"], APPLIED_8L))
        return 2
    log("  G-H1d  FIRES  (applied-8L %.6f > ternary-8L %.6f, delta %+.6f)"
        % (ap["bpb"]["applied-8L"], ap["bpb"]["ternary-8L"], ap["G_H1d"]["delta"]))

    # ---- 3. the router setting, READ not typed ----------------------------------------------
    if not os.path.exists(selj):
        log("REFUSING TO PACK: %s not found.  Run h1_router_select.py -- addendum D.3's rule "
            "is the only thing allowed to choose --router-lr and --aux."
            % os.path.basename(selj))
        return 2
    sel = json.load(open(selj, encoding="utf-8"))
    if not sel.get("launch"):
        log("REFUSING TO PACK: the router selection says H1 DOES NOT LAUNCH "
            "(addendum A consequence 3).  The hours go back unspent.")
        return 2
    rlr, aux = sel["router_lr"], sel["aux"]
    log("  router setting from addendum D.3: --router-lr %g --aux %g   (%d of %d layers%s)"
        % (rlr, aux, len(sel["layers_won"]), len(sel["layers"]),
           ", SHORTFALL recorded" if sel.get("rule4_shortfall") else ""))

    # ---- 4. the null control on the assembled eval -------------------------------------------
    if os.path.exists(nulj):
        nl = json.load(open(nulj, encoding="utf-8"))
        d = nl.get("decomposition_DIAGNOSTIC_NOT_A_GATE")
        if not d:
            log("REFUSING TO PACK: the null control ran WITHOUT the decomposition.")
            return 2
        okn = abs(d["experts"]) < 1e-12 and abs(d["router"]) < 1e-12
        log("  null control  experts %+.12f  router %+.12f  gate-form %+.6f  -> %s"
            % (d["experts"], d["router"], d["gate_form"], "FIRES" if okn else "*** FAILS ***"))
        if not okn:
            log("REFUSING TO PACK: an UNTRAINED bundle must decompose to exactly zero experts")
            log("and zero router.  It does not, so h1_eval.py's arms are not isolating what")
            log("their names claim and the decomposition would be an attribution artefact.")
            return 2
        log("     gate-form %+.6f is the price of the SOFT gate with zero training."
            % d["gate_form"])
        log("     Addendum F: that is why G-H1 is scored HARD -- engine.c runs the hard gate")
        log("     and applied-8L is hard, so a soft score would charge H1 that much before")
        log("     training counted.  The term stays as the train/eval mismatch.")
    elif os.environ.get("H1_SKIP_NULLCHECK") == "1":
        log("  *** WARNING: packing WITHOUT h1_eval.py's null control (H1_SKIP_NULLCHECK=1).")
        log("  *** The decomposition is UNVERIFIED and may attribute the delta to the wrong")
        log("  *** cause.  The gate itself is unaffected.")
    else:
        log("REFUSING TO PACK: %s not found.  Build it with h1_nullbundle.py and run "
            "h1_eval.py --tag NULL-CONTROL on it." % os.path.basename(nulj))
        return 2

    # ---- 5. the streams ----------------------------------------------------------------------
    hj = json.load(open(os.path.join(RES, "h1_heldout.json"), encoding="utf-8"))
    if hj["slice"]["ids_sha256"] != EXPECT_IDS_SHA:
        log("REFUSING TO PACK: h1_heldout.npz is not the frozen slice.")
        return 2
    cj = json.load(open(os.path.join(RES, "h1_calib.json"), encoding="utf-8"))
    if cj["slice"]["ids_sha256"] == EXPECT_IDS_SHA:
        log("REFUSING TO PACK: the calib stream IS the eval slice -- STATIC would be fitted "
            "on the tokens it is scored on.")
        return 2
    log("  streams  heldout %s (%d bytes)  calib %s -- disjoint"
        % (hj["slice"]["ids_sha256"][:12], hj["scored_bytes"],
           cj["slice"]["ids_sha256"][:12]))

    # ---- assemble ------------------------------------------------------------------------------
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    files = {}
    for src, dst in ((qat, "h1_qat.py"), (QO_BASE, "h0_trained3.npz"),
                     (LABELS, "labels_E256.npz"), (stats, "h1_actstats.npz"),
                     (train, "h1_train.npz"), (held, "h1_heldout.npz"),
                     (calib, "h1_calib.npz")):
        shutil.copy2(src, os.path.join(OUT, dst))
        files[dst] = {"sha256": sha(src), "bytes": os.path.getsize(src), "from": src}
        log("  + %-18s %8.1f MB" % (dst, files[dst]["bytes"] / 1e6))

    d = dict(DEFAULTS)
    d["rlr"], d["aux"] = "%g" % rlr, "%g" % aux
    d["budget"] = 2 * DEFAULTS["hours"]
    d["window"] = 40
    d["bundle_mb"] = files["h0_trained3.npz"]["bytes"] / 1e6
    with open(os.path.join(OUT, "RUN.md"), "w", encoding="utf-8") as f:
        f.write(RUN_MD.format(**d))

    json.dump({"brief": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md",
               "stage": "H1", "budget_gpu_hours": 2 * DEFAULTS["hours"], "sessions": 2,
               "gate": {"metric": "heldout_BPB_cpu_fp32", "threshold": APPLIED_8L,
                        "ordinal": True, "start_h0_run3": 0.810022,
                        "ternary_8L": 0.947851, "dense": 0.767595},
               "command": d,
               "router_setting": {"router_lr": rlr, "aux": aux,
                                  "chosen_by": "addendum D.3, h1_router_select.py",
                                  "layers_won": sel["layers_won"],
                                  "rule4_shortfall": sel.get("rule4_shortfall"),
                                  "G_H1e_expected": "FAIL -- re-registered in addendum H on the CORRECTED grid; addendum E's version was withdrawn in G as derived from a gate-confounded table"},
               "gates_fired": {"selftest_T1_T8": True,
                               "G_H1d": True, "G_H1a_real_shape": True,
                               "null_control_decomposition": os.path.exists(nulj)},
               "gate_form_price_soft_vs_hard": (
                   json.load(open(nulj, encoding="utf-8"))
                   ["decomposition_DIAGNOSTIC_NOT_A_GATE"]["gate_form"]
                   if os.path.exists(nulj) else None),
               "applied_meta": appj, "select_meta": selj, "null_meta": nulj,
               "files": files},
              open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    tot = sum(v["bytes"] for v in files.values()) / 1e6
    log("")
    log("  bundle %s   %.1f MB total" % (OUT, tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
