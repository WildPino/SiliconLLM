#!/usr/bin/env python3
"""S1 scale arm -- the Kaggle T4x2 entry point.  Implements BRIEF_S1 AMENDMENT 1.

This runs the SAME `s1_sparsity_bpb.py` the local CPU anchor ran.  It is a driver, not a
second implementation: if the rule, the arms or the controls were re-coded here the CPU/GPU
anchor control of A1.2 would compare two programs instead of two platforms.

------------------------------------------------------------------------------------------
A1.2, THE CONTROL THAT MAKES THE TREND READABLE -- non-negotiable and enforced here:

    the 1.5B point is re-run on this GPU path and compared to the local CPU fp32 number.
    If |BPB_gpu - BPB_cpu| > sigma_seed = 0.005 at the same p, this script STOPS and does
    not run the larger sizes, because the instrument would be measuring the platform.

The local CPU baseline is carried in as `ANCHOR_CPU` below, written from the completed local
artefact; it is not recomputed here and not guessed.

------------------------------------------------------------------------------------------
A1.3: fp16 weights on GPU, fp32 on CPU, AND NOTHING ELSE.  No 8-bit, no 4-bit -- this probe
measures ACTIVATION statistics and quantising the weights would change the object under test.
If 14B does not fit in fp16 across the two cards it is SKIPPED and reported as not fitting.

Kaggle duplicates subprocess stdout 2-3x, so nothing here counts log lines: every claim is
read back from the JSON artefacts this script writes.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
# On Kaggle the bundle is mounted read-only under /kaggle/input/<slug>/; results must go to
# /kaggle/working.  Locally both resolve to the repo.
ON_KAGGLE = os.path.isdir("/kaggle/working")
if ON_KAGGLE:
    # RECURSIVE, and asserted UNIQUE.  `--dir-mode zip` puts the top-level .py files directly on
    # the mount, but that is a property of the uploader, not a guarantee; a nested layout would
    # make a one-level glob return NOTHING and this loop would then silently leave HERE pointing
    # at the kernel's scratch dir, where the pinned eval slice does not exist -- and a redrawn
    # slice is exactly the failure the A1.2 control cannot see.  Fail loudly instead.
    cands = glob.glob("/kaggle/input/**/s1_sparsity_bpb.py", recursive=True)
    assert len(cands) == 1, f"expected exactly one code bundle on the mount, found {cands}"
    HERE = os.path.dirname(cands[0])
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
    os.environ.setdefault("S1_RESULTS", "/kaggle/working/s1_results")

import torch  # noqa: E402

sys.path.insert(0, HERE)
import s1_sparsity_bpb as S  # noqa: E402

# ---------------------------------------------------------------- the anchor, from the local run
# Written from `results/s1_run_qwen2.5-1.5b.json` produced on this machine (CPU, fp32).
# `null` here means the local run had not finished when this bundle was built; the anchor check
# then reports UNAVAILABLE and the script still refuses to proceed past 1.5B.
ANCHOR_CPU_PATH = os.path.join(HERE, "anchor_cpu.json")

# The sizes, smallest first.  0.5B is the SMOKE.  1.5B is the ANCHOR and gates everything after.
LADDER = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-7b", "qwen2.5-14b"]
BIG = {"qwen2.5-7b", "qwen2.5-14b"}          # need device_map across both cards


def log(*a):
    print(*a, flush=True)


def gpu_report():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devs = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        devs.append({"index": i, "name": p.name,
                     "total_mem_gb": round(p.total_memory / 2**30, 2),
                     "capability": f"{p.major}.{p.minor}"})
    return {"cuda_available": torch.cuda.is_available(), "n_devices_achieved": n,
            "devices": devs, "torch": torch.__version__,
            "torch_cuda": getattr(torch.version, "cuda", None),
            "bf16_supported_achieved": bool(n and torch.cuda.is_bf16_supported())}


def load_anchor():
    if os.path.exists(ANCHOR_CPU_PATH):
        try:
            return json.load(open(ANCHOR_CPU_PATH))
        except Exception:
            return None
    return None


MIN_ANCHOR_COMPARISONS = 20   # the bundled anchor carries 50 arm comparisons + baseline = 51.
                              # A gate that silently compares nothing must not be able to pass.


def _bad(v):
    """True if v cannot participate in a threshold comparison at all."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return True
    return math.isnan(float(v)) or math.isinf(float(v))


def check_anchor(gpu_run, anchor, sigma=S.SIGMA_SEED_BPB):
    """A1.2.  Compare the GPU fp16 1.5B point to the local CPU fp32 one, p by p.

    Returns (ok, report).  `ok` False means STOP -- do not run the larger sizes.

    -----------------------------------------------------------------------------------
    REPAIR 2026-09-04, after this gate reported PASS over a dead anchor on a real run.

    The original implementation accumulated `worst = max(worst, abs(d))`.  In Python
    `max(0.0, nan)` returns **0.0** -- `nan > 0.0` is False, so max() keeps its first
    argument.  The GPU 1.5B run came back with `baseline_bpb = NaN`; all 51 deltas were
    NaN, all were swallowed, `worst` stayed 0.0, and the gate certified "PASS -- fp16/GPU
    reproduces fp32/CPU within sigma_seed" while the probe's own C1_IDENTITY control in
    the same manifest said "FAIL -- HARNESS IS WRONG, STOP".

    A second, latent hole found while fixing the first: with an empty `results` the loop
    adds no comparisons and the same `worst = 0.0` passes the gate having compared
    nothing.  MIN_ANCHOR_COMPARISONS closes it.

    This is a REPAIR, not an amendment to the pre-registration.  A1.2 says stop unless
    every comparison is within sigma_seed; a NaN comparison is not "within" anything, and
    certifying it as such was the code failing to implement its own spec.  The threshold
    is untouched.
    -----------------------------------------------------------------------------------
    """
    if not anchor:
        return False, {"verdict": "UNAVAILABLE -- no local CPU anchor was bundled; "
                                  "cannot certify the platform, so the ladder is NOT run",
                       "sigma_seed": sigma}
    rows = []
    deltas = []
    nonfinite = []

    def add(row, cpu, gpu):
        if _bad(cpu) or _bad(gpu):
            nonfinite.append(dict(row, cpu=cpu, gpu=gpu,
                                  why="cpu or gpu value is NaN/Inf/missing"))
            rows.append(dict(row, cpu=cpu, gpu=gpu, abs_delta=None, NONFINITE=True))
            return
        d = abs(float(gpu) - float(cpu))
        deltas.append(d)
        rows.append(dict(row, cpu=cpu, gpu=gpu, abs_delta=d))

    add({"p": "baseline"}, anchor.get("baseline_bpb"), gpu_run.get("baseline_bpb"))
    for ps, arec in anchor.get("results", {}).items():
        grec = gpu_run.get("results", {}).get(ps)
        if not grec:
            continue
        for arm in ("A", "B", "C", "D", "D2"):
            if arm in arec and arm in grec:
                add({"p": ps, "arm": arm}, arec[arm].get("bpb"), grec[arm].get("bpb"))
                if rows[-1].get("abs_delta") is not None:
                    rows[-1]["cpu_achieved"] = arec[arm]["achieved"]["aggregate"]
                    rows[-1]["gpu_achieved"] = grec[arm]["achieved"]["aggregate"]

    worst = max(deltas) if deltas else None

    # Three independent ways to fail. Any one of them stops the ladder.
    if nonfinite:
        ok = False
        verdict = ("FAIL -- STOP.  %d of %d comparisons are NaN/Inf/missing, so the anchor "
                   "certifies nothing.  A non-finite comparison is NOT 'within sigma_seed'; "
                   "it means the GPU run is broken.  (This is the failure mode that used to "
                   "report PASS: max(0.0, nan) == 0.0 in Python.)  A1.2 requires the ladder "
                   "NOT be run." % (len(nonfinite), len(rows)))
    elif len(deltas) < MIN_ANCHOR_COMPARISONS:
        ok = False
        verdict = ("FAIL -- STOP.  Only %d finite comparisons were made, below the required "
                   "minimum of %d.  A gate that compares nothing must not be able to pass."
                   % (len(deltas), MIN_ANCHOR_COMPARISONS))
    elif worst > sigma:
        ok = False
        verdict = ("FAIL -- STOP.  The GPU path and the CPU path disagree by more than "
                   "sigma_seed, so a cross-size trend measured here would confound scale "
                   "with platform.  A1.2 requires the ladder NOT be run.")
    else:
        ok = True
        verdict = ("PASS -- fp16/GPU reproduces fp32/CPU within sigma_seed over %d finite "
                   "comparisons with zero non-finite; the trend that follows is a SCALE "
                   "trend, not a platform artefact" % len(deltas))

    return ok, {
        "sigma_seed": sigma,
        "max_abs_bpb_delta_cpu_vs_gpu_ACHIEVED": worst,
        "n_comparisons": len(rows),
        "n_finite_comparisons": len(deltas),
        "n_nonfinite_comparisons": len(nonfinite),
        "nonfinite": nonfinite[:20],
        "min_required_comparisons": MIN_ANCHOR_COMPARISONS,
        "rows": rows,
        "verdict": verdict,
    }


def selftest_anchor_gate(verbose=True):
    """PLANTED CONTROL FOR THE GATE ITSELF.

    Standing project law: an instrument must be shown to FIRE on a known positive before
    its passes mean anything.  This gate had never been shown to fire, and it did not --
    it certified a run whose every value was NaN.  These cases are that proof, and they
    run in the entry path before any real check, so a broken gate cannot reach a real run.
    """
    def mk(base, arm, n_p=10):
        r = {"baseline_bpb": base, "results": {}}
        for i in range(n_p):
            r["results"]["%.4f" % (1.0 - i * 0.05)] = dict(
                (a, {"bpb": arm, "achieved": {"aggregate": 0.5}})
                for a in ("A", "B", "C", "D", "D2"))
        return r

    nan, inf = float("nan"), float("inf")
    cases = [
        ("clean identical            -> must PASS", mk(0.75, 0.80), mk(0.75, 0.80), True),
        ("all-NaN GPU (the real bug) -> must FAIL", mk(0.75, 0.80), mk(nan, nan), False),
        ("NaN baseline only          -> must FAIL", mk(0.75, 0.80), mk(nan, 0.80), False),
        ("+Inf arm                   -> must FAIL", mk(0.75, 0.80), mk(0.75, inf), False),
        ("delta 0.02 > sigma 0.005   -> must FAIL", mk(0.75, 0.80), mk(0.75, 0.82), False),
        ("delta 0.001 < sigma        -> must PASS", mk(0.75, 0.80), mk(0.75, 0.801), True),
        ("empty results (compares 0) -> must FAIL", mk(0.75, 0.80),
         {"baseline_bpb": 0.75, "results": {}}, False),
    ]
    if verbose:
        print("== A1.2 anchor-gate planted control ==")
    bad = 0
    for name, anchor, gpu, want in cases:
        got, rep = check_anchor(gpu, anchor)
        if got != want:
            bad += 1
        if verbose:
            print("   %-6s %-42s got=%-5s want=%-5s  finite=%s nonfinite=%s"
                  % ("ok" if got == want else "BROKEN", name, got, want,
                     rep.get("n_finite_comparisons"), rep.get("n_nonfinite_comparisons")))
    if verbose:
        print("   %s (%d/%d)" % ("GATE CONTROL PASS" if not bad else "GATE CONTROL FAILED",
                                 len(cases) - bad, len(cases)))
    return bad == 0


def main():
    t0 = time.time()

    # The gate must prove it FIRES before it is allowed to certify anything. This runs before
    # any model is loaded: a gate that cannot detect NaN, Inf, an over-threshold delta or an
    # empty comparison set has no business gating a 5.8 GPU-hour ladder. On 2026-09-04 the
    # previous implementation passed a run whose every value was NaN; these cases reproduce
    # that failure exactly and the old code fails 3 of them.
    if not selftest_anchor_gate():
        raise SystemExit("A1.2 gate self-test FAILED -- refusing to run. The gate is broken.")

    outdir = os.environ.get("S1_RESULTS", S.RESULTS)
    os.makedirs(outdir, exist_ok=True)
    S.RESULTS = outdir

    only = os.environ.get("S1_ONLY")           # e.g. "qwen2.5-0.5b" for the smoke
    ladder = [only] if only else LADDER
    ladder_grid = S.P_GRID_LADDER if os.environ.get("S1_LADDER_GRID") else S.P_GRID

    man = {"stage": "kaggle_scale_arm", "gpu": gpu_report(), "env": S.env_block(),
           "on_kaggle": ON_KAGGLE, "ladder_requested": ladder,
           "p_grid": list(ladder_grid), "results_dir": outdir,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sizes": {}, "anchor_check": None}
    log("GPU: " + json.dumps(man["gpu"]))

    use_gpu = man["gpu"]["cuda_available"]
    dtype = torch.float16 if use_gpu else torch.float32
    man["dtype_policy"] = ("fp16 on GPU, fp32 on CPU, and nothing else -- no 8-bit, no 4-bit "
                           "(Amendment 1 A1.3: quantising weights would change the activation "
                           "statistics this probe measures)")

    # ---- selfcheck first, always.  It is device-independent and cheap.
    try:
        sc = S.stage_selfcheck()
        man["selfcheck_pass"] = True
        man["selfcheck_fast_path"] = sc["checks"]["FAST_PATH_VERDICT"]
    except Exception:
        man["selfcheck_pass"] = False
        man["selfcheck_traceback"] = traceback.format_exc()
        json.dump(man, open(os.path.join(outdir, "s1_kaggle_manifest.json"), "w"), indent=2,
                  default=float)
        log("SELFCHECK FAILED -- stopping.\n" + man["selfcheck_traceback"])
        return 1

    anchor = load_anchor()
    man["anchor_bundled"] = bool(anchor)

    for key in ladder:
        if key not in S.MODELS:
            man["sizes"][key] = {"status": "SKIPPED -- not in MODELS table"}
            continue
        dm = "auto" if (key in BIG and use_gpu) else None
        log(f"\n=== {key}  (dtype={dtype}, device_map={dm}) ===")
        try:
            res = S.stage_run(key, ladder_grid,
                              do_plant=True, arms=S.ARMS, resume=True,
                              dtype=dtype, device=("cuda" if use_gpu else "cpu"),
                              device_map=dm)
            man["sizes"][key] = {
                "status": "OK", "baseline_bpb": res["baseline_bpb"],
                "n_params_achieved": res["donor"]["n_params_achieved"],
                "dtype_achieved": res["donor"]["dtype_achieved"],
                "devices_in_use_achieved": res["donor"].get("devices_in_use_achieved"),
                "elapsed_s": res["elapsed_s"],
                "json": os.path.join(outdir, f"s1_run_{key}.json")}
        except torch.cuda.OutOfMemoryError:
            man["sizes"][key] = {"status": "DID NOT FIT in fp16 across the available cards -- "
                                           "SKIPPED, and NOT retried in 8-bit or 4-bit "
                                           "(Amendment 1 A1.3)"}
            log(f"{key}: OOM in fp16 -- skipped, reported as not fitting")
            torch.cuda.empty_cache()
            continue
        except Exception:
            man["sizes"][key] = {"status": "ERROR", "traceback": traceback.format_exc()}
            log(f"{key}: ERROR\n" + man["sizes"][key]["traceback"])
            continue
        finally:
            json.dump(man, open(os.path.join(outdir, "s1_kaggle_manifest.json"), "w"),
                      indent=2, default=float)

        # ---- A1.2 GATE: after the 1.5B point, compare to the local CPU anchor and STOP if it fails
        if key == "qwen2.5-1.5b":
            ok, rep = check_anchor(res, anchor)
            man["anchor_check"] = rep
            log("ANCHOR (A1.2): " + rep["verdict"])
            json.dump(man, open(os.path.join(outdir, "s1_kaggle_manifest.json"), "w"),
                      indent=2, default=float)
            if not ok:
                man["ladder_aborted"] = ("A1.2 anchor control did not pass; larger sizes were "
                                         "NOT run.")
                log("STOPPING before the larger sizes, as Amendment 1 A1.2 requires.")
                break

    man["elapsed_s"] = time.time() - t0
    json.dump(man, open(os.path.join(outdir, "s1_kaggle_manifest.json"), "w"), indent=2,
              default=float)
    log(f"\nDONE ({man['elapsed_s']:.0f}s) -> {outdir}/s1_kaggle_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
