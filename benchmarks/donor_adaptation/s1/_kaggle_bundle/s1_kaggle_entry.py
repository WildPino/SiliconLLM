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


def check_anchor(gpu_run, anchor, sigma=S.SIGMA_SEED_BPB):
    """A1.2.  Compare the GPU fp16 1.5B point to the local CPU fp32 one, p by p.

    Returns (ok, report).  `ok` False means STOP -- do not run the larger sizes.
    """
    if not anchor:
        return False, {"verdict": "UNAVAILABLE -- no local CPU anchor was bundled; "
                                  "cannot certify the platform, so the ladder is NOT run",
                       "sigma_seed": sigma}
    rows = []
    worst = 0.0
    db = gpu_run["baseline_bpb"] - anchor["baseline_bpb"]
    worst = max(worst, abs(db))
    rows.append({"p": "baseline", "cpu": anchor["baseline_bpb"],
                 "gpu": gpu_run["baseline_bpb"], "abs_delta": abs(db)})
    for ps, arec in anchor.get("results", {}).items():
        grec = gpu_run["results"].get(ps)
        if not grec:
            continue
        for arm in ("A", "B", "C", "D", "D2"):
            if arm in arec and arm in grec:
                d = grec[arm]["bpb"] - arec[arm]["bpb"]
                worst = max(worst, abs(d))
                rows.append({"p": ps, "arm": arm, "cpu": arec[arm]["bpb"],
                             "gpu": grec[arm]["bpb"], "abs_delta": abs(d),
                             "cpu_achieved": arec[arm]["achieved"]["aggregate"],
                             "gpu_achieved": grec[arm]["achieved"]["aggregate"]})
    ok = worst <= sigma
    return ok, {
        "sigma_seed": sigma,
        "max_abs_bpb_delta_cpu_vs_gpu_ACHIEVED": worst,
        "n_comparisons": len(rows),
        "rows": rows,
        "verdict": ("PASS -- fp16/GPU reproduces fp32/CPU within sigma_seed; the trend that "
                    "follows is a SCALE trend, not a platform artefact")
        if ok else
        ("FAIL -- STOP.  The GPU path and the CPU path disagree by more than sigma_seed, so a "
         "cross-size trend measured here would confound scale with platform.  Amendment 1 "
         "A1.2 requires the ladder NOT be run."),
    }


def main():
    t0 = time.time()
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
