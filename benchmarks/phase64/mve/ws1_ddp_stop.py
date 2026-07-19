#!/usr/bin/env python3
"""WS1 acceptance for the collective time-budget stop: force a budget hit under 2-rank DDP and require
that BOTH ranks leave the loop together.

Run 3's arm A is what this test exists for: rank 0 hit the budget and returned while rank 1 entered the
next step, then sat in an all-reduce with no partner until the 600 s NCCL watchdog took the process down
(SIGABRT). Survivable -- the checkpoint was already written -- but it must not recur, and "it did not
crash this time" is not evidence. Here the budget is set so small that it fires mid-stage on every run.

torchrun cannot disable libuv on Windows, so the rendezvous is set up by hand via env://; the backend
falls back to gloo on CPU. The collective algebra under test is backend-independent.

PASS = both ranks exit 0, both print the TIME-BUDGET line, a resume checkpoint exists, and no rank
reports a watchdog/abort.

Run: python benchmarks/phase64/mve/ws1_ddp_stop.py
"""
import os, sys, subprocess, tempfile, shutil, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "kaggle_mve", "account_1", "mve_data", "data")
TRAIN = os.path.join(HERE, "mve_train.py")

if not os.path.isdir(DATA):
    print("SKIP: pilot data not present at", DATA); sys.exit(0)

work = tempfile.mkdtemp(prefix="ws1_ddp_")
try:
    env0 = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT="29677", WORLD_SIZE="2",
                USE_LIBUV="0", OMP_NUM_THREADS="1")   # no GLOO_SOCKET_IFNAME: 'lo' is not a Windows interface
    args = [sys.executable, TRAIN, "--tag", "full", "--arm", "kd", "--kd", "span", "--recall", "on",
            "--smoke", "--stages", "CD", "--device", "cpu",
            "--time-budget-min", "0.05",          # ~3 s: fires mid-stage, every time
            "--ckpt-min", "0.01",
            "--data-dir", DATA, "--ckpt-dir", work,
            "--out", os.path.join(work, "ws1.pt"),
            "--resume-ckpt", os.path.join(work, "ws1_resume.pt")]

    import time as _t
    t0 = _t.time()
    procs = []
    for r in range(2):
        e = dict(env0, RANK=str(r), LOCAL_RANK=str(r))
        procs.append(subprocess.Popen(args, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=900)[0] for p in procs]
    wall = _t.time() - t0

    ok = True
    for r, (p, o) in enumerate(zip(procs, outs)):
        abort = any(w in o for w in ("Watchdog", "SIGABRT", "ProcessGroupNCCL", "Traceback"))
        # only rank 0 logs (P0 guard), so the budget LINE is expected on rank 0 alone. The evidence that rank 1
        # AGREED is that it exited 0 on its own: had the flag stayed local, rank 1 would have entered the next
        # step and blocked forever in a collective with no partner. A clean exit is the invariant, not the log.
        print(f"rank {r}: exit={p.returncode} abort/traceback={abort}"
              + (f" time-budget-line={'TIME-BUDGET' in o}" if r == 0 else " (silent by design)"))
        if abort:
            print("   ---- tail ----"); print("\n".join(o.strip().splitlines()[-12:]))
        ok &= (p.returncode == 0) and not abort
    ok &= "TIME-BUDGET" in outs[0]
    # a stranded rank would have burned the watchdog timeout before dying; finishing fast is part of the proof
    print(f"wall clock: {wall:.1f}s (a stranded rank would sit ~600s in the watchdog)")
    ok &= wall < 300

    ck = glob.glob(os.path.join(work, "ws1_resume.pt"))
    print(f"resume checkpoint written: {bool(ck)}")
    ok &= bool(ck)
    print("\n==== WS1 collective-stop acceptance: " + ("PASS" if ok else "FAIL") + " ====")
    sys.exit(0 if ok else 1)
finally:
    shutil.rmtree(work, ignore_errors=True)
