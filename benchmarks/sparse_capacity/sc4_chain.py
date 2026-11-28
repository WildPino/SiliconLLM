#!/usr/bin/env python3
# Sparse Capacity, phase 4: the chain of F4_BEYOND_VRAM_PROTOCOL.md section 5, in order.
#   1. pace cells (machine free): m1, e2k, e8k, e32k, e32k with the host update skipped; the routing parts cell (e32k)
#   2. sc4_verdict.py --pace-only: if G1 does not pass, the chain stops (no long run)
#   3. e2k-h, e2k; sc4_verdict.py --g0-only: if G0 fails, the chain stops (e32k and e8k do not start)
#   4. e32k, e8k; sc4_verdict.py
# A run that exits with STOP or REFUSED stops the chain; any other failure is resumed (--resume) up to twice.
#
#   python benchmarks/sparse_capacity/sc4_chain.py --data-dir <dir> [--from-step N]
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "sparse_capacity")
LOGS = "E:/SiliconLLM_Artifacts/sparse_capacity/f4"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--from-step", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    chain = open(os.path.join(RES, "f4_chain.jsonl"), "a")
    def note(**kw):
        kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S"); chain.write(json.dumps(kw) + "\n"); chain.flush(); print(kw, flush=True)
    py = sys.executable
    tr = [py, os.path.join(HERE, "sc4_train.py")]
    dd = ["--data-dir", a.data_dir]
    steps = [("time_m1", tr + ["--time", "--arm", "m1"] + dd, False),
             ("time_e2k", tr + ["--time", "--arm", "e2k"] + dd, False),
             ("time_e8k", tr + ["--time", "--arm", "e8k"] + dd, False),
             ("time_e32k", tr + ["--time", "--arm", "e32k"] + dd, False),
             ("time_e32k_nohost", tr + ["--time", "--arm", "e32k", "--skip-host"] + dd, False),
             ("parts_e32k", tr + ["--time-parts", "--arm", "e32k"] + dd, False),
             ("verdict_pace", [py, os.path.join(HERE, "sc4_verdict.py"), "--pace-only"], False),
             ("run_e2k-h", tr + ["--run", "--arm", "e2k-h", "--resume"] + dd, True),
             ("run_e2k", tr + ["--run", "--arm", "e2k", "--resume"] + dd, True),
             ("verdict_g0", [py, os.path.join(HERE, "sc4_verdict.py"), "--g0-only"], False),
             ("run_e32k", tr + ["--run", "--arm", "e32k", "--resume"] + dd, True),
             ("run_e8k", tr + ["--run", "--arm", "e8k", "--resume"] + dd, True),
             ("verdict", [py, os.path.join(HERE, "sc4_verdict.py")], False)]
    note(kind="chain_start", pid=os.getpid(), from_step=a.from_step)
    for k, (name, cmd, resumable) in enumerate(steps):
        if k < a.from_step:
            continue
        for attempt in range(3 if resumable else 1):
            log = os.path.join(LOGS, f"{name}.log")
            note(kind="start", step=k, name=name, attempt=attempt)
            with open(log, "a") as f:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            tail = open(log, errors="replace").read()[-4000:]
            note(kind="end", step=k, name=name, rc=rc)
            if rc == 0:
                break
            if "STOP" in tail or "REFUSED" in tail or not resumable:
                note(kind="chain_stop", name=name, why="STOP/REFUSED or not resumable"); return
        else:
            note(kind="chain_stop", name=name, why="three failures"); return
        if name == "verdict_pace":
            txt = open(os.path.join(RES, "f4_verdict_pace.txt")).read()
            if "-> PASS" not in txt.split("G1 PACE:")[1].split("\n")[0]:
                note(kind="chain_stop", name=name, why="G1 did not pass (or missing): no long run"); return
        if name == "verdict_g0":
            txt = open(os.path.join(RES, "f4_verdict_g0.txt")).read()
            if "-> PASS" not in txt.split("G0 EQUIVALENCE:")[1].split("\n")[0]:
                note(kind="chain_stop", name=name, why="G0 did not pass (or missing): e32k and e8k do not start"); return
    note(kind="chain_done")


if __name__ == "__main__":
    main()
