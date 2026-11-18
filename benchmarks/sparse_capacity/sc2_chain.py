#!/usr/bin/env python3
# Sparse Capacity, phase 2: the ten runs, one at a time, in the protocol's order (F2_BANK_PROTOCOL.md section 10).
#
#   python benchmarks/sparse_capacity/sc2_chain.py --after-1a <f1a verdict.txt> [--data-dir ...]
#
# - Refuses unless the phase 1a verdict is PARI or VERDE (PLAN, phase 1a: a red flag is discussed before phase 2).
#   After a red flag, --red-flag-go "<date: decision>" records the go given on F1A_RUNG1_GATE_RESULT.md.
# - Waits while another trainer holds the GPU (S2N's GPU chain), never two trainers together.
# - Skips a run whose result.json exists; otherwise launches it with --resume, so re-launching the chain after a crash
#   or a freeze continues from the last checkpoint.
# - Stops at the first run that exits non-zero (a non-finite stop or a crash is read, not skipped).
# - Records whether S2N's CPU chain was running at the start and end of each run: the hours of those runs are not a
#   clean pace for phase 3.
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = "E:/SiliconLLM_Artifacts/workspace/kaggle_rung1/main_run/data"
CKPT = "E:/SiliconLLM_Artifacts/sparse_capacity/f2"
ORDER = [("m1", 0), ("m64", 0), ("m4", 0), ("m16", 0), ("m1", 1), ("m64", 1), ("m4", 1), ("m16", 1),
         ("dense", 0), ("dense-fp", 0)]
GPU_HOLDERS = ("s2n_chain.py gpu", "s2n_train.py", "s2n_chain.py second")
CPU_HOLDERS = ("s2n_chain.py cpu", "s2n_run.py")


def cmdlines():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    me = str(os.getpid())
    return [l for l in out.splitlines() if l.strip() and not l.startswith(me + " ")]


def holding(keys):
    return [l for l in cmdlines() if any(k in " ".join(l.split()) for k in keys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-1a", required=True)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--red-flag-go", default="")
    a = ap.parse_args()
    v = open(a.after_1a, encoding="utf-8").read()
    red = "VERDICT: BANDIERA ROSSA" in v
    if not ("VERDICT: PARI" in v or "VERDICT: VERDE" in v or (red and a.red_flag_go.strip())):
        sys.exit("REFUSED: the phase 1a verdict is not PARI or VERDE and no --red-flag-go was given "
                 "(PLAN: a red flag is discussed first)")
    log = open(os.path.join(ROOT, "results", "sparse_capacity", "f2_chain.jsonl"), "a")
    def emit(**r):
        r["time"] = time.strftime("%Y-%m-%d %H:%M:%S"); log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True)
    emit(kind="gate_1a", verdict=[l for l in v.splitlines() if l.startswith("VERDICT")][0][:80],
         red_flag_go=a.red_flag_go if red else "")
    for arm, seed in ORDER:
        tag = f"f2_{arm}_s{seed}"
        if os.path.exists(os.path.join(ROOT, "results", "sparse_capacity", tag, "result.json")):
            emit(kind="skip", run=tag); continue
        while True:
            h = holding(GPU_HOLDERS)
            if not h: break
            emit(kind="wait_gpu", run=tag, holders=[x[:120] for x in h]); time.sleep(600)
        cpu0 = bool(holding(CPU_HOLDERS))
        emit(kind="start", run=tag, s2n_cpu_running=cpu0)
        r = subprocess.run([sys.executable, os.path.join(HERE, "sc2_train.py"), "--run", "--arm", arm, "--seed", str(seed),
                            "--data-dir", a.data_dir, "--ckpt-dir", f"{CKPT}/{tag}", "--resume"], cwd=ROOT)
        emit(kind="end", run=tag, exit=r.returncode, s2n_cpu_running=bool(holding(CPU_HOLDERS)))
        if r.returncode:
            sys.exit(f"STOP: {tag} exited {r.returncode}")
    emit(kind="done")
    print("verdict: python benchmarks/sparse_capacity/sc2_verdict.py")


if __name__ == "__main__":
    main()
