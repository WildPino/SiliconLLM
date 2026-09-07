# -*- coding: utf-8 -*-
"""E5 run 5 driver -- brief s11.4.  One engine process per record, ten arms interleaved inside it.

Runs 1-4 put each arm in its own process and differenced across minutes; at T10 @800 the weight
path is 301 ms, so a 1% gate is 3.0 ms and the component being estimated is 1.9 ms.  Here the
engine rotates the arm per token under a palindrome schedule and reports one organ line per arm,
so every difference this script records comes from a single model load.

cores_busy is still recorded, as a WITNESS.  s11.2: promoted to a gate in run 4 it discarded
nothing the weight-path gate had not already discarded, so it is not a gate here.
"""
import argparse, json, os, re, subprocess, sys, time

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donor_engine.exe")
ORGANS = ("qkv_proj", "rope", "attention", "o_proj", "ffn", "head", "norm+glue")


def _system_busy():
    """Mean cores burning MACHINE-WIDE: (kernel + user - idle) CPU-seconds.  Kernel time includes
    idle on Windows, so idle is subtracted exactly once.  A clean 6-thread run sits just above 6."""
    try:
        import ctypes
        from ctypes import wintypes
        i, k, u = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k),
                                                     ctypes.byref(u)):
            return None
        f = lambda t: ((t.dwHighDateTime << 32) | t.dwLowDateTime) / 1e7
        return f(k) + f(u) - f(i)
    except Exception:
        return None


def run_once(weights, bench, threads, prio, mode, narms):
    cmd = [ENGINE, "--weights", weights, "--threads", str(threads), "--profile",
           "--" + mode, "--bench", str(bench)]
    flags = 0
    if prio == "high" and hasattr(subprocess, "HIGH_PRIORITY_CLASS"):
        flags = subprocess.HIGH_PRIORITY_CLASS
    b0, t0 = _system_busy(), time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    b1, t1 = _system_busy(), time.time()
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise SystemExit("engine failed (%d)" % p.returncode)
    cb = ((b1 - b0) / (t1 - t0)) if (b0 is not None and b1 is not None and t1 > t0) else None

    arms = {}
    for m in re.finditer(r"^SWEEP (\S+) n=(\d+)(.*)$", p.stdout, re.M):
        org = dict(re.findall(r"([\w+]+)=([0-9.eE+-]+)", m.group(3)))
        arms[m.group(1)] = {"n": int(m.group(2)),
                            "organs": {k: float(org[k]) for k in ORGANS if k in org}}
    mt = re.search(r"BENCH\s+\d+ tokens\s+[\d.]+ s\s+([\d.]+) tok/s", p.stdout)
    missing = [k for k in arms if len(arms[k]["organs"]) != len(ORGANS)]
    if len(arms) != narms or missing:
        raise SystemExit("parsed %d arms, incomplete: %s" % (len(arms), missing))
    return {"arms": arms, "tok_s": float(mt.group(1)) if mt else None,
            "cores_busy": cb, "wall_s": t1 - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--bench", type=int, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--prio", default="")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--out", required=True)
    # s13.4: sweep6 is the ten avx4 arms (X now inside the family); sweepd is the four-entry
    # d-probe, three of whose entries run identical `none` code in different neighbourhoods.
    ap.add_argument("--mode", default="sweep", choices=("sweep", "sweep6", "sweepd"))
    a = ap.parse_args()

    narms = {"sweep": 10, "sweep6": 10, "sweepd": 4}[a.mode]
    r = run_once(a.weights, a.bench, a.threads, a.prio, a.mode, narms)
    r.update(shape=a.shape, bench=a.bench, run=a.run, threads=a.threads, prio=a.prio,
             mode=a.mode, label="%s_b%d_r%d_%s" % (a.shape, a.bench, a.run, a.mode))
    rec = []
    if os.path.exists(a.out):
        rec = json.load(open(a.out, encoding="utf-8"))
    rec.append(r)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
    print("%-16s %6.2f tok/s  cores_busy %.2f  attention ms/token: %s"
          % (r["label"], r["tok_s"] or 0.0, r["cores_busy"] or 0.0,
             "  ".join("%s=%.3f" % (k, r["arms"][k]["organs"]["attention"])
                       for k in sorted(r["arms"]))), flush=True)


if __name__ == "__main__":
    main()
