#!/usr/bin/env python3
"""WS4 / open number of the declared deviation: what does seq=2048 cost against seq=512?

WHY IT IS OPEN. The MQAR gate reads a distance curve at 128/512/2048, and a probe distance beyond the
training sequence length measures extrapolation, not recall -- so the gated ladder forces seq >= 2048.
The scan is LINEAR in L, so tokens/step and memory should not move when B.L is held constant. What does
move is batch parallelism: B drops 8 -> 2, and a GPU fed fewer, longer rows may not saturate. That is
the number, and it is wanted BEFORE the launch, not after.

PROTOCOL. One variable: (B, L). B.L is held at 4096 per micro-batch and accum at 2, so both sides see
IDENTICAL tokens per step and identical steps -- equal tokens, as required for the comparison to be
about throughput rather than about how much work each side did.

    seq= 512  batch=8  accum=2   ->  8192 tok/step
    seq=2048  batch=2  accum=2   ->  8192 tok/step

Reported: tok/s, peak allocated MiB (the linear-in-L claim is falsifiable here), and the BPB pair.
The BPB values are NOT a quality comparison -- at equal tokens a longer context is a different task with
a different loss floor, and reading them as an A/B would be a category error. They are reported only so
that a run which silently failed to train is visible.

Run: python benchmarks/phase64/mve/ws4_seqbench.py [--steps 120] [--stage E]
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PY): PY = sys.executable
DATA = os.path.join(ROOT, "kaggle_mve", "account_1", "mve_data", "data")
TMP = os.environ.get("TEMP", "/tmp")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="E")
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--configs", default="512x8,2048x2", help="comma list of SEQxBATCH; B.L should be constant")
a = ap.parse_args()

cfgs = [tuple(int(v) for v in c.split("x")) for c in a.configs.split(",")]
bl = {s * b for s, b in cfgs}
if len(bl) > 1:
    print(f"WARNING: B.L is NOT constant across configs ({sorted(bl)}). The comparison then confounds\n"
          f"         sequence length with how much work each micro-batch does. Proceeding, but the number\n"
          f"         below is not the one the deviation asked for.\n")

rows = []
for seq, batch in cfgs:
    out = os.path.join(TMP, f"ws4_seq{seq}x{batch}.pt")
    for f in (out, out + ".done", out + ".r"):
        if os.path.exists(f): os.remove(f)
    cmd = [PY, os.path.join(HERE, "mve_train.py"), "--tag", "full", "--arm", "kd", "--kd", "span",
           "--recall", "on", "--sparse-moe", "--fp16", "--device", "cuda",
           "--stages", a.stage, "--steps", str(a.steps), "--seq", str(seq), "--batch", str(batch),
           "--accum", "2", "--eval-tok", "20000",
           "--data-dir", DATA, "--ckpt-dir", TMP, "--out", out, "--resume-ckpt", out + ".r"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    m = re.search(r"stage . done: val BPB ([\d.]+) -> ([\d.]+) \([-+\d.]+\) \| (\d+) tok/s \| ([\d.]+) min"
                  r"(?: \| peak (\d+) MiB)?", txt)
    rows.append(dict(seq=seq, batch=batch,
                     bpb_in=float(m.group(1)) if m else float("nan"),
                     bpb_out=float(m.group(2)) if m else float("nan"),
                     tps=int(m.group(3)) if m else 0,
                     mins=float(m.group(4)) if m else float("nan"),
                     peak=int(m.group(5)) if (m and m.group(5)) else 0))
    if not m:
        print(f"[seq={seq} batch={batch}] no stage line -- tail:")
        print("\n".join(txt.strip().splitlines()[-10:]))

print(f"\nWS4 / sequence-length throughput   stage={a.stage} steps={a.steps} accum=2 (3060, fp16)")
print(f"  {'seq':>6} {'batch':>6} {'tok/step':>9} {'tok/s':>8} {'peak MiB':>9} {'min':>7} "
      f"{'BPB in':>9} {'BPB out':>9}")
for r in rows:
    print(f"  {r['seq']:6d} {r['batch']:6d} {r['seq']*r['batch']*2:9d} {r['tps']:8d} "
          f"{r['peak'] or -1:9d} {r['mins']:7.1f} {r['bpb_in']:9.4f} {r['bpb_out']:9.4f}")

if len(rows) == 2 and all(r["tps"] for r in rows):
    d = rows[1]["tps"] / rows[0]["tps"] - 1
    print(f"\n  seq={rows[1]['seq']} vs seq={rows[0]['seq']} at equal tokens: throughput {100*d:+.1f}%")
    if rows[0]["peak"] and rows[1]["peak"]:
        dm = rows[1]["peak"] / rows[0]["peak"] - 1
        verdict = ("consistent with linear-in-L at constant B.L" if abs(dm) < 0.10
                   else "NOT flat -- the linear-in-L assumption does not hold as stated")
        print(f"  peak memory {100*dm:+.1f}%  ({verdict})")
    print(f"\n  Cost of the 2048 ladder, if adopted: a rung-1 budget of H hours at seq=512 becomes "
          f"{1/(1+d):.2f}x H.")
print("\nSTOP. WS4 sequence-length number above. No commit.")
