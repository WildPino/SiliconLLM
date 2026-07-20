#!/usr/bin/env python3
"""WS5 item 1: AdamW-8bit vs fp32 AdamW, A/B on one protocol.

The rules this must satisfy (they are the brief's, not mine):
  * same protocol both sides, one variable;
  * applied symmetrically to all arms if adopted;
  * the MODEL maths must be unchanged -- so the evidence is a LOSS-CURVE OVERLAY, not just a speed number.
    The optimizer's state is quantized, the model's forward is not; if the curves separate beyond seed
    noise the win is not free and the decision changes.
  * resume-compatible.

Reported: tok/s, peak memory, optimizer-state bytes, per-step loss overlay + the end BPB.

Run: python benchmarks/phase64/mve/ws5_optbench.py [--stage E] [--steps 120]
"""
import argparse, os, subprocess, sys, re, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PY): PY = sys.executable
DATA = os.path.join(ROOT, "kaggle_mve", "account_1", "mve_data", "data")
TMP = os.environ.get("TEMP", "/tmp")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="E")
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()

rows = []
for tag, extra in (("AdamW-8bit", []), ("fp32 AdamW", ["--fp32-opt"])):
    out = os.path.join(TMP, f"ws5_{tag.split('-')[0].replace(' ', '')}.pt")
    for f in (out, out + ".done", out + ".r"):
        if os.path.exists(f): os.remove(f)
    cmd = [PY, os.path.join(HERE, "mve_train.py"), "--tag", "full", "--arm", "kd", "--kd", "span",
           "--recall", "on", "--sparse-moe", "--fp16", "--device", "cuda",
           "--stages", a.stage, "--steps", str(a.steps), "--seq", str(a.seq), "--batch", str(a.batch),
           "--accum", "2", "--eval-tok", "20000",
           "--data-dir", DATA, "--ckpt-dir", TMP, "--out", out, "--resume-ckpt", out + ".r"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    opt = re.search(r"optimizer: (\w+)", txt)
    done = re.search(r"stage . done: val BPB ([\d.]+) -> ([\d.]+) \(([-+\d.]+)\) \| (\d+) tok/s", txt)
    losses = [float(m) for m in re.findall(r"loss=([\d.]+)", txt)]
    rows.append(dict(tag=tag, opt=opt.group(1) if opt else "?",
                     bpb_in=float(done.group(1)) if done else float("nan"),
                     bpb_out=float(done.group(2)) if done else float("nan"),
                     tps=int(done.group(4)) if done else 0, losses=losses))
    if not done:
        print(f"[{tag}] run did not report a stage line -- tail:")
        print("\n".join(txt.strip().splitlines()[-8:]))

print(f"\nWS5 / optimizer A/B   stage={a.stage} steps={a.steps} seq={a.seq} batch={a.batch} accum=2 (3060)")
print(f"  {'variant':12s} {'optimizer':16s} {'tok/s':>8s} {'BPB in':>9s} {'BPB out':>9s}")
for r in rows:
    print(f"  {r['tag']:12s} {r['opt']:16s} {r['tps']:8d} {r['bpb_in']:9.4f} {r['bpb_out']:9.4f}")

if len(rows) == 2 and all(r["tps"] for r in rows):
    d = rows[0]["tps"] / rows[1]["tps"] - 1
    db = rows[0]["bpb_out"] - rows[1]["bpb_out"]
    print(f"\n  8-bit vs fp32: throughput {d*100:+.1f}%   BPB {db:+.4f}  (sigma_seed = 0.005)")
    n = min(len(rows[0]["losses"]), len(rows[1]["losses"]))
    if n:
        dif = [abs(rows[0]["losses"][i] - rows[1]["losses"][i]) for i in range(n)]
        print(f"  loss-curve overlay over {n} logged points: mean |d loss| = {sum(dif)/n:.4f}, "
              f"max = {max(dif):.4f}")
        print("  " + " ".join(f"{rows[0]['losses'][i]:.3f}/{rows[1]['losses'][i]:.3f}" for i in range(min(n, 6))))
print("\nSTOP. WS5 item 1 above. No commit.")
