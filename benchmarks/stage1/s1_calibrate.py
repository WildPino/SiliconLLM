"""Stage 1 (a) calibration of the Top-P threshold: the p at which Top-P keeps 4.0 experts per token on average.

Data: the first 200 items of A4's PIQA inputs (a likelihood panel of C50-A4 that the stage 1 panel does not use), run
through s1_llama_score in pgrid mode (observe only).  The rule: the grid p (0.30..0.95, step 0.01) whose mean number of
kept slots, over every token of every MoE layer, is closest to 4.0; ties to the lower p.  The chosen p is registered
in the protocol and in s1_run.py before the first arm runs.

Usage:
    python -I benchmarks/stage1/s1_calibrate.py --work E:/SiliconLLM_Artifacts/stage1/calib \
        --out benchmarks/stage1/results/s1a_calibrate.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_run import CAMPAIGN, HERE, MODEL, sha256  # noqa: E402

ITEMS, TARGET = 200, 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    src = CAMPAIGN / "a4" / "inputs" / "piqa.tsv"
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()][:ITEMS]
    ids = work / "piqa_first200.tsv"
    ids.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    exe = HERE / "bin" / "s1_llama_score_avx2.exe"
    cmd = [str(exe), "--model", str(MODEL[0]), "--ids", str(ids), "--out", str(work / "pgrid"), "--threads", "6",
           "--ubatch", "512", "--ctx", "4096", "--moe-mode", "pgrid"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines_out = proc.stdout.splitlines()
    grid_line = [ln for ln in lines_out if ln.startswith("PGRID ")]
    if proc.returncode != 0 or len(grid_line) != 1:
        raise SystemExit(f"calibration run failed: {proc.returncode} {proc.stderr[-400:]}")
    grid = {float(p): float(m) for p, m in (kv.split(":") for kv in grid_line[0].split()[2:])}
    best = min(sorted(grid), key=lambda p: (abs(grid[p] - TARGET), p))
    out = {"source": str(src), "items": ITEMS, "ids_sha256": sha256(ids), "exe_sha256": sha256(exe), "cmd": cmd,
           "config": [ln for ln in lines_out if ln.startswith(("CONFIG ", "MOE "))], "grid_mean_kept": grid,
           "p": best, "mean_kept_at_p": grid[best]}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"p={best:.2f} mean_kept={grid[best]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
