"""Reads w0_ref_speed.py's record: per engine, cell, file and depth, the median over the OK processes of each process's
median tok/s over its repetitions 2-5 (the first is dropped: it runs on pages the warm-up token did not touch), with
their min..max; the best cell per engine; the desk model's figure for the same file beside it. Prints numbers only.

A run that stopped VOID is refused unless --void-as-lower-bound is given. Then only its OK processes are read, and
the record says the figures are lower bounds: contention can only slow a process.

    python -I benchmarks/wallmap/w0_ref_read.py --run benchmarks/wallmap/results/w0_ref_speed_1.json \
        --desk benchmarks/wallmap/results/w0_desk.json --out benchmarks/wallmap/results/w0_ref_read_1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--desk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--void-as-lower-bound", action="store_true")
    a = ap.parse_args()
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    desk = json.loads(Path(a.desk).read_text(encoding="utf-8"))
    status = "COMPLETE"
    if run.get("verdict_input") != "COMPLETE":
        if not a.void_as_lower_bound:
            print("not a complete run:", run.get("verdict_input"))
            return 1
        status = f"LOWER BOUND, read from a run that stopped {run.get('verdict_input')}"
        print(status)
    desk_today = {r["ctx"]: r for r in desk["rows"] if r["scenario"] == "today_q4km"}
    cells: dict[tuple, list[float]] = {}
    for p in run["processes"]:
        if p.get("phase") != "run" or p["status"] != "OK":
            continue
        for r in p["rows"]:
            v = statistics.median(float(x) for x in r["samples_ts"][1:])
            cells.setdefault((p["engine"], p["cell"], p["file"], int(r["depth"])), []).append(v)
    rows = []
    for (e, c, f, d), v in sorted(cells.items()):
        rows.append({"engine": e, "cell": c, "file": f, "depth": d, "n": len(v), "median": statistics.median(v),
                     "min": min(v), "max": max(v)})
    best = {}
    for r in rows:
        k = (r["engine"], r["file"], r["depth"])
        if k not in best or r["median"] > best[k]["median"]:
            best[k] = r
    out = {"run": a.run, "status": status, "rows": rows,
           "best": [dict(v, desk_today_tok_s=(desk_today[2048][v["file"]]["tok_s"] if v["depth"] == 2048 else None))
                    for v in best.values()]}
    for r in rows:
        print(f"{r['engine']:6s} {r['cell']:8s} {r['file']:6s} d{r['depth']:<5d} n={r['n']} median {r['median']:7.2f} "
              f"[{r['min']:.2f}..{r['max']:.2f}]")
    print("best cell per engine, file and depth:")
    for v in out["best"]:
        print(f"  {v['engine']:6s} {v['file']:6s} d{v['depth']:<5d} {v['cell']:8s} {v['median']:7.2f}"
              + (f"  desk(Q4_K_M) {v['desk_today_tok_s']:.1f}" if v["desk_today_tok_s"] else ""))
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
