#!/usr/bin/env python3
"""WS3 -- how many parallel fetchers does the Software Heritage path actually reward?

The first smoke measured 0.63 MiB/s at 64 workers, which extrapolates to 18 h for the plan's ~40 GB.
That number is meaningless without this scan, because the path is REQUEST-LATENCY bound, not bandwidth
bound: the median Stack file is a few KiB, so throughput is (files/s x mean size) and files/s is set by
how many round trips are in flight. Reporting the 64-worker figure as "the" download cost would be a
measurement of an arbitrary flag value.

Run: python benchmarks/phase64/data/ws3_fetchscan.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import swh_fetch as F   # noqa: E402

SHARD = os.path.join(ROOT, "data", "phase64", "raw", "shard0.jsonl")


def main():
    rows = [json.loads(l) for l in open(SHARD, encoding="utf-8")]
    for r in rows: r.pop("text", None)
    n = min(1500, len(rows))
    mean_kb = sum(r["length"] for r in rows[:n]) / n / 1024
    print(f"WS3 fetch scaling   {n} blobs/point, mean {mean_kb:.1f} KiB, anonymous\n")
    print(f"  {'workers':>8} {'MiB/s':>8} {'files/s':>9} {'lost':>6} {'40GB hours':>11}")
    prev = None
    for w in (32, 64, 128, 256, 512):
        st, _ = F.fetch_many(rows[:n], w, True, report_every=1e9)
        fps = st["ok"] / max(st["seconds"], 1e-9)
        gain = "" if prev is None else f"   {100*(st['mib_s']/prev-1):+.0f}% vs prev"
        print(f"  {w:8d} {st['mib_s']:8.2f} {fps:9.1f} {st['missing']+st['undecodable']:6d} "
              f"{40*1024/max(st['mib_s'],1e-9)/3600:11.1f}{gain}")
        prev = st["mib_s"]
    print("\n  Read the knee, not the last row: past it the extra threads only add connections.")
    print("\nSTOP. WS3 fetch scaling above. No commit.")


if __name__ == "__main__":
    main()
