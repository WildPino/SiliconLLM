"""S2L, a check added after the verdict: does ks_org's file hold 2-bit values?  On every BF16 organ tensor, the number of distinct values in each block of 32 weights.
IQ2_KS writes 4 levels per block, so a decoding fault or an untouched tensor reads more.  The routed experts must still be P0's Q4_K.

Usage: python -I benchmarks/stage2/s2y_ks_levels.py --model E:/SiliconLLM_Artifacts/stage2/s2y/q4/ks_org.gguf --json benchmarks/stage2/results/s2y_ks_levels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1"))
from s1t_export import gguf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    organs, experts = {}, {}
    for t in gguf.GGUFReader(str(a.model)).tensors:
        if "_exps" in t.name:
            experts[t.name] = t.tensor_type.name
            continue
        if t.tensor_type.name != "BF16":
            continue
        w = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, 32)
        w = np.sort(w, axis=1)
        n = 1 + (np.diff(w, axis=1) != 0).sum(axis=1)
        organs[t.name] = {"blocks": int(len(n)), "max": int(n.max()), "mean": float(n.mean())}
    rec = {"organ_tensors": len(organs), "max_distinct_per_32": max(v["max"] for v in organs.values()),
           "experts_types": sorted(set(experts.values())), "tensors": organs}
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "tensors"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
