"""C50-A8 verdict: applies the registered rule to a c50_a8_verify.py record (it chooses nothing itself).

PASS iff T1, T2, T3, T4, T5_text and T5_ids each have 0 mismatches.  The "newer" group (unassigned code
points) is printed, never gated.  T6 (loop smoke) is judged from the c50_chat.exe run, not here.

Usage:
    python -I benchmarks/chatbot50/c50_a8_verdict.py results/<record>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GATED = ("T1", "T2", "T3", "T4", "T5_text", "T5_ids")


def main() -> int:
    rec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    c = rec["checks"]
    fails = [k for k in GATED if c[k]["mismatches"] != 0]
    for k in GATED:
        print(f"{k:8s} {c[k]['items']:5d} items {c[k]['mismatches']:4d} mismatches")
    print(f"newer    {c['newer']['items']:5d} items {c['newer']['mismatches']:4d} mismatches (reported, not gated)")
    print("VERDICT", "PASS" if not fails else "FAIL: " + ", ".join(fails))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
