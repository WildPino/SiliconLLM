"""S2M's panels, numbers only: s2z_verdict.py and s2z_k4_verdict.py decide.  S2L's runner with S2M's arms.  ks_org is S2L's file: it adds the five panels S2L did not run, at k = 8 and
mode none as S2L ran its documents and chat, which stay in s2y_run.json.  ks_stk4 and ks_all4 generate at k = 4, as stk4.
Amendment 1: ks_org4 is ks_org's file generating at k = 4 as bal4; its score panels are ks_org's, copied, as tq404's.  Its record is s2z_k4_run.json.

Usage: python -I benchmarks/stage2/s2z_run.py --arm A --model F --model-sha S --out benchmarks/stage2/results/s2z_run.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2y_run as R  # noqa: E402

R.ARMS.clear()
R.ARMS.update({"ks_org": ("nq", "tqa", "gsm8k", "arc", "mmlu"), "ks_stk4": R.FULL, "ks_all4": R.FULL, "ks_org4": ("nq", "tqa", "gsm8k", "chat")})
R.K4 = ("ks_stk4", "ks_all4", "ks_org4")
R.COPY.clear()
R.COPY["ks_org4"] = "ks_org"

if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    arm = sys.argv[sys.argv.index("--arm") + 1]
    if not out.exists():
        cell = "S2M-amendment-1-ks_org-at-k4" if arm == "ks_org4" else "S2M-first-stack-at-2-bits"
        out.write_text(json.dumps({"cell": cell, "arms": {}}) + "\n", encoding="utf-8")
    raise SystemExit(R.main())
