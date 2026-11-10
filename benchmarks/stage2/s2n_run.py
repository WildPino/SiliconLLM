"""S2N's panels, numbers only: s2n_verdict.py decides.  S2L's runner with S2N's arms, all seven panels, generating at k = 4 (mass 16, decode only) as ks_stk4,
scoring at k = 8.  e2e_row2 is e2e_row's second data order, run only if s2n_verdict.py asks for it.

Usage: python -I benchmarks/stage2/s2n_run.py --arm A --model F --model-sha S --out benchmarks/stage2/results/s2n_run.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2y_run as R  # noqa: E402

ARMS = ("e2e_row", "e2e_grp", "glob", "e2e_row2")
R.ARMS.clear()
R.ARMS.update({a: R.FULL for a in ARMS})
R.K4 = ARMS
R.COPY.clear()

if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    if not out.exists():
        out.write_text(json.dumps({"cell": "S2N-stack-trained-together", "arms": {}}) + "\n", encoding="utf-8")
    raise SystemExit(R.main())
