"""Stage 2 part 1, test 2 runner: s2h_run.py's runner (stage 1's panels, k = 8, mode none, resumable, numbers only)
with test 2's arms.  Each arm is Ling-mini with one trained ternary organ (s2o_export.py); the whole organs (the dense
FFN, the head) also run the chat panel.  s2o_verdict.py decides.

Usage:
    python -I benchmarks/stage2/s2o_run.py --work E:/SiliconLLM_Artifacts/stage1/s1a --arm org_attn \
        --model <q4.gguf> --model-sha <sha256> --out benchmarks/stage2/results/s2o_run.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2h_run  # noqa: E402

s2h_run.CELL = "S2O-one-organ-ternary"
s2h_run.ARMS = {"org_attn": ("docs",), "org_shexp": ("docs",), "org_exps": ("docs",), "org_attn_shuf": ("docs",),
                "org_dense": ("docs", "chat"), "org_head": ("docs", "chat")}

if __name__ == "__main__":
    raise SystemExit(s2h_run.main())
