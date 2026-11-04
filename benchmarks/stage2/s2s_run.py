"""Stage 2 test 2c runner: s2h_run.py's runner (stage 1's panels, k = 8, mode none, resumable, numbers only) with test 2c's
arms.  st_attn, st_ffn and st_both run the whole panel; st_attn_lo and st_attn_hi run the documents, ph_both the documents and chat.  The limit on
projected time is three times stage 1's, because the panels share the CPU with the trainer.  s2s_verdict.py decides.

Usage: python -I benchmarks/stage2/s2s_run.py --work E:/SiliconLLM_Artifacts/stage1/s1a --arm st_both \
           --model <q4.gguf> --model-sha <sha256> --out benchmarks/stage2/results/s2s_run.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2h_run  # noqa: E402

FULL = ("docs", "nq", "tqa", "gsm8k", "arc", "mmlu", "chat")
s2h_run.CELL = "S2S-stacked-organs"
s2h_run.MAX_PROJECTED_S = 3 * s2h_run.MAX_PROJECTED_S
s2h_run.ARMS = {"st_attn": FULL, "st_ffn": FULL, "st_both": FULL, "st_attn_lo": ("docs",), "st_attn_hi": ("docs",),
                "ph_both": ("docs", "chat")}

if __name__ == "__main__":
    raise SystemExit(s2h_run.main())
