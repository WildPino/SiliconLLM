"""Stage 2 test 2b: Ling-flash's desk ceiling with the head in two passes, from stage 1's s1_ceiling.py.  DESK-MODEL.

Every organ but the head is as in stage 1's row "all organs ternary, IQ2_BN measured, full q8_0 KV".  The head costs:
    the preview    ternary: its weights at IQ2_BN's rate on the attention cell, as stage 1's head; every other preview:
                   its bytes at stage 0's 32 GB/s wall;
    the exact rows K rows of the stored head (Q6_K, 210 bytes per 256 weights) at the same wall.  Rows are 3,360 bytes
                   apart at random: the wall is a bound that the gather has not been measured against;
    the selection  the K largest of 157,184 logits, SELECT_MS, a stated guess until the engine times it.
A low-rank preview's rank on flash is twice mini's: SVD-softmax's Table 1 finds the window proportional to the hidden
size, and flash's is twice mini's.  This is an assumption about flash, not a measurement on it.

Usage: python -I benchmarks/stage2/s2t_ceiling.py --json benchmarks/stage2/results/s2t_ceiling.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
import s1_ceiling as C  # noqa: E402

SELECT_MS = 0.2
STORED_ROW_BYTES_PER_WEIGHT = 210 / 256
FULL_KV = (1.0625, 1.0)


def preview_ms(kind: str, V: int, d: int, R: float, BW: float) -> tuple[float, float]:
    """(MB, ms) of one preview on flash."""
    if kind == "ternary":
        return V * d / 4 / 1e6, 1e3 * V * d / R
    if kind == "sign":
        b = V * d / 8
    elif kind == "int4":
        b = V * d / 2
    elif kind.startswith("rank"):
        r = 2 * int(kind[4:])                                # mini's rank, doubled for flash's width
        b = V * r + 2 * d * r
    else:
        raise SystemExit(f"unknown preview {kind}")
    return b / 1e6, 1e3 * b / BW


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(HERE.parents[0] / "wallmap" / "results" / "w0_byte_map.json"))
    ap.add_argument("--verdict", default=str(HERE.parents[0] / "stage1" / "results" / "s1k_verdict.json"))
    ap.add_argument("--json", required=True)
    ap.add_argument("--extra-ms", type=float, default=0.0, help="other organs' added time, for the row with test 3b")
    a = ap.parse_args()
    flash = [m for m in json.loads(Path(a.map).read_text(encoding="utf-8"))["models"] if m["model"] == "flash"][0]
    s = json.loads(Path(a.verdict).read_text(encoding="utf-8"))["summary"]
    iq2 = C.rates(s, "ik_llama.cpp|iq2_bn")
    forms = {o: "nibble" for o in C.TERNARY_ORGANS} | {"router": "int8"}
    base = C.ceiling(flash, forms, iq2, kv=FULL_KV)
    stored = C.ceiling(flash, {o: f for o, f in forms.items() if o != "head"}, iq2, kv=FULL_KV)
    h = flash["hparams"]
    V, d = int(h["vocab_size"]), int(h["embedding_length"])
    BW = C.D.BW
    rest = base["ms_per_token"] - base["ms"]["head"]
    rows = {}
    for kind in ("ternary", "sign", "int4", "rank64", "rank128", "rank256", "rank512"):
        mb, ms = preview_ms(kind, V, d, iq2["head"], BW)
        for k in (256, 1024, 4096):
            exact = k * d * STORED_ROW_BYTES_PER_WEIGHT
            head = ms + 1e3 * exact / BW + SELECT_MS
            total = rest + head + a.extra_ms
            rows[f"{kind} K={k}"] = {"preview_mb": round(mb, 1), "exact_mb": round(exact / 1e6, 2),
                                     "head_ms": round(head, 2), "ms_per_token": round(total, 2),
                                     "tok_s": round(1e3 / total, 2)}
    out = {"note": "DESK-MODEL, Ling-flash, context 2048, decode-only k = 4; every other organ ternary IQ2_BN, full q8_0 KV",
           "select_ms": SELECT_MS, "extra_ms": a.extra_ms, "wall_gb_s": BW / 1e9,
           "reference": {"head ternary (stage 1)": {"head_ms": round(base["ms"]["head"], 2), "tok_s": round(base["tok_s"], 2)},
                         "head as stored (Q6_K)": {"head_ms": round(stored["ms"]["head"], 2), "tok_s": round(stored["tok_s"], 2)}},
           "rows": rows}
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["reference"]))
    for n, r in rows.items():
        print(f"{r['tok_s']:6.2f} tok/s  head {r['head_ms']:6.2f} ms  preview {r['preview_mb']:7.1f} MB  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
