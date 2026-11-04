"""Stage 2 test 2c: Ling-flash's desk ceiling with one group of the organs without a router left as stored (Q4_K_M), the
head in two passes (test 2b's cell 1).  DESK-MODEL, from s1_ceiling.py and s2t_ceiling.py: k = 4, context 2048, full q8_0
KV; every other organ ternary IQ2_BN at its measured rate.

Usage: python -I benchmarks/stage2/s2s_ceiling.py --json benchmarks/stage2/results/s2s_ceiling.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
import s1_ceiling as C  # noqa: E402
from s2t_ceiling import FULL_KV, SELECT_MS, STORED_ROW_BYTES_PER_WEIGHT, preview_ms  # noqa: E402

K = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(HERE.parents[0] / "wallmap" / "results" / "w0_byte_map.json"))
    ap.add_argument("--verdict", default=str(HERE.parents[0] / "stage1" / "results" / "s1k_verdict.json"))
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    flash = [m for m in json.loads(Path(a.map).read_text(encoding="utf-8"))["models"] if m["model"] == "flash"][0]
    iq2 = C.rates(json.loads(Path(a.verdict).read_text(encoding="utf-8"))["summary"], "ik_llama.cpp|iq2_bn")
    h = flash["hparams"]
    V, d = int(h["vocab_size"]), int(h["embedding_length"])
    _, ms = preview_ms("ternary", V, d, iq2["head"], C.D.BW)
    head = ms + 1e3 * K * d * STORED_ROW_BYTES_PER_WEIGHT / C.D.BW + SELECT_MS
    rows = {}
    for name, stored in (("all ternary", ()), ("attention stored", ("attention",)), ("shared expert stored", ("shared",)),
                         ("dense FFN stored", ("dense_ffn",)), ("shared + dense FFN stored", ("shared", "dense_ffn")),
                         ("attention + shared + dense FFN stored", ("attention", "shared", "dense_ffn"))):
        forms = {o: "nibble" for o in C.TERNARY_ORGANS if o not in stored} | {"router": "int8"}
        c = C.ceiling(flash, forms, iq2, kv=FULL_KV)
        total = c["ms_per_token"] - c["ms"]["head"] + head
        rows[name] = {"ms_per_token": round(total, 2), "tok_s": round(1e3 / total, 2),
                      "organ_ms": {o: round(c["ms"][o], 2) for o in ("attention", "shared", "dense_ffn")}}
    out = {"note": "DESK-MODEL, Ling-flash, context 2048, decode-only k = 4, head in two passes K = 1,024, full q8_0 KV", "rows": rows}
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for n, r in rows.items():
        print(f"{r['tok_s']:6.2f} tok/s {r['ms_per_token']:6.2f} ms  {n}  {r['organ_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
