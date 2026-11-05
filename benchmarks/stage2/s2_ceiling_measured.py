"""Stage 2 delivery: Ling-flash's ceiling at test 2b's cell 1, each desk row replaced by the idle session's measure where one exists.

Rows replaced: the KV (q8_0, and the f16 reading beside it), the router, the fixed term, the shared expert, and the head's selection and gather at
K = 1,024.  Every other row stays DESK-MODEL.  It prints the numbers against the line and labels nothing; a sum of cells is still not a run of an
engine on flash, which does not exist.

Usage: python -I benchmarks/stage2/s2k_cells.py report --run benchmarks/stage2/results/s2k_cells.json --json benchmarks/stage2/results/s2k_report.json
       python -I benchmarks/stage2/s2_ceiling_measured.py --cells benchmarks/stage2/results/s2k_report.json \
           --gather benchmarks/stage2/results/s2t_gather_time.json --extra-ms 1.03 --json benchmarks/stage2/results/s2_ceiling_measured.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2t_ceiling as T  # noqa: E402

C = T.C
LINE = 24.7
K = 1024


def desk() -> dict:
    root = HERE.parents[0]
    flash = [m for m in json.loads((root / "wallmap" / "results" / "w0_byte_map.json").read_text(encoding="utf-8"))["models"] if m["model"] == "flash"][0]
    s = json.loads((root / "stage1" / "results" / "s1k_verdict.json").read_text(encoding="utf-8"))["summary"]
    iq2 = C.rates(s, "ik_llama.cpp|iq2_bn")
    base = C.ceiling(flash, {o: "nibble" for o in C.TERNARY_ORGANS} | {"router": "int8"}, iq2, kv=T.FULL_KV)
    d = int(flash["hparams"]["embedding_length"])
    rows = {o: v for o, v in base["ms"].items() if o != "head"}
    rows |= {"fixed": base["fixed_ms"], "preview": base["ms"]["head"], "gather": 1e3 * K * d * T.STORED_ROW_BYTES_PER_WEIGHT / C.D.BW, "select": T.SELECT_MS}
    return rows


def measured(cells: dict, gather: dict) -> dict:
    ms = cells["ms_at_32_layers"]
    g = [c for c in gather["cells"] if c["k"] == K][0]
    return {"kv_cache": ms["kv_q8"], "router": ms["router"], "fixed": ms["fixed"], "shared": ms["shared"], "gather": g["gather_ms"][1], "select": g["select_ms"][1]}


def table(rows: dict, seen: dict, extra: float) -> dict:
    now = rows | seen
    total = sum(now.values()) + extra
    return {"rows": {o: {"desk": round(rows[o], 3), "now": round(now[o], 3), "measured": o in seen} for o in rows}, "extra_ms": extra,
            "desk_ms": round(sum(rows.values()) + extra, 2), "desk_tok_s": round(1e3 / (sum(rows.values()) + extra), 2),
            "ms": round(total, 2), "tok_s": round(1e3 / total, 2), "line": LINE, "room_ms": round(1e3 / LINE - total, 2)}


def selftest() -> int:
    rows = {"a": 30.0, "b": 10.0}
    t = table(rows, {"b": 12.0}, 1.0)
    assert t["ms"] == 43.0 and t["desk_ms"] == 41.0 and t["rows"]["b"]["measured"] and not t["rows"]["a"]["measured"], t
    assert round(sum(desk().values()), 2) == 39.37, sum(desk().values())
    print("selftest PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path)
    ap.add_argument("--gather", type=Path)
    ap.add_argument("--extra-ms", type=float, default=0.0, help="other organs' added time, test 3b's")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    cells = json.loads(a.cells.read_text(encoding="utf-8"))
    if cells.get("status") != "COMPLETE":
        raise SystemExit(f"cells not COMPLETE: {cells.get('status')}")
    out = table(desk(), measured(cells, json.loads(a.gather.read_text(encoding="utf-8"))), a.extra_ms)
    out["kv_f16_ms"] = cells["ms_at_32_layers"]["kv_f16"]
    out["note"] = "cell 1, K = 1,024; rows not marked measured are DESK-MODEL; a sum of cells, not a run on flash"
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
