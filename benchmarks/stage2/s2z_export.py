"""S2M's exports, on S2L's compose: replaced tensors must read back to the intended values, every other tensor must equal the reference's bytes.

    ks_stk4   bal-q4's experts (Q4_0, bytes copied) and router, the organs at S2L's ks_org values (IQ2_KS with the imatrix, held in BF16).  Reference: bal-q4.
    ks_all4   the organs as ks_stk4, the routed experts at IQ2_KS's values with the imatrix (s2z_ks.py), held in Q8_0.  Reference: P0.
              Q8_0 cannot hold IQ2_KS's levels exactly, so the experts are checked as a container: per tensor, the squared error of Q8_0 against the IQ2_KS values must stay
              under 1% of the IQ2_KS values' own error against the original weights.  Q8_0 multiplies by Q8_0 activations, as bal's Q4_0 does.

Usage: python -I benchmarks/stage2/s2z_export.py --arm A --out-q4 F --json F
       python -I benchmarks/stage2/s2z_export.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s2y_export import BAL_BF16, BAL_Q4, BF16, M, Y, compose, gguf, gguf_name, organ_names  # noqa: E402

KS_ORG, KS_ALL = Y / "ks-im-bf16.gguf", Y / "ks-all-bf16.gguf"
CONTAINER = 0.01


def bf16(a) -> np.ndarray:
    return (np.asarray(a).reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def container_ok(held: np.ndarray, want: np.ndarray, orig: np.ndarray) -> tuple[bool, float, float]:
    """held: what the file returns; want: the IQ2_KS values; orig: the original weights.  Relative squared errors."""
    c = float(((held - want) ** 2).sum() / (want * want).sum())
    q = float(((want - orig) ** 2).sum() / (orig * orig).sum())
    return c <= CONTAINER * q, c, q


def selftest() -> int:
    rng = np.random.default_rng(1)
    o = rng.standard_normal(1 << 16).astype(np.float32)
    s = np.abs(o).reshape(-1, 32).max(1, keepdims=True) / 2
    want = (np.clip(np.round(o.reshape(-1, 32) / s * 1.5) / 1.5, -1, 1) * s).reshape(-1)
    d = np.abs(want).reshape(-1, 32).max(1, keepdims=True) / 127
    q8 = (np.round(want.reshape(-1, 32) / np.where(d > 0, d, 1)) * d).reshape(-1)
    cases = [(container_ok(q8, want, o)[0], True), (container_ok(o, want, o)[0], False), (container_ok(want * 1.05, want, o)[0], False)]
    ok = all(a == b for a, b in cases)
    print("selftest", "PASS" if ok else "FAIL", len(cases), [round(container_ok(x, want, o)[1], 6) for x in (q8, o, want * 1.05)])
    return 0 if ok else 1


def organs(path: Path) -> dict[str, torch.Tensor]:
    r = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}
    names = [gguf_name(i, n) for i in range(M.N_LAYER) for n in organ_names("both", i)]
    return {g: torch.from_numpy(np.asarray(r[g].data).reshape(-1).view(np.int16).copy()).view(torch.bfloat16) for g in names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("ks_stk4", "ks_all4"))
    ap.add_argument("--out-q4", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    a.out_q4.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if a.arm == "ks_stk4":
        r = {t.name: t for t in gguf.GGUFReader(str(BAL_Q4)).tensors}
        over = {n: np.asarray(t.data).tobytes() for n, t in r.items() if "_exps" in n and t.tensor_type.name == "Q4_0"}
        rec = compose(organs(KS_ORG), "bf16", a.out_q4, base=BAL_BF16, ref=BAL_Q4, over=over)
        extra = {"experts_copied": len(over)}
    else:
        ks = {t.name: t for t in gguf.GGUFReader(str(KS_ALL)).tensors}
        names = [n for n in ks if "_exps" in n]
        pre = {n: (lambda t=ks[n]: np.asarray(t.data).tobytes()) for n in names}
        rec = compose(organs(KS_ALL), "bf16", a.out_q4, pre=pre, retype={n: "q8_0" for n in names})
        got = {t.name: t for t in gguf.GGUFReader(str(a.out_q4)).tensors}
        src = {t.name: t for t in gguf.GGUFReader(str(BF16)).tensors}
        cont = {}
        for n in names:
            held = gguf.quants.dequantize(got[n].data, got[n].tensor_type).reshape(-1).astype(np.float32)
            ok, c, q = container_ok(held, bf16(ks[n].data), bf16(src[n].data))
            cont[n] = {"ok": ok and got[n].tensor_type.name == "Q8_0", "container": c, "iq2_ks": q}
        extra = {"experts_container": cont, "experts_ok": all(v["ok"] for v in cont.values()), "worst_container_over_iq2_ks": max(v["container"] / v["iq2_ks"] for v in cont.values())}
        rec["all_exact"] = rec["all_exact"] and extra["experts_ok"]
    rec = {"arm": a.arm, **rec, **extra}
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cmd", "exact", "experts_container")}))
    return 0 if rec["all_exact"] and not rec["other_tensors_differ"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
