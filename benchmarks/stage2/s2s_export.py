"""Stage 2 test 2c export: the trained layers (s2s_train.py's layer_<i>.pt) into Ling-mini's GGUF, quantized as P0 was
(Q4_K_M, fit24 imatrix) with the trained tensors as TQ2_0.  The path and its checks are test 2's (s2o_export.py, with its
amendment 1): every other quantized tensor is pinned to P0's type by exact name; the layout is checked against the HF
checkpoint; each trained tensor must dequantize to the trained values bit for bit; every other tensor must equal P0's.
--layers L-H exports a part of the trained layers (st_attn_lo: st_attn's 0-9).  The bf16 copy goes to --work and is deleted.

Usage: python -I benchmarks/stage2/s2s_export.py --trained DIR [--layers 0-9] --out-q4 F --json F
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
import s1t_model as M  # noqa: E402
from s1t_export import BF16, IMATRIX, P0_SHA, as_bytes, gguf, sha256, tensors  # noqa: E402
from s2h_head import BIN, P0, run  # noqa: E402
from s2o_export import gguf_name  # noqa: E402


def export(trained: Path, layers: str, out_q4: Path) -> dict:
    t0 = time.time()
    files = sorted(trained.glob("layer_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if layers:
        lo, hi = (int(x) for x in layers.split("-"))
        files = [p for p in files if lo <= int(p.stem.split("_")[1]) <= hi]
    new = {}
    for p in files:
        i = int(p.stem.split("_")[1])
        st = torch.load(p, map_location="cpu", weights_only=True)
        new.update({(i, n): v for n, v in st.items() if not n.startswith("ternary_scale:")})
    tt = tensors()
    W = M.Weights()
    bf = out_q4.with_name(out_q4.stem + "-bf16.gguf")
    shutil.copyfile(BF16, bf)
    layout = {}
    with bf.open("r+b") as f:
        for (i, n), v in new.items():
            g = gguf_name(i, n)
            off, nb, kind = tt[g]
            f.seek(off)
            hf = f"model.layers.{i}.{n}"
            layout[g] = f.read(nb) == as_bytes(W.get([hf])[hf], kind)
            b = as_bytes(v, kind)
            assert len(b) == nb, g
            f.seek(off)
            f.write(b)
    if not all(layout.values()):
        raise SystemExit(f"layout differs from the HF checkpoint: {[k for k, v in layout.items() if not v]}")
    cmd = [str(BIN / "llama-quantize.exe"), "--imatrix", IMATRIX.as_posix()]
    changed = {gguf_name(i, n) for i, n in new}
    for g in sorted(changed):
        cmd += ["--tensor-type", g.replace(".", r"\.").removesuffix(r"\.weight") + "=tq2_0"]
    ref = {t.name: t for t in gguf.GGUFReader(str(P0)).tensors}
    pins = out_q4.with_name(out_q4.stem + "-pins.txt")
    pins.write_text("".join(f"^{n.replace('.', chr(92) + '.')}$={t.tensor_type.name.lower()}\n" for n, t in ref.items()
                            if n not in changed and t.tensor_type.name not in ("F32", "F16", "BF16")), encoding="ascii")
    cmd += ["--tensor-type-file", pins.as_posix(), bf.as_posix(), out_q4.as_posix(), "Q4_K_M", "6"]
    run(cmd, out_q4.with_name(out_q4.name + ".quantize.log"))
    bf.unlink()
    got = {t.name: t for t in gguf.GGUFReader(str(out_q4)).tensors}
    exact = {}
    for (i, n), v in new.items():
        t = got[gguf_name(i, n)]
        exact[gguf_name(i, n)] = t.tensor_type.name == "TQ2_0" and bool(np.array_equal(
            gguf.quants.dequantize(t.data, t.tensor_type).reshape(v.shape), v.float().numpy()))
    others = [n for n in ref if n not in changed]
    return {"trained": str(trained), "layers": layers or "all", "changed": len(changed), "layout_checked": layout,
            "tq2_0_exact": exact, "all_exact": all(exact.values()), "other_tensors": len(others),
            "other_tensors_differ": [n for n in others if got[n].tensor_type != ref[n].tensor_type
                                     or np.asarray(got[n].data).tobytes() != np.asarray(ref[n].data).tobytes()],
            "q4": str(out_q4), "q4_sha256": sha256(out_q4), "p0_sha256_checked": sha256(P0) == P0_SHA, "cmd": cmd,
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", type=Path, required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--out-q4", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    rec = export(a.trained, a.layers, a.out_q4)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cmd", "layout_checked", "tq2_0_exact")}))
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0 if rec["all_exact"] and not rec["other_tensors_differ"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
