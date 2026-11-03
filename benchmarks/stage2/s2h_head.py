"""Stage 2 part 1, test 1: Ling-mini's head (output.weight, 157,184 rows of 2,048) ternary with one scale per row,
IQ2_BN's layout, and no training; the rest of the model is P0 as stored.

    imatrix  llama-imatrix on P0 with --process-output over the fit24 sample (-c 512, 6 threads, the settings of
             fit24 itself): the mean square of each of the head's 2,048 inputs.
    build    --arm tern_im | tern_col | shuf: the ternary head written into a copy of the bf16 GGUF, which is quantized
             as P0 was (Q4_K_M, fit24 imatrix) with --output-tensor-type tq2_0.

Ternarization, per row, with column weights h: the support S is the k weights largest in |w|, and k and the scale s
maximize (sum_S h|w|)^2 / sum_S h, the least-squares ternary fit; s = sum_S h|w| / sum_S h, rounded to bf16, and the
row becomes s * sign(w) on S, 0 elsewhere.  With h = 1 this is the exact optimum; with other h the support by |w| is a
heuristic.
    tern_im  h = the imatrix of the head's input: the error weighted by how large each input is;
    tern_col a scale per input as well as per row: the head becomes W diag(c) and the final norm's gain g / c, an
             exact change of the float model, before the same fit, so the effective head is diag(s) T diag(1 / c).
             c = rms^alpha (rms = the square root of the imatrix), normalized to geometric mean 1; the rows are fit
             with h / c^2, so the error stays weighted in the original space.  alpha is the grid point (-1, -0.95,
             ..., 1) with the least weighted error sum_j h_j (w_j - t_j / c_j)^2 over 16,384 rows (seed 20261103).
             alpha > 0 is AWQ's per-input scaling (arXiv 2306.00978, section 3.2, which searches [0, 1] on the output
             error of cached inputs); a row and a column vector around a low-bit matrix is OneBit's form (arXiv
             2402.11295, section 3.2, with signs); here the column vector lives in the norm, so the matrix stays
             IQ2_BN;
    shuf     tern_col's rows in a random order (seed 20261103): the planted control, which every token's logit must
             feel.
IQ2_BN keeps the scale as an f32 per row and the codes as {-1, 0, 1}, so it holds these rows exactly.  The panel's
llama.cpp has no IQ2_BN; TQ2_0 holds them exactly too (its fp16 block maximum is s, and a bf16 s is an fp16 value).
Checks written to the build record: output.weight in the output file is TQ2_0 and dequantizes to the built rows bit
for bit; output_norm.weight equals the built gain; every other tensor equals P0's, byte for byte.

Usage:
    python -I benchmarks/stage2/s2h_head.py imatrix --out C:/SiliconLLM_fast/s2h/head-imatrix.gguf
    python -I benchmarks/stage2/s2h_head.py build --arm tern_col --imatrix C:/SiliconLLM_fast/s2h/head-imatrix.gguf \
        --out-dir C:/SiliconLLM_fast/s2h --json benchmarks/stage2/results/s2h_build_tern_col.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
LLAMA = CAMPAIGN / "tools" / "llama.cpp-5b335f4"
sys.path.insert(0, str(LLAMA / "gguf-py"))
import gguf  # noqa: E402

BF16 = CAMPAIGN / "gguf" / "ling-mini-2.0-a810f64-bf16.gguf"
P0 = CAMPAIGN / "gguf" / "ling-mini-2.0-a810f64-q4_k_m.gguf"
P0_SHA = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
FIT24 = CAMPAIGN / "gguf" / "ling-mini-2.0-a810f64-imatrix-fit24.gguf"
FIT24_TEXT = CAMPAIGN / "gguf" / "imatrix_fit24_input.txt"
BIN = LLAMA / "build-avx2" / "bin"
# the binaries need the MinGW runtime first on PATH (see benchmarks/stage1/s1t_export.py)
RUNTIME = (Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
           / "BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe" / "mingw64" / "bin")
SEED = 20261103
CHUNK = 4096
ALPHAS = [i / 20 for i in range(-20, 21)]
SAMPLE_ROWS = 16384


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def run(cmd: list[str], log: Path) -> None:
    env = {**os.environ, "PATH": os.pathsep.join([str(BIN), str(RUNTIME), os.environ.get("PATH", "")])}
    q = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    log.write_text(q.stdout + q.stderr, encoding="utf-8")
    if q.returncode:
        raise SystemExit(f"{Path(cmd[0]).name} failed ({q.returncode}), log {log}")


def head_bf16() -> tuple[np.ndarray, int, np.ndarray, int]:
    by = {t.name: t for t in gguf.GGUFReader(str(BF16)).tensors}
    t, g = by["output.weight"], by["output_norm.weight"]
    assert t.tensor_type.name == "BF16" and g.tensor_type.name == "F32"
    u = np.frombuffer(np.asarray(t.data).tobytes(), dtype="<u2").reshape(int(t.shape[1]), int(t.shape[0]))
    gain = np.frombuffer(np.asarray(g.data).tobytes(), dtype="<f4").copy()
    return u, int(t.data_offset), gain, int(g.data_offset)


def to_f32(u: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))


def input_ms(imatrix: Path) -> np.ndarray:
    by = {t.name: np.asarray(t.data, dtype=np.float64).ravel() for t in gguf.GGUFReader(str(imatrix)).tensors}
    s2, n = by["output.weight.in_sum2"], by["output.weight.counts"]
    return s2 / n[0]


def ternarize(w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """w [R, C] fp32, h [C] -> bf16 rows s * sign(w) on the best top-k support (module docstring)."""
    a = w.abs()
    order = a.argsort(dim=1, descending=True)
    sa, sh = a.gather(1, order), h[order]
    c1, c2 = (sh * sa).cumsum(1), sh.cumsum(1)
    k = (c1 * c1 / c2).argmax(1)
    s = (c1.gather(1, k[:, None]) / c2.gather(1, k[:, None])).squeeze(1)
    s = s.to(torch.bfloat16).float()
    keep = torch.zeros_like(w, dtype=torch.bool).scatter_(1, order, torch.arange(w.shape[1]) <= k[:, None])
    return (torch.sign(w) * keep * s[:, None]).to(torch.bfloat16)


def scale_of(h: torch.Tensor, alpha: float) -> torch.Tensor:
    c = h.sqrt() ** alpha
    return c / c.log().mean().exp()


def weighted_err(w: torch.Tensor, t: torch.Tensor, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Per row: the weighted error of the effective head t / c against w, over the row's weighted energy."""
    e = t.float() / c - w
    return (e * e * h).sum(1) / (w * w * h).sum(1)


def col_alpha(u: np.ndarray, h: torch.Tensor) -> tuple[float, dict]:
    idx = np.sort(np.random.default_rng(SEED).choice(u.shape[0], SAMPLE_ROWS, replace=False))
    w = to_f32(u[idx])
    err = {}
    for al in ALPHAS:
        c = scale_of(h, al)
        err[f"{al:.2f}"] = float(weighted_err(w, ternarize(w * c, h / c ** 2), c, h).mean())
    best = min(err, key=err.get)
    return float(best), err


def build(arm: str, imatrix: Path, out_dir: Path) -> dict:
    t0 = time.time()
    u, off, gain, goff = head_bf16()
    rows, cols = u.shape
    h = torch.from_numpy(input_ms(imatrix)).float()
    alpha, grid = (0.0, None) if arm == "tern_im" else col_alpha(u, h)
    c = scale_of(h, alpha)
    new_gain = (torch.from_numpy(gain) / c).numpy().astype("<f4") if alpha > 0 else gain
    out = np.empty_like(u)
    st = {"density": [], "rel_err": [], "rel_err_weighted": []}
    for r0 in range(0, rows, CHUNK):
        w = to_f32(u[r0:r0 + CHUNK])
        t = ternarize(w * c, h / c ** 2)
        out[r0:r0 + CHUNK] = t.view(torch.int16).numpy().view(np.uint16)
        e = t.float() / c - w   # the effective head's error, in the original space
        st["density"].append((t != 0).float().mean(1))
        st["rel_err"].append((e * e).sum(1) / (w * w).sum(1))
        st["rel_err_weighted"].append(weighted_err(w, t, c, h))
    if arm == "shuf":
        out = out[np.random.default_rng(SEED).permutation(rows)]
    st = {k: torch.cat(v) for k, v in st.items()}
    stats = {k: {"mean": float(v.mean()), "p5": float(v.quantile(0.05)), "p95": float(v.quantile(0.95))}
             for k, v in st.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    bf, q4 = out_dir / f"head_{arm}-bf16.gguf", out_dir / f"head_{arm}-q4.gguf"
    shutil.copyfile(BF16, bf)
    with bf.open("r+b") as f:
        f.seek(off)
        f.write(out.astype("<u2").tobytes())
        f.seek(goff)
        f.write(new_gain.tobytes())
    run([str(BIN / "llama-quantize.exe"), "--imatrix", FIT24.as_posix(), "--output-tensor-type", "tq2_0",
         bf.as_posix(), q4.as_posix(), "Q4_K_M", "6"], out_dir / f"head_{arm}-quantize.log")
    bf.unlink()
    # checks: the head and the gain exact, everything else P0's bytes
    got = {t.name: t for t in gguf.GGUFReader(str(q4)).tensors}
    ref = {t.name: t for t in gguf.GGUFReader(str(P0)).tensors}
    o = got["output.weight"]
    want = (out.astype(np.uint32) << 16).view(np.float32)
    head_exact = (o.tensor_type.name == "TQ2_0"
                  and bool(np.array_equal(gguf.quants.dequantize(o.data, o.tensor_type).reshape(rows, cols), want)))
    gain_exact = bool(np.array_equal(np.frombuffer(np.asarray(got["output_norm.weight"].data).tobytes(), "<f4"),
                                     new_gain))
    others = [n for n in ref if n not in ("output.weight", "output_norm.weight")]
    differ = [n for n in others if got[n].tensor_type != ref[n].tensor_type
              or np.asarray(got[n].data).tobytes() != np.asarray(ref[n].data).tobytes()]
    return {"arm": arm, "imatrix": str(imatrix), "seed": SEED, "alpha": alpha, "alpha_grid_weighted_err": grid,
            "gain_changed": bool(alpha > 0), "rows": rows, "cols": cols, "row_stats": stats, "q4": str(q4),
            "q4_sha256": sha256(q4), "p0_sha256_checked": sha256(P0) == P0_SHA, "head_tq2_0_exact": head_exact,
            "output_norm_exact": gain_exact, "other_tensors": len(others), "other_tensors_differ": differ,
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["imatrix", "build"])
    ap.add_argument("--arm", choices=["tern_im", "tern_col", "shuf"])
    ap.add_argument("--imatrix", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.what == "imatrix":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        run([str(BIN / "llama-imatrix.exe"), "-m", P0.as_posix(), "-f", FIT24_TEXT.as_posix(), "-c", "512", "-t", "6",
             "--process-output", "-o", a.out.as_posix()], a.out.with_suffix(".log"))
        h = input_ms(a.out)
        r = {"imatrix": str(a.out), "sha256": sha256(a.out), "seconds": round(time.time() - t0, 1),
             "head_input_ms": {"mean": float(h.mean()), "max": float(h.max()), "argmax": int(h.argmax()),
                               "top8": [int(i) for i in np.argsort(-h)[:8]],
                               "share_of_top8": float(np.sort(h)[-8:].sum() / h.sum())}}
    else:
        if a.imatrix is None:
            raise SystemExit("build needs --imatrix")
        r = build(a.arm, a.imatrix, a.out_dir)
    print(json.dumps(r))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
