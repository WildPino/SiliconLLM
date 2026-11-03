"""Stage 2 part 1, test 2 export: one trained ternary organ (s2o_train.py's organ.pt) into Ling-mini's GGUF, quantized
as P0 was (Q4_K_M, fit24 imatrix) with the organ's tensors as TQ2_0.  One scale per row: every 256-weight block of a
row has the row's scale as its maximum (or is all zero), so TQ2_0 holds the rows exactly, as IQ2_BN does.

    --organ O --trained DIR [--shuffle-rows T]: copy the bf16 GGUF, check that each tensor to replace holds the HF
        checkpoint's original bytes (the layout), overwrite it at its offset (the head also writes output_norm.weight =
        g / c), quantize, delete the bf16 copy.  --shuffle-rows T permutes the rows of tensor T (seed 20261103): the
        planted control.
Checks written to the record: every replaced tensor is TQ2_0 in the output and dequantizes to the trained values bit
for bit (the shuffled ones to the shuffled values); the head's gain equals P0's over c; every other tensor equals P0's
byte for byte.

Usage: python -I benchmarks/stage2/s2o_export.py --organ attn --trained DIR --out-q4 F --json F
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
from s2h_head import P0, run  # noqa: E402
from s2h_head import BIN  # noqa: E402
from s2o_train import ORGANS  # noqa: E402

SEED = 20261103
GGUF_NAME = {
    "attention.query_key_value.weight": "attn_qkv", "attention.dense.weight": "attn_output",
    **{f"mlp.shared_experts.{p}_proj.weight": f"ffn_{p}_shexp" for p in ("gate", "up", "down")},
    **{f"mlp.experts.{p}_proj": f"ffn_{p}_exps" for p in ("gate", "up", "down")},
    **{f"mlp.{p}_proj.weight": f"ffn_{p}" for p in ("gate", "up", "down")},
}


def gguf_name(layer: int, hf: str) -> str:
    return "output.weight" if hf == "lm_head.weight" else f"blk.{layer}.{GGUF_NAME[hf]}.weight"


def original(layer: int, hf: str) -> torch.Tensor:
    W = M.Weights()
    return W.get(["lm_head.weight"])["lm_head.weight"] if hf == "lm_head.weight" else W.layer(layer)[hf]


def export(organ: str, trained: Path, out_q4: Path, shuffle: str) -> dict:
    t0 = time.time()
    layer, names = ORGANS[organ]
    st = torch.load(trained / "organ.pt", map_location="cpu", weights_only=True)
    new = {n: st[n] for n in names}
    if shuffle:
        perm = torch.from_numpy(np.random.default_rng(SEED).permutation(new[shuffle].shape[-2]))
        new[shuffle] = new[shuffle][..., perm, :]
    tt = tensors()
    bf = out_q4.with_name(out_q4.stem + "-bf16.gguf")
    shutil.copyfile(BF16, bf)
    layout = {}
    with bf.open("r+b") as f:
        for n, v in new.items():
            off, nb, kind = tt[gguf_name(layer, n)]
            f.seek(off)
            layout[n] = f.read(nb) == as_bytes(original(layer, n), kind)
            b = as_bytes(v, kind)
            assert len(b) == nb, n
            f.seek(off)
            f.write(b)
        if organ == "head":
            off, nb, kind = tt["output_norm.weight"]
            f.seek(off)
            f.write(as_bytes(st["output_norm.weight"], kind))
    if not all(layout.values()):
        raise SystemExit(f"layout differs from the HF checkpoint: {layout}")
    cmd = [str(BIN / "llama-quantize.exe"), "--imatrix", IMATRIX.as_posix()]
    for n in names:
        g = gguf_name(layer, n)
        cmd += (["--output-tensor-type", "tq2_0"] if g == "output.weight"
                else ["--tensor-type", g.replace(".", r"\.").removesuffix(r"\.weight") + "=tq2_0"])
    # Amendment 1: a manual type skips llama-quantize's per-layer counters (i_attention_wv, i_ffn_down, ...), which
    # shifts Q4_K_M's mixture on the later layers of the same class (org_attn's first export: blk.11/13/14/16
    # attn_qkv).  Every other quantized tensor is pinned to P0's type by exact name, so none goes through the mixture.
    ref = {t.name: t for t in gguf.GGUFReader(str(P0)).tensors}
    changed = {gguf_name(layer, n) for n in names} | ({"output_norm.weight"} if organ == "head" else set())
    pins = out_q4.with_name(out_q4.stem + "-pins.txt")
    pins.write_text("".join(f"^{n.replace('.', chr(92) + '.')}$={t.tensor_type.name.lower()}\n" for n, t in ref.items()
                            if n not in changed and t.tensor_type.name not in ("F32", "F16", "BF16")), encoding="ascii")
    cmd += ["--tensor-type-file", pins.as_posix(), bf.as_posix(), out_q4.as_posix(), "Q4_K_M", "6"]
    run(cmd, out_q4.with_name(out_q4.name + ".quantize.log"))
    bf.unlink()
    got = {t.name: t for t in gguf.GGUFReader(str(out_q4)).tensors}
    exact = {}
    for n, v in new.items():
        t = got[gguf_name(layer, n)]
        exact[n] = t.tensor_type.name == "TQ2_0" and bool(np.array_equal(
            gguf.quants.dequantize(t.data, t.tensor_type).reshape(v.shape), v.float().numpy()))
    rec = {"organ": organ, "layer": layer, "trained": str(trained), "shuffled": shuffle or None, "seed": SEED,
           "layout_checked": layout, "tq2_0_exact": exact, "all_exact": all(exact.values())}
    if organ == "head":
        fg = np.frombuffer(np.asarray(got["output_norm.weight"].data).tobytes(), "<f4").astype(np.float64)
        pg = np.frombuffer(np.asarray(ref["output_norm.weight"].data).tobytes(), "<f4").astype(np.float64)
        rec["gain_ratio_err"] = float(np.abs(pg / fg / st["column_scale"].double().numpy() - 1).max())
    rec["pinned_tensors"] = len(pins.read_text(encoding="ascii").splitlines())
    others = [n for n in ref if n not in changed]
    rec["other_tensors"] = len(others)
    rec["other_tensors_differ"] = [n for n in others if got[n].tensor_type != ref[n].tensor_type
                                   or np.asarray(got[n].data).tobytes() != np.asarray(ref[n].data).tobytes()]
    rec.update(q4=str(out_q4), q4_sha256=sha256(out_q4), p0_sha256_checked=sha256(P0) == P0_SHA, cmd=cmd,
               seconds=round(time.time() - t0, 1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", required=True, choices=sorted(ORGANS))
    ap.add_argument("--trained", type=Path, required=True)
    ap.add_argument("--shuffle-rows", default="")
    ap.add_argument("--out-q4", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    rec = export(a.organ, a.trained, a.out_q4, a.shuffle_rows)
    print(json.dumps({k: v for k, v in rec.items() if k != "cmd"}))
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
