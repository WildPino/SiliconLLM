"""S2L's exports.  Every arm is P0 (Q4_K_M, fit24 imatrix, every other tensor pinned to its type) with some tensors replaced.  After writing, each replaced tensor must read
back to the intended values, and every other tensor must equal the reference file's bytes.

    as_both     st_both's organs in asym, as Q4_0 with bytes written over (d = s * sign per row, q in -1..2): quality only, the panels read the trained values.
    st_both2    st_both's second seed, TQ2_0, as test 2c's export.
    stk4        bal-q4's experts and router, st_attn's attention as TQ2_0, everything else as P0.  Reference: bal-q4.
    ff_nodense, ff_noearly, ff_nolate   st_ffn with a group of tensors back as stored: layer 0; shared experts 1-9; shared experts 10-19.
    ff_keepout  st_ffn with its outlier groups (experts' rule: group 256, max over 8 times the TWN scale) back to the original values, as BF16.
    ff_keeprnd  the same, the same number of groups per tensor, drawn at random: outliers' trivial control.
    ks_org      attention, shared expert and dense FFN at ik_llama.cpp's IQ2_KS values with the imatrix, held in BF16 (llama.cpp does not read IQ2_KS), no training.

Usage: python -I benchmarks/stage2/s2y_export.py --arm A --out-q4 F --json F [--ks-bf16 F]
       python -I benchmarks/stage2/s2y_export.py --outliers F     counts the outlier groups of every organ; reads st_ffn and the checkpoint
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
from s1t_export import BF16, HF, IMATRIX, as_bytes, bal_blocks, gguf, sha256, tensors  # noqa: E402
from s1t_train import OUTLIER, twn_scale  # noqa: E402
from s2h_head import BIN, P0, run  # noqa: E402
from s2o_export import gguf_name  # noqa: E402
from s2s_train import ATTN, organ_names  # noqa: E402

S2S = Path("C:/SiliconLLM_fast/s2s")
OUT = Path("E:/SiliconLLM_Artifacts/stage2")
Y = OUT / "s2y"
BAL_Q4, BAL_BF16 = OUT / "s2v" / "bal-q4.gguf", OUT / "s2v" / "bal-bf16.gguf"
GROUP, SEED = 256, 20261107


def trained(d: Path, layers: list[int] | None = None) -> dict[tuple[int, str], dict]:
    out = {}
    for p in sorted(d.glob("layer_*.pt"), key=lambda x: int(x.stem.split("_")[1])):
        i = int(p.stem.split("_")[1])
        if layers is None or i in layers:
            st = torch.load(p, map_location="cpu", weights_only=True)
            for n, v in st.items():
                if ":" not in n:
                    out[(i, n)] = {"w": v, "s": st.get("ternary_scale:" + n), "g": st.get("sign:" + n)}
    return out


def asym_blocks(e: dict) -> tuple[np.ndarray, int]:
    w, s, g = e["w"].float(), e["s"].float().reshape(-1, 1), e["g"].float().reshape(-1, 1)
    d = s * g
    q = (w / d).round()
    if q.min() < -8 or q.max() > 7 or not torch.equal((q * d).to(torch.bfloat16), e["w"]):
        raise SystemExit("asym: the stored weights are not s * sign * level")
    return bal_blocks(d.expand(-1, w.shape[1] // 32).contiguous(), q)


def compose(new: dict[str, torch.Tensor], kind: str, out_q4: Path, base: Path = BF16, ref: Path = P0, over: dict[str, bytes] | None = None) -> dict:
    """new: GGUF name -> values; kind: their type in the file; over: bytes written on the output afterwards."""
    t0 = time.time()
    tt = tensors()
    bf = out_q4.with_name(out_q4.stem + "-bf16.gguf")
    shutil.copyfile(base, bf)
    with bf.open("r+b") as f:
        for g, v in new.items():
            off, nb, k = tt[g]
            b = as_bytes(v, k)
            assert len(b) == nb, g
            f.seek(off)
            f.write(b)
    r0 = {t.name: t for t in gguf.GGUFReader(str(ref)).tensors}
    cmd = [str(BIN / "llama-quantize.exe"), "--imatrix", IMATRIX.as_posix()]
    for g in sorted(new):
        cmd += ["--tensor-type", g.replace(".", r"\.").removesuffix(r"\.weight") + "=" + kind]
    pins = out_q4.with_name(out_q4.stem + "-pins.txt")
    pins.write_text("".join(f"^{n.replace('.', chr(92) + '.')}$={t.tensor_type.name.lower()}\n" for n, t in r0.items()
                            if n not in new and t.tensor_type.name not in ("F32", "F16", "BF16")), encoding="ascii")
    cmd += ["--tensor-type-file", pins.as_posix(), bf.as_posix(), out_q4.as_posix(), "Q4_K_M", "6"]
    run(cmd, out_q4.with_name(out_q4.name + ".quantize.log"))
    bf.unlink()
    if over:
        offs = {t.name: (int(t.data_offset), int(t.n_bytes)) for t in gguf.GGUFReader(str(out_q4)).tensors}
        with out_q4.open("r+b") as f:
            for g, b in over.items():
                assert len(b) == offs[g][1], g
                f.seek(offs[g][0])
                f.write(b)
    got = {t.name: t for t in gguf.GGUFReader(str(out_q4)).tensors}
    exact = {}
    for g, v in new.items():
        t = got[g]
        ok = t.tensor_type.name == kind.upper()
        if ok:
            ok = bool(np.array_equal(gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1), v.float().numpy().reshape(-1)))
        exact[g] = ok
    others = [n for n in r0 if n not in new]
    return {"changed": len(new), "kind": kind, "exact": exact, "all_exact": all(exact.values()), "other_tensors": len(others), "reference": str(ref),
            "other_tensors_differ": [n for n in others if got[n].tensor_type != r0[n].tensor_type or np.asarray(got[n].data).tobytes() != np.asarray(r0[n].data).tobytes()],
            "q4": str(out_q4), "q4_sha256": sha256(out_q4), "cmd": cmd, "seconds": round(time.time() - t0, 1)}


def outliers(w: torch.Tensor) -> torch.Tensor:
    """The experts' rule on a [O, I] tensor: its groups of 256 whose maximum is over OUTLIER times the TWN scale."""
    s = twn_scale(w, GROUP)
    return w.float().abs().view(*w.shape[:-1], -1, GROUP).amax(-1) > OUTLIER * s


def mixed(random: bool) -> tuple[dict[str, torch.Tensor], dict]:
    """st_ffn's tensors with some groups back to the original values."""
    W, gen = M.Weights(), torch.Generator().manual_seed(SEED)
    new, count = {}, {}
    for (i, n), e in trained(S2S / "st_ffn").items():
        hf = f"model.layers.{i}.{n}"
        o = W.get([hf])[hf]
        m = outliers(o)
        c = int(m.sum())
        if random:
            flat = torch.zeros(m.numel(), dtype=torch.bool)
            flat[torch.randperm(m.numel(), generator=gen)[:c]] = True
            m = flat.view_as(m)
        v = e["w"].view(*m.shape, GROUP).clone()
        v[m] = o.view(*m.shape, GROUP)[m]
        new[gguf_name(i, n)] = v.view_as(o)
        count[gguf_name(i, n)] = [c, m.numel()]
    return new, count


def outlier_counts(path: Path) -> int:
    W = M.Weights()
    out = {}
    for i in range(M.N_LAYER):
        for n in organ_names("both", i):
            hf = f"model.layers.{i}.{n}"
            o = W.get([hf])[hf]
            m = outliers(o)
            out[gguf_name(i, n)] = [int(m.sum()), m.numel(), float(o.float().abs().max())]
    kinds = {"attention": ("attn_",), "dense": ("blk.0.ffn",), "shared": ("shexp",)}
    tot = {k: [sum(v[0] for g, v in out.items() if any(x in g for x in p)), sum(v[1] for g, v in out.items() if any(x in g for x in p))] for k, p in kinds.items()}
    path.write_text(json.dumps({"total": tot, "tensors": out}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(tot), sorted(((v[0], g) for g, v in out.items()), reverse=True)[:8])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--out-q4", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--ks-bf16", type=Path)
    ap.add_argument("--outliers", type=Path)
    a = ap.parse_args()
    if a.outliers:
        return outlier_counts(a.outliers)
    a.out_q4.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if a.arm == "as_both":
        t = trained(Y / "as_both")
        new = {gguf_name(i, n): e["w"] for (i, n), e in t.items()}
        blocks = {gguf_name(i, n): asym_blocks(e) for (i, n), e in t.items()}
        rec = compose(new, "q4_0", a.out_q4, over={g: b[0].tobytes() for g, b in blocks.items()})
        extra = {"blocks_under_fp16_normal": sum(b[1] for b in blocks.values())}
    elif a.arm == "st_both2":
        rec = compose({gguf_name(i, n): e["w"] for (i, n), e in trained(Y / "st_both2").items()}, "tq2_0", a.out_q4)
    elif a.arm == "stk4":
        r = {t.name: t for t in gguf.GGUFReader(str(BAL_Q4)).tensors}
        over = {n: np.asarray(t.data).tobytes() for n, t in r.items() if "_exps" in n and t.tensor_type.name == "Q4_0"}
        rec = compose({gguf_name(i, n): e["w"] for (i, n), e in trained(S2S / "st_attn").items()}, "tq2_0", a.out_q4, base=BAL_BF16, ref=BAL_Q4, over=over)
        extra = {"experts_copied": len(over)}
    elif a.arm in ("ff_nodense", "ff_noearly", "ff_nolate"):
        keep = {"ff_nodense": list(range(1, 20)), "ff_noearly": [0] + list(range(10, 20)), "ff_nolate": list(range(0, 10))}[a.arm]
        rec = compose({gguf_name(i, n): e["w"] for (i, n), e in trained(S2S / "st_ffn", keep).items()}, "tq2_0", a.out_q4)
    elif a.arm in ("ff_keepout", "ff_keeprnd"):
        new, count = mixed(a.arm == "ff_keeprnd")
        rec = compose(new, "bf16", a.out_q4)
        extra = {"groups": count}
    elif a.arm == "ks_org":
        r = {t.name: t for t in gguf.GGUFReader(str(a.ks_bf16)).tensors}
        names = [gguf_name(i, n) for i in range(M.N_LAYER) for n in organ_names("both", i)]
        new = {g: torch.from_numpy(np.asarray(r[g].data).reshape(-1).view(np.int16).copy()).view(torch.bfloat16) for g in names}
        W, err = M.Weights(), {}
        for i in range(M.N_LAYER):
            for n in organ_names("both", i):
                hf, g = f"model.layers.{i}.{n}", gguf_name(i, n)
                o = W.get([hf])[hf].float().reshape(-1)
                err[g] = float((new[g].float() - o).pow(2).sum() / o.pow(2).sum())
        rec = compose(new, "bf16", a.out_q4)
        extra = {"rel_weight_error": err}
    else:
        raise SystemExit(f"unknown arm {a.arm}")
    rec = {"arm": a.arm, **rec, **extra}
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cmd", "exact", "groups", "rel_weight_error")}))
    return 0 if rec["all_exact"] and not rec["other_tensors_differ"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
