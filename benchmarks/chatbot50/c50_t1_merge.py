"""C50-T1 MVE-0: merge an S4 LoRA into T0's BF16 GGUF in place of the covered tensors, and check a Q4_K_M's layout.

The patched file is a byte copy of the BF16 GGUF in which only the LoRA's tensors are overwritten with
bf16(W + B A), W read as fp32 and rounded to nearest-even; nothing else is written. `--zero` writes the same bytes
back unchanged (K1). `layout` compares a Q4_K_M with P0 tensor by tensor (L).
Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_T1_PROTOCOL_20261024.md

    python -I benchmarks/chatbot50/c50_t1_merge.py merge --bf16 IN.gguf --lora final.pt --out OUT.gguf --json REC [--zero]
    python -I benchmarks/chatbot50/c50_t1_merge.py quantize --bf16 IN.gguf --out OUT.gguf --json REC [--build avx2|generic]
    python -I benchmarks/chatbot50/c50_t1_merge.py layout --ref P0.gguf --new T1.gguf --lora final.pt --json REC
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
sys.path.insert(0, str(C / "tools/llama.cpp-5b335f4/gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf import GGUFReader  # noqa: E402

IMATRIX = C / "gguf/ling-mini-2.0-a810f64-imatrix-fit24.gguf"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def deltas(lora: Path) -> dict[str, torch.Tensor]:
    """GGUF tensor name -> B A in fp32, [n_out, n_in]."""
    sd = torch.load(lora, map_location="cpu", weights_only=True)
    out = {}
    for k in sd:
        if k.endswith(".A"):
            i, m = k[len("lora."):-len(".A")].split("_", 1)
            out[f"blk.{i}.{m}.weight"] = sd[k[:-2] + ".B"].double() @ sd[k].double()
    assert len(out) * 2 == len(sd), "every key is a LoRA factor"
    return {n: d.float() for n, d in out.items()}


def bf16_to_f32(u: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32))


def f32_to_bf16(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def cmd_merge(a) -> int:
    d = deltas(a.lora)
    r = GGUFReader(str(a.bf16))
    tens = {t.name: t for t in r.tensors}
    assert set(d) <= set(tens), sorted(set(d) - set(tens))
    shutil.copyfile(a.bf16, a.out)
    rec = {"cell": "C50-T1 MVE-0 merge", "bf16": str(a.bf16), "lora": str(a.lora), "lora_sha256": sha256(a.lora),
           "zero": a.zero, "tensors": {}}
    print("CONFIG", json.dumps({k: rec[k] for k in ("bf16", "lora", "lora_sha256", "zero")}), flush=True)
    with open(a.out, "r+b") as f:
        for name, dw in sorted(d.items()):
            t = tens[name]
            assert t.tensor_type.name == "BF16", (name, t.tensor_type.name)
            u = np.array(t.data).view(np.uint16).reshape(dw.shape)       # [n_out, n_in], ne0 = n_in fastest
            w = bf16_to_f32(u)
            if a.zero:
                dw = torch.zeros_like(dw)
            new = f32_to_bf16(w + dw)
            got = bf16_to_f32(new) - w
            f.seek(t.data_offset)
            f.write(new.tobytes())
            n = float(dw.norm())
            rec["tensors"][name] = {"shape": list(dw.shape), "delta_norm": n, "w_norm": float(w.norm()),
                                    "kept": float(got.norm()) / n if n else None,
                                    "cos": float((got * dw).sum() / (got.norm() * n)) if n and float(got.norm()) else None,
                                    "changed": int((new != u).sum())}
    rec["out_sha256"] = sha256(a.out)
    if a.zero:
        rec["bf16_sha256"] = sha256(a.bf16)
        rec["K1"] = rec["out_sha256"] == rec["bf16_sha256"]
        print("K1", "PASS" if rec["K1"] else "FAIL", flush=True)
    else:
        k = [v["kept"] for v in rec["tensors"].values()]
        c = [v["cos"] for v in rec["tensors"].values()]
        rec["kept_median"], rec["cos_median"] = float(np.median(k)), float(np.median(c))
        print("MERGED", len(d), "tensors; kept median", round(rec["kept_median"], 4), "cos median",
              round(rec["cos_median"], 4), flush=True)
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0 if (not a.zero or rec["K1"]) else 1


def cmd_quantize(a) -> int:
    """P0's recipe (A2): llama-quantize --imatrix <imatrix-fit24> <bf16> <out> Q4_K_M 6, at the pinned commit."""
    import subprocess                                                   # noqa: PLC0415
    exe = C / f"tools/llama.cpp-5b335f4/build-{a.build}/bin/llama-quantize.exe"
    cmd = [str(exe), "--imatrix", str(IMATRIX), str(a.bf16), str(a.out), "Q4_K_M", "6"]
    print("CONFIG", json.dumps({"cmd": cmd}), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    Path(str(a.json) + ".log").write_text(r.stdout + r.stderr, encoding="utf-8")
    rec = {"cell": "C50-T1 MVE-0 quantize", "cmd": cmd, "exe_sha256": sha256(exe), "imatrix_sha256": sha256(IMATRIX),
           "bf16": str(a.bf16), "returncode": r.returncode,
           "sha256": sha256(a.out) if r.returncode == 0 else None, "size": a.out.stat().st_size if a.out.exists() else None}
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("QUANTIZED", r.returncode, rec["sha256"], flush=True)
    return r.returncode


def cmd_layout(a) -> int:
    """L: the same tensors, types, shapes and byte counts as P0; only the merged tensors' data may differ. Also the
    share of ΔW that survives Q4_K_M on the merged tensors, against P0's dequantized tensor."""
    from gguf.quants import dequantize                                  # noqa: PLC0415
    d = deltas(a.lora)
    ref, new = GGUFReader(str(a.ref)), GGUFReader(str(a.new))
    rt, nt = {t.name: t for t in ref.tensors}, {t.name: t for t in new.tensors}
    rec = {"cell": "C50-T1 MVE-0 layout", "ref": str(a.ref), "new": str(a.new), "n_ref": len(rt), "n_new": len(nt),
           "size_ref": a.ref.stat().st_size, "size_new": a.new.stat().st_size, "problems": [], "differs": [],
           "quant_kept": {}}
    if list(rt) != list(nt):
        rec["problems"].append("tensor names or order differ")
    # The header (metadata and tensor infos) is compared as bytes, and tensor offsets relative to the data section,
    # so that a metadata change and a data change are told apart (L's planted M0 control, 24 October).
    r0, n0 = ref.tensors[0].data_offset, new.tensors[0].data_offset
    with a.ref.open("rb") as f, a.new.open("rb") as g:
        rec["header_equal"] = r0 == n0 and f.read(r0) == g.read(n0)
    if not rec["header_equal"]:
        rec["problems"].append("headers differ (metadata or tensor infos)")
    for name, t in rt.items():
        u = nt.get(name)
        if u is None:
            continue
        if (t.tensor_type != u.tensor_type or list(t.shape) != list(u.shape) or t.n_bytes != u.n_bytes
                or t.data_offset - r0 != u.data_offset - n0):
            rec["problems"].append(f"{name}: type, shape, size or offset differs")
            continue
        if not np.array_equal(np.asarray(t.data), np.asarray(u.data)):
            rec["differs"].append(name)
            if name in d:
                got = torch.from_numpy(dequantize(u.data, u.tensor_type)
                                       - dequantize(t.data, t.tensor_type)).float().reshape(d[name].shape)
                dw = d[name]
                rec["quant_kept"][name] = {"kept": float(got.norm() / dw.norm()),
                                           "cos": float((got * dw).sum() / (got.norm() * dw.norm()))}
    extra = sorted(set(rec["differs"]) - set(d))
    if extra:
        rec["problems"].append(f"tensors outside the LoRA differ: {extra[:5]}")
    if rec["size_ref"] != rec["size_new"]:
        rec["problems"].append("file sizes differ")
    rec["L"] = not rec["problems"]
    if rec["quant_kept"]:
        rec["quant_kept_median"] = float(np.median([v["kept"] for v in rec["quant_kept"].values()]))
        rec["quant_cos_median"] = float(np.median([v["cos"] for v in rec["quant_kept"].values()]))
    print("L", "PASS" if rec["L"] else "FAIL", "differs", len(rec["differs"]), "of", len(rt), rec["problems"][:3],
          {k: rec.get(k) for k in ("quant_kept_median", "quant_cos_median")}, flush=True)
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0 if rec["L"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--bf16", type=Path, required=True)
    m.add_argument("--lora", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--json", required=True)
    m.add_argument("--zero", action="store_true", help="K1: write W back unchanged")
    m.set_defaults(f=cmd_merge)
    q = sub.add_parser("quantize")
    q.add_argument("--bf16", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--json", required=True)
    q.add_argument("--build", default="avx2", choices=["avx2", "generic"])
    q.set_defaults(f=cmd_quantize)
    lay = sub.add_parser("layout")
    lay.add_argument("--ref", type=Path, required=True)
    lay.add_argument("--new", type=Path, required=True)
    lay.add_argument("--lora", type=Path, required=True)
    lay.add_argument("--json", required=True)
    lay.set_defaults(f=cmd_layout)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    raise SystemExit(main())
