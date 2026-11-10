"""S2N's export: a trained arm's multipliers into a copy of ks_stk4.gguf, patched in place, then read back.

    organs   BF16 rows: bf16(value x multiplier), rounded to nearest even as torch does (the row's multiplier; glob: its class's scalar);
    experts  Q4_0: every block's d (fp16) x its multiplier (row: the expert row's; grp: its 256-group's; glob: the class's), nibbles untouched;
             bal's zero blocks stay zero, exception blocks scale with their row as in the student;
    else     every other byte is ks_stk4's.
The read-back recomputes every patched tensor from ks_stk4 and the scales and compares bytes; every other tensor and the header compared byte for byte.
The selftest exports random multipliers, plants one changed byte in an expert's d and in an untouched tensor, and needs the read-back to catch both.

Usage: python -I benchmarks/stage2/s2n_export.py --arm e2e_row --scales DIR/scales.pt --out FILE.gguf --json benchmarks/stage2/results/s2n_export_e2e_row.json
       python -I benchmarks/stage2/s2n_export.py --selftest --out E:/SiliconLLM_Artifacts/stage2/s2n/q4/selftest.gguf
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2n_train as T  # noqa: E402
from s2n_train import EXPERTS, GLOB, ORGANS, gguf  # noqa: E402


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def to_bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def from_bf16(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def multiplier(scales: dict, kind: str, i: int, n: str) -> np.ndarray | None:
    if kind == "none":
        return None
    if kind == "glob":
        return np.float32(scales[f"glob.{GLOB[n]}"].item())
    return scales[f"{i}.{n}"].float().numpy()


def patched(t, s, src: np.ndarray) -> np.ndarray:
    """The expected bytes of tensor t after its multiplier s, from ks_stk4's bytes src."""
    n = t.name.split(".")[2]
    if n in ORGANS:
        r, c = int(t.shape[1]), int(t.shape[0])
        a = src.view(np.uint16).reshape(r, c)
        m = s if np.ndim(s) == 0 else s.reshape(r, 1)
        return to_bf16(from_bf16(a) * m).view(np.uint8).reshape(-1)
    e, r, i = int(t.shape[2]), int(t.shape[1]), int(t.shape[0])
    b = src.reshape(-1, 18).copy()
    d = b[:, :2].copy().view(np.float16).reshape(e, r, i // 32).astype(np.float32)
    if np.ndim(s) == 0:
        m = s
    elif s.ndim == 2:
        m = s[..., None]
    else:
        m = np.repeat(s, 8, axis=-1)
    nd = (d * m).astype(np.float16)
    if not np.isfinite(nd).all():
        raise SystemExit(f"{t.name}: a d overflows fp16")
    b[:, :2] = nd.reshape(-1, 1).view(np.uint8).reshape(-1, 2)
    return b.reshape(-1)


def targets(rd, scales: dict, arm: str):
    kind_org, kind_exp = T.ARMS[arm]
    for t in rd.tensors:
        p = t.name.split(".")
        if len(p) != 4 or p[0] != "blk" or p[2] not in ORGANS + EXPERTS:
            continue
        kind = kind_org if p[2] in ORGANS else kind_exp
        s = multiplier(scales, kind, int(p[1]), p[2])
        if s is not None:
            yield t, s


def export(arm: str, scales: dict, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(T.KS, out)
    rd = gguf.GGUFReader(str(T.KS))
    src = np.memmap(T.KS, mode="r")
    mm = np.memmap(out, mode="r+")
    n = 0
    for t, s in targets(rd, scales, arm):
        o = int(t.data_offset)
        mm[o:o + int(t.n_bytes)] = patched(t, s, np.asarray(src[o:o + int(t.n_bytes)]))
        n += 1
    mm.flush()
    del mm
    return n


def verify(arm: str, scales: dict, out: Path) -> dict:
    rd = gguf.GGUFReader(str(T.KS))
    src = np.memmap(T.KS, mode="r")
    dst = np.memmap(out, mode="r")
    bad = []
    if src.shape != dst.shape:
        return {"ok": False, "bad": ["file size"]}
    want = {t.name: s for t, s in targets(rd, scales, arm)}
    first = min(int(t.data_offset) for t in rd.tensors)
    if not np.array_equal(src[:first], dst[:first]):
        bad.append("header")
    for t in rd.tensors:
        o, nb = int(t.data_offset), int(t.n_bytes)
        exp = patched(t, want[t.name], np.asarray(src[o:o + nb])) if t.name in want else src[o:o + nb]
        if not np.array_equal(exp, dst[o:o + nb]):
            bad.append(t.name)
    return {"ok": not bad, "bad": bad[:20], "patched": len(want), "tensors": len(rd.tensors)}


def selftest(out: Path) -> int:
    torch.manual_seed(0)
    rd = gguf.GGUFReader(str(T.KS))
    scales = {}
    for t in rd.tensors:
        p = t.name.split(".")
        if len(p) == 4 and p[0] == "blk" and p[2] in ORGANS + EXPERTS:
            shp = (int(t.shape[1]),) if p[2] in ORGANS else (int(t.shape[2]), int(t.shape[1]), int(t.shape[0]) // 256)
            scales[f"{p[1]}.{p[2]}"] = 1 + 0.1 * torch.randn(*shp)
    ok = {}
    n = export("e2e_grp", scales, out)
    v = verify("e2e_grp", scales, out)
    ok["exported and read back"] = v["ok"] and n == v["patched"] and n > 0
    ok["the exported file is not read back as ks_stk4"] = not verify("none", scales, out)["ok"]
    t = next(x for x in rd.tensors if x.name.endswith("ffn_up_exps.weight"))
    u = next(x for x in rd.tensors if x.name == "output_norm.weight")
    for name, off in (("planted d", int(t.data_offset) + 18 * 1000), ("planted untouched", int(u.data_offset) + 1)):
        mm = np.memmap(out, mode="r+")
        old = int(mm[off])
        mm[off] = old ^ 0x40
        mm.flush()
        del mm
        ok[f"{name} caught"] = not verify("e2e_grp", scales, out)["ok"]
        mm = np.memmap(out, mode="r+")
        mm[off] = old
        mm.flush()
        del mm
    ok["restored reads back"] = verify("e2e_grp", scales, out)["ok"]
    for k, x in ok.items():
        print("selftest", "ok  " if x else "FAIL", k)
    print("selftest", "PASS" if all(ok.values()) else "FAIL", len(ok))
    out.unlink()
    return 0 if all(ok.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", choices=sorted(T.ARMS))
    ap.add_argument("--scales", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if sha256(T.KS) != T.KS_SHA:
        raise SystemExit("ks_stk4 differs from its pin")
    if a.arm in ("damage", "damage_ref"):
        raise SystemExit("damage starts from a changed stack: a dry run only, never exported")
    if a.selftest:
        return selftest(a.out)
    scales = torch.load(a.scales, map_location="cpu")
    n = export(a.arm, scales, a.out)
    v = verify(a.arm, scales, a.out)
    rec = {"arm": a.arm, "scales": str(a.scales), "scales_sha256": sha256(a.scales), "out": str(a.out), "out_sha256": sha256(a.out),
           "source_sha256": T.KS_SHA, "exported": n, **v}
    if a.json:
        a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("EXPORT", "OK" if v["ok"] else "VOID", json.dumps(rec)[:600], flush=True)
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
