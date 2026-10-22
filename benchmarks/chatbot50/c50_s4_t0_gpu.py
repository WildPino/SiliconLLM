"""C50-S4: T0 on the GPUs from its quantized blocks, and MVE-2 (S4 design note, feasibility checks).

`T0Quant` keeps T0's raw GGUF blocks on the devices (layers 0-9 and the embedding on the first, layers 10-19 and the
output head on the second) and dequantizes a tensor when the forward asks for it (`c50_s4_dequant.py`, bit-identical to
the pinned gguf-py), in fp16. It is a drop-in for `T0Weights` in `c50_s4_t0_torch.py`, whose forward is unchanged.

    python -I benchmarks/chatbot50/c50_s4_t0_gpu.py --fingerprints OUT.json   # local: gguf-py's fp32 output per tensor
    python -I benchmarks/chatbot50/c50_s4_t0_gpu.py --mve2 --gguf G --refs DIR --fingerprints F.json --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_t0_torch as t0  # noqa: E402
from c50_s4_dequant import DEQUANT, QK_K  # noqa: E402

gguf = t0.gguf
T0_SHA256 = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
SPLIT = 10            # layers [0, SPLIT) on the first device
CHUNK = 1 << 25       # values per dequantization chunk (peak transient ~0.5 GB in fp32)


class T0Quant:
    def __init__(self, path: Path, devs: tuple[str, str], wdtype: torch.dtype = torch.float16,
                 keep: tuple[str, ...] = ("output.weight",)):
        self.devs = tuple(torch.device(d) for d in devs)
        self.wdtype = wdtype
        self.raw: dict[str, torch.Tensor] = {}
        self.qt: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.kept: dict[str, torch.Tensor] = {}
        r = gguf.GGUFReader(str(path))
        for t in r.tensors:
            dev = self._place(t.name)
            if t.tensor_type.name in DEQUANT:
                nbytes = DEQUANT[t.tensor_type.name][1]
                shape = tuple(t.data.shape[:-1]) + (t.data.shape[-1] // nbytes * QK_K,)
                self.raw[t.name] = torch.from_numpy(np.array(t.data)).to(dev)
                self.qt[t.name] = (t.tensor_type.name, shape)
            else:
                assert t.tensor_type.name == "F32", t.name
                self.raw[t.name] = torch.from_numpy(np.array(t.data, dtype=np.float32)).to(dev)
        for name in keep:
            self.kept[name] = self.dequant(name, wdtype)

    def _place(self, name: str) -> torch.device:
        if name.startswith("blk."):
            return self.devs[0] if int(name.split(".")[1]) < SPLIT else self.devs[1]
        return self.devs[0] if name == "token_embd.weight" else self.devs[1]

    def dev(self, i: int) -> torch.device:
        return self.devs[0] if i < SPLIT else self.devs[1]

    def dequant(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        qtype, shape = self.qt[name]
        fn, nbytes = DEQUANT[qtype]
        blocks = self.raw[name].reshape(-1, nbytes)
        out = torch.empty((blocks.shape[0], QK_K), dtype=dtype, device=blocks.device)
        step = CHUNK // QK_K
        for a in range(0, blocks.shape[0], step):
            out[a:a + step] = fn(blocks[a:a + step])
        return out.reshape(shape)

    def get(self, name: str) -> torch.Tensor:
        if name in self.kept:
            return self.kept[name]
        if name in self.qt:
            return self.dequant(name, self.wdtype)
        return self.raw[name]

    def embed_rows(self, ids: torch.Tensor) -> torch.Tensor:
        qtype, shape = self.qt["token_embd.weight"]
        fn, nbytes = DEQUANT[qtype]
        rows = self.raw["token_embd.weight"][ids.to(self.devs[0])]
        return fn(rows.reshape(-1, nbytes)).reshape(len(ids), shape[-1])


def fp32_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


def fingerprints(gguf_path: Path, out: Path) -> None:
    """The pinned gguf-py's fp32 dequantization of every quantized T0 tensor, as sha256 of its bytes. The PyTorch
    port is checked against it on the CPU on the way (every tensor, not a sample)."""
    from c50_s4_dequant import dequant
    r = gguf.GGUFReader(str(gguf_path))
    fp, bad = {}, []
    for t in r.tensors:
        if t.tensor_type.name in DEQUANT:
            ref = gguf.quants.dequantize(t.data, t.tensor_type)
            fp[t.name] = fp32_sha(ref)
            mine = fp32_sha(dequant(torch.from_numpy(np.array(t.data)), t.tensor_type.name, ref.shape).numpy())
            if mine != fp[t.name]:
                bad.append(t.name)
            print(t.name, fp[t.name][:12], "torch-cpu", "same" if mine == fp[t.name] else "DIFFERENT", flush=True)
    print("tensors", len(fp), "torch-cpu mismatched", bad)
    assert not bad
    out.write_text(json.dumps({"gguf_sha256": T0_SHA256, "gguf_py": "llama.cpp-5b335f4", "tensors": fp}, indent=1) + "\n",
                   encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def mve2(gguf_path: Path, refs: Path, fp_path: Path, out: Path, n_docs: int, n_tok: int) -> dict:
    rec: dict = {"cell": "C50-S4 MVE-2", "torch": torch.__version__, "python": platform.python_version(),
                 "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                 "docs": n_docs, "tokens": n_tok, "split": SPLIT, "wdtype": "float16",
                 "fp16_reduced_precision_reduction": False, "verdict": "c50_s4_mve2_verdict.py"}
    print("CONFIG", json.dumps(rec), flush=True)
    assert rec["gpus"] == ["Tesla T4", "Tesla T4"], rec["gpus"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    t = time.time()
    rec["gguf_sha256"] = sha256_file(gguf_path)
    assert rec["gguf_sha256"] == T0_SHA256, rec["gguf_sha256"]
    fp = json.loads(fp_path.read_text(encoding="utf-8"))
    assert fp["gguf_sha256"] == T0_SHA256
    W = T0Quant(gguf_path, ("cuda:0", "cuda:1"))
    rec["load_s"] = round(time.time() - t, 1)
    rec["mem_gib_after_load"] = [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(2)]
    print("loaded", rec["load_s"], rec["mem_gib_after_load"], flush=True)

    # 1. Dequantization: every quantized tensor on its GPU, fp32, against gguf-py's fingerprint.
    bad, t = [], time.time()
    assert set(fp["tensors"]) == set(W.qt), "fingerprint set differs from the GGUF's quantized tensors"
    for name in sorted(W.qt):
        got = W.dequant(name, torch.float32).cpu().numpy()
        if fp32_sha(got) != fp["tensors"][name]:
            bad.append(name)
        del got
    rec["dequant"] = {"tensors": len(W.qt), "mismatched": bad, "seconds": round(time.time() - t, 1)}
    print("dequant", json.dumps(rec["dequant"]), flush=True)

    # Time of one MoE layer's dequantization to fp16 (what every training step pays per layer, forward and backward).
    for dev_i, lay in ((0, 5), (1, 15)):
        torch.cuda.synchronize(dev_i)
        t = time.time()
        for n in ("gate", "up", "down"):
            W.get(f"blk.{lay}.ffn_{n}_exps.weight")
        torch.cuda.synchronize(dev_i)
        rec[f"moe_layer_{lay}_dequant_fp16_s"] = round(time.time() - t, 4)

    # 2. Fidelity: the same positions as MVE-1, fp16 weights, against A4's llama.cpp scores.
    P0 = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"), ("logp_target", "<f4"),
                   ("logp_top1", "<f4"), ("lse", "<f8")])
    ref = {k: np.fromfile(refs / f"{k}.docs.pos.bin", dtype=P0) for k in ("p0", "bf16")}
    lines = (refs / "docs.tsv").read_text(encoding="ascii").splitlines()[:n_docs]
    acc = {k: [] for k in ("p0", "bf16")}
    rows, t = [], time.time()
    for line in lines:
        d, s = line.split("\t", 1)
        ids = torch.tensor([int(x) for x in s.split(",")][:n_tok])
        t1 = time.time()
        lp = torch.log_softmax(t0.logits(ids, W).double(), -1).cpu()
        assert torch.isfinite(lp).all(), "non-finite log-probabilities"
        for k, rf in ref.items():
            r = rf[(rf["doc"] == int(d)) & (rf["pos"] < len(ids))]
            assert np.array_equal(r["target"], ids[r["pos"]].numpy()), "targets differ"
            mine = lp[torch.from_numpy(r["pos"] - 1), torch.from_numpy(r["target"]).long()].numpy()
            top1 = lp[torch.from_numpy(r["pos"] - 1)].argmax(-1).numpy()
            dl = mine - r["logp_target"]
            acc[k].append((dl, top1 == r["top1"]))
            rows.append({"doc": int(d), "ref": k, "positions": int(len(r)), "mean_dlogp": float(dl.mean()),
                         "mean_abs_dlogp": float(np.abs(dl).mean()), "max_abs_dlogp": float(np.abs(dl).max()),
                         "top1_agree": float((top1 == r["top1"]).mean()), "forward_s": round(time.time() - t1, 2)})
            print(json.dumps(rows[-1]), flush=True)
    rec["fidelity_rows"] = rows
    rec["fidelity"] = {}
    for k, v in acc.items():
        dl, ag = np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])
        rec["fidelity"][k] = {"positions": int(len(dl)), "mean_dlogp": round(float(dl.mean()), 4),
                              "mean_abs_dlogp": round(float(np.abs(dl).mean()), 4), "top1_agree": round(float(ag.mean()), 4)}
    rec["fidelity_s"] = round(time.time() - t, 1)
    rec["mem_gib_peak"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(2)]
    print("fidelity", json.dumps(rec["fidelity"]), flush=True)
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints", type=Path)
    ap.add_argument("--mve2", action="store_true")
    ap.add_argument("--gguf", type=Path, default=t0.T0_GGUF)
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.mve2:
        mve2(a.gguf, a.refs, a.fingerprints, a.out, a.docs, a.tokens)
    elif a.fingerprints:
        fingerprints(a.gguf, a.fingerprints)
