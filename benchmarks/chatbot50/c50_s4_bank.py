"""C50-S4: arm (b)'s memory bank, from the training split (block 1, shards 0-469), on the GPUs (S4 protocol).

- **Occurrence:** an n-gram ending at token t of a training document, with t below the document's last token (so
  F0's top-32 at position t + 1 exists). A trigram row exists if it occurs at least twice, and so does a bigram row.
- **Value:** the mean over the row's occurrences of sum_k p_k W_out[k], with F0's top-32 probabilities at position
  t + 1 and T0's output embedding dequantized to fp32. Sums are taken in fp32 on the GPUs (`embedding_bag`), trigram
  rows on the first, bigram rows on the second; the mean is stored in fp16.
- **Outputs in --out:** `bank_values.f16` (rows x 2048: trigram rows, then bigram rows, each in key order),
  `bank_index.npz` (tri_keys, bi_keys: sorted int64, key = (a << 36) | (b << 18) | c and (a << 18) | b; tri_count,
  bi_count: int32), `bank.json` (CONFIG, sizes, coverage, hashes, the check).
- **Check (`--check N`):** N rows of each kind are recomputed on the CPU in float64 from their occurrences; the max abs
  difference against the stored fp16 value is recorded.

    python -I benchmarks/chatbot50/c50_s4_bank.py --gguf G --teacher DIR --corpus TSV --out DIR [--shards 470]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_t0_torch as t0  # noqa: E402
from c50_s4_dequant import DEQUANT, QK_K  # noqa: E402
from c50_s4_student import Teacher  # noqa: E402

H, MIN_COUNT, TRAIN_SHARDS, CHUNK = 2048, 2, 470, 8192
W_OUT_FP32_SHA256 = "ccaba272e010156024af67ed354b3ba287f73861b0a5d14acfd4d5d18321a562"  # MVE-2 fingerprint of T0


def tri_key(a, b, c):
    return (a << 36) | (b << 18) | c


def bi_key(a, b):
    return (a << 18) | b


def output_embedding(gguf_path: Path, dev: str) -> torch.Tensor:
    """T0's `output.weight`, dequantized to fp32 on `dev` (the same port as MVE-2's, bit-identical to gguf-py)."""
    r = t0.gguf.GGUFReader(str(gguf_path))
    t = next(x for x in r.tensors if x.name == "output.weight")
    fn, nbytes = DEQUANT[t.tensor_type.name]
    blocks = torch.from_numpy(np.array(t.data)).to(dev).reshape(-1, nbytes)
    out = torch.empty((blocks.shape[0], QK_K), dtype=torch.float32, device=dev)
    step = (1 << 25) // QK_K
    for a in range(0, blocks.shape[0], step):
        out[a:a + step] = fn(blocks[a:a + step])
    return out.reshape(-1, H)


def occurrences(T: Teacher):
    """Every training position t with a next position: its trigram and bigram keys (-1 where t is too early) and its
    F0 top-32 at t + 1, concatenated over documents in stored order."""
    tri, bi, ids32, lp32, n_tok = [], [], [], [], 0
    for _, ids, top_id, top_lp in T.docs():
        L = len(ids)
        n_tok += L
        t = np.arange(1, L - 1)                       # 1..L-2: a bigram exists and position t + 1 exists
        bi.append(bi_key(ids[t - 1], ids[t]))
        k3 = np.full(len(t), -1, dtype=np.int64)
        k3[1:] = tri_key(ids[t[1:] - 2], ids[t[1:] - 1], ids[t[1:]])
        tri.append(k3)
        ids32.append(top_id[t].astype(np.int32))      # record index t holds position t + 1
        lp32.append(top_lp[t])
    return (np.concatenate(tri), np.concatenate(bi), np.concatenate(ids32), np.concatenate(lp32), n_tok)


def rows_of(keys: np.ndarray):
    """Keys seen at least MIN_COUNT times, their counts, and for each occurrence its row (-1 if none)."""
    valid = keys >= 0
    u, c = np.unique(keys[valid], return_counts=True)
    keep = c >= MIN_COUNT
    u, c = u[keep], c[keep].astype(np.int32)
    row = np.full(len(keys), -1, dtype=np.int64)
    j = np.searchsorted(u, keys[valid])
    j = np.minimum(j, len(u) - 1)
    hit = u[j] == keys[valid]
    row[np.flatnonzero(valid)[hit]] = j[hit]
    return u, c, row


def build(gguf_path: Path, teacher: Path, corpus: Path, out: Path, shards: int, n_check: int,
          devs=("cuda:0", "cuda:1")) -> dict:
    rec = {"cell": "C50-S4 bank", "torch": torch.__version__, "shards": shards, "min_count": MIN_COUNT,
           "devs": list(devs), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    print("CONFIG", json.dumps(rec), flush=True)
    t_all = time.time()
    T = Teacher(teacher, corpus, list(range(shards)))
    tri, bi, ids32, lp32, n_tok = occurrences(T)
    del T
    tri_u, tri_c, tri_row = rows_of(tri)
    bi_u, bi_c, bi_row = rows_of(bi)
    n = len(bi)
    rec.update({"train_tokens": n_tok, "occurrences": n,
                "trigram": {"rows": len(tri_u), "occurrences_covered": float((tri_row >= 0).sum() / n)},
                "bigram": {"rows": len(bi_u), "occurrences_covered": float((bi_row >= 0).sum() / n)},
                "read_hit": float(((tri_row >= 0) | (bi_row >= 0)).sum() / n),
                "rows": len(tri_u) + len(bi_u)})
    print("ROWS", json.dumps({k: rec[k] for k in ("occurrences", "trigram", "bigram", "read_hit", "rows")}), flush=True)
    W = [output_embedding(gguf_path, devs[0])]
    rec["w_out_fp32_sha256"] = hashlib.sha256(W[0].cpu().numpy().astype("<f4").tobytes()).hexdigest()
    assert rec["w_out_fp32_sha256"] == W_OUT_FP32_SHA256, "not T0's output embedding"
    W.append(W[0].to(devs[1]))
    sums = [torch.zeros(len(tri_u), H, device=devs[0]), torch.zeros(len(bi_u), H, device=devs[1])]
    t1 = time.time()
    for a in range(0, n, CHUNK):
        ids = torch.from_numpy(ids32[a:a + CHUNK].astype(np.int64))
        p = torch.from_numpy(np.exp(lp32[a:a + CHUNK].astype(np.float64)).astype(np.float32))
        for k, row in enumerate((tri_row, bi_row)):
            r = torch.from_numpy(row[a:a + CHUNK])
            sel = r >= 0
            if sel.any():
                u = F.embedding_bag(ids[sel].to(devs[k]), W[k], per_sample_weights=p[sel].to(devs[k]),
                                    mode="sum")
                sums[k].index_add_(0, r[sel].to(devs[k]), u)
        if a // CHUNK % 200 == 0:
            print(f"chunk {a // CHUNK} of {n // CHUNK + 1}, {time.time() - t1:.0f}s", flush=True)
    for k in range(2):
        if devs[k].startswith("cuda"):
            torch.cuda.synchronize(devs[k])
    rec["accumulate_s"] = round(time.time() - t1, 1)
    out.mkdir(parents=True, exist_ok=True)
    vals = out / "bank_values.f16"
    with vals.open("wb") as f:
        for k, cnt in enumerate((tri_c, bi_c)):
            c = torch.from_numpy(cnt).to(devs[k]).float()[:, None]
            for a in range(0, len(cnt), 1 << 16):
                f.write((sums[k][a:a + (1 << 16)] / c[a:a + (1 << 16)]).half().cpu().numpy().tobytes())
    np.savez(out / "bank_index.npz", tri_keys=tri_u, tri_count=tri_c, bi_keys=bi_u, bi_count=bi_c)
    rec["values_bytes"] = vals.stat().st_size
    assert rec["values_bytes"] == rec["rows"] * H * 2
    # Check: rows recomputed from their occurrences in float64 on the CPU, against the stored fp16 values.
    Wc = W[0].cpu().double()
    mm = np.memmap(vals, dtype=np.float16, mode="r", shape=(rec["rows"], H))
    rng = np.random.default_rng(20261016)
    worst = 0.0
    for k, (u, row, base) in enumerate(((tri_u, tri_row, 0), (bi_u, bi_row, len(tri_u)))):
        for j in rng.choice(len(u), size=min(n_check, len(u)), replace=False):
            occ = np.flatnonzero(row == j)
            v = (torch.from_numpy(np.exp(lp32[occ].astype(np.float64)))[:, :, None] *
                 Wc[torch.from_numpy(ids32[occ].astype(np.int64))]).sum((0, 1)) / len(occ)
            worst = max(worst, float((v - torch.from_numpy(mm[base + j].astype(np.float64))).abs().max()))
    rec["check"] = {"rows_per_kind": n_check, "max_abs_diff_vs_float64": worst}
    rec["value_norm_mean"] = float(np.linalg.norm(mm[:: max(1, rec["rows"] // 4096)].astype(np.float32), axis=1).mean())
    h = hashlib.sha256()
    with vals.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    rec["values_sha256"] = h.hexdigest()
    rec["index_sha256"] = hashlib.sha256((out / "bank_index.npz").read_bytes()).hexdigest()
    rec["wall_s"] = round(time.time() - t_all, 1)
    (out / "bank.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("RESULT", json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=t0.T0_GGUF)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=TRAIN_SHARDS)
    ap.add_argument("--check", type=int, default=8)
    ap.add_argument("--cpu-smoke", action="store_true", help="local: both devices on the CPU")
    a = ap.parse_args()
    build(a.gguf, a.teacher, a.corpus, a.out, a.shards, a.check,
          **({"devs": ("cpu", "cpu")} if a.cpu_smoke else {}))
