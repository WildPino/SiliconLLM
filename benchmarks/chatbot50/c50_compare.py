"""Compare two c50_llama_score-format runs on the same source-tokenizer IDs.

Prints numbers only (no verdict): per-position top-1 disagreement, mean KL(ref || other)
in nats from the full log-softmax rows (when both runs dumped them), and dBPB =
(bits_other - bits_ref) / bytes of the scored tokens, each with a document-level
bootstrap 95% interval (documents resampled with replacement, 10,000 draws, fixed seed).
Scored-token bytes come from the source tokenizer's byte-level vocabulary.

Usage:
    python -I benchmarks/chatbot50/c50_compare.py --ref PREFIX --other PREFIX --ids IDS.tsv \
        --tokenizer tokenizer.json [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
DRAWS = 10_000
SEED = 20261016


def byte_lengths(tokenizer_json: Path) -> np.ndarray:
    tok = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}
    vocab = tok["model"]["vocab"]
    out = np.zeros(max(vocab.values()) + 1 + len(tok.get("added_tokens", [])), dtype=np.int64)
    for s, i in vocab.items():
        out[i] = len(bytes(dec[ch] for ch in s))
    return out


def load(prefix: str) -> np.ndarray:
    return np.fromfile(prefix + ".pos.bin", dtype=REC)


def kl_rows(ref: str, other: str, n_pos: int, n_vocab: int) -> np.ndarray:
    a = np.memmap(ref + ".logp.f32", dtype="<f4", mode="r", shape=(n_pos, n_vocab))
    b = np.memmap(other + ".logp.f32", dtype="<f4", mode="r", shape=(n_pos, n_vocab))
    out = np.empty(n_pos)
    for s in range(0, n_pos, 512):
        la = a[s:s + 512].astype(np.float64)
        lb = b[s:s + 512].astype(np.float64)
        out[s:s + 512] = (np.exp(la) * (la - lb)).sum(axis=1)
    return out


def boot(per_doc_num: np.ndarray, per_doc_den: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_doc_num), size=(DRAWS, len(per_doc_num)))
    stats = per_doc_num[idx].sum(axis=1) / per_doc_den[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    r, o = load(args.ref), load(args.other)
    if len(r) != len(o) or not (np.array_equal(r["doc"], o["doc"]) and np.array_equal(r["pos"], o["pos"])
                                and np.array_equal(r["target"], o["target"])):
        raise SystemExit("runs do not score the same positions")
    meta_r = json.loads(Path(args.ref + ".json").read_text())
    meta_o = json.loads(Path(args.other + ".json").read_text())
    blen = byte_lengths(Path(args.tokenizer))
    nbytes = blen[r["target"]].astype(np.float64)
    docs = np.unique(r["doc"])
    d_index = np.searchsorted(docs, r["doc"])

    def per_doc(values: np.ndarray) -> np.ndarray:
        return np.bincount(d_index, weights=values, minlength=len(docs))

    disagree = (r["top1"] != o["top1"]).astype(np.float64)
    ones = np.ones(len(r))
    dbits = (-(o["logp_target"].astype(np.float64)) + r["logp_target"].astype(np.float64)) / np.log(2)
    out = {
        "ref": args.ref, "other": args.other, "ref_meta": meta_r, "other_meta": meta_o,
        "positions": int(len(r)), "documents": int(len(docs)), "scored_bytes": float(nbytes.sum()),
        "top1_disagreement": float(disagree.mean()),
        "top1_disagreement_ci95": boot(per_doc(disagree), per_doc(ones)),
        "bpb_ref": float((-r["logp_target"].astype(np.float64)).sum() / np.log(2) / nbytes.sum()),
        "dbpb": float(dbits.sum() / nbytes.sum()),
        "dbpb_ci95": boot(per_doc(dbits), per_doc(nbytes)),
        "max_abs_dlogp_target": float(np.abs(dbits).max() * np.log(2)),
        "bit_identical_logp_target": bool(np.array_equal(r["logp_target"], o["logp_target"])),
    }
    if meta_r.get("dump") and meta_o.get("dump"):
        kl = kl_rows(args.ref, args.other, len(r), int(meta_r["n_vocab"]))
        out["mean_kl_nats"] = float(kl.mean())
        out["mean_kl_ci95"] = boot(per_doc(kl), per_doc(ones))
        out["p99_kl_nats"] = float(np.percentile(kl, 99))
        out["max_kl_nats"] = float(kl.max())
    for k, v in out.items():
        if not isinstance(v, dict):
            print(f"{k}: {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
