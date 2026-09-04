#!/usr/bin/env python3
"""Shared harness for the D0-D3 density probes.

Project laws honoured here:
  * the artefact is the authority  -> every dimension is read from config.json, never asserted;
  * BPB, byte-normalised           -> the unit is bits per UTF-8 byte of the held-out slice;
  * the eval slice is pinned       -> sha256 of the exact token ids and of the exact byte span;
  * planted controls               -> `planted_control_*` helpers, each of which must FIRE.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CORPUS = os.path.join(HERE, "corpus")
RESULTS = os.path.join(HERE, "results")

MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"   # pinned

torch.set_grad_enabled(False)


# --------------------------------------------------------------------------- model
def load_model(dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=dtype, attn_implementation="eager")
    model.eval()
    return model, tok


def arch(model) -> dict:
    """Dimensions read off the loaded artefact, not off the config prose."""
    cfg = model.config
    layer0 = model.model.layers[0]
    return {
        "n_layers": len(model.model.layers),
        "d_model": layer0.self_attn.q_proj.in_features,
        "d_ffn": layer0.mlp.gate_proj.out_features,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads,
        "vocab": model.get_input_embeddings().weight.shape[0],
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "hidden_act": cfg.hidden_act,
        "revision": REVISION,
    }


ORGANS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


ATTN_ORGANS = ("q_proj", "k_proj", "v_proj", "o_proj")


def get_linear(model, layer: int, organ: str):
    L = model.model.layers[layer]
    parent = L.self_attn if organ in ATTN_ORGANS else L.mlp
    return getattr(parent, organ)


# --------------------------------------------------------------------------- data
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_slice(tok, part: str, n_seq: int, seq_len: int, seed: int):
    """Draw `n_seq` sequences of `seq_len` tokens from a pinned corpus half.

    Offsets are drawn deterministically and uniformly over the file, which is itself a
    global shuffle of 8 KiB chunks, so neither file order nor block order can be a confound.
    A sequence is kept only if decode(ids[1:]) re-encodes to exactly ids[1:] -- that makes
    the byte count an exact property of the scored span, not an approximation.
    Returns (ids[n_seq, seq_len], byte_counts[n_seq], meta).
    """
    import random
    path = os.path.join(CORPUS, f"{part}.txt")
    size = os.path.getsize(path)
    rng = random.Random(seed)
    span = seq_len * 8          # ~4 bytes/token here; 8x is always enough
    ids_all, byts, offsets = [], [], []
    rejected = 0
    with open(path, "rb") as f:
        tries = 0
        while len(ids_all) < n_seq and tries < n_seq * 40:
            tries += 1
            off = rng.randrange(0, size - span)
            f.seek(off)
            txt = f.read(span).decode("utf-8", "ignore")
            if len(txt) < seq_len * 2:
                rejected += 1; continue
            enc = tok(txt, add_special_tokens=False)["input_ids"]
            if len(enc) < seq_len:
                rejected += 1; continue
            ids = enc[:seq_len]
            dec = tok.decode(ids[1:])
            if tok(dec, add_special_tokens=False)["input_ids"] != ids[1:]:
                rejected += 1; continue          # byte accounting would be inexact -> drop
            nb = len(dec.encode("utf-8"))
            if nb == 0:
                rejected += 1; continue
            ids_all.append(ids); byts.append(nb); offsets.append(off)
    ids = torch.tensor(ids_all, dtype=torch.long)
    meta = {
        "part": part, "n_seq": len(ids_all), "seq_len": seq_len, "seed": seed,
        "n_rejected": rejected,
        "total_scored_bytes": int(sum(byts)),
        "bytes_per_token": sum(byts) / max(1, len(ids_all) * (seq_len - 1)),
        "ids_sha256": _sha(ids.numpy().tobytes()),
        "offsets_sha256": _sha(json.dumps(offsets).encode()),
        "corpus_sha256": json.load(open(os.path.join(CORPUS, "manifest.json")))
                              ["parts"][part]["sha256"],
    }
    return ids, torch.tensor(byts, dtype=torch.float64), meta


def get_slice(tok, part: str, n_seq: int, seq_len: int, seed: int):
    """Disk-cached slice so every probe scores the byte-identical span."""
    os.makedirs(RESULTS, exist_ok=True)
    key = f"slice_{part}_{n_seq}x{seq_len}_s{seed}.pt"
    p = os.path.join(RESULTS, key)
    if os.path.exists(p):
        d = torch.load(p)
        return d["ids"], d["byts"], d["meta"]
    ids, byts, meta = make_slice(tok, part, n_seq, seq_len, seed)
    torch.save({"ids": ids, "byts": byts, "meta": meta}, p)
    return ids, byts, meta


# --------------------------------------------------------------------------- eval
def bpb(model, ids: torch.Tensor, byts: torch.Tensor, batch: int = 4,
        return_per_seq: bool = False):
    """Bits per UTF-8 byte on the pinned slice.  Deterministic given (model, ids)."""
    tot_nats = torch.zeros((), dtype=torch.float64)
    per_seq = []
    for i in range(0, ids.shape[0], batch):
        chunk = ids[i:i + batch]
        out = model(chunk).logits.float()
        lp = torch.nn.functional.log_softmax(out[:, :-1], dim=-1)
        tgt = chunk[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(dim=1).double()
        per_seq.append(nll)
        tot_nats += nll.sum()
    per_seq = torch.cat(per_seq)
    val = float(tot_nats / (math.log(2.0) * byts.sum()))
    if return_per_seq:
        return val, (per_seq / (math.log(2.0) * byts)).numpy()
    return val


def bootstrap_se(per_seq_bpb, byts, n_boot: int = 2000, seed: int = 7) -> float:
    """Slice-resampling SE: how much of a BPB delta is just which sequences we drew."""
    import numpy as np
    rng = np.random.default_rng(seed)
    b = byts.numpy()
    nats = per_seq_bpb * math.log(2.0) * b
    n = len(b)
    vals = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        vals.append(nats[k].sum() / (math.log(2.0) * b[k].sum()))
    return float(np.std(vals))


# --------------------------------------------------------------------------- results io
def dump(name: str, obj) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return p


class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a):
        print(f"  [{self.label}] {time.time()-self.t:.1f}s", flush=True)
