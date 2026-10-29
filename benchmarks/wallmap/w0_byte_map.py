"""Per-organ byte map of a bailingmoe2 GGUF (Ling-mini-2.0, Ling-flash-2.0), from the headers only.

For each organ: parameters (from the BF16 file), bytes stored and bytes read per decoded token (from the quantized
file), batch 1, at the declared contexts. Rules, the same for every model:
- every tensor is read once per token, except routed experts (top-k of E per MoE layer, so k/E of their bytes) and
  the token embedding (one row);
- the KV cache adds n_layers * 2 * n_kv_heads * head_dim * context * kv_bytes per token (F16 = 2, as llama.cpp
  stores it by default);
- "small" is norms, q/k norms and the expert-bias vectors.
Nothing here is measured on the box: it is SOURCE-DERIVED from the files' own headers.

    python -I benchmarks/wallmap/w0_byte_map.py --pair mini BF16.gguf Q4KM.gguf --pair flash BF16.gguf Q4KM.gguf \
        --ctx 2048 8192 32768 --out benchmarks/wallmap/results/w0_byte_map.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

LLAMA_PIN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4")
ORGANS = ("attention", "kv_cache", "shared", "dense_ffn", "router", "routed", "head", "embedding", "small")


def organ(name: str) -> str:
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "head"
    if "_exps" in name:
        return "routed"
    if "_shexp" in name:
        return "shared"
    if "ffn_gate_inp" in name:
        return "router"
    if name.endswith(("attn_qkv.weight", "attn_output.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight")):
        return "attention"
    if name.startswith("blk.") and name.endswith(("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")):
        return "dense_ffn"
    return "small"


def header_sha(path: Path, n: int = 1 << 26) -> str:
    """sha256 of the first 64 MiB: the header and tensor table, enough to pin which file was read."""
    with path.open("rb") as f:
        return hashlib.sha256(f.read(n)).hexdigest()


def read(path: Path):
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    hp = {k[len("bailingmoe2."):]: r.fields[k].contents() for k in r.fields if k.startswith("bailingmoe2.")}
    tensors = [(t.name, int(t.n_elements), int(t.n_bytes), t.tensor_type.name, [int(x) for x in t.shape])
               for t in r.tensors]
    return hp, tensors


def one(name: str, bf16: Path, quant: Path, ctxs: list[int]) -> dict:
    hp, tb = read(bf16)
    hq, tq = read(quant)
    assert {k: v for k, v in hp.items() if k != "nextn_predict_layers"} == \
           {k: v for k, v in hq.items() if k != "nextn_predict_layers"}, "the two files are different models"
    qmap = {t[0]: t for t in tq}
    assert sorted(t[0] for t in tb) == sorted(qmap), "the two files hold different tensors"
    assert all(qmap[t[0]][1] == t[1] for t in tb), "a tensor's element count differs between the files"
    E, k = int(hp["expert_count"]), int(hp["expert_used_count"])
    params, stored, per_tok, types = defaultdict(int), defaultdict(int), defaultdict(float), defaultdict(lambda: defaultdict(int))
    for n, ne, _, _, shape in tb:
        _, _, nb, ty, _ = qmap[n]
        o = organ(n)
        params[o] += ne
        stored[o] += nb
        types[o][ty] += nb
        if o == "routed":
            per_tok[o] += nb * k / E
        elif o == "embedding":
            per_tok[o] += nb / shape[1]                    # one row of n_vocab
        else:
            per_tok[o] += nb
    L, kvh, hd = int(hp["block_count"]), int(hp["attention.head_count_kv"]), int(hp["attention.key_length"])
    kv_per_pos = L * 2 * kvh * hd * 2
    act = sum(params[o] for o in ORGANS if o not in ("routed", "embedding")) + params["routed"] * k / E \
        + params["embedding"] / int(hp["vocab_size"])
    rec = {"model": name, "bf16": str(bf16), "quant": str(quant), "bf16_header_sha256": header_sha(bf16),
           "quant_header_sha256": header_sha(quant), "hparams": {kk: (v if not hasattr(v, "item") else v.item())
                                                                 for kk, v in hp.items()},
           "params": dict(params), "params_total": sum(params.values()), "params_active_per_token": act,
           "stored_bytes": dict(stored), "stored_total": sum(stored.values()),
           "types_by_organ": {o: dict(v) for o, v in types.items()},
           "weight_bytes_per_token": dict(per_tok), "kv_bytes_per_position": kv_per_pos, "by_context": {}}
    for c in ctxs:
        row = dict(per_tok)
        row["kv_cache"] = kv_per_pos * c
        rec["by_context"][str(c)] = {"bytes_per_token": row, "total": sum(row.values())}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NAME", "BF16", "QUANT"))
    ap.add_argument("--ctx", nargs="+", type=int, default=[2048, 8192, 32768])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(LLAMA_PIN / "gguf-py"))
    recs = [one(n, Path(b), Path(q), a.ctx) for n, b, q in a.pair]
    for r in recs:
        print(f"== {r['model']}: {r['params_total'] / 1e9:.3f} B params, {r['params_active_per_token'] / 1e9:.3f} B "
              f"active, {r['stored_total'] / 1e9:.2f} GB stored")
        for c, v in r["by_context"].items():
            print(f"  ctx {c}: {v['total'] / 1e6:.1f} MB/token  " +
                  "  ".join(f"{o} {v['bytes_per_token'].get(o, 0) / 1e6:.1f}" for o in ORGANS))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"models": recs}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
