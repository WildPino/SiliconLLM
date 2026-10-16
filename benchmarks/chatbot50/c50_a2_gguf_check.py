"""C50-A2 GGUF artifact record: file hash, source-vocabulary identity, tensor types and
per-token weight bytes of each artifact.

- The GGUF token list must equal the source tokenizer's ID -> token mapping for every ID
  (model vocabulary plus added tokens); runtimes are fed source-tokenizer IDs, so this
  identity is what makes those IDs mean the same tokens.
- Per-token weight bytes (batch-1 decode, context-free part): every tensor once, except
  routed experts (top-k / n_experts of their bytes) and the token embedding (one row).

Usage:
    python -I benchmarks/chatbot50/c50_a2_gguf_check.py --tokenizer tokenizer.json \
        --gguf a.gguf [b.gguf ...] --out results/c50_a2_gguf_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

LLAMA_PIN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def source_tokens(tokenizer_json: Path) -> dict[int, str]:
    tok = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    ids = {i: s for s, i in tok["model"]["vocab"].items()}
    for a in tok.get("added_tokens", []):
        if a["id"] in ids and ids[a["id"]] != a["content"]:
            raise SystemExit(f"added token {a['id']} conflicts with the model vocabulary")
        ids[a["id"]] = a["content"]
    return ids


def organ(name: str) -> str:
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "head"
    if "_exps" in name:
        return "routed"
    if "_shexp" in name:
        return "shared"
    if "ffn_gate_inp" in name or "exp_probs_b" in name:
        return "router"
    if "attn" in name:
        return "attention"
    if name.startswith("blk.") and ("ffn_gate" in name or "ffn_up" in name or "ffn_down" in name):
        return "dense_ffn"
    return "small"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(LLAMA_PIN / "gguf-py"))
    from gguf import GGUFReader

    src = source_tokens(Path(args.tokenizer))
    report = {"tokenizer_sha256": sha256_file(Path(args.tokenizer)), "artifacts": []}
    for g in args.gguf:
        path = Path(g)
        reader = GGUFReader(str(path))
        tokens = reader.get_field("tokenizer.ggml.tokens").contents()
        n_exp = int(reader.get_field("bailingmoe2.expert_count").contents())
        k_exp = int(reader.get_field("bailingmoe2.expert_used_count").contents())
        mismatches = [i for i in range(max(len(tokens), max(src) + 1))
                      if (tokens[i] if i < len(tokens) else None) != src.get(i)]
        # IDs absent from the source mapping are padding rows of the GGUF vocabulary; they are
        # listed separately and are never fed (the source tokenizer cannot emit them).
        padding = [i for i in mismatches if i not in src]
        real = [i for i in mismatches if i in src]
        by_type, by_organ = defaultdict(int), defaultdict(int)
        n_embd = None
        for t in reader.tensors:
            by_type[t.tensor_type.name] += int(t.n_bytes)
            by_organ[organ(t.name)] += int(t.n_bytes)
            if t.name == "token_embd.weight":
                n_embd = [int(x) for x in t.shape]
                embd_row = int(t.n_bytes) // int(t.shape[1])
        per_token = (sum(v for k, v in by_organ.items() if k not in ("routed", "embedding"))
                     + by_organ["routed"] * k_exp / n_exp + embd_row)
        file_type = reader.get_field("general.file_type")
        rec = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "file_type": int(file_type.contents()) if file_type else None,
            "tensors": len(reader.tensors), "gguf_tokens": len(tokens), "source_tokens": len(src),
            "vocab_mismatch_on_source_ids": real[:20], "vocab_mismatch_count": len(real),
            "gguf_ids_outside_source": len(padding),
            "experts": n_exp, "experts_used": k_exp, "token_embd_shape": n_embd,
            "bytes_by_type": dict(sorted(by_type.items())), "bytes_by_organ": dict(sorted(by_organ.items())),
            "per_token_weight_bytes": per_token,
        }
        report["artifacts"].append(rec)
        print(f"{path.name}: {rec['bytes'] / 1e9:.2f} GB sha256 {rec['sha256'][:16]}.. vocab mismatches "
              f"{len(real)} (GGUF-only ids {len(padding)}); per-token weight bytes {per_token / 1e6:.1f} MB")
        print("  by type (GB):", {k: round(v / 1e9, 3) for k, v in rec["bytes_by_type"].items()})
        print("  by organ (GB):", {k: round(v / 1e9, 3) for k, v in rec["bytes_by_organ"].items()})
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if all(a["vocab_mismatch_count"] == 0 for a in report["artifacts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
