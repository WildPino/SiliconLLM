"""C50-A0: pinned metadata screen of Phase-A donor candidates (CHATBOT50 roadmap section 5).

Reads, at a pinned Hugging Face revision, only metadata: the model API record,
small text files (config, generation config, tokenizer files, chat template)
and the JSON headers of every safetensors shard (HTTP range requests; no tensor
payload is requested).  From the headers it counts coefficients per organ and
prices one decode token with the same policies and constants as
byte_budget_ledger.py.  It also hashes tokenizer files so that tokenizer
identity between family members is decided by bytes, not by names.

DESK MODEL for every rate.  Standard library only.  Usage:
    python -I benchmarks/chatbot50/a0_donor_screen.py --json out.json --cache <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_budget_ledger as L  # noqa: E402

ORIGIN = "https://huggingface.co/"
API = ORIGIN + "api/models/"
SMALL_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja", "tokenizer.json")
MAX_SMALL = 64 * 1024 * 1024
MAX_HEADER = 32 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "U32": 4,
               "I64": 8, "F8_E4M3": 1}

# Candidates: (repo, revision pinned at screen time 2026-10-16, role).
CANDIDATES = [
    ("inclusionAI/Ling-mini-2.0", "a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab", "candidate"),
    ("inclusionAI/Ling-3.0-tiny", "9a98e35fe1c9ee255f78dd64771c7ae15a799481", "candidate"),
    ("utter-project/EuroMoE-2.6B-A0.6B-Instruct-2512", "96e19829c2fe57c4d5d484ecbd6b0ad6b950cac6", "candidate"),
    ("arcee-ai/Trinity-Nano-Preview", "b576b32b8bba7f5218312c9509e494537f8d282d", "candidate"),
    ("LiquidAI/LFM2.5-8B-A1B", "5dd22602c2e9f6a097b1de4c4efe0658b605015c", "candidate"),
]
# Family members whose tokenizer must be compared with the Phase-A donor (Phase C teacher).
TOKENIZER_ONLY = [
    ("inclusionAI/Ling-flash-2.0", "96f9b7a73e531b0ca05f8839ea9f271948ad8ea4"),
    ("inclusionAI/Ling-3.0-flash", "fcc60a52b34c69c6271d0ac8a8cb0815020b6a43"),
    ("inclusionAI/Ling-3.1-flash", "a4ccd6b45c002ddb236f9cb77d52cc427482bf19"),
]

LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


class ScreenError(RuntimeError):
    pass


def fetch(url: str, cap: int, headers: dict | None = None) -> bytes:
    req = Request(url, headers={"Accept-Encoding": "identity", **(headers or {})})
    with urlopen(req, timeout=60) as h:
        data = h.read(cap + 1)
    if len(data) > cap:
        raise ScreenError(f"oversize response: {url}")
    return data


def fetch_range(url: str, first: int, last: int) -> bytes:
    req = Request(url, headers={"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=60) as h:
        if h.status != 206:
            raise ScreenError(f"range refused ({h.status}): {url}")
        body = h.read(last - first + 2)
    if len(body) != last - first + 1:
        raise ScreenError(f"range length mismatch: {url}")
    return body


def resolve(repo: str, rev: str, name: str) -> str:
    return f"{ORIGIN}{repo}/resolve/{rev}/{name}"


def small_files(repo: str, rev: str, siblings: set[str], cache: Path | None) -> dict:
    out = {}
    for name in SMALL_FILES:
        if name not in siblings:
            continue
        data = fetch(resolve(repo, rev, name), MAX_SMALL)
        out[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "data": data}
        if cache is not None:
            target = cache / repo.replace("/", "__") / rev[:12] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return out


def shard_header(repo: str, rev: str, shard: str) -> dict:
    url = resolve(repo, rev, shard)
    n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    if n <= 0 or n > MAX_HEADER:
        raise ScreenError(f"bad header length {n}: {shard}")
    header = json.loads(fetch_range(url, 8, 8 + n - 1))
    header.pop("__metadata__", None)
    return header


def tensors(repo: str, rev: str, siblings: set[str]) -> dict:
    shards = sorted(s for s in siblings if s.endswith(".safetensors"))
    if not shards:
        raise ScreenError(f"no safetensors shards: {repo}")
    out = {}
    for shard in shards:
        for name, item in shard_header(repo, rev, shard).items():
            if item["dtype"] not in DTYPE_BYTES:
                raise ScreenError(f"unsupported dtype {item['dtype']}: {name}")
            numel = 1
            for x in item["shape"]:
                numel *= x
            if name in out:
                raise ScreenError(f"duplicate tensor {name}")
            out[name] = {"shape": item["shape"], "dtype": item["dtype"], "numel": numel, "shard": shard}
    return out


# ---------------------------------------------------------------- organs --
def organ_of(name: str, shape: list[int], cfg: dict) -> tuple[str, float]:
    """Return (organ, fraction streamed per decode token) for one tensor."""
    n_layers = cfg["num_hidden_layers"]
    experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    topk = cfg.get("num_experts_per_tok")
    low = name.lower()
    if "embed_tokens" in low or "word_embeddings" in low or low.endswith("wte.weight"):
        return "embedding_row", 1.0 / shape[0]
    if low.startswith("lm_head") or low.endswith("output.weight") and "layers" not in low:
        return "head", 1.0
    m = LAYER.search(name)
    if m is None:
        if "norm" in low:
            return "small", 1.0
        raise ScreenError(f"unclassified non-layer tensor: {name}")
    layer, suffix = int(m[1]), m[2].lower()
    if layer >= n_layers:
        return "mtp_excluded", 0.0
    if suffix.endswith("bias") and ("norm" in suffix or "correction" in suffix or "expert_bias" in suffix):
        return "small", 1.0
    if "norm" in suffix and len(shape) == 1:
        return "small", 1.0
    if re.search(r"experts\.\d+\.", suffix) or ".experts." in suffix and "shared" not in suffix:
        return "routed", topk / experts
    if "shared" in suffix:
        return "shared", 1.0
    if (suffix.endswith("gate.weight") or "router" in suffix) and shape and shape[0] == experts:
        return "router", 1.0
    if any(k in suffix for k in ("attn", "attention", "mixer", "conv", "self_attn", "linear_attn",
                                  "mamba", "kda")):
        return "attention_or_mixer", 1.0
    if "mlp" in suffix or "feed_forward" in suffix or "ffn" in suffix:
        return "dense_ffn", 1.0
    raise ScreenError(f"unclassified layer tensor: {name}")


def organ_counts(tens: dict, cfg: dict) -> dict:
    counts: dict[str, float] = {}
    stored: dict[str, int] = {}
    for name, t in tens.items():
        organ, frac = organ_of(name, t["shape"], cfg)
        counts[organ] = counts.get(organ, 0.0) + t["numel"] * frac
        stored[organ] = stored.get(organ, 0) + t["numel"]
    return {"per_token_coefficients": {k: round(v) for k, v in sorted(counts.items())},
            "stored_coefficients": dict(sorted(stored.items())),
            "total_stored": sum(stored.values())}


# ------------------------------------------------------------- cache shapes --
def cache_geometry(cfg: dict) -> dict:
    """KV values read per position at full context, sliding-window values, recurrent state."""
    mt = cfg.get("model_type", "")
    n = cfg["num_hidden_layers"]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    if mt in ("bailing_moe", "bailing_moe_v2", "mixtral"):
        return {"kv_values_per_pos": n * 2 * cfg["num_key_value_heads"] * hd, "kv_window_per_pos": 0,
                "kv_window": 0, "state_bytes": 0, "basis": "GQA, every layer full attention"}
    if mt in ("bailing_moe_v3", "bailing_hybrid"):
        group = cfg["layer_group_size"]
        mla_layers = n // group
        kda_layers = n - mla_layers
        latent = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
        heads = cfg["num_attention_heads"]
        state = kda_layers * heads * hd * hd * 4 * 2  # F32 state, read + write
        return {"kv_values_per_pos": mla_layers * latent, "kv_window_per_pos": 0, "kv_window": 0,
                "state_bytes": state,
                "basis": f"{mla_layers} MLA layers cache {latent} latent values/pos; {kda_layers} KDA "
                         f"layers keep a {heads}x{hd}x{hd} F32 state (read+write per token, ASSUMED "
                         f"head_dim {hd} for key and value)"}
    if mt == "afmoe":
        kinds = cfg["layer_types"]
        full = sum(k == "full_attention" for k in kinds)
        slide = len(kinds) - full
        per = 2 * cfg["num_key_value_heads"] * hd
        return {"kv_values_per_pos": full * per, "kv_window_per_pos": slide * per,
                "kv_window": cfg.get("sliding_window", 0), "state_bytes": 0,
                "basis": f"{full} full + {slide} sliding-window layers"}
    if mt == "lfm2_moe":
        kinds = cfg["layer_types"]
        full = sum(k == "full_attention" for k in kinds)
        conv = len(kinds) - full
        state = conv * cfg["hidden_size"] * cfg["conv_L_cache"] * 4 * 2
        return {"kv_values_per_pos": full * 2 * cfg["num_key_value_heads"] * hd, "kv_window_per_pos": 0,
                "kv_window": 0, "state_bytes": state,
                "basis": f"{full} attention layers; {conv} short-conv layers with a D x L_cache state"}
    raise ScreenError(f"no cache geometry for model_type {mt}")


def donor_from(repo: str, cfg: dict, organs: dict, cache: dict, total: int) -> L.Donor:
    c = organs["per_token_coefficients"]
    vocab = cfg["vocab_size"]
    head = c.get("head", 0)
    if head == 0 and cfg.get("tie_word_embeddings"):
        head = vocab * cfg["hidden_size"]
    return L.Donor(
        name=repo.split("/")[1], evidence="VERIFIED-HEADERS (C50-A0)",
        attention=c.get("attention_or_mixer", 0), dense=c.get("dense_ffn", 0),
        shared=c.get("shared", 0), routed=c.get("routed", 0), router=c.get("router", 0),
        head=head, vocab=vocab, d_model=cfg["hidden_size"],
        kv_values_per_pos=cache["kv_values_per_pos"], kv_window_per_pos=cache["kv_window_per_pos"],
        kv_window=cache["kv_window"], state_bytes=cache["state_bytes"], total_params=float(total),
    )


# ------------------------------------------------------------------ chat facts --
def chat_facts(files: dict, cfg: dict) -> dict:
    tc = json.loads(files["tokenizer_config.json"]["data"]) if "tokenizer_config.json" in files else {}
    gc = json.loads(files["generation_config.json"]["data"]) if "generation_config.json" in files else {}
    template = None
    if "chat_template.jinja" in files:
        template = "chat_template.jinja"
    elif tc.get("chat_template"):
        template = "tokenizer_config.json"
    return {
        "chat_template_source": template,
        "eos_token_id_config": cfg.get("eos_token_id"), "pad_token_id_config": cfg.get("pad_token_id"),
        "bos_token_id_config": cfg.get("bos_token_id"),
        "eos_token_id_generation": gc.get("eos_token_id"),
        "eos_token": tc.get("eos_token"), "pad_token": tc.get("pad_token"), "bos_token": tc.get("bos_token"),
        "tokenizer_class": tc.get("tokenizer_class"),
    }


def tokenizer_fingerprint(files: dict) -> dict:
    if "tokenizer.json" not in files:
        return {}
    tok = json.loads(files["tokenizer.json"]["data"])
    model = tok.get("model", {})
    vocab = model.get("vocab", {})
    vocab_items = sorted(vocab.items(), key=lambda kv: kv[1]) if isinstance(vocab, dict) else vocab
    added = sorted((a["id"], a["content"]) for a in tok.get("added_tokens", []))
    return {
        "tokenizer_json_sha256": files["tokenizer.json"]["sha256"],
        # same parsed content regardless of whitespace or key order in the file
        "canonical_json_sha256": hashlib.sha256(json.dumps(tok, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "model_type": model.get("type"), "vocab_entries": len(vocab_items),
        "vocab_sha256": hashlib.sha256(json.dumps(vocab_items, ensure_ascii=False).encode()).hexdigest(),
        # merges are stored either as "a b" strings or as ["a", "b"] pairs; hash the normalised form
        "merges_sha256": hashlib.sha256(json.dumps(
            [m if isinstance(m, str) else " ".join(m) for m in model.get("merges", [])],
            ensure_ascii=False).encode()).hexdigest(),
        "merges": len(model.get("merges", [])), "ignore_merges": model.get("ignore_merges"),
        "added_tokens": len(added),
        "added_tokens_sha256": hashlib.sha256(json.dumps(added, ensure_ascii=False).encode()).hexdigest(),
        "pre_tokenizer_sha256": hashlib.sha256(json.dumps(tok.get("pre_tokenizer"), sort_keys=True).encode()).hexdigest(),
        "normalizer_sha256": hashlib.sha256(json.dumps(tok.get("normalizer"), sort_keys=True).encode()).hexdigest(),
    }


def api_record(repo: str, rev: str) -> dict:
    rec = json.loads(fetch(API + repo + "/revision/" + rev, 8 * 1024 * 1024))
    if rec.get("sha") != rev:
        raise ScreenError(f"revision mismatch for {repo}: {rec.get('sha')}")
    card = rec.get("cardData") or {}
    return {"sha": rec["sha"], "license": card.get("license"), "license_name": card.get("license_name"),
            "gated": rec.get("gated"), "last_modified": rec.get("lastModified"),
            "architectures": (rec.get("config") or {}).get("architectures"),
            "base_model": card.get("base_model"),
            "siblings": sorted(s["rfilename"] for s in rec.get("siblings", []))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--cache", default=None, help="directory for the fetched small files")
    args = ap.parse_args()
    cache = Path(args.cache) if args.cache else None
    report = {"cell": "C50-A0", "constants": {"BW_PLAN": L.BW_PLAN, "T_FIXED": L.T_FIXED,
              "CALIBRATION": L.CALIBRATION, "CALIBRATION_W4": L.CALIBRATION_W4,
              "SHORTLIST_RANK": L.SHORTLIST_RANK, "SHORTLIST_RERANK": L.SHORTLIST_RERANK},
              "candidates": [], "tokenizers": {}}
    print("CONFIG cell=C50-A0 bw_plan=%.1e t_fixed=%.1e policies=%s contexts=%s" % (
        L.BW_PLAN, L.T_FIXED, ",".join(p.key for p in L.POLICIES), L.CONTEXTS))
    for repo, rev, role in CANDIDATES:
        api = api_record(repo, rev)
        sib = set(api["siblings"])
        files = small_files(repo, rev, sib, cache)
        cfg = json.loads(files["config.json"]["data"])
        tens = tensors(repo, rev, sib)
        organs = organ_counts(tens, cfg)
        geom = cache_geometry(cfg)
        donor = donor_from(repo, cfg, organs, geom, organs["total_stored"])
        rows = [L.ledger(donor, p, ctx) for p in L.POLICIES for ctx in L.CONTEXTS]
        entry = {"repo": repo, "revision": rev, "role": role, "api": api,
                 "files": {k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in files.items()},
                 "chat": chat_facts(files, cfg), "tokenizer": tokenizer_fingerprint(files),
                 "tensor_count": len(tens), "organs": organs, "cache_geometry": geom, "ledger": rows}
        report["candidates"].append(entry)
        report["tokenizers"][repo] = entry["tokenizer"]
        p2 = next(r for r in rows if r["policy"] == "P2" and r["context"] == 2048)
        print(f"{repo}@{rev[:7]} license={api['license']}/{api['license_name']} tensors={len(tens)} "
              f"stored={organs['total_stored'] / 1e9:.3f}B P2@2048={p2['total_bytes'] / 1e6:.1f}MB "
              f"desk={p2['desk_tok_s']:.1f}")
    for repo, rev in TOKENIZER_ONLY:
        api = api_record(repo, rev)
        files = small_files(repo, rev, set(api["siblings"]), cache)
        cfg = json.loads(files["config.json"]["data"])
        report["tokenizers"][repo] = {**tokenizer_fingerprint(files), "revision": rev,
                                      "license": api["license"], "chat": chat_facts(files, cfg)}
    base = report["tokenizers"]["inclusionAI/Ling-mini-2.0"]
    keys = ("vocab_sha256", "merges_sha256", "added_tokens_sha256", "pre_tokenizer_sha256", "normalizer_sha256")
    report["tokenizer_identity_vs_ling_mini"] = {
        repo: {k: fp.get(k) == base.get(k) for k in keys + ("tokenizer_json_sha256", "canonical_json_sha256")}
        for repo, fp in report["tokenizers"].items() if repo != "inclusionAI/Ling-mini-2.0" and fp}
    base_added = {a["id"]: a["content"] for a in json.loads(
        (Path(args.cache) / "inclusionAI__Ling-mini-2.0" / "a810f6416bc4" / "tokenizer.json").read_text(
            encoding="utf-8"))["added_tokens"]} if cache is not None else {}
    for repo, rev in [(r, v) for r, v, _ in CANDIDATES] + TOKENIZER_ONLY:
        if not repo.startswith("inclusionAI/") or repo == "inclusionAI/Ling-mini-2.0" or cache is None:
            continue
        tok = json.loads((cache / repo.replace("/", "__") / rev[:12] / "tokenizer.json").read_text(encoding="utf-8"))
        renamed = [[a["id"], base_added.get(a["id"]), a["content"]] for a in tok["added_tokens"]
                   if base_added.get(a["id"]) != a["content"]]
        report["tokenizer_identity_vs_ling_mini"][repo]["added_token_changes"] = renamed
        print(f"added-token changes vs Ling-mini-2.0: {repo}: {len(renamed)} "
              + " ".join(f"{i}:{a}->{b}" for i, a, b in renamed))
    for repo, same in report["tokenizer_identity_vs_ling_mini"].items():
        print(f"tokenizer vs Ling-mini-2.0: {repo}: " + " ".join(
            f"{k}={v}" for k, v in same.items() if k != "added_token_changes"))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
