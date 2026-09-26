#!/usr/bin/env python3
"""Freeze source-disjoint documents for an independent C96 route audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth19_half_residual_independent_audit as M19
import meth25_fresh_r8_manifest as M25


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
SOURCES = (
    ("strat02_document_holdout_v1/heldout.jsonl",
     "450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e"),
    ("strat01_gigachat_fresh_v2/heldout.jsonl",
     "04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e"),
)
PRIOR = (
    (DOCS / "meth17_fresh_document_manifest.json",
     "7e0593d6c56c28398e3440f9b80db15d506a31131a21480a87578c23b0a043a8"),
    (DOCS / "meth19_half_residual_independent_manifest.json",
     "1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70"),
    (DOCS / "meth25_fresh_r8_manifest.json",
     "20890f2c4614287dfdb7527b0935436bb88a035a348e45e0c1e4f13f6d068f77"),
)
SEED = "meth41-41041"
COUNTS = {"code": 8, "prose": 8, "technical_general": 8}
CODE_REF = "9fb7791"


def rank(source_id):
    return M17.sha((SEED + "|" + source_id).encode("utf-8"))


def select(tokenizer):
    for path, expected in PRIOR:
        assert M17.sha(path.read_bytes()) == expected, path
    _, m17_items = M17.build_selection()
    _, m19_items = M19.select()
    _, m25_items = M25.select()
    prior = m17_items + m19_items + m25_items
    old_ids = {item["source_id"] for item in prior}
    base = [(M17.CORPUS / name).read_bytes()
            for name in ("calib.txt", "heldout.txt")]
    assert [M17.sha(x) for x in base] == [M17.CALIB_SHA, M17.HELDOUT_SHA]
    base.extend(item["text"].encode("utf-8") for item in prior)
    pool = defaultdict(list)
    stats = defaultdict(Counter)
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", CODE_REF],
        cwd=ROOT, text=True).splitlines()
    for path in paths:
        if not (path.startswith("benchmarks/donor_adaptation/s1/")
                and path.endswith(".py")):
            continue
        stats["code"]["source_rows"] += 1
        if path in old_ids:
            stats["code"]["prior_source_id"] += 1
            continue
        raw = subprocess.check_output(["git", "show", f"{CODE_REF}:{path}"],
                                      cwd=ROOT)
        if len(raw) < M17.SPAN:
            stats["code"]["short"] += 1
            continue
        start = int(rank(path)[:16], 16) % (len(raw) - M17.SPAN + 1)
        text = raw[start:start + M17.SPAN].decode("utf-8", errors="ignore")
        data = text.encode("utf-8")
        if len(data) < 4000 or M17.fragment_overlap(data, base):
            stats["code"]["short_or_fragment_overlap"] += 1
            continue
        pool["code"].append({"category": "code", "source_id": path,
                             "source_kind": "git", "source_ref": CODE_REF,
                             "source_sha256": M17.sha(raw),
                             "span_start_byte": start, "text": text})
    for path in paths:
        if not (path.startswith("docs/") and path.count("/") == 1
                and path.endswith(".md")):
            continue
        stats["technical_general"]["source_rows"] += 1
        if path in old_ids:
            stats["technical_general"]["prior_source_id"] += 1
            continue
        raw = subprocess.check_output(["git", "show", f"{CODE_REF}:{path}"],
                                      cwd=ROOT)
        if len(raw) < M17.SPAN:
            stats["technical_general"]["short"] += 1
            continue
        start = int(rank(path)[:16], 16) % (len(raw) - M17.SPAN + 1)
        text = raw[start:start + M17.SPAN].decode("utf-8", errors="ignore")
        data = text.encode("utf-8")
        if len(data) < 4000 or M17.fragment_overlap(data, base):
            stats["technical_general"]["short_or_fragment_overlap"] += 1
            continue
        pool["technical_general"].append(
            {"category": "technical_general", "source_id": path,
             "source_kind": "git", "source_ref": CODE_REF,
             "source_sha256": M17.sha(raw), "span_start_byte": start,
             "text": text})
    for rel, expected in SOURCES:
        source = M17.CORPUS / rel
        assert M17.sha(source.read_bytes()) == expected
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cat = row["category"]
            if cat != "prose":
                continue
            stats[cat]["source_rows"] += 1
            source_id = row["source_document_id"]
            if source_id in old_ids:
                stats[cat]["prior_source_id"] += 1
                continue
            text = row["text"].encode("utf-8")[:M17.SPAN].decode(
                "utf-8", errors="ignore")
            data = text.encode("utf-8")
            if len(data) < 4000 or M17.fragment_overlap(data, base):
                stats[cat]["short_or_fragment_overlap"] += 1
                continue
            pool[cat].append({"category": cat, "source_id": source_id,
                              "source_kind": row["source_kind"],
                              "source_ref": expected,
                              "source_sha256": row["source_content_sha256"],
                              "span_start_byte": row["span_start_byte"],
                              "text": text})
    items = []
    seen_ids = set()
    seen_hashes = set()
    for cat, count in COUNTS.items():
        candidates = sorted(pool[cat], key=lambda x: rank(x["source_id"]))
        for item in candidates:
            data = item["text"].encode("utf-8")
            digest = M17.sha(data)
            if item["source_id"] in seen_ids or digest in seen_hashes:
                stats[cat]["duplicate"] += 1
                continue
            if M17.fragment_overlap(data, [x["text"].encode("utf-8")
                                           for x in items]):
                stats[cat]["selected_overlap"] += 1
                continue
            items.append(item)
            seen_ids.add(item["source_id"])
            seen_hashes.add(digest)
            if sum(x["category"] == cat for x in items) == count:
                break
        assert sum(x["category"] == cat for x in items) == count, (
            cat, len(candidates), dict(stats[cat]))
    chosen = []
    for item in items:
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        assert len(ids) >= 256
        prompt = ids[:256]
        chosen.append({**item, "ids": ids, "prompt_ids": prompt,
                       "text_sha256": M17.sha(item["text"].encode("utf-8")),
                       "bytes": len(item["text"].encode("utf-8")),
                       "tokens": len(ids),
                       "prompt_ids_sha256": M17.sha(np.asarray(
                           prompt, dtype=np.int32).tobytes())})
    manifest = {"experiment": "METH-41", "seed": SEED,
                "code_ref": CODE_REF,
                "source_sha256": {rel: sha for rel, sha in SOURCES},
                "prior_manifest_sha256": {path.name: sha for path, sha in PRIOR},
                "excluded_qwen_corpus_sha256": [M17.CALIB_SHA, M17.HELDOUT_SHA],
                "overlap_rule": "source-ID exclusion plus first/middle/final 256-byte fragment exclusion against Qwen corpus and METH-17/19/25 selected spans; selected-set cross-screen",
                "selected_counts": COUNTS,
                "source_stats": {k: dict(v) for k, v in stats.items()},
                "span_max_bytes": M17.SPAN,
                "tokenizer_fingerprint": M15.M13.TOK_FP,
                "items": [{k: v for k, v in item.items()
                           if k not in ("text", "ids", "prompt_ids")}
                          for item in chosen]}
    return manifest, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        M15.M13.MODEL, revision=M15.M13.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    manifest, _ = select(tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_sha256": M17.sha(args.out.read_bytes()),
                      "selected_counts": manifest["selected_counts"],
                      "source_stats": manifest["source_stats"]}, indent=2))


if __name__ == "__main__":
    main()
