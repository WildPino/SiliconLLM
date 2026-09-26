#!/usr/bin/env python3
"""METH-25: freeze new documents before evaluating the packed R8 artifact."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import meth17_fresh_transfer_audit as M17
import meth19_half_residual_independent_audit as M19


ROOT = Path(__file__).resolve().parents[3]
CORPORA = (
    ("strat01_gigachat_fresh_v2/calib.jsonl",
     "68d9327a823328b1104c843afe986efeefcdd88c911c02895d577f65bffd8f81"),
    ("strat02_document_holdout_v1/calib.jsonl",
     "f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f"),
)
SEED = "meth25-25025"
COUNTS = {"code": 24, "prose": 16, "technical_general": 8}
SPAN = M17.SPAN


def rank(key):
    return M17.sha((SEED + "|" + key).encode("utf-8"))


def select():
    _, m17_items = M17.build_selection()
    _, m19_items = M19.select()
    previous = m17_items + m19_items
    old_ids = {item["source_id"] for item in previous}
    base = [(M17.CORPUS / name).read_bytes()
            for name in ("calib.txt", "heldout.txt")]
    assert [M17.sha(x) for x in base] == [M17.CALIB_SHA, M17.HELDOUT_SHA]
    base.extend(item["text"].encode("utf-8") for item in previous)
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", M17.CODE_REF],
        cwd=ROOT, text=True).splitlines()
    pool = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(int))
    for path in paths:
        if not (path.startswith("benchmarks/phase") and path.endswith((".c", ".py"))):
            continue
        raw = M17.git_bytes(path)
        if len(raw) < SPAN:
            continue
        stats["code"]["eligible_size"] += 1
        if path in old_ids:
            stats["code"]["old_id"] += 1
            continue
        start = int(rank(path)[:16], 16) % (len(raw) - SPAN + 1)
        selected = raw[start:start+SPAN].decode("utf-8", errors="ignore")
        data = selected.encode("utf-8")
        if len(data) < 4000 or M17.fragment_overlap(data, base):
            stats["code"]["overlap_or_short"] += 1
            continue
        pool["code"].append({"category": "code", "source_id": path,
                             "source_kind": "git", "source_ref": M17.CODE_REF,
                             "source_sha256": M17.sha(raw),
                             "span_start_byte": start, "text": selected})
    for rel, expected_sha in CORPORA:
        source = M17.CORPUS / rel
        assert M17.sha(source.read_bytes()) == expected_sha
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            category = row["category"]
            if category not in ("prose", "technical_general"):
                continue
            stats[category]["source_rows"] += 1
            source_id = row["source_document_id"]
            if source_id in old_ids:
                stats[category]["old_id"] += 1
                continue
            full = row["text"].encode("utf-8")
            selected = full[:SPAN].decode("utf-8", errors="ignore")
            data = selected.encode("utf-8")
            if len(data) < 4000 or M17.fragment_overlap(data, base):
                stats[category]["overlap_or_short"] += 1
                continue
            pool[category].append({"category": category,
                                   "source_id": source_id,
                                   "source_kind": row["source_kind"],
                                   "source_ref": expected_sha,
                                   "source_sha256": row["source_content_sha256"],
                                   "span_start_byte": row["span_start_byte"],
                                   "text": selected})
    items = []
    seen_ids = set()
    seen_hashes = set()
    for category, count in COUNTS.items():
        candidates = sorted(pool[category], key=lambda x: rank(x["source_id"]))
        for item in candidates:
            digest = M17.sha(item["text"].encode("utf-8"))
            if item["source_id"] in seen_ids or digest in seen_hashes:
                stats[category]["duplicate"] += 1
                continue
            # The selected set itself must satisfy the same fragment screen.
            if M17.fragment_overlap(item["text"].encode("utf-8"),
                                    [x["text"].encode("utf-8") for x in items]):
                stats[category]["selected_overlap"] += 1
                continue
            items.append(item)
            seen_ids.add(item["source_id"])
            seen_hashes.add(digest)
            if sum(x["category"] == category for x in items) == count:
                break
        assert sum(x["category"] == category for x in items) == count, (
            category, len(candidates), count, dict(stats[category]))
    manifest_items = []
    for item in items:
        row = {k: v for k, v in item.items() if k != "text"}
        row["text_sha256"] = M17.sha(item["text"].encode("utf-8"))
        row["text_bytes"] = len(item["text"].encode("utf-8"))
        manifest_items.append(row)
    manifest = {"experiment": "METH-25", "seed": SEED,
                "code_ref": M17.CODE_REF,
                "external_calib_sha256": {rel: sha for rel, sha in CORPORA},
                "excluded_qwen_corpus_sha256": [M17.CALIB_SHA, M17.HELDOUT_SHA],
                "excluded_prior_experiments": ["METH-17", "METH-19"],
                "overlap_rule": "first/middle/final 256 bytes absent from both Qwen corpus files and METH-17/19 selected spans, plus selected-set cross-screen",
                "source_stats": {k: dict(v) for k, v in stats.items()},
                "selected_counts": COUNTS, "span_max_bytes": SPAN,
                "items": manifest_items}
    return manifest, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest, items = select()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_sha256": M17.sha(out.read_bytes()),
                      "selected_counts": manifest["selected_counts"],
                      "selected_bytes": sum(x["text_bytes"] for x in manifest["items"]),
                      "source_stats": manifest["source_stats"]}, indent=2))


if __name__ == "__main__":
    main()
