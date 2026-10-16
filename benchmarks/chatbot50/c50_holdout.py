#!/usr/bin/env python3
"""Build the CHATBOT50 document holdout with FIT / DEV / RESERVED roles (roadmap A2).

The document machinery (source discovery, exact global dedup, eligibility, UTF-8-safe
bounded spans, item hashing) is the committed ``build_document_holdout.py`` loaded
from a pinned git blob, so local uncommitted edits to that file cannot change this
build.  What is new here:

- three roles instead of two, assigned to every category (code, technical markdown,
  PG19 prose) from the complete document content hash under a C50-specific salt;
  the PG19 publisher train/test label is ignored (nobody here trains on it);
- every document used by an earlier project holdout (``--exclude`` JSONL files) is
  removed by source ID and by content hash before role assignment;
- the source tokenizer (``tokenizer.json`` of the pinned checkpoint, no special
  tokens added) turns each span into IDs, and decode(IDs) must reproduce the span;
  that tokenizer normalises to NFC, so spans NFC would change are skipped in rank
  order and counted (``nfc_unstable_spans_skipped``).

Raw text and IDs are written under ``--out-dir`` (gitignored); ``manifest.json``
carries hashes only and is the tracked record.

Usage:
    python -I benchmarks/chatbot50/c50_holdout.py build --out-dir benchmarks/chatbot50/corpus/c50_holdout_v1 \
        --tokenizer <checkpoint>/tokenizer.json \
        --exclude benchmarks/donor_adaptation/density/corpus/strat02_document_holdout_v1/calib.jsonl ...
    python -I benchmarks/chatbot50/c50_holdout.py verify --out-dir ... --tokenizer ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BDH_PATH = "benchmarks/donor_adaptation/density/build_document_holdout.py"
BDH_COMMIT = "44cda684e5526f829bfaf564cc62481cc6478f5e"
SCHEMA = "c50_document_holdout_v1"
ROLES = ("FIT", "DEV", "RESERVED")
CONFIG = {
    "name": "c50_holdout_v1",
    "seed": 20261016,
    "role_salt": "c50-role-v1",
    "span_bytes": 8192,
    "min_source_bytes": 8192,
    "FIT": {"code": 48, "technical_general": 48, "prose": 48},
    "DEV": {"code": 32, "technical_general": 32, "prose": 32},
    "RESERVED": {"code": 32, "technical_general": 32, "prose": 32},
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_bdh():
    src = subprocess.run(["git", "show", f"{BDH_COMMIT}:{BDH_PATH}"], cwd=REPO, check=True,
                         capture_output=True).stdout
    mod = types.ModuleType("c50_bdh_pinned")
    mod.__file__ = str(REPO / BDH_PATH)
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"{BDH_COMMIT}:{BDH_PATH}", "exec"), mod.__dict__)
    return mod, sha(src)


def role_of(content_sha256: str) -> str:
    h = sha(f"{CONFIG['role_salt']}:{content_sha256}".encode("ascii"))
    return ROLES[int(h[:16], 16) % 3]


def read_exclusions(paths: list[str]) -> tuple[set[str], set[str], list[dict]]:
    ids, contents, record = set(), set(), []
    for p in paths:
        raw = Path(p).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        ids |= {r["source_document_id"] for r in rows}
        contents |= {r["source_content_sha256"] for r in rows}
        record.append({"path": Path(p).as_posix(), "jsonl_sha256": sha(raw),
                       "rows": len(rows)})
    return ids, contents, record


def tokenize(tok, text: str) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False).ids
    if tok.decode(ids, skip_special_tokens=False) != text:
        raise RuntimeError("source tokenizer does not round-trip a span")
    return ids


def build(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer
    bdh, bdh_sha = load_bdh()
    tok_raw = Path(args.tokenizer).read_bytes()
    tok = Tokenizer.from_str(tok_raw.decode("utf-8"))
    ex_ids, ex_contents, ex_record = read_exclusions(args.exclude)

    refs, counters = bdh.discover_documents()
    eligible, eligibility = bdh._eligible_refs(refs, CONFIG["min_source_bytes"])
    unique, dedup = bdh._deduplicate_globally(eligible)
    excluded = [r for r in unique if r.source_document_id in ex_ids or r.source_content_sha256 in ex_contents]
    pool = [r for r in unique if not (r.source_document_id in ex_ids or r.source_content_sha256 in ex_contents)]

    pools = {role: {c: [] for c in bdh.CATEGORIES} for role in ROLES}
    for r in pool:
        pools[role_of(r.source_content_sha256)][r.category].append(r)
    span_cfg = {"seed": CONFIG["seed"], "span_bytes": CONFIG["span_bytes"],
                "min_source_bytes": CONFIG["min_source_bytes"]}
    availability, nfc_rejected, file_hashes, built = {}, {}, {}, {}
    for role in ROLES:
        availability[role], nfc_rejected[role], built[role] = {}, {}, []
        for c in bdh.CATEGORIES:
            cand = sorted(pools[role][c], key=lambda r: (bdh._selection_key(r, role, CONFIG["seed"]),
                                                         r.source_document_id))
            availability[role][c] = len(cand)
            need, taken, rejected, start = CONFIG[role][c], [], 0, 0
            while len(taken) < need:
                batch = cand[start:start + 2 * need]
                if not batch:
                    raise SystemExit(f"insufficient {role}/{c}: need {need}, found {len(taken)}")
                start += len(batch)
                texts, hashes = bdh._hydrate_selected(batch)
                file_hashes.update(hashes)
                for r in batch:
                    if len(taken) == need:
                        break
                    item = bdh._item(r, role, texts[r.source_document_id], hashes[r.source_path], span_cfg)
                    # The source tokenizer normalises to NFC; a span that NFC changes cannot be fed as
                    # source IDs that decode to the same bytes, so it is skipped (counted) in rank order.
                    if unicodedata.normalize("NFC", item["text"]) != item["text"]:
                        rejected += 1
                        continue
                    taken.append(item)
            nfc_rejected[role][c] = rejected
            built[role].extend(taken)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = {}
    for role in ROLES:
        items, id_rows = [], []
        for item in built[role]:
            item["schema"] = SCHEMA
            item.pop("item_sha256")
            item["item_sha256"] = sha(bdh._canonical(item))
            ids = tokenize(tok, item["text"])
            items.append(item)
            id_rows.append({"item_sha256": item["item_sha256"], "category": item["category"],
                            "n_tokens": len(ids), "span_byte_count": item["span_byte_count"], "ids": ids})
        text_bytes = b"".join(bdh._canonical(i) + b"\n" for i in items)
        ids_bytes = b"".join(json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in id_rows)
        (out / f"{role.lower()}.jsonl").write_bytes(text_bytes)
        (out / f"{role.lower()}_ids.jsonl").write_bytes(ids_bytes)
        parts[role] = {
            "items": len(items),
            "category_counts": dict(Counter(i["category"] for i in items)),
            "span_bytes_total": sum(i["span_byte_count"] for i in items),
            "tokens_total": sum(r["n_tokens"] for r in id_rows),
            "tokens_by_category": {c: sum(r["n_tokens"] for r in id_rows if r["category"] == c)
                                   for c in bdh.CATEGORIES},
            "jsonl_sha256": sha(text_bytes), "ids_jsonl_sha256": sha(ids_bytes),
            "items_aggregate_sha256": sha("".join(i["item_sha256"] + "\n" for i in items).encode()),
        }
    check_disjoint(built)
    manifest = {
        "schema": SCHEMA, "config": CONFIG,
        "document_machinery": {"path": BDH_PATH, "commit": BDH_COMMIT, "blob_sha256": bdh_sha},
        "tokenizer": {"path": args.tokenizer, "sha256": sha(tok_raw), "add_special_tokens": False,
                      "roundtrip": "decode(ids) == span for every item"},
        "role_rule": "ROLES[int(sha256(role_salt + ':' + content_sha256)[:16], 16) % 3], all categories; "
                     "PG19 publisher split ignored",
        "exclusions": {"sources": ex_record, "excluded_unique_documents": len(excluded)},
        "source_discovery_counts": dict(sorted(counters.items())),
        "planning": {"eligibility": eligibility, "deduplication": dedup, "availability": availability,
                     "nfc_unstable_spans_skipped": nfc_rejected},
        "parts": parts,
        "source_files": [{"path": p, "sha256": h} for p, h in sorted(file_hashes.items())],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    for role in ROLES:
        print(role, json.dumps(parts[role]["category_counts"]), "tokens", parts[role]["tokens_total"],
              "bytes", parts[role]["span_bytes_total"])
    print("excluded earlier-holdout documents:", len(excluded))


def check_disjoint(chosen: dict) -> None:
    for a in ROLES:
        for b in ROLES:
            if a < b:
                ia = {i["source_document_id"] for i in chosen[a]} | {i["source_content_sha256"] for i in chosen[a]}
                ib = {i["source_document_id"] for i in chosen[b]} | {i["source_content_sha256"] for i in chosen[b]}
                if ia & ib:
                    raise SystemExit(f"leakage between {a} and {b}")


def verify(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer
    out = Path(args.out_dir)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    tok_raw = Path(args.tokenizer).read_bytes()
    if sha(tok_raw) != m["tokenizer"]["sha256"]:
        raise SystemExit("tokenizer hash differs from the manifest")
    tok = Tokenizer.from_str(tok_raw.decode("utf-8"))
    seen = {}
    for role in ROLES:
        tb = (out / f"{role.lower()}.jsonl").read_bytes()
        ib = (out / f"{role.lower()}_ids.jsonl").read_bytes()
        if sha(tb) != m["parts"][role]["jsonl_sha256"] or sha(ib) != m["parts"][role]["ids_jsonl_sha256"]:
            raise SystemExit(f"{role}: file hash differs from the manifest")
        items = [json.loads(x) for x in tb.decode().splitlines()]
        rows = [json.loads(x) for x in ib.decode().splitlines()]
        for it, r in zip(items, rows, strict=True):
            if it["item_sha256"] != r["item_sha256"] or tokenize(tok, it["text"]) != r["ids"]:
                raise SystemExit(f"{role}: ids do not match the span of {it['source_document_id']}")
            for key in (it["source_document_id"], it["source_content_sha256"]):
                if seen.setdefault(key, role) != role:
                    raise SystemExit(f"leakage: {key} in {seen[key]} and {role}")
    print("verify PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--exclude", nargs="*", default=[])
    v = sub.add_parser("verify")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--tokenizer", required=True)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else verify(args)


if __name__ == "__main__":
    main()
