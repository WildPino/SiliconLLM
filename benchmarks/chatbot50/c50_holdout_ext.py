"""C50 holdout extension (A5 part 5): more items in the same rank order as c50_holdout_v1.

Reruns the c50_holdout_v1 selection (pinned document machinery, same seed, salt, exclusions, span rule and
NFC skip) with larger counts, then:
- asserts that the regenerated FIT ranks 1-48 and DEV ranks 1-32 have exactly the item_sha256 of the
  committed fit.jsonl / dev.jsonl (the extension cannot have moved an assigned item);
- writes SEL = FIT ranks 49-248 per category and DEV-B = DEV ranks 33-44 per category, as text JSONL, IDs
  JSONL and engine ids.tsv ("<index>\\t<ids>"), plus a manifest of hashes.
Outputs go under --out-dir (gitignored); the manifest is the tracked record.
A4-bis adds the part FRESH = FIT ranks 249-272 per category (--parts fresh). Its proof check compares the
regenerated FIT ranks 1-248 with the committed fit.jsonl plus the SEL items of c50_holdout_ext_v1, whose
jsonl hash must equal the tracked ext_v1 manifest.

Usage:
    python -I benchmarks/chatbot50/c50_holdout_ext.py --out-dir benchmarks/chatbot50/corpus/c50_holdout_ext_v1 \
        --tokenizer <checkpoint>/tokenizer.json --manifest-out benchmarks/chatbot50/results/c50_holdout_ext_v1_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_holdout as H  # noqa: E402

PARTS = {"sel": ("FIT", 48, 248), "devb": ("DEV", 32, 44),    # (role, already assigned ranks, last rank)
         "fresh": ("FIT", 248, 272)}
EXT_V1_MANIFEST = Path(__file__).resolve().parent / "results" / "c50_holdout_ext_v1_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--parts", default="sel,devb")
    ap.add_argument("--ext-v1-dir", default=str(Path(__file__).resolve().parent / "corpus" / "c50_holdout_ext_v1"))
    args = ap.parse_args()
    from tokenizers import Tokenizer
    base = H.HERE / "corpus" / "c50_holdout_v1"
    man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    excl = [str(H.REPO / s["path"]) for s in man["exclusions"]["sources"]]
    bdh, _ = H.load_bdh()
    tok = Tokenizer.from_str(Path(args.tokenizer).read_bytes().decode("utf-8"))
    ex_ids, ex_contents, ex_record = H.read_exclusions(excl)
    if [r["jsonl_sha256"] for r in ex_record] != [s["jsonl_sha256"] for s in man["exclusions"]["sources"]]:
        raise SystemExit("exclusion files differ from the c50_holdout_v1 manifest")
    refs, _ = bdh.discover_documents()
    eligible, _ = bdh._eligible_refs(refs, H.CONFIG["min_source_bytes"])
    unique, _ = bdh._deduplicate_globally(eligible)
    pool = [r for r in unique if not (r.source_document_id in ex_ids or r.source_content_sha256 in ex_contents)]
    pools = {role: {c: [] for c in bdh.CATEGORIES} for role in H.ROLES}
    for r in pool:
        pools[H.role_of(r.source_content_sha256)][r.category].append(r)
    span_cfg = {"seed": H.CONFIG["seed"], "span_bytes": H.CONFIG["span_bytes"], "min_source_bytes": H.CONFIG["min_source_bytes"]}
    committed = {role: [json.loads(x)["item_sha256"] for x in (base / f"{role.lower()}.jsonl").read_text(encoding="utf-8").splitlines()]
                 for role in ("FIT", "DEV")}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {"schema": "c50_holdout_ext_v1", "base_manifest": "c50_holdout_v1", "parts": {}}
    if "fresh" in args.parts.split(","):
        sel_bytes = (Path(args.ext_v1_dir) / "sel.jsonl").read_bytes()
        if H.sha(sel_bytes) != json.loads(EXT_V1_MANIFEST.read_text(encoding="utf-8"))["parts"]["sel"]["jsonl_sha256"]:
            raise SystemExit("ext_v1 sel.jsonl differs from its tracked manifest")
        fit = [json.loads(x) for x in (base / "fit.jsonl").read_text(encoding="utf-8").splitlines()]
        sel = [json.loads(x) for x in sel_bytes.decode("utf-8").splitlines()]
        committed["FIT248"] = [i["item_sha256"] for c in bdh.CATEGORIES for i in fit + sel if i["category"] == c]
    for part, (role, assigned, last) in [(k, v) for k, v in PARTS.items() if k in args.parts.split(",")]:
        regenerated_first, items = [], []
        for c in bdh.CATEGORIES:
            cand = sorted(pools[role][c], key=lambda r: (bdh._selection_key(r, role, H.CONFIG["seed"]), r.source_document_id))
            taken, start = [], 0
            while len(taken) < last:
                batch = cand[start:start + 2 * last]
                if not batch:
                    raise SystemExit(f"insufficient {role}/{c}")
                start += len(batch)
                texts, hashes = bdh._hydrate_selected(batch)
                for r in batch:
                    if len(taken) == last:
                        break
                    item = bdh._item(r, role, texts[r.source_document_id], hashes[r.source_path], span_cfg)
                    if unicodedata.normalize("NFC", item["text"]) != item["text"]:
                        continue
                    item["schema"] = H.SCHEMA
                    item.pop("item_sha256")
                    item["item_sha256"] = H.sha(bdh._canonical(item))
                    taken.append(item)
            regenerated_first += [t["item_sha256"] for t in taken[:assigned]]
            for rank, t in enumerate(taken[assigned:], start=assigned + 1):
                t["rank"] = rank
                items.append(t)
        if regenerated_first != committed["FIT248" if part == "fresh" else role]:
            raise SystemExit(f"{role}: regenerated ranks 1-{assigned} differ from the committed holdout")
        rows, tsv = [], []
        for i, t in enumerate(items):
            ids = H.tokenize(tok, t["text"])
            rows.append({"item_sha256": t["item_sha256"], "category": t["category"], "rank": t["rank"], "ids": ids})
            tsv.append(f"{i}\t" + ",".join(map(str, ids)))
        text_bytes = b"".join(bdh._canonical(t) + b"\n" for t in items)
        ids_bytes = b"".join(json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in rows)
        (out / f"{part}.jsonl").write_bytes(text_bytes)
        (out / f"{part}_ids.jsonl").write_bytes(ids_bytes)
        (out / f"{part}_ids.tsv").write_text("\n".join(tsv) + "\n", encoding="ascii")
        record["parts"][part] = {"role": role, "ranks": [assigned + 1, last], "items": len(items),
                                 "tokens": sum(len(r["ids"]) for r in rows),
                                 "jsonl_sha256": H.sha(text_bytes), "ids_jsonl_sha256": H.sha(ids_bytes),
                                 "ids_tsv_sha256": H.sha((out / f"{part}_ids.tsv").read_bytes()),
                                 "proof_check": f"regenerated {role} ranks 1-{assigned} equal the committed item_sha256"}
        print(part, record["parts"][part]["items"], "items", record["parts"][part]["tokens"], "tokens; proof check ok")
    Path(args.manifest_out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
