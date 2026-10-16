"""Write a scorer input (one document per line: "<index>\t<id>,<id>,...") from a C50 holdout role.

The subset is fixed by rule, never by inspection: the first N items of each category in
the role file's order (category, then seeded rank).  Prints the item hashes it used.

Usage:
    python -I benchmarks/chatbot50/c50_make_ids.py --holdout benchmarks/chatbot50/corpus/c50_holdout_v1 \
        --role fit --per-category 4 --out ids.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--role", required=True, choices=["fit", "dev", "reserved"])
    ap.add_argument("--per-category", type=int, default=0, help="0 = every item")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in (Path(args.holdout) / f"{args.role}_ids.jsonl").read_text().splitlines()]
    taken, seen = [], {}
    for r in rows:
        seen[r["category"]] = seen.get(r["category"], 0) + 1
        if args.per_category == 0 or seen[r["category"]] <= args.per_category:
            taken.append(r)
    lines = "".join(f"{i}\t{','.join(map(str, r['ids']))}\n" for i, r in enumerate(taken))
    Path(args.out).write_text(lines, encoding="ascii", newline="\n")
    print(f"{len(taken)} documents, {sum(r['n_tokens'] for r in taken)} tokens, "
          f"{sum(r['n_tokens'] - 1 for r in taken)} scored positions; file sha256 "
          f"{hashlib.sha256(lines.encode()).hexdigest()}")
    for i, r in enumerate(taken):
        print(i, r["category"], r["item_sha256"], r["n_tokens"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
