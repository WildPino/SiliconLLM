"""C50-S4: splits S2's donor-gap panel into an S4 half and an S5 half, before any S4 training (S2 protocol).

Within each stratum (an item panel; for the documents, a document category), the indices are ordered by
sha256("c50-s4-split|<panel>|<index>") and the first ceil(n / 2) go to S4. No score is read.

    python -I benchmarks/chatbot50/c50_s4_split.py --docs-meta E:/.../a4/inputs/docs_meta.jsonl --out results/c50_s4_split.json
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def key(panel: str, i: int) -> str:
    return hashlib.sha256(f"c50-s4-split|{panel}|{i}".encode()).hexdigest()


def split(items: dict, category: dict) -> dict:
    out = {"s4": {}, "s5": {}}
    for p, ix in items.items():
        strata = collections.defaultdict(list)
        for i in ix:
            strata[category[i] if p == "docs" else p].append(i)
        s4 = []
        for _, v in sorted(strata.items()):
            v = sorted(v, key=lambda i: key(p, i))
            s4 += v[:math.ceil(len(v) / 2)]
        out["s4"][p] = sorted(s4)
        out["s5"][p] = sorted(set(ix) - set(s4))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", type=Path, default=HERE / "results/c50_s2_verdict.json")
    ap.add_argument("--docs-meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = json.loads(a.verdict.read_text(encoding="utf-8"))["donor_gap_panel"]["items"]
    category = {}
    with a.docs_meta.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            category[m["index"]] = m["category"]
    rec = {"cell": "C50-S4 panel split", "rule": __doc__.split("\n\n")[1].replace("\n", " "),
           "verdict_sha256": hashlib.sha256(a.verdict.read_bytes()).hexdigest(),
           "docs_meta_sha256": hashlib.sha256(a.docs_meta.read_bytes()).hexdigest(), **split(items, category)}
    rec["sizes"] = {h: {p: len(v) for p, v in rec[h].items()} for h in ("s4", "s5")}
    a.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec["sizes"]))
