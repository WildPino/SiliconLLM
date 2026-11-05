"""Stage 2 test 2b-bis, reported: the documents' dBPB without the ones that share a run of 32 tokens with the preview's corpus or the panel.

The overlap is `s2u_run.py sets`' own rule; this names the documents, which `sets` only counted, and repeats s1_verdict's per-document dBPB and CI.

Usage: python -I benchmarks/stage2/s2u_overlap.py --json benchmarks/stage2/results/s2u_overlap.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE))
import s2u_run as U  # noqa: E402
import s1_verdict as V  # noqa: E402


def hits() -> list[int]:
    import s1t_train as S
    rows = [json.loads(x) for x in (U.CORPUS / "reserved_ids.jsonl").read_text(encoding="utf-8").splitlines()]
    lines = (U.INPUTS["a4"] / "docs.tsv").read_text(encoding="ascii").splitlines()
    seen: set = set()
    tr, he = S.load_corpus(1_500_000, 75_000)
    for s in list(tr) + list(he) + [[int(x) for x in ln.split("\t")[1].split(",")] for ln in lines]:
        s = [int(x) for x in s]
        seen.update(hash(tuple(s[i:i + U.NGRAM])) for i in range(0, max(len(s) - U.NGRAM, 0)))
    return [n for n, r in enumerate(rows) if any(hash(tuple(r["ids"][i:i + U.NGRAM])) in seen for i in range(0, len(r["ids"]) - U.NGRAM))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    from c50_compare import byte_lengths
    import c50_a4_checks as A4
    out_docs = hits()
    r = np.fromfile(U.WORK / "ref" / "docs.pos.bin", dtype=V.REC)
    x = np.fromfile(U.WORK / "two_pass_a" / "docs.pos.bin", dtype=V.REC)
    blen = byte_lengths(Path(A4.TOKENIZER))[r["target"]].astype(np.float64)

    def bpb(z, d):
        m = z["doc"] == d
        return -z["logp_target"][m].astype(np.float64).sum() / np.log(2) / blen[m].sum()
    docs = [int(d) for d in np.unique(r["doc"])]
    dd = {d: bpb(x, d) - bpb(r, d) for d in docs}

    def stat(keep):
        v = np.array([dd[d] for d in keep])
        rng = np.random.default_rng(V.SEED)
        bb = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(V.DRAWS)]
        return {"documents": len(v), "dbpb": float(v.mean()), "dbpb_ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}
    rep = {"overlapping_documents": out_docs, "their_dbpb": {str(d): dd[d] for d in out_docs}, "all": stat(docs), "without_overlap": stat([d for d in docs if d not in out_docs])}
    a.json.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
