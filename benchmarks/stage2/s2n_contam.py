"""S2N: does the 1.6 M teacher pass that e2e_rowx trained on share text with the panels or with its own held-out documents?

The check is the one the amendment-2 review ran on the extension: 13-token windows of the training documents (s2n_train.split: the first pass's first
1.5 M positions) against each target.  Two rules, both reported: the corpus's own (a window counts only with >= 8 tokens holding a letter,
c50_s3_corpus.py, the rule the corpus was decontaminated with against every panel input), and every 13-token window (no letter rule).
Targets: every panel input the stage-1 scores read (A4: gsm8k, arc, chat, docs; S2: nq, tqa, mmlu), and the held-out 'pick' and 'acts' documents.
Per target: items (or documents) hit, positions or windows covered, and the hit items' ids, for re-reading the panels without them.
Numbers only.

Usage: python -I benchmarks/stage2/s2n_contam.py --json benchmarks/stage2/results/s2n_contam.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chatbot50"))
import types  # noqa: E402
for _m in ("pyarrow", "pyarrow.parquet"):          # the builder imports pyarrow for its sources; its helpers here do not need it
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["pyarrow"].parquet = sys.modules["pyarrow.parquet"]
import c50_s3_corpus as K  # noqa: E402

TEACH = Path("E:/SiliconLLM_Artifacts/stage2/s2n/teacher")
CAMP = K.CAMPAIGN
PANELS = {"gsm8k": "a4", "arc": "a4", "chat": "a4", "docs": "a4", "nq": "s2", "tqa": "s2", "mmlu": "s2"}
TRAIN_POS = 1_500_000
META = CAMP / "s3/corpus_v1/meta.jsonl"



def all_windows(ids: np.ndarray, letters: np.ndarray | None) -> np.ndarray:
    """Every window's hash (positions kept), with a mask of those the rule counts."""
    n = len(ids) - K.NGRAM + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=bool)
    x = ids.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(K.NGRAM):
            h = h * np.uint64(1_000_003) + x[j:n + j]
    if letters is None:
        return h, np.ones(n, dtype=bool)
    c = np.concatenate([[0], np.cumsum(letters[ids])])
    return h, (c[K.NGRAM:] - c[:n]) >= K.MIN_LETTER_TOKENS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(K.TOKENIZER))
    letters = K.letter_table(tok)
    docs = []
    for line in (TEACH / "ids.tsv").read_text(encoding="utf-8").splitlines():
        k, v = line.split("\t", 1)
        docs.append((int(k), np.array(v.split(","), dtype=np.int64)))
    train, n = [], 0
    for d in docs:
        if n >= TRAIN_POS:
            break
        train.append(d)
        n += len(d[1]) - 1
    ho = docs[len(train):]
    meta = {}
    with open(META, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= len(docs):
                break
            m = json.loads(line)
            meta[m["doc"]] = f"{m['source']}:{m['key']}:{m['chunk']}"
    targets = {p: [(int(l.split("\t", 1)[0]), np.array(l.split("\t", 1)[1].split(","), dtype=np.int64))
                   for l in (CAMP / f"{s}/inputs/{p}.tsv").read_text(encoding="ascii").splitlines() if "\t" in l]
               for p, s in PANELS.items()}
    targets["heldout_pick"], targets["heldout_acts"] = ho[0::2], ho[1::2]
    out = {"train_docs": len(train), "train_positions": n, "heldout_docs": [len(ho[0::2]), len(ho[1::2])], "ngram": K.NGRAM,
           "min_letter_tokens": K.MIN_LETTER_TOKENS, "rules": {}}
    for rule, lt in (("letters", letters), ("any", None)):
        bank = {}
        for d, ids in train:
            h, m = all_windows(ids, lt)
            for x in np.unique(h[m]):
                bank.setdefault(int(x), d)
        res = {}
        for name, items in targets.items():
            hit, cov_num, cov_den, examples = [], 0, 0, []
            for key, ids in items:
                h, m = all_windows(ids, lt)
                h = h[m]
                if len(h) == 0:
                    continue
                found = np.array([int(x) in bank for x in h])
                cov_den += len(h)
                cov_num += int(found.sum())
                if found.any():
                    hit.append(key)
                    if len(examples) < 5:
                        src = bank[int(h[found][0])]
                        examples.append({"item": key, "covered": round(float(found.mean()), 3), "train_doc": src, "train_source": meta.get(src)})
            res[name] = {"items": len(items), "hit": len(hit), "windows_covered": cov_num / cov_den if cov_den else 0.0,
                         "hit_items": hit, "examples": examples}
        out["rules"][rule] = res
        print(rule, {k: f"{v['hit']}/{v['items']} cov {v['windows_covered']:.4f}" for k, v in res.items()}, flush=True)
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
