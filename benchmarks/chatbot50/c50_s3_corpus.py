"""C50-S3 corpus v1: the text the teacher (F0) scores, 50 M tokens in three equal parts, from pinned sources.

Decided on 20 October 2026 (ROADMAP §12): option (a), a mix shaped on S2's gap map, 50 M tokens.
- wiki: English Wikipedia (wikimedia/wikipedia 20231101.en, shard 0 of 41): knowledge (nq, tqa).
- edu: FineWeb-Edu (sample/10BT, file 000_00000): school knowledge (mmlu, arc).
- math: worked solutions, GSM8K train (all of it, first) then MetaMathQA: reasoning (gsm8k).

Rules (fixed in the S3 protocol, before any teacher data is kept):
- Items are walked in sha256("<SEED>:<source>:<key>") order, except GSM8K train, which comes first in its own hash order.
- Text: wiki "title\\n\\ntext"; edu "text"; math "Question: q\\nAnswer: a" (GSM8K's <<...>> calculator notes removed).
  Tokenized with Ling-mini-2.0's tokenizer, no special tokens (as every C50 panel).
- Decontamination: an item is dropped whole if any of its 13-token windows with >= 8 tokens holding a letter (digits
  are one token each, so number runs are not evidence) occurs in any evaluation input: every
  `*/inputs/*.tsv` of the campaign (A4, A4-bis, A6 DEV, S2: nq, tqa, gsm8k, arc, mmlu, piqa, ifeval, chat, rollout,
  docs) and the c50_holdout_v1 / c50_holdout_ext_v1 id files.  This covers S2's donor-gap panel, a subset of them.
- Chunks: an item's ids are cut into consecutive windows of CHUNK tokens; a last window shorter than MIN_CHUNK is
  dropped.  A source stops at the first item that takes it to its share (TARGET // 3 tokens).
- Order of the output: the three sources interleaved by tokens: the next chunk always comes from the source with the
  fewest tokens written so far (ties in the order wiki, edu, math), so that any prefix the teacher has finished holds
  the three parts in near-equal token shares.

Output: <out>/teacher.tsv (the teacher tool's input: "doc<TAB>ids"), <out>/meta.jsonl (source, key, chunk, tokens),
and a manifest with every hash and count.

Usage:
    python -I benchmarks/chatbot50/c50_s3_corpus.py --out E:/.../s3/corpus_v1 \
        --manifest-out benchmarks/chatbot50/results/c50_s3_corpus_v1_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

SEED = 20261016
TARGET = 50_000_000
CHUNK, MIN_CHUNK, NGRAM, MIN_LETTER_TOKENS = 2048, 64, 13, 8
CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
TOKENIZER = CAMPAIGN / "sources/Ling-mini-2.0_a810f64/tokenizer.json"
SRC = CAMPAIGN / "s3/sources"
FILES = {
    "wiki": ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.en/train-00000-of-00041.parquet",
             "382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677"),
    "edu": ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "sample/10BT/000_00000.parquet",
            "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/train-00000-of-00001.parquet",
              "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"),
    "metamath": ("meta-math/MetaMathQA", "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18", "MetaMathQA-395K.json",
                 "fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725"),
}
HOLDOUT = Path(__file__).resolve().parent / "corpus"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def key_order(source: str, keys: list[str]) -> list[int]:
    h = [hashlib.sha256(f"{SEED}:{source}:{k}".encode("utf-8")).hexdigest() for k in keys]
    return sorted(range(len(keys)), key=h.__getitem__)


def letter_table(tok: Tokenizer) -> np.ndarray:
    """True for the ids whose decoded text contains a letter.  The tokenizer splits digits one per token, so a window
    of 13 tokens can be a date range or a list of numbers; such windows are not evidence of overlap."""
    return np.array([any(c.isalpha() for c in tok.decode([i])) for i in range(tok.get_vocab_size())], dtype=bool)


def ngram_hashes(ids: np.ndarray, letters: np.ndarray) -> np.ndarray:
    """64-bit polynomial hash (wrapping uint64 arithmetic) of every NGRAM window holding at least MIN_LETTER_TOKENS
    tokens with a letter."""
    n = len(ids) - NGRAM + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    x = ids.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(NGRAM):
            h = h * np.uint64(1_000_003) + x[j:n + j]
    c = np.concatenate([[0], np.cumsum(letters[ids])])
    return h[(c[NGRAM:] - c[:n]) >= MIN_LETTER_TOKENS]


def eval_sequences() -> tuple[list[np.ndarray], list[str]]:
    seqs, files = [], []
    for p in sorted(CAMPAIGN.glob("*/inputs/*.tsv")):
        files.append(str(p))
        for line in p.read_text(encoding="ascii").splitlines():
            if "\t" in line:
                seqs.append(np.array(line.split("\t", 1)[1].split(","), dtype=np.int64))
    for p in sorted(HOLDOUT.glob("*/*_ids.jsonl")):
        files.append(str(p))
        for line in p.read_text(encoding="utf-8").splitlines():
            seqs.append(np.array(json.loads(line)["ids"], dtype=np.int64))
    return seqs, files


def items(source: str):
    """Yield (key, text) in the source's fixed order."""
    if source == "wiki":
        t = pq.read_table(SRC / "wikimedia__wikipedia" / FILES["wiki"][2], columns=["id", "title", "text"])
        ids, title, text = t.column("id").to_pylist(), t.column("title"), t.column("text")
        for i in key_order("wiki", ids):
            yield ids[i], title[i].as_py() + "\n\n" + text[i].as_py()
    elif source == "edu":
        t = pq.read_table(SRC / "HuggingFaceFW__fineweb-edu" / FILES["edu"][2], columns=["id", "text"])
        ids, text = t.column("id").to_pylist(), t.column("text")
        for i in key_order("edu", ids):
            yield ids[i], text[i].as_py()
    else:
        g = pq.read_table(SRC / "openai__gsm8k" / FILES["gsm8k"][2]).to_pydict()
        for i in key_order("gsm8k", [str(i) for i in range(len(g["question"]))]):
            yield f"gsm8k:{i}", "Question: " + g["question"][i] + "\nAnswer: " + re.sub(r"<<[^>]*>>", "", g["answer"][i])
        m = json.loads((SRC / "meta-math__MetaMathQA" / FILES["metamath"][2]).read_text(encoding="utf-8"))
        for i in key_order("metamath", [str(i) for i in range(len(m))]):
            yield f"metamath:{i}", "Question: " + m[i]["query"] + "\nAnswer: " + m[i]["response"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()
    for name, (repo, rev, fn, pin) in FILES.items():
        got = sha256_file(SRC / repo.replace("/", "__") / fn)
        if got != pin:
            raise SystemExit(f"{name}: sha256 {got} != pin {pin}")
    tok = Tokenizer.from_file(str(TOKENIZER))
    seqs, eval_files = eval_sequences()
    letters = letter_table(tok)
    bad = np.unique(np.concatenate([ngram_hashes(s, letters) for s in seqs]))
    print(time.strftime("%H:%M:%S"), f"eval: {len(eval_files)} files, {len(seqs)} sequences, {len(bad)} distinct {NGRAM}-grams",
          flush=True)
    share = TARGET // 3
    chunks: dict[str, list[tuple[str, int, np.ndarray]]] = {}
    stats = {}
    for source in ("wiki", "edu", "math"):
        out, n_tok, seen, dropped, short = [], 0, 0, 0, 0
        for key, text in items(source):
            seen += 1
            ids = np.array(tok.encode(text, add_special_tokens=False).ids, dtype=np.int64)
            h = ngram_hashes(ids, letters)
            if len(h) and (bad[np.minimum(np.searchsorted(bad, h), len(bad) - 1)] == h).any():
                dropped += 1
                continue
            k = 0
            for s in range(0, len(ids), CHUNK):
                w = ids[s:s + CHUNK]
                if len(w) < MIN_CHUNK:
                    short += 1
                    continue
                out.append((str(key), k, w))
                k += 1
                n_tok += len(w)
            if n_tok >= share:
                break
        chunks[source] = out
        stats[source] = {"items_seen": seen, "items_dropped_contaminated": dropped, "windows_dropped_short": short,
                         "chunks": len(out), "tokens": n_tok, "positions": n_tok - len(out)}
        print(time.strftime("%H:%M:%S"), source, stats[source], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    order, nxt, written = [], {s: 0 for s in chunks}, {s: 0 for s in chunks}
    while True:
        live = [s for s in ("wiki", "edu", "math") if nxt[s] < len(chunks[s])]
        if not live:
            break
        s = min(live, key=lambda s: written[s])
        key, k, w = chunks[s][nxt[s]]
        order.append((s, key, k, w))
        nxt[s] += 1
        written[s] += len(w)
    with open(a.out / "teacher.tsv", "w", encoding="ascii", newline="\n") as ft, \
            open(a.out / "meta.jsonl", "w", encoding="utf-8", newline="\n") as fm:
        for d, (source, key, k, w) in enumerate(order):
            ft.write(f"{d}\t{','.join(map(str, w.tolist()))}\n")
            fm.write(json.dumps({"doc": d, "source": source, "key": key, "chunk": k, "tokens": len(w)}) + "\n")
    man = {
        "cell": "C50-S3 corpus v1", "seed": SEED, "target_tokens": TARGET, "chunk": CHUNK, "min_chunk": MIN_CHUNK,
        "ngram": NGRAM, "min_letter_tokens": MIN_LETTER_TOKENS, "tokenizer": {"path": str(TOKENIZER), "sha256": sha256_file(TOKENIZER),
                                      "add_special_tokens": False},
        "sources": {k: {"repo": v[0], "revision": v[1], "file": v[2], "sha256": v[3]} for k, v in FILES.items()},
        "eval_files": eval_files, "eval_sequences": len(seqs), "eval_ngrams": int(len(bad)), "stats": stats,
        "docs": len(order), "tokens": int(sum(len(o[3]) for o in order)),
        "positions": int(sum(len(o[3]) - 1 for o in order)),
        "teacher_tsv_sha256": sha256_file(a.out / "teacher.tsv"), "meta_sha256": sha256_file(a.out / "meta.jsonl"),
        "wall_s": round(time.time() - t0, 1),
    }
    a.manifest_out.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print("S3 CORPUS", man["docs"], "docs", man["tokens"], "tokens", man["teacher_tsv_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
