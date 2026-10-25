"""C50-T1 part 2b review, addendum 2: fresh general-panel items, never read by any C50 cell, written once and hashed.

The same pinned sources, prompts and formats as S2 (nq, tqa, mmlu) and A4 (arc), on the rows no panel has used:
  nq     NQ-open validation rows 600..1199          (S2 used rows 0..599)
  tqa    TriviaQA rc.nocontext validation 600..1199 (S2 used 0..599)
  mmlu   MMLU all/test, items 11..20 of every subject in file order (S2 used items 1..10)
  arc    ARC-Challenge test rows 600..end            (A4 used rows 0..599; A6 DEV copied A4's)
An item is dropped whole, before anything is scored, if
- any of its input lines has a 13-token window with >= 8 letter tokens that occurs in S3's corpus v1 (the text every
  T1 arm trained on; c50_s3_corpus.py's rule, turned the other way), or
- any of its input lines is identical to a line of the same panel in an earlier input set.
Kept items are numbered 0..n-1 in source order; the meta keeps the source row.

    python -I benchmarks/chatbot50/c50_t1_fresh_tasks.py --out E:/.../t1/fresh/inputs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_a4_tasks as A4T  # noqa: E402
import c50_s2_tasks as S2T  # noqa: E402
from c50_s3_corpus import NGRAM, letter_table, ngram_hashes, sha256_file  # noqa: E402

CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
CORPUS = CAMPAIGN / "s3" / "corpus_v1" / "teacher.tsv"
OLD = {"nq": CAMPAIGN / "s2/inputs", "tqa": CAMPAIGN / "s2/inputs", "mmlu": CAMPAIGN / "s2/inputs",
       "arc": CAMPAIGN / "a4/inputs"}
ROWS = {"nq": (600, 1200), "tqa": (600, 1200), "mmlu": (11, 20), "arc": (600, None)}


def ids_of(line: str) -> np.ndarray:
    return np.array(line.rstrip("\n").split("\t", 1)[1].split(","), dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    srcs = {k: S2T.TASKS / rel for k, (rel, _) in S2T.PINNED.items()}
    pins = {k: sha256_file(p) for k, p in srcs.items()}
    for k, (_, prefix) in S2T.PINNED.items():
        if not pins[k].startswith(prefix):
            raise SystemExit(f"{k}: sha256 differs from S2's pin")
    arc_p = A4T.PATHS["arc_challenge_test.parquet"]
    if sha256_file(arc_p) != A4T.PINNED["arc_challenge_test.parquet"]:
        raise SystemExit("arc: sha256 differs from A4's pin")
    pins["arc_challenge_test.parquet"] = A4T.PINNED["arc_challenge_test.parquet"]
    tok = Tokenizer.from_file(str(S2T.SRC / "tokenizer.json"))
    cfg = json.loads((S2T.SRC / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False).from_string(cfg["chat_template"])
    letters = letter_table(tok)

    def enc(text: str) -> list[int]:
        ids = tok.encode(text, add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != text:
            raise SystemExit(f"round trip failed: {text[:80]!r}")
        return ids

    def chat(user: str) -> list[int]:
        text = template.render(messages=[{"role": "user", "content": user}], add_generation_prompt=True, tools=None)
        return tok.encode(text, add_special_tokens=False).ids

    # the corpus's 13-grams, by c50_s3_corpus.py's rule
    hs = []
    with open(CORPUS, encoding="ascii") as f:
        for line in f:
            hs.append(ngram_hashes(ids_of(line), letters))
    bank = np.unique(np.concatenate(hs))
    del hs
    print(time.strftime("%H:%M:%S"), f"corpus v1: {len(bank)} distinct {NGRAM}-grams", flush=True)

    def in_corpus(seq: list[int]) -> bool:
        h = ngram_hashes(np.array(seq, dtype=np.int64), letters)
        return bool(len(h)) and bool((bank[np.minimum(np.searchsorted(bank, h), len(bank) - 1)] == h).any())

    # planted control: a span of the corpus itself must be flagged, a shuffled copy of it must not
    with open(CORPUS, encoding="ascii") as f:
        span = ids_of(f.readline())[100:164]
    shuffled = np.random.default_rng(0).permutation(span)
    planted = {"corpus_span_flagged": in_corpus(span.tolist()), "shuffled_span_flagged": in_corpus(shuffled.tolist())}
    print("PLANTED", planted, flush=True)
    if not planted["corpus_span_flagged"] or planted["shuffled_span_flagged"]:
        raise SystemExit("the contamination check does not discriminate")

    old_lines = {p: {ln.split("\t", 1)[1] for ln in (d / f"{p}.tsv").read_text(encoding="ascii").splitlines()}
                 for p, d in OLD.items()}
    manifest = {"cell": "C50-T1 addendum 2 fresh inputs", "rows": ROWS, "sources": pins,
                "tokenizer_sha256": sha256_file(S2T.SRC / "tokenizer.json"),
                "chat_template_sha256": S2T.sha256_bytes(cfg["chat_template"].encode("utf-8")),
                "stop_ids": S2T.STOP_IDS, "qa_prompt": S2T.QA_PROMPT, "corpus": str(CORPUS),
                "corpus_sha256": sha256_file(CORPUS), "corpus_ngrams": int(len(bank)),
                "old_inputs": {p: S2T.sha256_bytes((d / "manifest.json").read_bytes()) for p, d in OLD.items()},
                "planted": planted,
                "dropped": {}, "files": {}}

    def write(name: str, cand: list[tuple[int, list[list[int]], dict]]) -> None:
        """cand: (source row, the item's input lines' ids, meta without "item"); lines are numbered here."""
        lines, meta, drop = [], [], {"corpus": [], "duplicate": []}
        mc = name in ("mmlu", "arc")
        for row, seqs, m in cand:
            if any(in_corpus(s) for s in seqs):
                drop["corpus"].append(row)
                continue
            if any(",".join(map(str, s)) in old_lines[name] for s in seqs):
                drop["duplicate"].append(row)
                continue
            i = len(meta)
            for k, s in enumerate(seqs):
                lines.append(f"{i * 8 + k if mc else i}\t{','.join(map(str, s))}\n")
            meta.append({"item": i, "row": row, **m})
        data = "".join(lines).encode("ascii")
        (out / f"{name}.tsv").write_bytes(data)
        mdata = ("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in meta)).encode("utf-8")
        (out / f"{name}_meta.jsonl").write_bytes(mdata)
        manifest["dropped"][name] = drop
        manifest["files"][name] = {"tsv_sha256": S2T.sha256_bytes(data), "meta_sha256": S2T.sha256_bytes(mdata),
                                   "lines": len(lines), "items": len(meta), "candidates": len(cand)}
        print(time.strftime("%H:%M:%S"), name, manifest["files"][name], {k: len(v) for k, v in drop.items()},
              flush=True)

    lo, hi = ROWS["nq"]
    nq = pq.read_table(srcs["nq_open_validation.parquet"]).to_pylist()
    write("nq", [(r, [chat(S2T.QA_PROMPT.format(q=x["question"]))],
                  {"question": x["question"], "aliases": list(x["answer"])}) for r, x in enumerate(nq) if lo <= r < hi])

    lo, hi = ROWS["tqa"]
    tqa = pq.read_table(srcs["trivia_qa_rc_nocontext_validation.parquet"],
                        columns=["question", "question_id", "answer"]).to_pylist()
    write("tqa", [(r, [chat(S2T.QA_PROMPT.format(q=x["question"]))],
                   {"id": x["question_id"], "question": x["question"],
                    "aliases": sorted(set(x["answer"]["aliases"]) | {x["answer"]["value"]})})
                  for r, x in enumerate(tqa) if lo <= r < hi])

    lo, hi = ROWS["mmlu"]
    mm = pq.read_table(srcs["mmlu_all_test.parquet"]).to_pylist()
    seen: dict[str, int] = {}
    cand = []
    for r, x in enumerate(mm):
        seen[x["subject"]] = seen.get(x["subject"], 0) + 1
        if not lo <= seen[x["subject"]] <= hi:
            continue
        pre = enc(f"Question: {x['question']}\nAnswer:")
        opts = [enc(f" {c}") for c in x["choices"]]
        if len(opts) != 4 or not 0 <= x["answer"] < 4:
            raise SystemExit(f"MMLU row {r}: options or answer")
        cand.append((r, [pre + s for s in opts], {"subject": x["subject"], "label": int(x["answer"]),
                                                   "prefix_len": len(pre), "suffix_lens": [len(s) for s in opts]}))
    if len(cand) != 57 * (hi - lo + 1):
        raise SystemExit(f"MMLU: {len(cand)} candidates")
    write("mmlu", cand)

    lo, _ = ROWS["arc"]
    arc = pq.read_table(arc_p).to_pylist()
    cand = []
    for r, x in enumerate(arc):
        if r < lo:
            continue
        pre = enc(f"Question: {x['question']}\nAnswer:")
        texts, labs = x["choices"]["text"], x["choices"]["label"]
        if x["answerKey"] not in labs or len(labs) > 8:
            raise SystemExit(f"ARC row {r}: answer key or option count")
        opts = [enc(f" {t}") for t in texts]
        cand.append((r, [pre + s for s in opts], {"id": x["id"], "label": labs.index(x["answerKey"]),
                                                   "prefix_len": len(pre), "suffix_lens": [len(s) for s in opts]}))
    write("arc", cand)
    manifest["wall_s"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
