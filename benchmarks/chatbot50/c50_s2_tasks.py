"""C50-S2 task inputs for the panels A4 does not have: source-tokenizer IDs, written once and hashed.

Panels (sizes fixed by the S2 protocol):
  nq     NQ-open validation, first N in file order, chat format, short-answer prompt   -> nq.tsv + meta   (generate)
  tqa    TriviaQA rc.nocontext validation, first N in file order, same prompt          -> tqa.tsv + meta  (generate)
  mmlu   MMLU all/test, the first K items of every subject in file order; prefix "Question: {q}\nAnswer:",
         suffix " {choice}" (the A4 ARC format)                                        -> mmlu.tsv + meta (score)
The documents, ARC, GSM8K and chat panels are A4's inputs, used unchanged (their manifest hash is checked by the
runner).  Ling-flash-2.0 at 96f9b7a has the same tokenizer as Ling-mini (equal as canonical JSON; tokenizer_config
and special_tokens_map byte-identical; ROADMAP §12), so one ID file serves every arm.

Usage:
    python -I benchmarks/chatbot50/c50_s2_tasks.py --out E:/.../s2/inputs --nq 600 --tqa 600 --mmlu-per-subject 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
SRC = CAMPAIGN / "sources" / "Ling-mini-2.0_a810f64"
TASKS = CAMPAIGN / "sources" / "tasks_s2_20261017"
PINNED = {
    "nq_open_validation.parquet": ("google-research-datasets__nq_open/nq_open/validation-00000-of-00001.parquet",
                                   "b074bed0bccb"),
    "trivia_qa_rc_nocontext_validation.parquet": ("mandarjoshi__trivia_qa/rc.nocontext/validation-00000-of-00001.parquet",
                                                  "48a5005c0eb4"),
    "mmlu_all_test.parquet": ("cais__mmlu/all/test-00000-of-00001.parquet", "74a41822ce7d"),
}
QA_PROMPT = "Answer the question with a short answer only, without explanation.\nQuestion: {q}"
STOP_IDS = [156892, 156895]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nq", type=int, required=True)
    ap.add_argument("--tqa", type=int, required=True)
    ap.add_argument("--mmlu-per-subject", type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man_src = json.loads((TASKS / "download_manifest.json").read_text(encoding="utf-8"))
    paths = {}
    for name, (rel, prefix) in PINNED.items():
        p = TASKS / rel
        h = sha256_bytes(p.read_bytes())
        if not h.startswith(prefix) or h not in {v["sha256"] for v in man_src.values()}:
            raise SystemExit(f"{name}: sha256 differs from the pin")
        paths[name] = p
    tok = Tokenizer.from_file(str(SRC / "tokenizer.json"))
    cfg = json.loads((SRC / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False).from_string(cfg["chat_template"])

    def enc(text: str) -> list[int]:
        ids = tok.encode(text, add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != text:
            raise SystemExit(f"round trip failed: {text[:80]!r}")
        return ids

    def chat(user: str) -> list[int]:
        text = template.render(messages=[{"role": "user", "content": user}], add_generation_prompt=True, tools=None)
        return tok.encode(text, add_special_tokens=False).ids

    manifest = {"cell": "C50-S2 inputs", "tokenizer_sha256": sha256_bytes((SRC / "tokenizer.json").read_bytes()),
                "chat_template_sha256": sha256_bytes(cfg["chat_template"].encode("utf-8")), "stop_ids": STOP_IDS,
                "qa_prompt": QA_PROMPT, "sources": {k: sha256_bytes(p.read_bytes()) for k, p in paths.items()},
                "args": vars(args), "files": {}}

    def write(name: str, lines: list[str], meta: list[dict]) -> None:
        data = "".join(lines).encode("ascii")
        (out / f"{name}.tsv").write_bytes(data)
        mdata = ("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta)).encode("utf-8")
        (out / f"{name}_meta.jsonl").write_bytes(mdata)
        manifest["files"][name] = {"tsv_sha256": sha256_bytes(data), "meta_sha256": sha256_bytes(mdata),
                                   "lines": len(lines), "items": len(meta)}
        print(name, len(meta), "items", len(lines), "lines", manifest["files"][name]["tsv_sha256"])

    nq = pq.read_table(paths["nq_open_validation.parquet"]).to_pylist()[: args.nq]
    lines, meta = [], []
    for i, r in enumerate(nq):
        ids = chat(QA_PROMPT.format(q=r["question"]))
        lines.append(f"{i}\t{','.join(map(str, ids))}\n")
        meta.append({"item": i, "question": r["question"], "aliases": list(r["answer"]), "prompt_tokens": len(ids)})
    write("nq", lines, meta)

    tqa = pq.read_table(paths["trivia_qa_rc_nocontext_validation.parquet"], columns=["question", "question_id", "answer"]).to_pylist()[: args.tqa]
    lines, meta = [], []
    for i, r in enumerate(tqa):
        ids = chat(QA_PROMPT.format(q=r["question"]))
        aliases = sorted(set(r["answer"]["aliases"]) | {r["answer"]["value"]})
        lines.append(f"{i}\t{','.join(map(str, ids))}\n")
        meta.append({"item": i, "id": r["question_id"], "question": r["question"], "aliases": aliases, "prompt_tokens": len(ids)})
    write("tqa", lines, meta)

    mm = pq.read_table(paths["mmlu_all_test.parquet"]).to_pylist()
    seen: dict[str, int] = {}
    lines, meta = [], []
    for r in mm:
        seen[r["subject"]] = seen.get(r["subject"], 0) + 1
        if seen[r["subject"]] > args.mmlu_per_subject:
            continue
        i = len(meta)
        pre = enc(f"Question: {r['question']}\nAnswer:")
        opts = [enc(f" {c}") for c in r["choices"]]
        if len(opts) != 4 or not 0 <= r["answer"] < 4:
            raise SystemExit(f"MMLU item {i}: options or answer")
        for k, suf in enumerate(opts):
            lines.append(f"{i * 8 + k}\t{','.join(map(str, pre + suf))}\n")
        meta.append({"item": i, "subject": r["subject"], "label": int(r["answer"]), "prefix_len": len(pre),
                     "suffix_lens": [len(s) for s in opts]})
    if len(seen) != 57 or len(meta) != 57 * args.mmlu_per_subject:
        raise SystemExit(f"MMLU: {len(seen)} subjects, {len(meta)} items")
    write("mmlu", lines, meta)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
