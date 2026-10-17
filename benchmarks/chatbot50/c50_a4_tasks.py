"""C50-A4 task inputs: source-tokenizer IDs for every A4 panel, written once and hashed.

Panels (sizes are arguments; the A4 protocol fixes them):
  docs     FIT documents outside the imatrix sample: per category, items 25..24+N in role order
                                                           -> docs.tsv           (c50_llama_score)
  piqa     PIQA validation, original order, 2 options      -> piqa.tsv + meta     (c50_llama_score)
  arc      ARC-Challenge test, original order, 3-5 options -> arc.tsv + meta      (c50_llama_score)
  gsm8k    GSM8K test, first N in file order, chat format  -> gsm8k.tsv + meta    (c50_llama_generate)
  ifeval   IFEval, single-instruction items of the checked types, file order, chat format
                                                           -> ifeval.tsv + meta   (c50_llama_generate)
  chat     the frozen IT/EN chat panel (a4_chat_panel_v1.json), chat format
                                                           -> chat.tsv + meta     (c50_llama_generate)
  rollout  the same non-imatrix FIT documents, items 25..24+N per category, first 256 tokens as
           prompt                                          -> rollout.tsv + meta  (c50_llama_generate)
The imatrix was computed on the first 24 FIT items of each category (A2/A2b addendum 1), so those
documents are excluded from every A4 panel: P0 must not be judged on its own calibration text.

Likelihood panels (piqa, arc) score each option as one document "prefix + suffix" and read the
suffix positions from the scorer's records (prefix and suffix tokenized separately, no special
token, as in STRAT-01).  Ling-mini adds no BOS (tokenizer add_bos_token False), so the first prefix
token is context only.  Chat prompts use the source chat template with thinking off, rendered in
a sandboxed Jinja environment, one user turn plus the generation prompt.

Usage:
    python -I benchmarks/chatbot50/c50_a4_tasks.py --out E:/.../a4/inputs --docs-per-category 16 \
        --gsm8k 100 --rollout-per-category 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
SRC = CAMPAIGN / "sources" / "Ling-mini-2.0_a810f64"
TASKS = CAMPAIGN / "sources" / "tasks_20261017"
HOLDOUT = HERE / "corpus" / "c50_holdout_v1"
PIQA = HERE.parent / "donor_adaptation" / "density" / "results" / "strat01_task_sources_20260919"
PINNED = {  # sha256 of every task source file
    "piqa_valid.jsonl": "93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d",
    "piqa_valid-labels.lst": "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb",
    "gsm8k_test.parquet": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    "arc_challenge_test.parquet": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
    "ifeval_input_data.jsonl": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
}
PATHS = {
    "piqa_valid.jsonl": PIQA / "piqa_valid.jsonl",
    "piqa_valid-labels.lst": PIQA / "piqa_valid-labels.lst",
    "gsm8k_test.parquet": TASKS / "openai__gsm8k" / "main" / "test-00000-of-00001.parquet",
    "arc_challenge_test.parquet": TASKS / "allenai__ai2_arc" / "ARC-Challenge" / "test-00000-of-00001.parquet",
    "ifeval_input_data.jsonl": TASKS / "google__IFEval" / "ifeval_input_data.jsonl",
}
# IFEval instruction types checked by c50_a4_checks.py (deterministic, no language model, no language ID)
IFEVAL_TYPES = {
    "punctuation:no_comma", "change_case:english_lowercase", "change_case:english_capital",
    "startend:quotation", "startend:end_checker", "detectable_format:json_format", "detectable_format:title",
    "keywords:forbidden_words", "keywords:existence", "detectable_content:postscript",
    "detectable_format:number_bullet_lists", "detectable_format:constrained_response",
    "combination:repeat_prompt",
}
GSM8K_SUFFIX = "\n\nSolve the problem step by step. End with a final line of the form \"#### <number>\"."
IMATRIX_PER_CATEGORY = 24  # A2/A2b addendum 1
STOP_IDS = [156892, 156895]  # <|endoftext|>, <|role_end|> (generation_config eos_token_id)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs-per-category", type=int, required=True)
    ap.add_argument("--gsm8k", type=int, required=True)
    ap.add_argument("--rollout-per-category", type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, path in PATHS.items():
        if sha256_bytes(path.read_bytes()) != PINNED[name]:
            raise SystemExit(f"{name}: sha256 differs from the pin")
    tok = Tokenizer.from_file(str(SRC / "tokenizer.json"))
    cfg = json.loads((SRC / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False).from_string(cfg["chat_template"])

    def enc(text: str) -> list[int]:
        ids = tok.encode(text, add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != text:
            raise SystemExit(f"round trip failed: {text[:80]!r}")
        return ids

    def chat(user: str) -> tuple[str, list[int]]:
        text = template.render(messages=[{"role": "user", "content": user}], add_generation_prompt=True, tools=None)
        return text, tok.encode(text, add_special_tokens=False).ids

    manifest = {"cell": "C50-A4 inputs", "tokenizer_sha256": sha256_bytes((SRC / "tokenizer.json").read_bytes()),
                "chat_template_sha256": sha256_bytes(cfg["chat_template"].encode("utf-8")), "stop_ids": STOP_IDS,
                "sources": PINNED, "args": vars(args), "files": {}}

    def write(name: str, lines: list[str], meta: list[dict]) -> None:
        data = "".join(lines).encode("ascii")
        (out / f"{name}.tsv").write_bytes(data)
        mdata = ("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta)).encode("utf-8")
        (out / f"{name}_meta.jsonl").write_bytes(mdata)
        manifest["files"][name] = {"tsv_sha256": sha256_bytes(data), "meta_sha256": sha256_bytes(mdata),
                                   "lines": len(lines), "items": len(meta)}
        print(name, len(meta), "items", len(lines), "lines", manifest["files"][name]["tsv_sha256"])

    # docs and rollout from the FIT role
    rows = [json.loads(x) for x in (HOLDOUT / "fit_ids.jsonl").read_text().splitlines()]

    def after_imatrix(n: int) -> list[dict]:
        seen, taken = {}, []
        for r in rows:
            seen[r["category"]] = seen.get(r["category"], 0) + 1
            if IMATRIX_PER_CATEGORY < seen[r["category"]] <= IMATRIX_PER_CATEGORY + n:
                taken.append(r)
        if len(taken) != 3 * n:
            raise SystemExit(f"fewer than {n} non-imatrix FIT items in some category")
        return taken

    docs = after_imatrix(args.docs_per_category)
    write("docs", [f"{i}\t{','.join(map(str, r['ids']))}\n" for i, r in enumerate(docs)],
          [{"index": i, "category": r["category"], "item_sha256": r["item_sha256"], "n_tokens": r["n_tokens"]}
           for i, r in enumerate(docs)])
    roll = after_imatrix(args.rollout_per_category)
    write("rollout", [f"{i}\t{','.join(map(str, r['ids'][:256]))}\n" for i, r in enumerate(roll)],
          [{"index": i, "category": r["category"], "item_sha256": r["item_sha256"]} for i, r in enumerate(roll)])

    # PIQA: prefix "Question: {goal}\nAnswer:", suffixes " {sol1}", " {sol2}"
    goals = [json.loads(x) for x in PATHS["piqa_valid.jsonl"].read_text(encoding="utf-8").splitlines() if x.strip()]
    labels = [int(x) for x in PATHS["piqa_valid-labels.lst"].read_text().split()]
    if len(goals) != 1838 or len(labels) != 1838:
        raise SystemExit("PIQA size differs from 1,838")
    lines, meta = [], []
    for i, (g, lab) in enumerate(zip(goals, labels)):
        pre = enc(f"Question: {g['goal']}\nAnswer:")
        opts = [enc(f" {g['sol1']}"), enc(f" {g['sol2']}")]
        for k, suf in enumerate(opts):
            lines.append(f"{i * 8 + k}\t{','.join(map(str, pre + suf))}\n")
        meta.append({"item": i, "label": lab, "prefix_len": len(pre), "suffix_lens": [len(s) for s in opts]})
    write("piqa", lines, meta)

    # ARC-Challenge: prefix "Question: {q}\nAnswer:", suffix " {choice}"
    arc = pq.read_table(PATHS["arc_challenge_test.parquet"]).to_pylist()
    lines, meta = [], []
    for i, a in enumerate(arc):
        pre = enc(f"Question: {a['question']}\nAnswer:")
        texts, labs = a["choices"]["text"], a["choices"]["label"]
        if a["answerKey"] not in labs or len(labs) > 8:
            raise SystemExit(f"ARC item {i}: answer key or option count")
        opts = [enc(f" {t}") for t in texts]
        for k, suf in enumerate(opts):
            lines.append(f"{i * 8 + k}\t{','.join(map(str, pre + suf))}\n")
        meta.append({"item": i, "id": a["id"], "label": labs.index(a["answerKey"]), "prefix_len": len(pre),
                     "suffix_lens": [len(s) for s in opts]})
    write("arc", lines, meta)

    # GSM8K: first N in file order
    gsm = pq.read_table(PATHS["gsm8k_test.parquet"]).to_pylist()[: args.gsm8k]
    lines, meta = [], []
    for i, g in enumerate(gsm):
        text, ids = chat(g["question"] + GSM8K_SUFFIX)
        lines.append(f"{i}\t{','.join(map(str, ids))}\n")
        meta.append({"item": i, "answer": g["answer"].split("####")[-1].strip().replace(",", ""), "prompt_tokens": len(ids)})
    write("gsm8k", lines, meta)

    # IFEval: single-instruction items of the checked types, file order
    ife = [json.loads(x) for x in PATHS["ifeval_input_data.jsonl"].read_text(encoding="utf-8").splitlines() if x.strip()]
    lines, meta = [], []
    for r in ife:
        if len(r["instruction_id_list"]) == 1 and r["instruction_id_list"][0] in IFEVAL_TYPES:
            text, ids = chat(r["prompt"])
            lines.append(f"{len(meta)}\t{','.join(map(str, ids))}\n")
            meta.append({"item": len(meta), "key": r["key"], "instruction": r["instruction_id_list"][0],
                         "kwargs": r["kwargs"][0], "prompt": r["prompt"], "prompt_tokens": len(ids)})
    write("ifeval", lines, meta)

    # IT/EN chat panel
    panel = json.loads((HERE / "a4_chat_panel_v1.json").read_text(encoding="utf-8"))
    manifest["chat_panel_sha256"] = sha256_bytes((HERE / "a4_chat_panel_v1.json").read_bytes())
    lines, meta = [], []
    for i, p in enumerate(panel["items"]):
        text, ids = chat(p["prompt"])
        lines.append(f"{i}\t{','.join(map(str, ids))}\n")
        meta.append({"item": i, **p, "prompt_tokens": len(ids)})
    write("chat", lines, meta)
    manifest["chat_example_rendering"] = chat("EXAMPLE")[0]

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
