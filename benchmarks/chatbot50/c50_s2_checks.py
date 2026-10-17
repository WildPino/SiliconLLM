"""C50-S2 per-item scoring of one arm (numbers only; c50_s2_verdict.py compares the arms).

    nq, tqa (closed-book short answers): the first non-empty line of the response, with a leading "Answer:" removed,
        is normalized as in SQuAD (NFKC, lower case, punctuation deleted, the articles a/an/the removed, whitespace
        collapsed).  match (primary) = some normalized alias occurs in it as a whole-word span; em (reported) = it
        equals a normalized alias.
    mmlu, arc: the A4 likelihood rule (lowest mean NLL over the suffix tokens; ties to the lowest index).
    gsm8k, chat: the A4 checks, unchanged.
The documents are scored in the verdict (paired bits per byte).

Usage:
    python -I benchmarks/chatbot50/c50_s2_checks.py --a4-inputs E:/.../a4/inputs --s2-inputs E:/.../s2/inputs \
        --arm-a4 <dir with arc.pos.bin, gsm8k.jsonl, chat.jsonl> --arm-s2 <dir with nq.jsonl, tqa.jsonl, mmlu.pos.bin> \
        --out arm_scores.json
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_a4_checks as A4  # noqa: E402

ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(ch for ch in s if not (ch in string.punctuation or unicodedata.category(ch).startswith("P")))  # deleted, as SQuAD
    return " ".join(ARTICLES.sub(" ", s).split())


def first_line(text: str) -> str:
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            return re.sub(r"^answer\s*:\s*", "", ln, flags=re.IGNORECASE)
    return ""


def qa_check(text: str, aliases: list[str]) -> dict:
    line = normalize(first_line(text))
    gold = [g for g in (normalize(a) for a in aliases) if g]
    return {"match": any(f" {g} " in f" {line} " for g in gold), "em": line in gold, "line": line}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4-inputs", required=True)
    ap.add_argument("--s2-inputs", required=True)
    ap.add_argument("--arm-a4", required=True)
    ap.add_argument("--arm-s2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a4_in, s2_in, arm4, arm2 = Path(args.a4_inputs), Path(args.s2_inputs), Path(args.arm_a4), Path(args.arm_s2)
    tok = A4.Tokenizer.from_file(A4.TOKENIZER)
    res: dict = {"a4_inputs": args.a4_inputs, "s2_inputs": args.s2_inputs, "arm_a4": args.arm_a4, "arm_s2": args.arm_s2}
    res["arc"] = A4.likelihood(a4_in, arm4, "arc")
    res["mmlu"] = A4.likelihood(s2_in, arm2, "mmlu")
    for name in ("nq", "tqa"):
        rows = A4.gen_rows(arm2, name)
        items = []
        for m in A4.meta(s2_in, name):
            r = rows[m["item"]]
            text = A4.decode(tok, r["gen"])
            c = qa_check(text, m["aliases"])
            items.append({"item": m["item"], "correct": c["match"], "em": c["em"], "line": c["line"], "stop": r["stop"],
                          "generated": len(r["gen"]), "text": text})
        res[name] = {"items": items, "correct": sum(i["correct"] for i in items),
                     "em": sum(i["em"] for i in items), "n": len(items)}
    rows = A4.gen_rows(arm4, "gsm8k")
    items = []
    for m in A4.meta(a4_in, "gsm8k"):
        text = A4.decode(tok, rows[m["item"]]["gen"])
        got = A4.gsm8k_answer(text)
        items.append({"item": m["item"], "answer": m["answer"], "got": got, "correct": A4.num_equal(got, m["answer"])})
    res["gsm8k"] = {"items": items, "correct": sum(i["correct"] for i in items), "n": len(items)}
    rows = A4.gen_rows(arm4, "chat")
    items = []
    for m in A4.meta(a4_in, "chat"):
        r = rows[m["item"]]
        c = A4.chat_check(m, A4.decode(tok, r["gen"]), r["stop"] == "EOS")
        items.append({"item": m["item"], "id": m["id"], "lang": m["lang"], "correct": c["pass"]})
    res["chat"] = {"items": items, "correct": sum(i["correct"] for i in items), "n": len(items)}
    for name in ("arc", "mmlu", "nq", "tqa", "gsm8k", "chat"):
        print(name, res[name]["correct"], "/", res[name]["n"])
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
