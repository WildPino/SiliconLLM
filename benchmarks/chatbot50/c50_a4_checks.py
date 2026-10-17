"""C50-A4 per-item scoring of one arm's raw outputs (numbers only; c50_a4_verdict.py compares the arms).

    likelihood panels (piqa, arc): from c50_llama_score records, the option with the lowest mean NLL
        over its suffix tokens is chosen; exact ties go to the lowest option index.
    gsm8k: the number after the last "####" of the response, else the last number in it; exact
        numeric equality with the reference answer.
    ifeval: the strict IFEval checks for the instruction types listed in c50_a4_tasks.IFEVAL_TYPES,
        transcribed from google-research instruction_following_eval (no language detection).
    chat: keyword (whole word, case- and accent-insensitive) + language (stop-word majority) +
        finished on a stop ID.
    rollout: run32 / loop8x3 / empty on the generated IDs (STRAT-01 definitions); the stop ID, if
        any, is excluded from the flags and recorded as eos.

Usage:
    python -I benchmarks/chatbot50/c50_a4_checks.py --inputs E:/.../a4/inputs --arm E:/.../a4/P0 --out arm_scores.json
(--arm is the prefix directory holding docs.pos.bin, piqa.pos.bin, arc.pos.bin, gsm8k.jsonl,
 ifeval.jsonl, chat.jsonl, rollout.jsonl written by the runner)
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TOKENIZER = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/sources/Ling-mini-2.0_a810f64/tokenizer.json"
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
STOP_IDS = {156892, 156895}
IT_WORDS = {"il", "lo", "la", "gli", "le", "di", "che", "è", "e", "un", "una", "per", "del", "della", "dei",
            "con", "non", "sono", "si", "al", "nel", "nella", "questo", "come", "anche", "più", "ed"}
EN_WORDS = {"the", "is", "of", "and", "an", "to", "that", "it", "was", "for", "with", "are", "on", "by",
            "this", "as", "be", "from", "which", "has"}


def meta(inputs: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (inputs / f"{name}_meta.jsonl").read_text(encoding="utf-8").splitlines()]


def likelihood(inputs: Path, arm: Path, name: str) -> dict:
    recs = np.fromfile(arm / f"{name}.pos.bin", dtype=REC)
    by_doc: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    order = np.argsort(recs["doc"], kind="stable")
    docs, starts = np.unique(recs["doc"][order], return_index=True)
    bounds = list(starts) + [len(order)]
    for k, d in enumerate(docs):
        sl = order[bounds[k]:bounds[k + 1]]
        by_doc[int(d)] = (recs["pos"][sl], recs["logp_target"][sl].astype(np.float64))
    items = []
    for m in meta(inputs, name):
        nll = []
        for k, slen in enumerate(m["suffix_lens"]):
            pos, lp = by_doc[m["item"] * 8 + k]
            sel = pos >= m["prefix_len"]
            if int(sel.sum()) != slen:
                raise SystemExit(f"{name} item {m['item']} option {k}: {int(sel.sum())} suffix positions, expected {slen}")
            nll.append(float(-lp[sel].sum()))
        mean = [n / s for n, s in zip(nll, m["suffix_lens"])]
        choice = int(np.argmin(mean))  # argmin returns the first minimum: ties -> lowest index
        items.append({"item": m["item"], "label": m["label"], "choice": choice, "correct": choice == m["label"],
                      "mean_nll": mean, "total_nll": nll})
    return {"items": items, "correct": sum(i["correct"] for i in items), "n": len(items)}


def gen_rows(arm: Path, name: str) -> dict[int, dict]:
    return {r["item"]: r for r in (json.loads(x) for x in (arm / f"{name}.jsonl").read_text().splitlines() if x)}


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode([i for i in ids if i not in STOP_IDS], skip_special_tokens=True)


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_answer(text: str) -> str | None:
    if "####" in text:
        nums = NUM.findall(text.split("####")[-1])
        pick = nums[0] if nums else None
    else:
        nums = NUM.findall(text)
        pick = nums[-1] if nums else None
    return None if pick is None else pick.replace(",", "").rstrip(".")


def num_equal(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def ifeval_check(inst: str, kw: dict, r: str) -> bool:
    if inst == "punctuation:no_comma":
        return not re.search(r"\,", r)
    if inst == "change_case:english_lowercase":
        return r == r.lower()
    if inst == "change_case:english_capital":
        return r == r.upper()
    if inst == "startend:quotation":
        v = r.strip()
        return len(v) > 1 and v[0] == '"' and v[-1] == '"'
    if inst == "startend:end_checker":
        return r.strip().strip("\"").lower().endswith(kw["end_phrase"].strip().lower())
    if inst == "detectable_format:json_format":
        v = r.strip()
        for p in ("```json", "```Json", "```JSON", "```"):
            if v.startswith(p):
                v = v[len(p):]
                break
        if v.endswith("```"):
            v = v[:-3]
        try:
            json.loads(v.strip())
        except ValueError:
            return False
        return True
    if inst == "detectable_format:title":
        return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", r))
    if inst == "keywords:forbidden_words":
        return not any(re.search(r"\b" + w + r"\b", r, flags=re.IGNORECASE) for w in kw["forbidden_words"])
    if inst == "keywords:existence":
        return all(re.search(k, r, flags=re.IGNORECASE) for k in kw["keywords"])
    if inst == "detectable_content:postscript":
        marker = kw["postscript_marker"].strip()
        if marker.lower() == "p.p.s":
            pat = r"\s*p\.\s?p\.\s?s.*$"
        elif marker.lower() == "p.s.":
            pat = r"\s*p\.\s?s\..*$"
        else:
            pat = r"\s*" + marker.lower() + r".*$"
        return bool(re.findall(pat, r.lower(), flags=re.MULTILINE))
    if inst == "detectable_format:number_bullet_lists":
        n = len(re.findall(r"^\s*\*[^\*].*$", r, flags=re.MULTILINE)) + len(re.findall(r"^\s*-.*$", r, flags=re.MULTILINE))
        return n == kw["num_bullets"]
    if inst == "detectable_format:constrained_response":
        return any(o in r for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if inst == "combination:repeat_prompt":
        return r.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())
    raise SystemExit(f"unchecked IFEval type {inst}")


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def chat_check(item: dict, text: str, finished: bool) -> dict:
    t = fold(text)
    kw = any(re.search(r"(?<!\w)" + re.escape(fold(k)) + r"(?!\w)", t) for k in item["must_contain_any"])
    words = re.findall(r"\w+", text.lower())
    it = sum(w in IT_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    lang = (it > en) if item["lang"] == "it" else (en > it)
    return {"keyword": kw, "language": lang, "finished": finished, "it_words": it, "en_words": en,
            "pass": kw and lang and finished}


def rollout_flags(ids: list[int]) -> dict:
    eos = bool(ids) and ids[-1] in STOP_IDS
    g = ids[:-1] if eos else ids
    run32 = any(len(set(g[s:s + 32])) == 1 for s in range(0, len(g) - 31))
    loop8x3 = any(g[s:s + 8] == g[s + 8:s + 16] == g[s + 16:s + 24] for s in range(0, len(g) - 23))
    return {"run32": run32, "loop8x3": loop8x3, "empty": not g, "eos": eos, "generated": len(g),
            "degenerate": run32 or loop8x3 or not g}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    inputs, arm = Path(args.inputs), Path(args.arm)
    tok = Tokenizer.from_file(TOKENIZER)
    res: dict = {"inputs": args.inputs, "arm": args.arm}
    for name in ("piqa", "arc"):
        res[name] = likelihood(inputs, arm, name)
        print(name, res[name]["correct"], "/", res[name]["n"])
    rows = gen_rows(arm, "gsm8k")
    items = []
    for m in meta(inputs, "gsm8k"):
        r = rows[m["item"]]
        text = decode(tok, r["gen"])
        got = gsm8k_answer(text)
        items.append({"item": m["item"], "answer": m["answer"], "got": got, "correct": num_equal(got, m["answer"]),
                      "stop": r["stop"], "generated": len(r["gen"]), "text": text})
    res["gsm8k"] = {"items": items, "correct": sum(i["correct"] for i in items), "n": len(items)}
    print("gsm8k", res["gsm8k"]["correct"], "/", len(items))
    rows = gen_rows(arm, "ifeval")
    items = []
    for m in meta(inputs, "ifeval"):
        r = rows[m["item"]]
        text = decode(tok, r["gen"])
        items.append({"item": m["item"], "key": m["key"], "instruction": m["instruction"],
                      "correct": ifeval_check(m["instruction"], m["kwargs"], text), "stop": r["stop"],
                      "generated": len(r["gen"]), "text": text})
    res["ifeval"] = {"items": items, "correct": sum(i["correct"] for i in items), "n": len(items)}
    print("ifeval", res["ifeval"]["correct"], "/", len(items))
    rows = gen_rows(arm, "chat")
    items = []
    for m in meta(inputs, "chat"):
        r = rows[m["item"]]
        text = decode(tok, r["gen"])
        c = chat_check(m, text, r["stop"] == "EOS")
        items.append({"item": m["item"], "id": m["id"], "lang": m["lang"], **c, "correct": c["pass"],
                      "generated": len(r["gen"]), "text": text})
    res["chat"] = {"items": items, "correct": sum(i["correct"] for i in items), "n": len(items)}
    print("chat", res["chat"]["correct"], "/", len(items))
    rows = gen_rows(arm, "rollout")
    items = []
    for m in meta(inputs, "rollout"):
        r = rows[m["item"]]
        items.append({"item": m["item"], "category": m["category"], **rollout_flags(r["gen"]),
                      "text": decode(tok, r["gen"])})
    res["rollout"] = {"items": items, "degenerate": sum(i["degenerate"] for i in items), "n": len(items)}
    print("rollout degenerate", res["rollout"]["degenerate"], "/", len(items))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
