"""Known-answer tests of the C50-A4 checkers (each check must accept a known positive and reject a
planted negative).  Run: python -I benchmarks/chatbot50/test_c50_a4_checks.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_a4_checks as C  # noqa: E402

CASES = [  # (instruction, kwargs, response, expected)
    ("punctuation:no_comma", {}, "No commas here.", True),
    ("punctuation:no_comma", {}, "One, two.", False),
    ("change_case:english_lowercase", {}, "all lower case.", True),
    ("change_case:english_lowercase", {}, "Not all lower.", False),
    ("change_case:english_capital", {}, "ALL CAPS 12.", True),
    ("change_case:english_capital", {}, "ALL CAPs.", False),
    ("startend:quotation", {}, '"wrapped in quotes"', True),
    ("startend:quotation", {}, '"only at the start', False),
    ("startend:end_checker", {"end_phrase": "Is there anything else I can help with?"},
     "Text. Is there anything else I can help with?", True),
    ("startend:end_checker", {"end_phrase": "Is there anything else I can help with?"}, "Text. Anything else?", False),
    ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', True),
    ("detectable_format:json_format", {}, "{a: 1}", False),
    ("detectable_format:title", {}, "<<My Title>>\nbody", True),
    ("detectable_format:title", {}, "My Title\nbody", False),
    ("keywords:forbidden_words", {"forbidden_words": ["cat", "dog"]}, "A category of birds.", True),
    ("keywords:forbidden_words", {"forbidden_words": ["cat", "dog"]}, "The dog barked.", False),
    ("keywords:existence", {"keywords": ["river", "bank"]}, "The river bank.", True),
    ("keywords:existence", {"keywords": ["river", "bank"]}, "The river.", False),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body.\nP.S. a note", True),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body without a note.", False),
    ("detectable_content:postscript", {"postscript_marker": "P.P.S"}, "Body.\nP.P.S another", True),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* one\n* two", True),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* one\n* two\n- three", False),
    ("detectable_format:constrained_response", {}, "My answer is yes.", True),
    ("detectable_format:constrained_response", {}, "Yes.", False),
    ("combination:repeat_prompt", {"prompt_to_repeat": "Write a poem."}, "Write a poem. Roses...", True),
    ("combination:repeat_prompt", {"prompt_to_repeat": "Write a poem."}, "Roses are red.", False),
]


def main() -> int:
    bad = 0
    for inst, kw, r, exp in CASES:
        got = C.ifeval_check(inst, kw, r)
        if got != exp:
            bad += 1
            print("FAIL ifeval", inst, repr(r), got)
    gsm = [("Steps...\n#### 1,234", "1234", True), ("so the total is 18.", "18", True),
           ("#### 17", "18", False), ("no number", "3", False), ("#### 18.", "18", True)]
    for text, ans, exp in gsm:
        if C.num_equal(C.gsm8k_answer(text), ans) != exp:
            bad += 1
            print("FAIL gsm8k", repr(text))
    po = {"lang": "it", "must_contain_any": ["po"]}
    chats = [(po, "Il fiume più lungo d'Italia è il Po.", True, True),
             (po, "Dopo un po' di tempo non lo so.", True, True),     # "po'" is a whole word: keyword found
             (po, "Dopo il ponte, la risposta è il Tevere.", True, False),  # substrings only: rejected
             ({"lang": "it", "must_contain_any": ["lisbona"]}, "The capital is Lisbona, of course.", True, False),
             ({"lang": "en", "must_contain_any": ["ottawa"]}, "The capital of Canada is Ottawa.", True, True),
             ({"lang": "en", "must_contain_any": ["ottawa"]}, "The capital of Canada is Ottawa.", False, False)]
    for item, text, fin, exp in chats:
        if C.chat_check(item, text, fin)["pass"] != exp:
            bad += 1
            print("FAIL chat", repr(text), C.chat_check(item, text, fin))
    # loop8x3 (STRAT-01 definition) sees periods dividing 8 only: a period-3 loop is not flagged
    rolls = [([1, 2, 3, 4] * 8, True), ([1, 2, 3] * 10, False), ([5] * 32, True), (list(range(100)), False), ([], True),
             (list(range(50)) + [156895], False), ([156895], True)]
    for ids, exp in rolls:
        if C.rollout_flags(ids)["degenerate"] != exp:
            bad += 1
            print("FAIL rollout", ids[:10], C.rollout_flags(ids))
    total = len(CASES) + len(gsm) + len(chats) + len(rolls)
    print(f"{total - bad}/{total} checks as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
