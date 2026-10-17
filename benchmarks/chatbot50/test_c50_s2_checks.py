"""Known-answer cases for the C50-S2 short-answer check: each rule has a known positive and a planted negative.

Usage:
    python -I benchmarks/chatbot50/test_c50_s2_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_s2_checks import first_line, normalize, qa_check  # noqa: E402

CASES = [
    # (response, aliases, match, em)
    ("David Seville", ["David Seville"], True, True),
    ("The answer is David Seville.", ["David Seville"], True, False),
    ("Answer: David Seville", ["David Seville"], True, True),
    ("\n\n  Ross Bagdasarian Sr.\nHe used the name David Seville.", ["David Seville"], False, False),   # second line only
    ("Davids Seville", ["David Seville"], False, False),                                               # not a whole word
    ("Seville", ["David Seville"], False, False),                                                      # partial alias
    ("December 1972", ["14 December 1972 UTC", "December 1972"], True, True),
    ("In December, 1972.", ["14 December 1972 UTC", "December 1972"], True, False),                   # comma removed
    ("U.S.A.", ["USA"], True, True),                                                                   # punctuation deleted, not spaced
    ("Jean-Paul Sartre", ["Jean Paul Sartre"], False, False),                                          # SQuAD: the hyphen joins
    ("1973", ["December 1972"], False, False),
    ("the Beatles", ["Beatles"], True, True),                                                          # article removed
    ("Élan", ["élan"], True, True),                                                                    # case, NFKC
    ("an", ["a"], False, False),                                                                       # aliases empty after normalization never match
    ("", ["Paris"], False, False),
]


def main() -> int:
    fail = 0
    for resp, aliases, m, e in CASES:
        c = qa_check(resp, aliases)
        ok = c["match"] == m and c["em"] == e
        fail += not ok
        print("ok  " if ok else "FAIL", repr(resp)[:50], aliases, "->", c["match"], c["em"])
    assert normalize("  The  U.S.A.!  ") == "usa", normalize("  The  U.S.A.!  ")
    assert first_line("\n answer:  Rome \n x") == "Rome"
    print(f"{len(CASES) - fail}/{len(CASES)} cases pass")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
