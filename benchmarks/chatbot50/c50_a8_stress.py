"""C50-A8 stress set S (protocol §Verification), generated from seed 20261016 before any comparison.

Groups (counts fixed by the protocol):
  random   3,000 strings of 1-40 code points assigned in Unicode 15.0; half of them biased towards combining
           marks, Hangul, CJK, emoji with ZWJ / variation selectors and White_Space;
  mixed    1,000 strings of ASCII words, digit runs, contractions ('s 'S 'll 'VE and the long s), punctuation
           runs before newlines, trailing whitespace runs and added-token texts inside and next to words;
  nfc      500 strings whose NFC differs from themselves (decomposed Latin, Hangul jamo, unordered marks);
  newer    500 strings with unassigned code points (reported, not gated).
Writes one JSON list of {"group", "text"}.

Usage:
    python -I benchmarks/chatbot50/c50_a8_stress.py --tokenizer <tokenizer.json> --out <stress.json>
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from pathlib import Path

SEED = 20261016
WS = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2007, 0x200A, 0x2028, 0x2029, 0x202F,
      0x205F, 0x3000, 0x1C, 0x1D, 0x1E, 0x1F, 0x200B, 0x180E, 0xFEFF]
BIASED = [(0x0300, 0x036F), (0x0591, 0x05C7), (0x0900, 0x097F), (0x1100, 0x11FF), (0xAC00, 0xD7A3), (0x3040, 0x30FF),
          (0x4E00, 0x9FFF), (0x1F300, 0x1F6FF), (0x1F900, 0x1F9FF), (0xFE00, 0xFE0F), (0x200C, 0x200D),
          (0x0600, 0x06FF), (0x0E00, 0x0E7F), (0x1D400, 0x1D7FF), (0xFF00, 0xFFEF)]


def assigned(cp: int) -> bool:
    return not (0xD800 <= cp <= 0xDFFF) and unicodedata.category(chr(cp)) != "Cn"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)
    tok = json.loads(Path(args.tokenizer).read_text(encoding="utf-8"))
    added = [a["content"] for a in tok["added_tokens"]]
    pool = [cp for cp in range(0x110000) if assigned(cp)]
    out = []

    def rand_cp(biased: bool) -> int:
        if biased:
            r = rng.random()
            if r < 0.25:
                return rng.choice(WS)
            lo, hi = rng.choice(BIASED)
            while True:
                cp = rng.randint(lo, hi)
                if assigned(cp):
                    return cp
        return rng.choice(pool) if rng.random() < 0.5 else rng.randint(0x20, 0x7E)

    for i in range(3000):
        n = rng.randint(1, 40)
        out.append({"group": "random", "text": "".join(chr(rand_cp(i % 2 == 1)) for _ in range(n))})

    words = ["the", "The", "cat", "isn", "don", "we", "they", "Hello", "perché", "città", "naïve", "Straße", "über",
             "ΑΒΓ", "дом", "x", "I", "you"]
    contractions = ["'s", "'S", "'t", "'T", "'d", "'m", "'ll", "'LL", "'Ll", "'ve", "'VE", "'re", "'Re", "'ſ", "'x", "'"]
    puncts = ["!", "?!", "...", "--", "()", "{", "}", ";", "::", "#", "@@", "%", "€", "«", "»", "\"", "''", "`", "~~"]
    spaces = [" ", "  ", "   ", "\t", "\n", "\r\n", "\n\n", " \n", " ", "　", " \t ", "\n  "]
    for _ in range(1000):
        parts = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.3:
                parts.append(rng.choice(words))
            elif r < 0.42:
                parts.append(str(rng.randint(0, 10 ** rng.randint(1, 12))))
            elif r < 0.55:
                parts.append(rng.choice(contractions))
            elif r < 0.68:
                parts.append(rng.choice(puncts) + rng.choice(["", "\n", "\r\n", "\n\n"]))
            elif r < 0.88:
                parts.append(rng.choice(spaces))
            else:
                parts.append(rng.choice(added))
        s = "".join(parts)
        if rng.random() < 0.3:
            s += rng.choice(spaces) * rng.randint(1, 3)
        out.append({"group": "mixed", "text": s})

    marks = [0x0301, 0x0300, 0x0308, 0x0327, 0x0323, 0x031B, 0x0302, 0x0303, 0x0331, 0x05B0, 0x093C]
    bases = "aeiouAEIOUnNcCsSyYzZ"
    while sum(1 for x in out if x["group"] == "nfc") < 500:
        r = rng.random()
        if r < 0.4:
            s = "".join(rng.choice(bases) + "".join(chr(rng.choice(marks)) for _ in range(rng.randint(1, 3)))
                        for _ in range(rng.randint(1, 6)))
        elif r < 0.7:
            s = "".join(chr(0x1100 + rng.randrange(19)) + chr(0x1161 + rng.randrange(21))
                        + (chr(0x11A8 + rng.randrange(27)) if rng.random() < 0.5 else "")
                        for _ in range(rng.randint(1, 5)))
        else:
            s = "".join(unicodedata.normalize("NFD", chr(rng.choice(pool))) for _ in range(rng.randint(2, 10)))
        if rng.random() < 0.5:
            s = rng.choice(words) + " " + s + rng.choice(spaces)
        if unicodedata.normalize("NFC", s) != s:
            out.append({"group": "nfc", "text": s})

    for _ in range(500):
        n = rng.randint(1, 20)
        chars = []
        for _ in range(n):
            if rng.random() < 0.5:
                while True:
                    cp = rng.randint(0x0, 0x10FFFF)
                    if not (0xD800 <= cp <= 0xDFFF) and unicodedata.category(chr(cp)) == "Cn":
                        break
                chars.append(chr(cp))
            else:
                chars.append(rng.choice(words + [" "]))
        out.append({"group": "newer", "text": "".join(chars)})
    Path(args.out).write_text(json.dumps(out, ensure_ascii=True), encoding="utf-8")
    print({g: sum(1 for x in out if x["group"] == g) for g in ("random", "mixed", "nfc", "newer")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
