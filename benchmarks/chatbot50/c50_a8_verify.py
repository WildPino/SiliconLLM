"""C50-A8 verification T1-T5 of the C tokenizer and chat template against the source (numbers only).

Reference: HF tokenizers on the pinned tokenizer.json (encode with add_special_tokens=False, decode with
skip_special_tokens=False) and transformers apply_chat_template with the pinned chat_template.  The C side is
bin/c50_tok_cli.exe on the P0 GGUF.  c50_a8_verdict.py applies the rule (0 mismatches on T1-T5).

  T1  FIT document texts (corpus/c50_holdout_v1/fit.jsonl)           encode IDs equal
  T2  A4 chat panel prompts (a4_chat_panel_v1.json)                  encode IDs equal
  T3  stress set S, groups random / mixed / nfc                      encode IDs equal
  T4  decode of every T1-T3 reference sequence + 2,000 random ID sequences   decoded bytes equal
  T5  conversation set C (12, add_generation_prompt)                 rendered text and IDs equal
  reported, not gated: stress group "newer" (unassigned code points)

Usage:
    python -I benchmarks/chatbot50/c50_a8_verify.py --stress <stress.json> --work <dir> --out results/<record>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMP = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
SRC = CAMP / "sources/Ling-mini-2.0_a810f64"
GGUF = CAMP / "gguf/ling-mini-2.0-a810f64-q4_k_m.gguf"
CLI = HERE / "bin" / "c50_tok_cli.exe"
SEED, N_RANDOM_SEQ, N_SOURCE_IDS = 20261016, 2000, 157153
SYSTEM, USER, ASSISTANT = 0, 1, 2
CONVERSATIONS = [
    [(USER, "Ciao! Come stai?")],
    [(SYSTEM, "Sei un assistente che risponde in italiano."), (USER, "Qual è la capitale della Francia?")],
    [(USER, "What is 2+2?"), (ASSISTANT, "2+2 = 4."), (USER, "And 3*7?")],
    [(SYSTEM, "You are terse."), (USER, "Name three primes."), (ASSISTANT, "2, 3, 5."), (USER, "Three more.")],
    [(USER, "")],
    [(USER, "Line one\nLine two\n\n  indented\ttab")],
    [(USER, "Explain <|role_end|> and <role>HUMAN</role> tokens literally.")],
    [(SYSTEM, "Rispondi sempre con una sola frase.\nNiente elenchi."), (USER, "Perché il cielo è blu?")],
    [(USER, "Scrivi una funzione Python:\n```python\ndef f(x):\n    return x**2\n```\nÈ corretta?")],
    [(USER, "Translate 'naïve café' to Italian."), (ASSISTANT, "«caffè ingenuo»"), (USER, "Grazie!"),
     (ASSISTANT, "Prego!")],
    [(SYSTEM, ""), (USER, "Hi")],
    [(USER, "First"), (SYSTEM, "A mid-conversation system note."), (USER, "Second")],
]
ROLE_NAME = {SYSTEM: "system", USER: "user", ASSISTANT: "assistant"}


def run_cli(mode: str, payload: bytes, work: Path) -> tuple[bytes, str]:
    inp, out = work / f"{mode}.in.bin", work / f"{mode}.out.bin"
    inp.write_bytes(payload)
    p = subprocess.run([str(CLI), "--model", str(GGUF), f"--{mode}", str(inp), "--out", str(out)],
                       capture_output=True, text=True)
    if p.returncode != 0 or "DONE" not in p.stdout:
        raise SystemExit(f"c50_tok_cli {mode} failed: {p.stdout} {p.stderr}")
    return out.read_bytes(), next(x for x in p.stdout.splitlines() if x.startswith("CONFIG"))


def read_id_records(b: bytes) -> list[list[int]]:
    out, o = [], 0
    while o < len(b):
        (n,) = struct.unpack_from("<I", b, o)
        o += 4
        out.append(list(struct.unpack_from(f"<{n}i", b, o)))
        o += 4 * n
    return out


def read_byte_records(b: bytes) -> list[bytes]:
    out, o = [], 0
    while o < len(b):
        (n,) = struct.unpack_from("<I", b, o)
        o += 4
        out.append(b[o:o + n])
        o += n
    return out


def compare(name, refs, got, items):
    bad = [i for i, (a, b) in enumerate(zip(refs, got)) if a != b]
    if len(refs) != len(got):
        raise SystemExit(f"{name}: {len(got)} outputs for {len(refs)} inputs")
    ex = []
    for i in bad[:5]:
        a, b = refs[i], got[i]
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        ex.append({"index": i, "input": ascii(items[i])[:300], "first_diff": k,
                   "ref": ascii(a[max(0, k - 3):k + 5]), "c": ascii(b[max(0, k - 3):k + 5])})
    print(f"{name}: {len(refs)} items, {len(bad)} mismatches")
    return {"items": len(refs), "mismatches": len(bad), "examples": ex}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    hf = Tokenizer.from_file(str(SRC / "tokenizer.json"))
    cfg = json.loads((SRC / "tokenizer_config.json").read_text(encoding="utf-8"))
    rec = {"cell": "C50-A8 verification", "cli_sha256": hashlib.sha256(CLI.read_bytes()).hexdigest(),
           "tokenizer_json_sha256": hashlib.sha256((SRC / "tokenizer.json").read_bytes()).hexdigest(),
           "gguf": str(GGUF), "checks": {}}

    fit = [json.loads(x)["text"] for x in (HERE / "corpus/c50_holdout_v1/fit.jsonl").read_text(encoding="utf-8").splitlines()]
    chat = [x["prompt"] for x in json.loads((HERE / "a4_chat_panel_v1.json").read_text(encoding="utf-8"))["items"]]
    stress = json.loads(Path(args.stress).read_text(encoding="utf-8"))
    sets = {"T1": fit, "T2": chat, "T3": [x["text"] for x in stress if x["group"] != "newer"],
            "newer": [x["text"] for x in stress if x["group"] == "newer"]}
    all_texts = [t for k in ("T1", "T2", "T3", "newer") for t in sets[k]]
    payload = b"".join(struct.pack("<I", len(t.encode("utf-8"))) + t.encode("utf-8") for t in all_texts)
    out, rec["config_encode"] = run_cli("encode", payload, work)
    got = read_id_records(out)
    o = 0
    ref_ids = {}
    for k in ("T1", "T2", "T3", "newer"):
        refs = [hf.encode(t, add_special_tokens=False).ids for t in sets[k]]
        ref_ids[k] = refs
        rec["checks"][k] = compare(k, refs, got[o:o + len(refs)], sets[k])
        o += len(refs)
    if "T3" in rec["checks"]:
        g = [x["group"] for x in stress if x["group"] != "newer"]
        bad = set()
        for i, (a, b) in enumerate(zip(ref_ids["T3"], got[len(fit) + len(chat):len(fit) + len(chat) + len(g)])):
            if a != b:
                bad.add(i)
        rec["checks"]["T3"]["mismatches_by_group"] = {grp: sum(1 for i in bad if g[i] == grp) for grp in sorted(set(g))}

    rng = random.Random(SEED)
    seqs = ref_ids["T1"] + ref_ids["T2"] + ref_ids["T3"]
    seqs += [[rng.randrange(N_SOURCE_IDS) for _ in range(rng.randint(1, 64))] for _ in range(N_RANDOM_SEQ)]
    payload = b"".join(struct.pack("<I", len(s)) + struct.pack(f"<{len(s)}i", *s) for s in seqs)
    out, rec["config_decode"] = run_cli("decode", payload, work)
    got = read_byte_records(out)
    refs = [hf.decode(s, skip_special_tokens=False).encode("utf-8") for s in seqs]
    rec["checks"]["T4"] = compare("T4", refs, got, seqs)

    fast = PreTrainedTokenizerFast(tokenizer_file=str(SRC / "tokenizer.json"))
    fast.chat_template = cfg["chat_template"]
    payload = b""
    ref_text, ref_tids = [], []
    for conv in CONVERSATIONS:
        msgs = [{"role": ROLE_NAME[r], "content": c} for r, c in conv]
        t = fast.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ref_text.append(t.encode("utf-8"))
        ref_tids.append(hf.encode(t, add_special_tokens=False).ids)
        payload += struct.pack("<I", len(conv))
        for r, c in conv:
            payload += struct.pack("<II", r, len(c.encode("utf-8"))) + c.encode("utf-8")
    out, rec["config_template"] = run_cli("template", payload, work)
    o, gt, gi = 0, [], []
    while o < len(out):
        (n,) = struct.unpack_from("<I", out, o); o += 4
        gt.append(out[o:o + n]); o += n
        (n,) = struct.unpack_from("<I", out, o); o += 4
        gi.append(list(struct.unpack_from(f"<{n}i", out, o))); o += 4 * n
    rec["checks"]["T5_text"] = compare("T5_text", ref_text, gt, CONVERSATIONS)
    rec["checks"]["T5_ids"] = compare("T5_ids", ref_tids, gi, CONVERSATIONS)
    Path(args.out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
