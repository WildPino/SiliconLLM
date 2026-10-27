"""C50 injection cell: the facts, F0's passages, the rewordings and every input file (local, no training).

Design: docs/research/CHATBOT50_ROADMAP_20261016/C50_INJECTION_REVIEW_20261027.md. Steps, in order:

  gen    run S2's pinned greedy generator (`bin/c50_llama_generate_avx2.exe`) with F0 or T0 on a prompt file, in
         chunks of 40 prompts that are skipped once COMPLETE (so a stopped run resumes)
  pool   the candidate facts: S2's donor-gap nq and tqa items (F0 right, T0 wrong), plus the same rule on T1 part 2b's
         fresh nq and tqa items (F0's answers from `gen`); the known-item panel (fresh items v0 answered right, not in
         the pool); prompts for v0's 8 samples and for F0's first passage (from the question alone)
  split  the filters, the same for every item: v0 right in 0 of 8 samples at temperature 0.7, and F0's first passage
         holding a gold alias. Facts sharing a normalized gold or question form one cluster. Then a seeded split of
         the clusters, stratified by panel: about 15 dev facts per panel (the lr pilot's), the rest alternately
         injected and twin; at least 100 per half or STOP. Prompts for F0's 4 more passages, the answer type and F0's
         rewording (every fact), T0's rewording
  build  the training files (E: passages; S: the same with each alias swapped for another injected fact's gold of the
         same answer type, a passage where a gold alias survives the swap is dropped; QA1: one chat line "original
         question -> gold"), and the evaluation sets: rewordings (rejected if it holds an alias, or is a near copy of
         its own fact's text: a shared 4-gram of content words with p1-p5, or content Jaccard >= 0.5 with p3's
         question; the same rule for every half), original question, cloze on the first passage, open book (first
         passage in the prompt, with the original question and with each rewording); a sha256 manifest

    python -I benchmarks/chatbot50/c50_inj_data.py gen --model f0|t0 --prompts TSV --out DIR --max-new N
    python -I benchmarks/chatbot50/c50_inj_data.py pool|split|build [--dir E:/.../inj]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_s2_checks import normalize, qa_check  # noqa: E402
from c50_s2_tasks import QA_PROMPT  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
SRC = CAMPAIGN / "sources" / "Ling-mini-2.0_a810f64"
DIR = CAMPAIGN / "inj"
MODELS = {"t0": (CAMPAIGN / "gguf" / "ling-mini-2.0-a810f64-q4_k_m.gguf",
                 "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"),
          "f0": (Path("C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf"),
                 "5348c73fe6497c5152e16fd792d6c7b5aeefa15de8b15b8c8e18bef691704b8e")}
EXE = HERE / "bin" / "c50_llama_generate_avx2.exe"
STOP, EOT = "156892,156895", 156895
CHUNK, SEED, MIN_HALF, SAMPLES, DEV_PER_PANEL = 40, 20261027, 100, 8, 15
PANELS = ("nq", "tqa")
P1 = ("Write one encyclopedia-style sentence that answers this question, stating the answer explicitly.\n"
      "Question: {q}")
MORE = {"p2": "Write a short paragraph of two or three sentences about the subject of this question, including the "
              "answer.\nQuestion: {q}",
        "p3": "Write one question-and-answer pair about the same fact, asking the question in different words from "
              "this one. Format:\nQ: ...\nA: ...\nQuestion: {q}",
        "p4": "Write the fact behind this question as a single bullet point, as in a list of facts.\nQuestion: {q}",
        "p5": "Write a two-line exchange in which one person asks about this and the other answers.\nQuestion: {q}"}
TYPE = ("Classify the type of the answer to this question. Reply with one word: PERSON, PLACE, DATE, NUMBER, "
        "ORGANIZATION, WORK or OTHER.\nQuestion: {q}")
TYPES = ("PERSON", "PLACE", "DATE", "NUMBER", "ORGANIZATION", "WORK", "OTHER")
REWORD = ("Rewrite this question in different words, keeping its meaning. Do not answer it. Reply with the question "
          "only.\nQuestion: {q}")
OPEN = "Context: {ctx}\n\n" + QA_PROMPT


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Tok:
    def __init__(self):
        self.tok = Tokenizer.from_file(str(SRC / "tokenizer.json"))
        cfg = json.loads((SRC / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.tmpl = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False).from_string(cfg["chat_template"])

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def chat(self, user: str) -> list[int]:
        return self.enc(self.tmpl.render(messages=[{"role": "user", "content": user}], add_generation_prompt=True,
                                         tools=None))

    def dec(self, ids) -> str:
        return self.tok.decode([i for i in ids if i not in (156892, 156895)], skip_special_tokens=True)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(p: Path, rows) -> None:
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_tsv(p: Path, rows: list[tuple[int, list[int]]]) -> None:
    p.write_bytes("".join(f"{i}\t{','.join(map(str, ids))}\n" for i, ids in rows).encode("ascii"))


def read_tsv(p: Path) -> dict[int, list[int]]:
    out = {}
    for ln in p.read_text(encoding="ascii").splitlines():
        i, s = ln.split("\t", 1)
        out[int(i)] = [int(x) for x in s.split(",")]
    return out


def gens(prefix_dir: Path) -> dict[int, list[int]]:
    """The `gen` step's chunks, merged: item -> generated ids. Every chunk must be COMPLETE."""
    st = json.loads((prefix_dir / "gen.json").read_text(encoding="utf-8"))
    assert all(c["status"] == "COMPLETE" for c in st["chunks"].values()), prefix_dir
    out = {}
    for c in sorted(st["chunks"]):
        for r in read_jsonl(prefix_dir / f"{c}.jsonl"):
            out[int(r["item"])] = r["gen"]
    assert len(out) == st["prompts"], (prefix_dir, len(out), st["prompts"])
    return out


def holds_alias(text: str, aliases: list[str]) -> str | None:
    """The first alias (in list order) that occurs in the whole text as a normalized whole-word span."""
    t = f" {normalize(text)} "
    for a in aliases:
        g = normalize(a)
        if g and f" {g} " in t:
            return a
    return None


# ---------------------------------------------------------------- gen
def cmd_gen(a) -> int:
    model, pin = MODELS[a.model]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stp = out / "gen.json"
    st = json.loads(stp.read_text(encoding="utf-8")) if stp.exists() else None
    rows = sorted(read_tsv(Path(a.prompts)).items())
    if st is None:
        print("hashing", model, flush=True)
        st = {"model": str(model), "model_sha256": sha256(model), "exe": str(EXE), "exe_sha256": sha256(EXE),
              "prompts": len(rows), "prompts_file": a.prompts, "prompts_sha256": sha256(Path(a.prompts)),
              "max_new": a.max_new, "chunks": {}}
        if st["model_sha256"] != pin:
            raise SystemExit(f"{a.model}: model sha256 differs from its pin")
        stp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    assert st["prompts_sha256"] == sha256(Path(a.prompts)) and st["max_new"] == a.max_new, "a different run"
    for j in range(0, len(rows), CHUNK):
        name = f"c{j // CHUNK:04d}"
        if st["chunks"].get(name, {}).get("status") == "COMPLETE":
            continue
        write_tsv(out / f"{name}.tsv", rows[j:j + CHUNK])
        cmd = [str(EXE), "--model", str(model), "--prompts", str(out / f"{name}.tsv"), "--out", str(out / name),
               "--threads", "6", "--ubatch", "512", "--max-projected-s", "0", "--max-new", str(a.max_new),
               "--stop", STOP]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        config = [ln for ln in p.stdout.splitlines() if ln.startswith("CONFIG ")]
        c = config[0] if len(config) == 1 else ""
        ok = (p.returncode == 0 and "AVX2 = 1" in c and "threads=6 " in c and "ubatch=512 " in c
              and "flash_attn=disabled" in c and f"stop={STOP} " in c
              and len(read_jsonl(out / f"{name}.jsonl")) == len(rows[j:j + CHUNK]))
        st["chunks"][name] = {"status": "COMPLETE" if ok else f"VOID: rc={p.returncode}", "wall_s": round(time.time() - t0, 1),
                              "config": c, "stderr_tail": p.stderr[-400:]}
        stp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        print(name, st["chunks"][name]["status"], st["chunks"][name]["wall_s"], flush=True)
        if not ok:
            return 2
    print("DONE", out, flush=True)
    return 0


# ---------------------------------------------------------------- pool
def cmd_pool(a) -> int:
    d = Path(a.dir)
    T = Tok()
    s2v = json.loads((HERE / "results" / "c50_s2_verdict.json").read_text(encoding="utf-8"))["donor_gap_panel"]["items"]
    pool, known = [], []
    for p in PANELS:
        meta = {m["item"]: m for m in read_jsonl(CAMPAIGN / "s2" / "inputs" / f"{p}_meta.jsonl")}
        ids = read_tsv(CAMPAIGN / "s2" / "inputs" / f"{p}.tsv")
        for i in s2v[p]:
            pool.append({"src": "s2", "panel": p, "item": i, "question": meta[i]["question"],
                         "aliases": meta[i]["aliases"], "qa_ids": ids[i]})
        fmeta = {m["item"]: m for m in read_jsonl(CAMPAIGN / "t1" / "fresh" / "inputs" / f"{p}_meta.jsonl")}
        fids = read_tsv(CAMPAIGN / "t1" / "fresh" / "inputs" / f"{p}.tsv")
        right = {}
        for arm, src in (("t0", CAMPAIGN / "t1" / "fresh" / "t0" / f"{p}.jsonl"),
                         ("v0", CAMPAIGN / "t1" / "fresh" / "a1" / f"{p}.jsonl")):
            right[arm] = {r["item"]: qa_check(T.dec(r["gen"]), fmeta[r["item"]]["aliases"])["match"] for r in read_jsonl(src)}
        f0 = gens(d / "f0_fresh" / p)
        right["f0"] = {i: qa_check(T.dec(g), fmeta[i]["aliases"])["match"] for i, g in f0.items()}
        assert set(right["f0"]) == set(right["t0"]) == set(right["v0"]) == set(fmeta), p
        for i in sorted(fmeta):
            if right["f0"][i] and not right["t0"][i]:
                pool.append({"src": "fresh", "panel": p, "item": i, "question": fmeta[i]["question"],
                             "aliases": fmeta[i]["aliases"], "qa_ids": fids[i]})
            elif right["v0"][i]:
                known.append({"src": "fresh", "panel": p, "item": i, "aliases": fmeta[i]["aliases"], "qa_ids": fids[i]})
    for k, r in enumerate(pool):
        r["fact"] = k
    write_jsonl(d / "pool.jsonl", pool)
    write_jsonl(d / "known.jsonl", [{**r, "k": k} for k, r in enumerate(known)])
    write_tsv(d / "qa.tsv", [(r["fact"], r["qa_ids"]) for r in pool])
    write_tsv(d / "known.tsv", [(k, r["qa_ids"]) for k, r in enumerate(known)])
    write_tsv(d / "p1.tsv", [(r["fact"], T.chat(P1.format(q=r["question"]))) for r in pool])
    rec = {"pool": len(pool), "by": {f"{s}:{p}": sum(r["src"] == s and r["panel"] == p for r in pool)
                                     for s in ("s2", "fresh") for p in PANELS}, "known": len(known)}
    print("POOL", json.dumps(rec), flush=True)
    return 0


# ---------------------------------------------------------------- split
def cmd_split(a) -> int:
    d = Path(a.dir)
    T = Tok()
    pool = read_jsonl(d / "pool.jsonl")
    samp = {r["item"]: r["gens"] for r in read_jsonl(d / "v0_sample" / "qa.samples.jsonl")}
    p1 = gens(d / "f0_p1")
    keep, why = [], {"v0_knows": 0, "p1_no_alias": 0}
    for r in pool:
        assert len(samp[r["fact"]]) == SAMPLES, r["fact"]
        if any(qa_check(T.dec(g), r["aliases"])["match"] for g in samp[r["fact"]]):
            why["v0_knows"] += 1
            continue
        gold = holds_alias(T.dec(p1[r["fact"]]), r["aliases"])
        if gold is None:
            why["p1_no_alias"] += 1
            continue
        keep.append({**r, "gold": gold})
    # clusters: facts that share a normalized gold or a normalized question go to the same half together
    parent = {r["fact"]: r["fact"] for r in keep}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen_key = {}
    for r in keep:
        for key in ("g:" + normalize(r["gold"]), "q:" + normalize(r["question"])):
            if key in seen_key:
                parent[root(r["fact"])] = root(seen_key[key])
            else:
                seen_key[key] = r["fact"]
    clusters = {}
    for r in keep:
        clusters.setdefault(root(r["fact"]), []).append(r)
    rng = np.random.default_rng(SEED)
    half = {}
    for p in PANELS:                          # a cluster's panel is its first member's
        cl = [c for c in clusters.values() if c[0]["panel"] == p]
        cl = [cl[j] for j in rng.permutation(len(cl))]
        n_dev, k = 0, 0
        for c in cl:
            if n_dev < DEV_PER_PANEL:
                h = "dev"
                n_dev += len(c)
            else:
                h = "injected" if k % 2 == 0 else "twin"
                k += 1
            for r in c:
                half[r["fact"]] = h
    n = {h: sum(v == h for v in half.values()) for h in ("injected", "twin", "dev")}
    rec = {"pool": len(pool), "dropped": why, "kept": len(keep), "clusters": len(clusters),
           "multi_member_clusters": sum(len(c) > 1 for c in clusters.values()), "halves": n, "seed": SEED,
           "min_half": MIN_HALF}
    print("SPLIT", json.dumps(rec), flush=True)
    if min(n["injected"], n["twin"]) < MIN_HALF:
        print("STOP: fewer than", MIN_HALF, "facts in a half; score more nq and tqa items before anything trains")
        return 3
    facts = [{"fact": r["fact"], "panel": r["panel"], "src": r["src"], "item": r["item"], "half": half[r["fact"]],
              "cluster": root(r["fact"]), "question": r["question"], "aliases": r["aliases"], "gold": r["gold"]}
             for r in keep]
    (d / "split.json").write_text(json.dumps({**rec, "facts": facts}, indent=1, ensure_ascii=False), encoding="utf-8")
    # p2-p5 for every half: the twin half's are never trained, they let the rewording rule be the same for both
    more = [(f["fact"] * 8 + j, T.chat(MORE[k].format(q=f["question"]))) for f in facts for j, k in enumerate(MORE)]
    write_tsv(d / "more.tsv", more)
    write_tsv(d / "type.tsv", [(f["fact"], T.chat(TYPE.format(q=f["question"]))) for f in facts])
    write_tsv(d / "reword.tsv", [(f["fact"], T.chat(REWORD.format(q=f["question"]))) for f in facts])
    return 0


# ---------------------------------------------------------------- build
STOP_WORDS = frozenset("a an the of in on at to for by with from and or is was are were be been what which who whom "
                       "whose when where why how did do does that this these those it its as into than then there "
                       "name named called".split())


def first_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def content(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w not in STOP_WORDS]


def near_copy(q: str, own: list[str], qa_q: str | None) -> bool:
    """A rewording is a near copy of its own fact's text if it shares a 4-gram of content words with any of the
    fact's passages, or if its content-word Jaccard with the question of the fact's Q&A passage (p3) is >= 0.5."""
    cq = content(q)
    g = {tuple(cq[i:i + 4]) for i in range(len(cq) - 3)}
    for t in own:
        ct = content(t)
        if g & {tuple(ct[i:i + 4]) for i in range(len(ct) - 3)}:
            return True
    if qa_q:
        a, b = set(cq), set(content(qa_q))
        if a and b and len(a & b) / len(a | b) >= 0.5:
            return True
    return False


def p3_question(t: str) -> str | None:
    for ln in t.splitlines():
        m = re.match(r"^\s*\**\s*q\s*[:.]\s*(.+)$", ln, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def swap(text: str, aliases: list[str], new: str) -> str | None:
    """Every occurrence of any alias (longest first, case-insensitive, whole word) replaced by `new`; None if none
    was replaced, or if a gold alias still holds afterwards (the normalized match is looser than the raw one)."""
    out, n = text, 0
    for al in sorted(aliases, key=len, reverse=True):
        if not al.strip():
            continue
        out, k = re.subn(rf"(?<!\w){re.escape(al)}(?!\w)", new, out, flags=re.IGNORECASE)
        n += k
    return out if n and holds_alias(out, aliases) is None else None


def cmd_build(a) -> int:
    d = Path(a.dir)
    T = Tok()
    sp = json.loads((d / "split.json").read_text(encoding="utf-8"))
    facts = {f["fact"]: f for f in sp["facts"]}
    inj = [f for f in sp["facts"] if f["half"] == "injected"]
    dev = [f for f in sp["facts"] if f["half"] == "dev"]
    p1, more, typ = gens(d / "f0_p1"), gens(d / "f0_more"), gens(d / "f0_type")
    rw = {"f0": gens(d / "f0_reword"), "t0": gens(d / "t0_reword")}
    for f in sp["facts"]:
        w = (T.dec(typ[f["fact"]]).strip().split() or ["OTHER"])[0].strip(".,:;*").upper()
        f["type"] = w if w in TYPES else "OTHER"
    # the passages, for every fact: p1, then p2-p5; a passage without a gold alias is dropped
    pas = {f["fact"]: [("p1", T.dec(p1[f["fact"]]).strip())] for f in sp["facts"]}
    dropped = {"injected": 0, "twin": 0, "dev": 0}
    for f in sp["facts"]:
        for j, k in enumerate(MORE):
            t = T.dec(more[f["fact"] * 8 + j]).strip()
            if holds_alias(t, f["aliases"]):
                pas[f["fact"]].append((k, t))
            else:
                dropped[f["half"]] += 1
    # S: within answer type, a seeded derangement of the injected facts' gold answers
    rng = np.random.default_rng(SEED + 1)
    partner = {}
    by = {}
    for f in inj:
        by.setdefault(f["type"], []).append(f["fact"])
    small = [t for t, v in by.items() if len(v) < 2]
    for t in small:
        by.setdefault("OTHER*", []).extend(by.pop(t))
    for t, v in by.items():
        if len(v) < 2:
            continue
        perm = [v[j] for j in rng.permutation(len(v))]
        for j, x in enumerate(perm):
            partner[x] = perm[(j + 1) % len(perm)]
    train = {"E": [], "S": [], "QA1": [], "dev_E": [{"fact": f["fact"], "kind": k, "ids": T.enc(t)}
                                                   for f in dev for k, t in pas[f["fact"]]]}
    s_lost = 0
    for f in inj:
        for k, t in pas[f["fact"]]:
            train["E"].append({"fact": f["fact"], "kind": k, "ids": T.enc(t)})
            if f["fact"] in partner:
                s = swap(t, f["aliases"], facts[partner[f["fact"]]]["gold"])
                if s is None:
                    s_lost += 1
                else:
                    train["S"].append({"fact": f["fact"], "kind": k, "ids": T.enc(s)})
        train["QA1"].append({"fact": f["fact"], "kind": "qa1",
                             "ids": T.chat(QA_PROMPT.format(q=f["question"])) + T.enc(f["gold"]) + [EOT]})
    for k, v in train.items():
        write_jsonl(d / f"train_{k}.jsonl", v)
    # the evaluation sets, one row per (fact, variant); index = row number, meta carries fact and variant.
    # The rewording rule is the same for every half: it reads only the fact's own text (p1-p5), which every fact has.
    sets = {"rew": [], "orig": [], "cloze": [], "open": [], "open_rew": []}
    rej = {h: {"alias": 0, "near_copy": 0, "same_as_original": 0, "empty": 0} for h in ("injected", "twin", "dev")}
    for f in sp["facts"]:
        own = [t for _, t in pas[f["fact"]]]
        qa_q = next((p3_question(t) for k, t in pas[f["fact"]] if k == "p3"), None)
        t1 = pas[f["fact"]][0][1]
        for src in ("f0", "t0"):
            q = first_line(T.dec(rw[src][f["fact"]]))
            q = re.sub(r"^(question|rewritten question)\s*:\s*", "", q, flags=re.IGNORECASE).strip()
            why = ("empty" if not q else "alias" if holds_alias(q, f["aliases"]) else
                   "same_as_original" if normalize(q) == normalize(f["question"]) else
                   "near_copy" if near_copy(q, own, qa_q) else None)
            if why:
                rej[f["half"]][why] += 1
                continue
            sets["rew"].append({"fact": f["fact"], "variant": src, "question": q, "ids": T.chat(QA_PROMPT.format(q=q))})
            sets["open_rew"].append({"fact": f["fact"], "variant": src, "ids": T.chat(OPEN.format(ctx=t1, q=q))})
        sets["orig"].append({"fact": f["fact"], "variant": "orig", "question": f["question"],
                             "ids": T.chat(QA_PROMPT.format(q=f["question"]))})
        al = holds_alias(t1, f["aliases"])
        m = re.search(rf"(?<!\w){re.escape(al)}(?!\w)", t1, flags=re.IGNORECASE)
        if m and m.start() > 0:
            sets["cloze"].append({"fact": f["fact"], "variant": "p1", "ids": T.enc(t1[:m.start()].rstrip())})
        sets["open"].append({"fact": f["fact"], "variant": "orig", "ids": T.chat(OPEN.format(ctx=t1, q=f["question"]))})
    man = {"cell": "C50 injection inputs", "split_sha256": sha256(d / "split.json"), "files": {}, "rejected": rej,
           "s_swap_lost": s_lost, "passages_dropped": dropped, "partner_pairs": len(partner),
           "types": {t: sum(f["type"] == t for f in sp["facts"]) for t in TYPES},
           "rewordings_kept": {h: {src: sum(1 for r in sets["rew"] if r["variant"] == src and facts[r["fact"]]["half"] == h)
                                   for src in ("f0", "t0")} for h in ("injected", "twin", "dev")}}
    for k, rows in sets.items():
        write_jsonl(d / f"eval_{k}_meta.jsonl", [{kk: vv for kk, vv in r.items() if kk != "ids"} | {"row": j}
                                                for j, r in enumerate(rows)])
        write_tsv(d / f"eval_{k}.tsv", [(j, r["ids"]) for j, r in enumerate(rows)])
    (d / "split_typed.json").write_text(json.dumps(sp, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in (sorted(d.glob("train_*.jsonl")) + sorted(d.glob("eval_*")) +
              [d / "known.tsv", d / "known.jsonl", d / "split_typed.json"]):
        man["files"][p.name] = sha256(p)
    man["counts"] = {**{f"train_{k}": len(v) for k, v in train.items()},
                     **{f"eval_{k}": len(v) for k, v in sets.items()},
                     "train_tokens": {k: sum(len(r["ids"]) for r in v) for k, v in train.items()}}
    (d / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("BUILD", json.dumps({k: man[k] for k in ("counts", "rejected", "rewordings_kept", "s_swap_lost",
                                                    "passages_dropped", "types")}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True, choices=list(MODELS))
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-new", type=int, required=True)
    for c in ("pool", "split", "build"):
        sub.add_parser(c).add_argument("--dir", default=str(DIR))
    a = ap.parse_args()
    return {"gen": cmd_gen, "pool": cmd_pool, "split": cmd_split, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
