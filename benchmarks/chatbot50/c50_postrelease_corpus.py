"""C50 A4-bis corpus: 72 documents written after Ling-mini-2.0's weights were published, from pinned rules.

Why: on the FIT text (PG19, The Stack, public markdown, very probably in pretraining) A4's document gate could
not see a Q2_K loss (A4 result, design review step 1).  This builds text the model cannot have trained on.

Date pin: the safetensors of Ling-mini-2.0 were last changed by commit b735a17 on 2025-09-08 (later commits touch
README and modelling code only).  Every source is created inside WINDOW, at least 5.7 months later.

Per category, 24 documents, the A4 shape (category labels as in c50_holdout_v1):
- code: one .py file per GitHub repository.  Candidate repositories: for each 7-day slice of WINDOW and each licence
  in LICENCES, the first page (30, sorted by stars) of `created:<slice> license:<l> language:python stars:>=20
  fork:false archived:false`.  Repositories are walked in sha256("<SEED>:code:<full_name>") order; in each, the
  eligible file with the smallest sha256("<SEED>:<full_name>:<path>") is taken, at the HEAD commit read at build time.
- technical_general: one .md file per repository, the same candidates walked in "<SEED>:technical:" order, skipping
  the repositories already used for code.
- prose: English Wikipedia articles created (creation log, namespace 0) on the 1st and 15th of each month of WINDOW,
  two per day, walked in sha256("<SEED>:prose:<title>") order; plain-text extract of the current revision.
Text rules (every category): strict UTF-8, NFC-stable, >= 90% ASCII characters, at least MIN_BYTES; no copyright
line naming a year 1990-2025; files <= MAX_FILE_BYTES; tokenizer round-trip.  Path rules: no path component starting
with "." or in SKIP_DIRS; code skips __init__.py, setup.py, conftest.py, test_*.py and *_pb2*.py; technical skips the
licence, changelog, contributing, conduct, security and agent-instruction files.  Wikipedia: no redirect, no
disambiguation page, no creation summary that names a translation, split, copy or merge, no ContentTranslation tag.
Span: the c50_holdout_v1 rule (pinned build_document_holdout._bounded_span, seed 20261016, 8,192 bytes).

The search results move with time, so a rebuild is not expected to reproduce the same items: the tracked manifest
(every item's URL, licence, creation date, pinned commit and git blob, or Wikipedia revid, and the hashes) is the
record, and it is committed before any arm scores the documents.

Usage:
    python -I benchmarks/chatbot50/c50_postrelease_corpus.py --tokenizer <checkpoint>/tokenizer.json \
        --corpus-dir E:/.../a4bis_post/corpus --inputs-dir E:/.../a4bis_post/inputs \
        --manifest-out benchmarks/chatbot50/results/c50_postrelease_v1_manifest.json
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_holdout as H  # noqa: E402

SEED = 20261016
SPAN_BYTES = 8192
MIN_BYTES = 8192
MAX_FILE_BYTES = 262144
PER_CAT = 24
WINDOW = ("2026-03-01", "2026-08-31")
WEIGHTS = {"repo": "inclusionAI/Ling-mini-2.0", "weights_commit": "b735a17", "weights_date": "2025-09-08"}
LICENCES = ("mit", "apache-2.0")
SPDX_OK = {"MIT", "Apache-2.0"}
ASCII_MIN = 0.90
SKIP_DIRS = {"test", "tests", "testing", "vendor", "vendored", "third_party", "thirdparty", "external", "extern",
             "node_modules", "site-packages", "migrations", "build", "dist"}
CODE_SKIP = re.compile(r"^(__init__\.py|setup\.py|conftest\.py|test_.*\.py|.*_pb2(_grpc)?\.py)$")
TECH_SKIP = re.compile(r"^(license|licence|copying|notice|changelog|changes|history|contributing|code_of_conduct|"
                       r"security|skill|agents|claude|gemini|copilot-instructions)([._-].*)?\.md$", re.I)
COPYRIGHT_OLD = re.compile(r"(?i)(copyright|\(c\)|©)[^\n]{0,80}\b(199\d|200\d|201\d|202[0-5])\b")
SPDX_LINE = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.+-]+)")
WP_API = "https://en.wikipedia.org/w/api.php"
WP_UA = "SiliconLLM-research/0.1 (one-off evaluation corpus build)"
WP_BAD_SUMMARY = re.compile(r"translat|split from|copied|merg|content from|attribution", re.I)
CATEGORIES = ("code", "technical_general", "prose")


def gh(path: str, **params) -> dict:
    cmd = ["gh", "api", "-X", "GET", path] + [x for k, v in params.items() for x in ("-f", f"{k}={v}")]
    for k in range(5):
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if p.returncode == 0:
            return json.loads(p.stdout)
        if "rate limit" in (p.stderr + p.stdout).lower() or "secondary" in (p.stderr + p.stdout).lower():
            time.sleep(60 * (k + 1))
            continue
        raise RuntimeError(f"gh api {path}: {p.stderr.strip()[:300]}")
    raise RuntimeError(f"gh api {path}: rate limited")


def wp(params: dict) -> dict:
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(f"{WP_API}?{q}", headers={"User-Agent": WP_UA})
    for k in range(6):
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(10 * (k + 1))
    raise RuntimeError("Wikipedia API: rate limited")


def key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def text_ok(raw: bytes, counts: dict) -> str | None:
    """The shared text rules; returns the text or None, counting the reason."""
    def no(reason):
        counts[reason] = counts.get(reason, 0) + 1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return no("not_utf8")
    if len(raw) < MIN_BYTES:
        return no("short")
    if unicodedata.normalize("NFC", text) != text:
        return no("not_nfc")
    if sum(ch < "\x80" for ch in text) < ASCII_MIN * len(text):
        return no("ascii_below_90pct")
    if COPYRIGHT_OLD.search(text):
        return no("copyright_year_1990_2025")
    if any(x not in SPDX_OK for x in SPDX_LINE.findall(text)):
        return no("spdx_other_licence")
    return text


def path_ok(path: str, cat: str) -> bool:
    parts = path.split("/")
    if any(p.startswith(".") or p.lower() in SKIP_DIRS for p in parts[:-1]) or parts[-1].startswith("."):
        return False
    name = parts[-1]
    if cat == "code":
        return name.endswith(".py") and not CODE_SKIP.match(name)
    return name.lower().endswith(".md") and not TECH_SKIP.match(name)


def slices() -> list[tuple[str, str]]:
    a, end = dt.date.fromisoformat(WINDOW[0]), dt.date.fromisoformat(WINDOW[1])
    out = []
    while a <= end:
        b = min(a + dt.timedelta(days=6), end)
        out.append((a.isoformat(), b.isoformat()))
        a = b + dt.timedelta(days=1)
    return out


def candidates(record: dict) -> dict:
    repos = {}
    record["searches"] = []
    for a, b in slices():
        for lic in LICENCES:
            q = f"created:{a}..{b} license:{lic} language:python stars:>=20 fork:false archived:false"
            d = gh("search/repositories", q=q, sort="stars", order="desc", per_page=30)
            record["searches"].append({"q": q, "total_count": d["total_count"], "returned": len(d["items"])})
            for it in d["items"]:
                spdx = (it.get("license") or {}).get("spdx_id")
                if spdx in SPDX_OK and it["created_at"][:10] >= WINDOW[0] and not it["fork"]:
                    repos[it["full_name"]] = {"full_name": it["full_name"], "created_at": it["created_at"],
                                              "licence": spdx, "stars_at_build": it["stargazers_count"]}
            time.sleep(2.5)
    return repos


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def github_items(cat: str, repos: dict, used: set, counts: dict, rejected: list) -> list[dict]:
    items = []
    for name in sorted(repos, key=lambda n: key(f"{SEED}:{'code' if cat == 'code' else 'technical'}:{n}")):
        if len(items) == PER_CAT:
            break
        if name in used:
            continue
        try:
            commit = gh(f"repos/{name}/commits/HEAD")["sha"]
            tree = gh(f"repos/{name}/git/trees/{commit}", recursive=1)
        except RuntimeError as e:
            rejected.append({"repo": name, "reason": f"api: {str(e)[:120]}"})
            continue
        if tree.get("truncated"):
            rejected.append({"repo": name, "reason": "tree truncated"})
            continue
        files = sorted((f for f in tree["tree"] if f["type"] == "blob" and MIN_BYTES <= f.get("size", 0) <= MAX_FILE_BYTES
                        and path_ok(f["path"], cat)), key=lambda f: key(f"{SEED}:{name}:{f['path']}"))
        taken = None
        for f in files:
            blob = gh(f"repos/{name}/git/blobs/{f['sha']}")
            raw = base64.b64decode(blob["content"])
            if git_blob_sha(raw) != f["sha"]:
                raise SystemExit(f"{name}:{f['path']}: blob content does not match its git sha")
            text = text_ok(raw, counts)
            if text is None:
                continue
            taken = (f, raw, text)
            break
        if taken is None:
            rejected.append({"repo": name, "reason": f"no eligible file ({len(files)} by path and size)"})
            continue
        f, raw, text = taken
        items.append({"category": cat, "source_kind": "github", "repo": name, "licence": repos[name]["licence"],
                      "created_at": repos[name]["created_at"], "commit": commit, "path": f["path"],
                      "git_blob_sha1": f["sha"], "url": f"https://github.com/{name}/blob/{commit}/{f['path']}",
                      "source_content_sha256": H.sha(raw), "source_byte_count": len(raw), "_text": text})
        used.add(name)
    if len(items) < PER_CAT:
        raise SystemExit(f"{cat}: only {len(items)} eligible repositories")
    return items


def prose_items(counts: dict, rejected: list) -> list[dict]:
    days = [f"2026-{m:02d}-{d:02d}" for m in range(3, 9) for d in (1, 15)]
    items = []
    for day in days:
        nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
        ev, cont = [], {}
        while True:
            d = wp({"action": "query", "list": "logevents", "letype": "create", "lenamespace": 0,
                    "lestart": f"{day}T00:00:00Z", "leend": f"{nxt}T00:00:00Z", "ledir": "newer", "lelimit": 500,
                    "leprop": "title|timestamp|comment|tags|ids", **cont})
            ev += d["query"]["logevents"]
            if "continue" not in d:
                break
            cont = d["continue"]
        ev = [e for e in ev if "contenttranslation" not in e.get("tags", []) and not WP_BAD_SUMMARY.search(e.get("comment", ""))]
        got = 0
        for e in sorted(ev, key=lambda e: key(f"{SEED}:prose:{e['title']}")):
            if got == 2:
                break
            d = wp({"action": "query", "titles": e["title"], "prop": "extracts|revisions|pageprops|info",
                    "explaintext": 1, "exsectionformat": "plain", "rvprop": "ids|timestamp", "ppprop": "disambiguation"})
            p = d["query"]["pages"][0]
            if p.get("missing") or p.get("redirect") or "pageprops" in p or p.get("ns") != 0:
                counts["wp_missing_redirect_disambig"] = counts.get("wp_missing_redirect_disambig", 0) + 1
                continue
            text = p.get("extract") or ""
            text = text_ok(text.encode("utf-8"), counts)
            if text is None:
                continue
            raw = text.encode("utf-8")
            rev = p["revisions"][0]
            items.append({"category": "prose", "source_kind": "wikipedia", "title": p["title"], "licence": "CC-BY-SA-4.0",
                          "created_at": e["timestamp"], "revid": rev["revid"], "rev_timestamp": rev["timestamp"],
                          "url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}",
                          "source_content_sha256": H.sha(raw), "source_byte_count": len(raw), "_text": text})
            got += 1
        if got < 2:
            rejected.append({"day": day, "reason": f"only {got} eligible articles"})
            raise SystemExit(f"prose {day}: only {got} eligible articles")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--manifest-out", required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(Path(args.tokenizer).read_bytes().decode("utf-8"))
    bdh, bdh_sha = H.load_bdh()
    record = {"schema": "c50_postrelease_v1", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "window": WINDOW,
              "weights": WEIGHTS, "seed": SEED, "span_bytes": SPAN_BYTES, "min_bytes": MIN_BYTES,
              "bdh_commit": H.BDH_COMMIT, "bdh_sha256": bdh_sha, "tokenizer_sha256": H.sha(Path(args.tokenizer).read_bytes())}
    counts, rejected = {}, []
    repos = candidates(record)
    record["candidate_repositories"] = len(repos)
    used = set()
    by_cat = {"code": github_items("code", repos, used, counts, rejected)}
    by_cat["technical_general"] = github_items("technical_general", repos, used, counts, rejected)
    by_cat["prose"] = prose_items(counts, rejected)
    rows, tsv, meta, items = [], [], [], []
    for cat in CATEGORIES:
        for rank, it in enumerate(by_cat[cat], start=1):
            text = it.pop("_text")
            span, start = bdh._bounded_span(text, it["source_content_sha256"], SEED, SPAN_BYTES)
            ids = H.tokenize(tok, span)
            it.update({"schema": "c50_postrelease_v1", "rank": rank, "span_start_byte": start,
                       "span_byte_count": len(span.encode("utf-8")), "text": span})
            it["item_sha256"] = H.sha(bdh._canonical(it))
            i = len(items)
            items.append(it)
            tsv.append(f"{i}\t" + ",".join(map(str, ids)))
            meta.append({"index": i, "category": cat, "rank": rank, "item_sha256": it["item_sha256"], "n_tokens": len(ids)})
            rows.append({k: v for k, v in it.items() if k != "text"} | {"n_tokens": len(ids)})
    corpus, inputs = Path(args.corpus_dir), Path(args.inputs_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    text_bytes = b"".join(bdh._canonical(t) + b"\n" for t in items)
    (corpus / "post.jsonl").write_bytes(text_bytes)
    (inputs / "docs.tsv").write_text("\n".join(tsv) + "\n", encoding="ascii")
    (inputs / "docs_meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta), encoding="utf-8")
    files = {"docs": {"tsv_sha256": H.sha((inputs / "docs.tsv").read_bytes()),
                      "meta_sha256": H.sha((inputs / "docs_meta.jsonl").read_bytes()),
                      "items": len(items), "tokens": sum(m["n_tokens"] for m in meta)}}
    (inputs / "manifest.json").write_text(json.dumps({"cell": "C50-A4-bis inputs", "source": "c50_postrelease_v1",
                                                      "files": files}, indent=1), encoding="utf-8")
    record.update({"jsonl_sha256": H.sha(text_bytes), "inputs": files,
                   "inputs_manifest_sha256": H.sha((inputs / "manifest.json").read_bytes()),
                   "rejection_counts": counts, "rejected": rejected, "items": rows})
    Path(args.manifest_out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    for cat in CATEGORIES:
        print(cat, len(by_cat[cat]), "items", sum(m["n_tokens"] for m in meta if m["category"] == cat), "tokens")
    print("candidate repositories", len(repos), "inputs manifest", record["inputs_manifest_sha256"][:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
