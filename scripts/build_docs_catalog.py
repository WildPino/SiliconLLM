#!/usr/bin/env python3
"""Build or check a deterministic navigation catalog for docs/**/*.md.

This is discovery metadata only. Filename-derived record types do not assert
scientific success, validity, or current operational status.

Reparse rule: follow only the approved NATIVE_EXPERT_SCALING_20260925 campaign
junction when it resolves to the same relative docs path under the fixed
E:/SiliconLLM_Artifacts/workspace/docs storage mirror. Catalog paths remain
logical paths under this repository's docs directory. All other symlinks and
junctions are skipped. This keeps catalog bytes stable across that migration.

Usage: python scripts/build_docs_catalog.py build
       python scripts/build_docs_catalog.py check
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote, unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NAV = DOCS / "navigation"
CATALOG = NAV / "catalog.json"
INDEX = NAV / "index.html"
START = ROOT / "START_HERE.md"
APPROVED_JUNCTION = DOCS / "research" / "NATIVE_EXPERT_SCALING_20260925"
APPROVED_STORAGE_TARGET = (Path("E:/SiliconLLM_Artifacts/workspace/docs") /
                           "research" / "NATIVE_EXPERT_SCALING_20260925")
HEADING = re.compile(r"^#\s+(.+?)\s*$")
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def markdown_files() -> list[Path]:
    found: list[Path] = []
    def visit(folder: Path) -> None:
        for path in sorted(folder.iterdir(), key=lambda p: p.name):
            if path == NAV:
                continue
            reparse = path.is_symlink() or path.is_junction()
            if path == APPROVED_JUNCTION and reparse:
                require(path.is_dir(), f"approved campaign junction is broken: {path}")
            if path.is_dir():
                if reparse:
                    if path != APPROVED_JUNCTION:
                        continue
                    require(path.resolve(strict=True) ==
                            APPROVED_STORAGE_TARGET.resolve(strict=True),
                            f"approved campaign junction target changed: {path}")
                visit(path)
            elif path.suffix.lower() == ".md" and not reparse:
                found.append(path)

    visit(DOCS)
    return sorted(found, key=lambda p: p.relative_to(ROOT).as_posix())


def title_of(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            match = HEADING.match(line.rstrip("\r\n"))
            if match:
                return match.group(1).strip()
    return path.stem


def record_type(path: Path) -> str:
    name = path.stem.upper()
    if name in ("INDEX", "README") or name.endswith("_INDEX"):
        return "index"
    if "HANDOFF" in name or name.startswith("PAUSE"):
        return "handoff"
    if "METHOD" in name:
        return "method"
    if "PROTOCOL" in name or "PREREG" in name:
        return "protocol"
    if "RESULT" in name or "OUTCOME" in name:
        return "result"
    if "DECISION" in name or "ADJUDICATION" in name:
        return "decision"
    if any(word in name for word in ("EVIDENCE", "LITERATURE", "REVIEW", "SOURCES")):
        return "evidence"
    if "STATUS" in name or name.endswith("_STATE"):
        return "status"
    if any(word in name for word in ("PLAN", "ROADMAP", "BRIEF")):
        return "plan"
    return "note"


def campaign_of(path: Path) -> str:
    parts = path.relative_to(DOCS).parts
    if len(parts) == 1:
        return "docs (general)"
    if parts[0] == "research":
        return parts[1] if len(parts) > 2 else "research (general)"
    return parts[0]


def catalog_document() -> dict:
    items = []
    for path in markdown_files():
        relative = path.relative_to(ROOT).as_posix()
        href = "../" + "/".join(quote(part) for part in path.relative_to(DOCS).parts)
        items.append({"title": title_of(path), "path": relative,
                      "campaign": campaign_of(path),
                      "record_type": record_type(path), "href": href})
    require(len(items) == len({item["path"] for item in items}), "duplicate catalog path")
    require(all(item["title"] for item in items), "empty document title")
    return {"schema": "SILICONLLM_DOCS_CATALOG_V1",
            "scope": "Markdown discovery; record types derive from filenames, not scientific outcomes",
            "count": len(items),
            "campaign_counts": dict(sorted(Counter(item["campaign"] for item in items).items())),
            "record_type_counts": dict(sorted(Counter(item["record_type"] for item in items).items())),
            "items": items}


def catalog_bytes(document: dict) -> bytes:
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def html_bytes(document: dict) -> bytes:
    # Escaping '<' keeps document titles from terminating the embedded script.
    embedded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SiliconLLM document catalog</title>
<style>
  :root { color-scheme: light; font: 16px/1.5 system-ui, sans-serif; }
  body { max-width: 82rem; margin: 2rem auto; padding: 0 1.25rem; color: #17212b; }
  h1 { margin-bottom: .25rem; }
  .note { color: #52606d; max-width: 65rem; }
  .controls { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1.5rem 0; }
  input, select { font: inherit; padding: .55rem; min-width: 14rem; }
  input { flex: 2 1 20rem; }
  select { flex: 1 1 14rem; }
  #count { color: #52606d; }
  ul { list-style: none; padding: 0; }
  li { border-top: 1px solid #d7dde2; padding: .7rem 0; }
  li a { color: #1558a5; font-weight: 600; text-decoration: none; }
  li a:hover, li a:focus { text-decoration: underline; }
  small { color: #52606d; display: block; overflow-wrap: anywhere; }
</style>
</head>
<body>
<p><a href="../../START_HERE.md">Start here</a></p>
<h1>Scientific document catalog</h1>
<p class="note">Search and filter the original Markdown files. The labels describe filenames and folders; they do not certify a result or override the current handoff.</p>
<div class="controls">
  <input id="query" type="search" aria-label="Search titles and paths" placeholder="Search titles and paths">
  <select id="campaign" aria-label="Campaign"><option value="">All campaigns</option></select>
  <select id="kind" aria-label="Record type"><option value="">All record types</option></select>
</div>
<p id="count" aria-live="polite"></p>
<ul id="results"></ul>
<script type="application/json" id="catalog">__CATALOG__</script>
<script>
const data = JSON.parse(document.getElementById('catalog').textContent);
const query = document.getElementById('query');
const campaign = document.getElementById('campaign');
const kind = document.getElementById('kind');
const results = document.getElementById('results');
const count = document.getElementById('count');
function options(select, values) {
  for (const value of values) {
    const option = document.createElement('option');
    option.value = value; option.textContent = value; select.append(option);
  }
}
options(campaign, Object.keys(data.campaign_counts));
options(kind, Object.keys(data.record_type_counts));
function render() {
  const text = query.value.trim().toLocaleLowerCase();
  const matching = data.items.filter(item =>
    (!text || (item.title + ' ' + item.path).toLocaleLowerCase().includes(text)) &&
    (!campaign.value || item.campaign === campaign.value) &&
    (!kind.value || item.record_type === kind.value));
  count.textContent = `${matching.length} of ${data.count} Markdown documents`;
  results.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (const item of matching) {
    const row = document.createElement('li');
    const link = document.createElement('a');
    link.href = item.href; link.textContent = item.title;
    const meta = document.createElement('small');
    meta.textContent = `${item.path} · ${item.campaign} · ${item.record_type}`;
    row.append(link, meta); fragment.append(row);
  }
  results.append(fragment);
}
query.addEventListener('input', render);
campaign.addEventListener('change', render);
kind.addEventListener('change', render);
render();
</script>
</body>
</html>
"""
    return page.replace("__CATALOG__", embedded).encode("utf-8")


def check_links(document: dict, generated_may_be_missing: bool) -> int:
    for item in document["items"]:
        target = ROOT / item["path"]
        require(target.is_file() and target.suffix.lower() == ".md",
                f"catalog path does not exist: {item['path']}")
        linked = (NAV / unquote(item["href"])).resolve()
        require(linked == target.resolve(), f"catalog link mismatch: {item['path']}")
    require(START.is_file(), "START_HERE.md is missing")
    checked = 0
    for link in MARKDOWN_LINK.findall(START.read_text(encoding="utf-8")):
        if "://" in link or link.startswith("#"):
            continue
        target = (START.parent / unquote(link.split("#", 1)[0])).resolve()
        if generated_may_be_missing and target in (CATALOG.resolve(), INDEX.resolve()):
            checked += 1
            continue
        require(target.is_file(), f"broken START_HERE link: {link}")
        checked += 1
    return document["count"] + checked


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".catalog-",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check"))
    args = parser.parse_args()
    document = catalog_document()
    checked_links = check_links(document, generated_may_be_missing=args.mode == "build")
    expected = {CATALOG: catalog_bytes(document), INDEX: html_bytes(document)}
    if args.mode == "build":
        for path, payload in expected.items():
            write_atomic(path, payload)
        checked_links = check_links(document, generated_may_be_missing=False)
    else:
        for path, payload in expected.items():
            require(path.is_file() and path.read_bytes() == payload,
                    f"catalog output is missing or stale: {path}")
    print(json.dumps({"mode": args.mode, "markdown_documents": document["count"],
                      "checked_links": checked_links, "broken_links": 0,
                      "catalog": str(CATALOG), "index": str(INDEX)}, sort_keys=True))


if __name__ == "__main__":
    main()
