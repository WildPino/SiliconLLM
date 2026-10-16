"""C50-A2: verify a downloaded Hugging Face snapshot against the pinned revision.

For every file of the pinned revision the API reports either an LFS sha256 or a
git blob id.  This hashes the local copy (sha256 for LFS files, git blob sha1
for small files) and writes a manifest; any mismatch or missing file fails.

Standard library only.  Usage:
    python -I benchmarks/chatbot50/a2_source_manifest.py --repo inclusionAI/Ling-mini-2.0 \
        --revision <sha> --dir <local snapshot> --json <manifest.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def api_files(repo: str, rev: str) -> dict:
    with urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true", timeout=60) as h:
        rec = json.load(h)
    if rec.get("sha") != rev:
        raise SystemExit(f"revision mismatch: {rec.get('sha')} != {rev}")
    return {s["rfilename"]: s for s in rec["siblings"]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    root = Path(args.dir)
    print(f"CONFIG cell=C50-A2-manifest repo={args.repo} revision={args.revision} dir={root}")
    rows, failures = [], 0
    for name, sib in sorted(api_files(args.repo, args.revision).items()):
        path = root / name
        row = {"file": name, "remote_size": sib.get("size")}
        if not path.is_file():
            row["status"] = "MISSING"
        elif sib.get("lfs"):
            row.update(local_size=path.stat().st_size, expected_sha256=sib["lfs"]["sha256"],
                       local_sha256=sha256_file(path))
            row["status"] = "OK" if (row["local_sha256"] == row["expected_sha256"]
                                     and row["local_size"] == sib["lfs"]["size"]) else "MISMATCH"
        else:
            row.update(local_size=path.stat().st_size, expected_blob=sib.get("blobId"),
                       local_blob=git_blob_sha1(path), local_sha256=sha256_file(path))
            row["status"] = "OK" if row["local_blob"] == row["expected_blob"] else "MISMATCH"
        failures += row["status"] != "OK"
        rows.append(row)
        print(f"{row['status']:8s} {name} {row.get('local_size')}")
    out = {"cell": "C50-A2", "repo": args.repo, "revision": args.revision, "files": rows,
           "verdict": "PASS" if failures == 0 else "FAIL"}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"verdict={out['verdict']} files={len(rows)} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
