"""C50-A6 part 2 inputs: DEV documents 13-32 per category plus the A4 task panels, byte for byte.

docs     DEV-role items ranked 13..32 per category in role order (c50_holdout_v1 dev_ids.jsonl): items 1-12 were
         read by A5 part 2, 33-44 are A5's DEV-B.  -> docs.tsv / docs_meta.jsonl (the A4 formats)
others   piqa, arc, gsm8k, ifeval, chat, rollout copied from the A4 inputs after checking each file against
         the A4 inputs manifest (the task panels are the A4 panels, unchanged).

Usage:
    python -I benchmarks/chatbot50/c50_a6_dev_inputs.py --a4-inputs E:/.../a4/inputs --out E:/.../a6dev/inputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOLDOUT = HERE / "corpus" / "c50_holdout_v1"
FIRST, LAST = 13, 32
TASKS = ("piqa", "arc", "gsm8k", "ifeval", "chat", "rollout")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4-inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a4, out = Path(args.a4_inputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a4man = json.loads((a4 / "manifest.json").read_text(encoding="utf-8"))
    man = {"cell": "C50-A6 part 2 inputs", "a4_inputs_manifest_sha256": sha((a4 / "manifest.json").read_bytes()),
           "holdout_dev_ids_sha256": sha((HOLDOUT / "dev_ids.jsonl").read_bytes()), "dev_ranks": [FIRST, LAST], "files": {}}
    rows = [json.loads(x) for x in (HOLDOUT / "dev_ids.jsonl").read_text(encoding="utf-8").splitlines()]
    seen, docs = {}, []
    for r in rows:
        seen[r["category"]] = seen.get(r["category"], 0) + 1
        if FIRST <= seen[r["category"]] <= LAST:
            docs.append(r)
    if len(docs) != 3 * (LAST - FIRST + 1):
        raise SystemExit("fewer DEV items than registered in some category")
    data = "".join(f"{i}\t{','.join(map(str, r['ids']))}\n" for i, r in enumerate(docs)).encode("ascii")
    mdata = "".join(json.dumps({"index": i, "category": r["category"], "item_sha256": r["item_sha256"],
                                "n_tokens": r["n_tokens"]}) + "\n" for i, r in enumerate(docs)).encode("utf-8")
    (out / "docs.tsv").write_bytes(data)
    (out / "docs_meta.jsonl").write_bytes(mdata)
    man["files"]["docs"] = {"tsv_sha256": sha(data), "meta_sha256": sha(mdata), "items": len(docs),
                            "tokens": sum(r["n_tokens"] for r in docs)}
    print("docs", len(docs), "items", man["files"]["docs"]["tokens"], "tokens", man["files"]["docs"]["tsv_sha256"])
    for name in TASKS:
        t, m = (a4 / f"{name}.tsv").read_bytes(), (a4 / f"{name}_meta.jsonl").read_bytes()
        rec = a4man["files"][name]
        if sha(t) != rec["tsv_sha256"] or sha(m) != rec["meta_sha256"]:
            raise SystemExit(f"{name}: A4 input differs from the A4 manifest")
        (out / f"{name}.tsv").write_bytes(t)
        (out / f"{name}_meta.jsonl").write_bytes(m)
        man["files"][name] = {"tsv_sha256": rec["tsv_sha256"], "meta_sha256": rec["meta_sha256"], "copied_from_a4": True}
        print(name, "copied", rec["tsv_sha256"])
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
