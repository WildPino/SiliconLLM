"""Freeze A/B continuations, then unblind a previously committed verdict."""

import argparse
import hashlib
import json
from pathlib import Path


SEED = "meth74-blind-74074"
MANIFEST_SHA = "11621dce9638aae01cc3bd8c76730de7676e4bbb704271056f164b9ad4d5b437"
AUDIT_SHA = "f43658cd8c8e1a791237a7bffa1203f518cc6836a7dab4f725b3d2fb0bef2696"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def donor_is_a(source_id):
    return int(sha((SEED + "|" + source_id).encode("utf-8"))[:16], 16) % 2 == 0


def blind(args):
    assert sha(args.manifest.read_bytes()) == MANIFEST_SHA
    assert sha(args.audit.read_bytes()) == AUDIT_SHA
    manifest = read(args.manifest)
    audit = read(args.audit)
    assert audit["external_manifest_sha256"] == MANIFEST_SHA
    assert audit["decision"] == "stop_instruct_parent_promotion"
    items = {r["source_id"]: r for r in manifest["items"]}
    generated = {r["source_id"]: r for r in audit["generation_rows"]}
    assert len(items) == len(generated) == 24
    rows = []
    for source_id, item in items.items():
        pair = generated[source_id]
        first, second = (("donor", "student") if donor_is_a(source_id)
                         else ("student", "donor"))
        rows.append({"source_id": source_id, "category": item["category"],
                     "excerpt": item["excerpt"],
                     "A": pair[first]["continuation_text"],
                     "B": pair[second]["continuation_text"]})
    output = {"experiment": "METH-74-blind-semantic-review",
              "manifest_sha256": MANIFEST_SHA,
              "audit_sha256": AUDIT_SHA,
              "review_rule": "Count unsupported, severe and missing-detail findings from the excerpt alone.",
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"rows": len(rows), "blind_sha256": sha(args.out.read_bytes())}))


def score(args):
    blind_data = read(args.blind)
    verdict = read(args.verdict)
    assert verdict["blind_sha256"] == sha(args.blind.read_bytes())
    records = {r["source_id"]: r for r in blind_data["rows"]}
    reviewed = {r["source_id"]: r for r in verdict["rows"]}
    assert len(records) == len(reviewed) == 24
    counts = {arm: {key: 0 for key in ("unsupported", "severe", "missing_detail", "ambiguous")}
              for arm in ("donor", "student")}
    paired = []
    for source_id, row in reviewed.items():
        assert source_id in records
        item = {"source_id": source_id, "category": records[source_id]["category"]}
        for side, arm in (("A", "donor" if donor_is_a(source_id) else "student"),
                          ("B", "student" if donor_is_a(source_id) else "donor")):
            finding = row[side]
            assert isinstance(finding["missing_detail"], bool)
            assert isinstance(finding["unsupported"], list)
            assert isinstance(finding["ambiguous"], list)
            for claim in finding["unsupported"]:
                assert claim["claim"].strip() and claim["evidence"].strip()
                assert isinstance(claim["severe"], bool)
            counts[arm]["unsupported"] += len(finding["unsupported"])
            counts[arm]["severe"] += sum(c["severe"] for c in finding["unsupported"])
            counts[arm]["missing_detail"] += int(finding["missing_detail"])
            counts[arm]["ambiguous"] += len(finding["ambiguous"])
            item[arm] = finding
        paired.append(item)
    gates = {key: counts["student"][key] <= counts["donor"][key]
             for key in ("unsupported", "severe", "missing_detail")}
    gates["joint_semantic"] = all(gates.values())
    output = {"experiment": "METH-74-unblinded-semantic-score",
              "blind_sha256": sha(args.blind.read_bytes()),
              "verdict_sha256": sha(args.verdict.read_bytes()),
              "counts": counts, "gates": gates, "paired_rows": paired}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"counts": counts, "gates": gates}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("blind")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--audit", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--blind", type=Path, required=True)
    s.add_argument("--verdict", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (blind if args.mode == "blind" else score)(args)


if __name__ == "__main__":
    main()
