"""Freeze A/B continuations, then unblind a previously committed verdict."""

import argparse
import hashlib
import json
from pathlib import Path


SEED = "meth123-zero-mean-child-blind-external-123000"
MANIFEST_SHA = "7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366"
AUDIT_SHA = "db24228ceabd2b00ecc30a24e9e83b5d20278c62edb204b23b5522473d0295c1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parent_is_a(source_id):
    return int(sha((SEED + "|" + source_id).encode("utf-8"))[:16], 16) % 2 == 0


def blind(args):
    assert sha(args.manifest.read_bytes()) == MANIFEST_SHA
    assert sha(args.audit.read_bytes()) == AUDIT_SHA
    manifest = read(args.manifest)
    audit = read(args.audit)
    assert audit["manifest_sha256"] == MANIFEST_SHA
    assert audit["decision"] == "automatic_pass_blind_semantic_review_pending"
    assert audit["transform"] == "parent_B_plus_child_B_minus_per_parent_child_mean"
    items = {r["source_id"]: r for r in manifest["items"]}
    generated = {r["source_id"]: r for r in audit["generation_rows"]}
    assert len(items) == len(generated) == 24
    rows = []
    for source_id, item in items.items():
        pair = generated[source_id]
        first, second = (("bf16_e128", "bf16_e1280") if parent_is_a(source_id)
                         else ("bf16_e1280", "bf16_e128"))
        rows.append({"source_id": source_id, "category": item["category"],
                     "excerpt": item["excerpt"],
                     "A": pair[first]["continuation_text"],
                     "B": pair[second]["continuation_text"]})
    output = {"experiment": "METH-123-zero-mean-E1280-blind-external-review",
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
              for arm in ("e128", "e1280")}
    paired = []
    for source_id, row in reviewed.items():
        assert source_id in records
        item = {"source_id": source_id, "category": records[source_id]["category"]}
        for side, arm in (("A", "e128" if parent_is_a(source_id) else "e1280"),
                          ("B", "e1280" if parent_is_a(source_id) else "e128")):
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
    gates = {key: counts["e1280"][key] <= counts["e128"][key]
             for key in ("unsupported", "severe", "missing_detail")}
    gates["joint_semantic"] = all(gates.values())
    output = {"experiment": "METH-123-zero-mean-E1280-unblinded-external-score",
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
