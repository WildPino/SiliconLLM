"""Freeze anonymous exact/Q7 responses, then score a committed verdict."""

import argparse
import hashlib
import json
from pathlib import Path
import secrets


MANIFEST_SHA = "84d8dbb400be133b6d1e440d55da5d09afd8ebbe0bd659474bdad8e708868ba9"
AUDIT_SHA = "6053a6d1f5977d6f1adfdee97daf45d3d376796ce4c41fd7d2737067a8375db8"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def blind(args):
    assert sha(args.manifest.read_bytes()) == MANIFEST_SHA
    assert sha(args.audit.read_bytes()) == AUDIT_SHA
    manifest, audit = load(args.manifest), load(args.audit)
    assert audit["manifest_sha256"] == MANIFEST_SHA
    assert audit["decision"] == "automatic_pass_blind_semantic_review_pending"
    assert len(manifest["items"]) == len(audit["generation_rows"]) == 24
    assert not args.out.exists() and not args.map.exists()
    generated = {row["source_id"]:row for row in audit["generation_rows"]}
    mapping = {}
    rows = []
    for item in manifest["items"]:
        source_id=item["source_id"]
        pair=generated[source_id]
        first,second = (("exact_e1280","q7_e1280") if secrets.randbits(1)
                        else ("q7_e1280","exact_e1280"))
        mapping[source_id]={"A":first,"B":second}
        rows.append({"source_id":source_id,"category":item["category"],
                     "excerpt":item["excerpt"],
                     "A":pair[first]["continuation_text"],
                     "B":pair[second]["continuation_text"]})
    save(args.map,{"experiment":"METH-131-hidden-arm-map",
                   "manifest_sha256":MANIFEST_SHA,"audit_sha256":AUDIT_SHA,
                   "mapping":mapping})
    mapping_sha=sha(args.map.read_bytes())
    save(args.out,{"experiment":"METH-131-arm-blind-excerpt-review",
                   "manifest_sha256":MANIFEST_SHA,"audit_sha256":AUDIT_SHA,
                   "hidden_mapping_sha256":mapping_sha,
                   "rubric":"From the visible excerpt alone, count unsupported claims, severe central inversions and answers lacking a requested supported detail.",
                   "rows":rows})
    print(json.dumps({"rows":len(rows),"blind_sha256":sha(args.out.read_bytes()),
                      "hidden_mapping_sha256":mapping_sha}))


def score(args):
    blind_data,verdict,mapping=load(args.blind),load(args.verdict),load(args.map)
    assert verdict["blind_sha256"] == sha(args.blind.read_bytes())
    assert blind_data["hidden_mapping_sha256"] == sha(args.map.read_bytes())
    rows={r["source_id"]:r for r in blind_data["rows"]}
    reviewed={r["source_id"]:r for r in verdict["rows"]}
    assert len(rows)==len(reviewed)==len(mapping["mapping"])==24
    counts={arm:{key:0 for key in ("unsupported","severe","missing_detail","ambiguous")}
            for arm in ("exact_e1280","q7_e1280")}
    paired=[]
    for source_id,row in reviewed.items():
        assert source_id in rows
        item={"source_id":source_id,"category":rows[source_id]["category"]}
        for side in "AB":
            arm=mapping["mapping"][source_id][side]
            finding=row[side]
            assert isinstance(finding["missing_detail"],bool)
            assert isinstance(finding["unsupported"],list)
            assert isinstance(finding["ambiguous"],list)
            for claim in finding["unsupported"]:
                assert claim["claim"].strip() and claim["evidence"].strip()
                assert isinstance(claim["severe"],bool)
            counts[arm]["unsupported"]+=len(finding["unsupported"])
            counts[arm]["severe"]+=sum(c["severe"] for c in finding["unsupported"])
            counts[arm]["missing_detail"]+=int(finding["missing_detail"])
            counts[arm]["ambiguous"]+=len(finding["ambiguous"])
            item[arm]=finding
        paired.append(item)
    gates={key:counts["q7_e1280"][key]<=counts["exact_e1280"][key]
           for key in ("unsupported","severe","missing_detail")}
    gates["joint_semantic"]=all(gates.values())
    output={"experiment":"METH-131-unblinded-Q7-grounding-score",
            "blind_sha256":sha(args.blind.read_bytes()),
            "verdict_sha256":sha(args.verdict.read_bytes()),
            "mapping_sha256":sha(args.map.read_bytes()),
            "counts":counts,"gates":gates,"paired_rows":paired}
    save(args.out,output)
    print(json.dumps({"counts":counts,"gates":gates},indent=2))


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="mode",required=True)
    b=sub.add_parser("blind")
    b.add_argument("--manifest",type=Path,required=True)
    b.add_argument("--audit",type=Path,required=True)
    b.add_argument("--out",type=Path,required=True)
    b.add_argument("--map",type=Path,required=True)
    s=sub.add_parser("score")
    s.add_argument("--blind",type=Path,required=True)
    s.add_argument("--verdict",type=Path,required=True)
    s.add_argument("--map",type=Path,required=True)
    s.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    (blind if args.mode=="blind" else score)(args)


if __name__=="__main__":
    main()
