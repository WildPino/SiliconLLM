"""Record excerpt-only METH-105 A/B judgments before arm unblinding."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "46aa702a2f337c7d9a1b304a0f7231c7a8f7b3b3d99ef58b7713ea137927e22a"
FINDINGS = {}
MISSING = {(17, "B")}


def finding(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append({
        "claim": claim, "evidence": evidence, "severe": severe})


finding(0, "A", "summarize returns a total divided by the number of rows",
        "Only out = {'items': len(rows)} is visible; no division or return is shown.")
finding(0, "B", "summarize returns a sum of NLL values over rows",
        "The excerpt shows the beginning of summarize with an item count, not an NLL sum.")
for side in "AB":
    finding(1, side, "descriptive takes only two inputs",
            "The shown signature is descriptive(candidate, reference, shape).")
finding(2, "A", "args is an optional argument to _run_parent",
        "The signature requires args; only smoke is keyword-only.")
finding(2, "A", "_run_parent returns the number of successful preflight runs",
        "The excerpt shows a return type int but no return statement or run count.")
finding(2, "B", "_launch_preflight is nested inside _run_parent",
        "The excerpt shows a call to _launch_preflight, not a nested definition.")
for side in "AB":
    finding(3, side, "Counter counts words in a file",
            "Counter is imported, but no word-count operation is visible.")
finding(3, "A", "the excerpt verifies file integrity with a hash",
        "Only hash constants and the start of a sha256(path) definition are visible.")
finding(3, "B", "the script checks that a file is valid JSON",
        "json is imported, but no JSON parsing or validity check appears.")
finding(3, "B", "the file hash is compared with an expected value",
        "The excerpt shows constants and a function signature, not a comparison.")
for side in "AB":
    finding(4, side, "an artifact is accepted by a model",
            "The variable model names a file path checked for existence and size; no model acceptance operation is shown.")
    finding(5, side, "both --out and --tensor-types-out are output directories",
            "They are Path arguments; the excerpt does not identify either as a directory.")
finding(6, "A", "attn_norm and q are used together in the reshaped tensor",
        "attn_norm-1 is a separate post-RMSNorm item; q-1 is the reshape item.")
finding(6, "B", "the attn input is reshaped and then normalized",
        "The entries show post-RMSNorm attn_norm-1 before direct Q after reshape q-1.")
finding(6, "B", "the whole attn tensor is reshaped to 192x32x8",
        "The 192x32x8 shape belongs to q-1, not the whole attn object.")
for side in "AB":
    finding(7, side, "quantization is not specified for the fold/rotation entries",
            "The comments explicitly say 'no quant' for perm, XH and XO.")
    finding(7, side, "Q is R3 after folding only",
            "Q is annotated 'R3, no fold, no rotation'; N is the folding-only entry.")
finding(8, "A", "the man is independent",
        "The excerpt says he stands apart and has divine calmness, not that he is independent.")
for side in "AB":
    finding(9, side, "the man had taken his weapons with him",
            "He intended to take weapons but could find only a large dagger before leaving.")
    finding(10, side, "Napoleon rivaled Joseph Bonaparte, described as his sister-in-law",
            "An unnamed man felt rivalry with Napoleon and married Joseph Bonaparte's sister-in-law.", True)
    finding(10, side, "Napoleon made the match with Joseph Bonaparte's sister-in-law",
            "The unnamed man made that match and later gained Sweden's throne.", True)
finding(12, "B", "the cited tribes are Native American",
        "The citation explicitly names Native Tribes of Australia.", True)
finding(12, "B", "Thomas's work is part of Hartland's collection",
        "Thomas and Hartland appear as separate citations.")
finding(12, "B", "Primitive Paternity was written by Dalton",
        "The citation attributes Primitive Paternity to Hartland.", True)
finding(12, "B", "Dalton's Primitive Paternity is part of Tylor's collection",
        "Dalton is cited for Ethnology of Bengal; Tylor is a separate later citation.")
finding(13, "A", "the woman stares at a birthmark behind her own ear",
        "Waite turns her toward the light and stares at the mark.", True)
finding(13, "A", "the sheriff is stricken dumb",
        "The excerpt says old Waite is seemingly stricken dumb.")
finding(13, "A", "the woman wipes her eyes with her coat sleeve",
        "Waite runs his sleeve across his eyes, then appears dazed.", True)
finding(13, "B", "the woman stares at a birthmark behind her own ear",
        "Waite is the person looking at her birthmark.", True)
finding(13, "B", "the sheriff fears for the woman's sanity",
        "The woman shrinks from Waite as though she fears he is crazed.")
finding(13, "B", "the woman turns and wipes her eyes with her coat sleeve",
        "Waite turns to face the sheriff and wipes his own eyes.", True)
for side in "AB":
    finding(14, side, "the woman left behind is the young man's elderly mother",
            "The excerpt names Mimotchka but gives no mother relationship or elderly age.", True)
    finding(14, side, "the young man's hard work causes his death",
            "His possible death is a hypothetical question, not a consequence of working hard.")
finding(15, "A", "the ringleader is being arrested",
        "He drops to his knees and implores pardon; no arrest is shown.")
finding(15, "A", "the guard covers his own body to show surrender",
        "The guard aims a pistol at the ringleader; the ringleader implores pardon.", True)
finding(15, "B", "the train-letter apparatus is used during the guard's confrontation",
        "The apparatus begins a new topic after that scene; no simultaneous use is stated.")
for side in "AB":
    finding(16, side, "the completed diagnostic ran on a benchmark dataset",
            "The excerpt names a protocol, producer commit, invocation count and raw directory, but no dataset.")
finding(17, "A", "the total has 32 entries",
        "The visible 32 counts belong to technical_general and prose category rows.")
finding(17, "A", "60.561 and 63.820 are Category 1 and Category 2 labels",
        "They are numeric entries in category rows, not category names.")
finding(17, "A", "the operational verdict equals 0.031954590155327",
        "That number appears in the prose row; the text states the total exceeds +0.02 by +0.027177891834178.", True)
finding(17, "B", "the report has 32 total entries",
        "The visible 32 values are category counts, not a report-wide count.")
finding(17, "B", "60.561 and 63.820 are category averages",
        "The table header is absent from the excerpt; their statistic is not identified.")
finding(17, "B", "0.5574 and 0.8536 are standard deviations",
        "No standard-deviation label appears in the excerpt.")
finding(17, "B", "the prose average is 85.3600576543356",
        "The excerpt shows 0.853600576543356 and 63.820, not 85.3600576543356.", True)
finding(20, "A", "the excerpt compares two gradient-based optimization methods",
        "It discusses mass selection, an observed E38 control and untested gradient/output-aligned criteria.")
finding(20, "A", "the counterexample results in G-E38C being used",
        "The visible counterexample states that mass chooses component 1, not which control is used.")
finding(20, "B", "the excerpt compares G-E38C with a mathematical theorem about BPB",
        "It explicitly calls G-E38C an observed comparison, not a mathematical theorem.")
for side in "AB":
    finding(21, side, "a specific bitnet.cpp code issue is tagged [S]",
            "The [S] rule applies to source-code findings generally; the visible bitnet.cpp TL1 heading is P1 published.")
    finding(22, side, "the project AVX2 transcription and pinned x86 oracle disagree",
            "The excerpt explicitly says they agree exactly and both differ from ffn_out-0.", True)
    finding(23, side, "HumanEval records the listed BF16 and Q4 scores",
            "HumanEval is pending; those 1,466/1,838 and 1,456/1,838 results are paired PIQA.", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert hashlib.sha256(args.blind.read_bytes()).hexdigest() == BLIND_SHA
    blind = json.loads(args.blind.read_text(encoding="utf-8"))
    assert len(blind["rows"]) == 24
    rows = []
    for index, pair in enumerate(blind["rows"]):
        row = {"source_id": pair["source_id"]}
        for side in "AB":
            claims = FINDINGS.get((index, side), [])
            row[side] = {"unsupported": claims,
                         "severe_count": sum(bool(c["severe"]) for c in claims),
                         "missing_detail": (index, side) in MISSING,
                         "ambiguous": []}
        rows.append(row)
    assert all(0 <= i < 24 and side in "AB" for i, side in FINDINGS)
    result = {"experiment": "METH-105-arm-blind-semantic-verdict",
              "blind_sha256": BLIND_SHA,
              "reviewer": "single agent, excerpt-only review",
              "rubric": "Count claims contradicted or not supported by the visible 384-character excerpt; severe means a central entity, event or result is inverted; missing_detail means no concrete supported detail is supplied.",
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"blind_sha256": BLIND_SHA,
                      "verdict_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "A_unsupported": sum(len(r["A"]["unsupported"]) for r in rows),
                      "B_unsupported": sum(len(r["B"]["unsupported"]) for r in rows),
                      "A_severe": sum(r["A"]["severe_count"] for r in rows),
                      "B_severe": sum(r["B"]["severe_count"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
