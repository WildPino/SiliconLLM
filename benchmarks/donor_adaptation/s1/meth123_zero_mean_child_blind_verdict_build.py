"""Freeze METH-123 excerpt-only A/B findings before arm unblinding."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "4503f9cf4b2ae7e4a76422ac8ae7726fb71b110f38a7bf2f1ded6dc05e435bd2"
FINDINGS = {}
MISSING = set()


def finding(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append(
        {"claim": claim, "evidence": evidence, "severe": severe})


for side in "AB":
    finding(1, side, "__call__ returns a standard HTTP request",
            "The visible signature returns object; it only begins constructing a rewritten Request.")
    finding(2, side, "C_INPUT_SHA is a hash for cross-calls",
            "The macro name identifies C input, while no calls or call hashes are shown.")
finding(2, "A", "descriptor_ok returns zero when the tensor matches",
        "The shown expression begins by returning a conjunction; a full match is truthy, not zero.", True)
for side in "AB":
    finding(3, side, "parity() calculates the parity of a given text file",
            "It parses parity-marked NATS_TOTAL entries from parity.txt; no parity calculation is shown.")
finding(3, "A", "dictionary keys are words and values are bit-identities",
        "Keys are regex group 1 labels and values are NATS_TOTAL numeric strings retained as text.")
finding(3, "B", "dictionary keys are bit-identity values and values are line numbers",
        "The loop stores group 1 as key and group 2 as value; it captures no line numbers.")
for side in "AB":
    finding(4, side, "the first visible function is named yrs",
            "The excerpt begins mid-signature at 'yers):'; the full function name is hidden.")
finding(5, "A", "the 163840-byte reference file is a current shared file",
        "That tuple is reference_routed; current_shared is the separate 40960-byte tuple.")
finding(5, "A", "the SHA-256 string is the shared size",
        "It is a hash in the reference_routed tuple, not a byte count.")
finding(5, "B", "the file belongs to a TensorFlow model",
        "No TensorFlow reference appears in the path or tuple.", True)
finding(5, "B", "the tuples are shared-memory locations",
        "They are paths, byte sizes, hashes and dtype labels for files.")
for side in "AB":
    finding(6, side, "g_e62d is the function containing the shown thresholds",
            "The displayed classification function ends before a new def g_e62d begins.")
finding(7, "A", "r_start_hash is compared to both C_START_SHA and REFERENCE_START_SHA",
        "c_start_hash is compared to C_START_SHA; r_start_hash only to REFERENCE_START_SHA.")
finding(7, "A", "negative controls are checked to be passable",
        "negative_pass requires all negative controls to have false pass values.", True)
finding(8, "A", "the young man is denied access to his own home",
        "The first speaker says he lets the addressed person dwell in his house; no denial of access appears.", True)
finding(8, "A", "someone is unjustly accused of a crime",
        "The excerpt mentions an offense in a rhetorical question, not a criminal accusation.")
finding(8, "A", "Antonio is the young man described in the first speech",
        "Antonio is identified only as the angry respondent.")
finding(8, "B", "the young man is denied access to his own home",
        "The speech says he permits dwelling in his house; no denial of access appears.", True)
finding(8, "B", "Antonio is the young man's friend",
        "No friendship relation is stated for Antonio and the young man.")
finding(8, "B", "Antonio asks the other person to understand",
        "Antonio says the other person would not understand his reasons.")
finding(9, "A", "one man holds a gun in the archery scene",
        "The passage discusses archers shooting at butts, not guns.", True)
finding(9, "A", "the others do not shoot",
        "The yeomen shoot two or three times, and their shots hit the prick.")
finding(9, "A", "William of Cloudesly has died",
        "'By Him that for me died' refers to a different figure; William then speaks.", True)
finding(9, "B", "the yeomen are likely knights or soldiers",
        "The visible stanza identifies them only as yeomen and archers.")
finding(11, "B", "the priest fears a house is about to fall",
        "A falling house is a hypothetical simile for the man's terrible feeling.", True)
for side in "AB":
    finding(12, side, "Latin edi is the common auxiliary verb",
            "The excerpt names fa as a common auxiliary; edi illustrates a completed action.")
    finding(13, side, "the man waits for the crow to move",
            "The crow waits to see whether the apparently dead man moves.", True)
finding(14, "A", "the poem is set in a school",
        "A weary school appears in a line; no school scene or setting is established.")
finding(14, "A", "the child has ready courage and the fool has wisdom",
        "The lines assign folly to the child and ready courage to the fool.", True)
finding(14, "A", "the poet criticizes the child's lack of courage",
        "No such criticism appears; the poem longs for humility and folly.")
finding(14, "B", "the child is called weary and a fool",
        "The text has 'weary school', 'folly of the child' and 'courage of the fool'.", True)
finding(14, "B", "the poet criticizes the education system",
        "The quoted lines contain no education-system critique.")
finding(14, "B", "the speaker wishes to cleanse the knowledge of the school",
        "The prayer is to crush 'our knowledge' and cleanse 'us' of wisdom.")
finding(18, "B", "the excerpt describes a specific data structure",
        "It names a manifest, two schedules and a table header, without identifying a data structure.")
finding(20, "B", "the stored types include 8-bit and 16-bit floating point",
        "The table lists BF16 and I64, which is a 64-bit integer type, not 8-bit floating point.", True)
for side in "AB":
    finding(21, side, "k=16 is the endpoint where every selector keeps everything",
            "The text places that structural endpoint at k=E; k=16 is merely a displayed row.", True)
    finding(21, side, "at k=1 the group is worth choosing",
            "The passage explicitly concludes there is nothing worth choosing.", True)
    finding(22, side, "the table concerns a trading strategy",
            "It measures neural-network carve-group timing and token rate, not financial trading.", True)
    finding(22, side, "TEN-B-NEAR-HUNDRED is tokens bought per minute",
            "The label names an approximately ten-billion-parameter configuration near 100 tokens/s.")
    finding(22, side, "straddles the edge at 80 marks maximum strategy performance",
            "The table only says a target edge is straddled at 80; no maximum is established.")
    finding(23, side, "the held-out slice contains 24 by 512 bytes",
            "The excerpt states a 24x512 slice and separately gives 51,870 bytes.")
    finding(23, side, "the slice contains 51,870 tokens",
            "The excerpt says 51,870 bytes and 12,264 tokens.")


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
    output = {"experiment": "METH-123-zero-mean-arm-blind-external-verdict",
              "blind_sha256": BLIND_SHA,
              "reviewer": "single agent, excerpt-only review",
              "rubric": "Count claims contradicted or unsupported by the visible excerpt; severe means a central entity, event or result is inverted; missing_detail means no concrete supported detail is supplied.",
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"blind_sha256": BLIND_SHA,
                      "verdict_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "A_unsupported": sum(len(r["A"]["unsupported"]) for r in rows),
                      "B_unsupported": sum(len(r["B"]["unsupported"]) for r in rows),
                      "A_severe": sum(r["A"]["severe_count"] for r in rows),
                      "B_severe": sum(r["B"]["severe_count"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
