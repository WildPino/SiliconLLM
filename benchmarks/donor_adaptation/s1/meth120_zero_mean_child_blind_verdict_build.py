"""Freeze excerpt-only METH-120 A/B findings before revealing arm identities."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "33a13dd05d17c274351359529d63c837d0b9943c5ab209e6d2a04b3d996186c2"
FINDINGS = {}
MISSING = set()


def finding(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append(
        {"claim": claim, "evidence": evidence, "severe": severe})


for side in "AB":
    finding(0, side, "BPB expands to Backpropagation Through Time Bias",
            "The visible method reads BPB from a held-out text slice; that expansion is not stated.")
finding(0, "A", "the SVD is computed on the small side of the network",
        "The excerpt says the small side of each weight/Gram matrix, not a network side.")
finding(0, "B", "the operation is already applied and takes minutes",
        "The excerpt presents a method and predicts minutes, not a completed timing.")
finding(1, "A", "the damaged blob has a different record format",
        "Only a damaged byte and an expected reserved-W4 error are shown.")
finding(1, "B", "the damaged blob has 1024 bytes",
        "No blob length is visible.")
finding(1, "B", "the scale region is a 17 by 17 by 2 byte array",
        "The code gives a scalar offset 17 * 17 * 2 and changes one byte, not an array shape.")
finding(2, "A", "the parser includes a build-directory path option",
        "The visible path is the default for --r512; no build-directory argument is shown.")
finding(2, "B", "the parser configures a model training process",
        "The flags show a probe/script interface, without a training process in the excerpt.")
finding(2, "B", "the binary file contains model data of a specified kind",
        "--r512 only shows a default .bin path; its content is not visible.")
finding(4, "A", "the supplied stats update the bundle's own RMS",
        "The help text says the bundle's own RMS overwrites the supplied stats.")
finding(4, "B", "the labels file is named E256.npz",
        "The help text explicitly names labels_E256.npz.")
finding(5, "A", "the visible function returns a new Wnew transformation matrix",
        "The shown lines evaluate a reconstruction and assign rec, per; no return is shown.")
finding(5, "A", "the function returns zero_frac_achieved",
        "zero_frac_achieved appears as an argument to evaluate, not a visible return.")
finding(5, "A", "Wnew is the same as L with updated weights",
        "Wnew[L] supplies weights copied into W; L is a layer index, not a weight matrix.")
finding(5, "B", "the visible function returns reconstructed data",
        "Only an evaluate call and assignment to rec, per are visible.")
for side in "AB":
    finding(6, side, "rat01_l1up_project is a Python function",
            "The excerpt is C syntax in a .h file, with pointer-like calls and memcpy.", True)
finding(6, "A", "execution continues after an unsuccessful projection",
        "Each false projection condition goes directly to finish.", True)
finding(7, "A", "M29 is the function iterating over counts",
        "M29 appears as a module namespace for add_count and finalize_count.")
finding(8, "A", "King Svend pledges faith to Si at the Isle of Svald",
        "The Isle of Svald belongs to the preceding stanza; the pledge line cuts off at Si.")
finding(8, "B", "Si is likely the Norse god Odin",
        "The line is truncated at Si and supplies no identity, let alone Odin.", True)
finding(8, "B", "the poem ends with the pledge",
        "The displayed excerpt ends mid-word and does not identify the poem's ending.")
finding(10, "A", "Delio doubts Antonio's intentions",
        "Delio says he doubts the hope of reconciliation, not Antonio's sincerity.")
for side in "AB":
    finding(10, side, "the Marquis of Pescara threatens to capture Antonio",
            "The visible text calls safe-conduct letters possible nets; it cuts off during the Marquis clause.")
finding(11, "A", "the conversation is with a deity",
        "The speakers address each other as monsieur and child; no deity appears.", True)
finding(11, "A", "the deity replies cryptically to a request for forgiveness",
        "The man says he has no reproaches and knows her motives; the woman states her love.", True)
finding(11, "B", "the man asks the woman for forgiveness",
        "The woman asks whether the man can forgive her for sending him to death.", True)
finding(11, "B", "the woman refuses to forgive or reproach him",
        "The man's reply says he has no reproaches for her.", True)
for side in "AB":
    finding(12, side, "the person sought is a friend",
            "The command explicitly says 'Finde out thy brother'.", True)
    finding(12, side, "absence is due to communication trouble or misunderstanding",
            "The speaker orders a search and threatens exile after twelve months.")
    finding(12, side, "the command is to avoid seeing the person alive in the territory",
            "The brother must be brought dead or living; the threatened exile concerns the searcher.", True)
for side in "AB":
    finding(13, side, "Jeanne and Hannibal are the two conversational speakers",
            "One speaker addresses Jeanne and proposes Hannibal as a name for her protege.", True)
    finding(13, side, "the man being discussed is named Jeanne",
            "Jeanne is addressed by name; the man's proposed name is Hannibal.", True)
    finding(13, side, "Hannibal suggests calling the man Hannibal",
            "Hannibal is the suggested name, not an identified speaker.")
finding(13, "B", "the author is clearly amused",
        "The excerpt records dialogue but no authorial reaction.")
for side in "AB":
    finding(14, side, "Farmer Green is the tailor",
            "The excerpt names Mr. Frog as the tailor.", True)
finding(17, "A", "the probe is already a working model",
        "The passage says the goal is a working model and still needs its two halves joined.")
finding(17, "B", "99.7 tokens/s is half of the goal speed",
        "The passage calls this the speed half of a two-part goal, not half the speed.", True)
for side in "AB":
    finding(19, side, "the table compares Kcur-2 and Vcur-2 values for the Vcur-2 variable",
            "The excerpt lists separate reference and C payload rows and hashes.")
    finding(19, side, "the kqv_out-2 hash is a Vcur-2 reference value",
            "It is the reference target hash for the separate kqv_out-2 payload.")
finding(21, "A", "the byte-identical top-four IDs prove a data structure is immutable",
        "Only the ID payload and fixed slot order are stated, not an immutable data structure.")
finding(21, "B", "the passage describes a specific data structure",
        "It discusses an evidence binding and top-four IDs, without naming a data structure.")
finding(22, "A", "the result shows gathered weights cost more than streamed weights",
        "The registered below-40 trigger does not fire: K16 is 41.41 and K4 is 41.67.", True)
finding(22, "B", "the result shows streamed weights cost more than gathered weights",
        "The excerpt only says the registered below-40 trigger does not fire; it does not establish the reverse.")
finding(23, "A", "a defect is shared between the two binaries",
        "The text explicitly says independently built binaries cannot share a harness defect.", True)


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
    output = {"experiment": "METH-120-zero-mean-arm-blind-development-verdict",
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
