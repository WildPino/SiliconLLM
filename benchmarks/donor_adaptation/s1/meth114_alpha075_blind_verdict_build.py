"""Record excerpt-only METH-114 alpha=0.75 A/B findings before unblinding."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "742099d40b7e8d68701125e9581dca8fc81677dd0f2d586e682b21ef35d66542"
FINDINGS = {}
MISSING = set()


def finding(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append(
        {"claim": claim, "evidence": evidence, "severe": severe})


for side in "AB":
    finding(0, side, "test_inventory_predecessors_and_source_controls returns the shown dictionary",
            "Its definition begins only after the return dictionary in the excerpt.", True)
    finding(1, side, "the q_proj and o_proj weights are stored in a dictionary",
            "The excerpt stops at mod.weight.da and shows no storage operation.")
    finding(1, side, "the loop's elapsed time is calculated",
            "Only t0 = time.time() is visible; no elapsed-time calculation appears.")
finding(2, "A", "source is an image",
        "The excerpt shows source rows and W2 BF16 encoding but never identifies an image.", True)
finding(2, "A", "source has shape (rows, groups, channels)",
        "Only source.shape[1] appears; no three-dimensional shape is shown.")
finding(2, "A", "tiles have shape (rows, groups, local_rows, local_groups, channels)",
        "The visible tile.a_bits.shape provides only local_rows and local_groups.")
finding(2, "A", "the offsets slice the source image for encoding",
        "The code encodes batch first and uses offsets for target_rows and a, not for slicing source afterwards.")
finding(2, "B", "source is an image",
        "The excerpt shows source rows and W2 BF16 encoding but never identifies an image.")
finding(3, "A", "h1_heldout.npz contains labels_E256.npz and h1_actstats.npz",
        "Those are separately loaded files after the heldout file.")
finding(3, "A", "the labels and stats arrays are checked for shape and values",
        "The visible shape/value assertions apply to ids and byts, before labels_np and stats_np are loaded.")
finding(5, "A", "the tie rule compares the absolute difference between a and b twice",
        "It compares |magnitude-b| with |magnitude-a|.", True)
finding(5, "B", "the function returns 0 when a<b and 1 when a>b",
        "The return depends on |magnitude-b| <= |magnitude-a|, not the ordering of a and b.", True)
finding(5, "B", "_w2_bf16_group_scalar_reference is nested in the prior function",
        "The visible def is at module level, after the prior return.")
finding(5, "B", "the group reference returns three integers from the comparison",
        "Its annotation is tuple[np.uint16, np.uint16, np.ndarray], with no such comparison result shown.")
finding(6, "A", "the loop checks that selected exists in the schema",
        "The excerpt ends at selected=selection.get(name), before any such check.")
finding(6, "B", "the loop checks that selected matches shape",
        "The excerpt ends at selected=selection.get(name), before any shape comparison.")
for side in "AB":
    finding(7, side, "the truncated first function is named n",
            "The excerpt begins mid-signature at n(value...); its full name is not visible.")
finding(7, "A", "that rounding expression calculates storage bytes for a value",
        "The shown expression rounds a number to an alignment; bytes are not assigned to it.")
finding(7, "B", "it converts to a 32-bit integer by leading-zero padding",
        "It rounds value upward to a multiple of alignment; no bit width or zero-padding operation appears.", True)
finding(7, "B", "it counts elements fitting in an alignment",
        "The expression rounds value, rather than counting elements per alignment block.")
for side in "AB":
    finding(8, side, "the debate is a Council plenary session",
            "The speaker says 'here in plenary'; the Presidency in the Council put forward a compromise.", True)
    finding(9, side, "Mr Schulz is the President giving the speech",
            "The speaker addresses 'Mr President' and separately names Martin Schulz as the group chairman.", True)
finding(9, "B", "Martin Schulz is a member of the group chairman",
        "The excerpt calls him the group chairman and spokesman, not a member of one.")
for side in "AB":
    finding(10, side, "the fishermen face this in many parts of the world",
            "The visible excerpt stops at 'many parts of the' and supplies no geographic completion.")
    finding(11, side, "a minority is already blocking the regime",
            "The excerpt predicts a blocking minority will oppose continuation.")
    finding(12, side, "Eurobonds and the CDU/CSU parties are parties to a debate",
            "Eurobonds are a proposal and the CDU/CSU are political parties opposing it.")
    finding(13, side, "the President is discussing Eurostat at a meeting",
            "The text is a speaker addressing Mr President and reporting the Committee's position.", True)
    finding(15, side, "Community wine production and consumption are generally below expectation",
            "The excerpt gives under-1% production and 10% consumption for a truncated subject, then discusses an obligatory wine burden.")
finding(12, "B", "G20 action on a speculation tax is infeasible",
        "The source frames EU action conditionally if worldwide G20 action cannot be achieved.")
finding(16, "A", "the donor revision changed to the displayed hash",
        "The excerpt says the donor revision is fixed; only codes and scale differ.")
finding(16, "A", "the model now uses the donor revision as a code",
        "The displayed 8faed761… identifies a fixed revision, not a quantization code.")
finding(16, "B", "the evaluation slice changed relative to a prior version",
        "The excerpt explicitly calls the heldout 24x512 slice shared and fixed.")
for side in "AB":
    finding(17, side, "F1 and E19 are two neural-network architectures",
            "They name prior tolerance questions/probes; the excerpt does not identify architectures.")
finding(17, "B", "E19's tolerance is 48%",
        "V52 drops 48% of FFN output for +0.141846 BPB; 48% is not a tolerance threshold.", True)
finding(18, "A", "MoE means Multi-Output Embedding",
        "MoE is used for mixture-of-experts conversion, with other layers left dense.")
finding(18, "A", "Neuralink and LLM-in-a-Flash are cited as the half-layer MoE precedent",
        "They start a new memory-layout section; the half-layer precedent is in the preceding paragraph.")
finding(18, "B", "the dense architecture converts 12/24 layers to MoE",
        "The excerpt says the rest of a partially converted architecture stay dense; it does not assign MoE layers to a dense architecture.", True)
finding(18, "B", "Neuralink and LLM-in-a-Flash exemplify the half-layer MoE precedent",
        "They are only named in the following memory-layout section heading.")
finding(19, "A", "the aux=0 setting is now observed to gain more points",
        "The paragraph is explicitly a registered prediction written before numbers exist.")
for side in "AB":
    finding(21, side, "the rotation has already removed outliers from weights and activations",
            "The excerpt states this as a conditional hypothesis: 'If a rotation removes them'.")
    finding(22, side, "the full export verification is for the baseline F32 heldout dataset",
            "The COMPLETE verifier belongs to the full W4 BF16 export; F32 heldout is a separately fixed baseline.")
finding(23, "A", "failure of the rise means E46 is not functioning",
        "The excerpt says absent rise would make run 3 contention; no E46-functioning conclusion is visible.")


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
    output = {"experiment": "METH-114-alpha075-arm-blind-development-verdict",
              "blind_sha256": BLIND_SHA,
              "reviewer": "single agent, excerpt-only review",
              "rubric": "Count claims contradicted or not supported by the visible 384-character excerpt; severe means a central entity, event or result is inverted; missing_detail means no concrete supported detail is supplied.",
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
