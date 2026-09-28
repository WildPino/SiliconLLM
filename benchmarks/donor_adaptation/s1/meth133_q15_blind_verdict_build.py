"""Freeze excerpt-only METH-133 findings before revealing arm identity."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "fee824711e58993cc6bbb0930ccc2f363a16d0ff5346044d09de54deb34cf791"
FINDINGS = {}


def add(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append(
        {"claim": claim, "evidence": evidence, "severe": severe})


def both(index, claim, evidence, severe=False):
    for side in "AB":
        add(index, side, claim, evidence, severe)


add(0, "A", "the excerpt shows _clone_model returning the cloned model",
    "The visible function stops immediately after load_state_dict; no return statement is shown.")
add(0, "B", "the returned instance omits the source state dictionary",
    "The visible function calls clone.load_state_dict(source.state_dict(), strict=True).", True)
add(0, "B", "the excerpt shows _clone_model returning an instance",
    "No return statement is visible after load_state_dict.")
both(1, "the passage is a process for evaluating model performance",
     "Its stated decision is REFUSES TO PACK unless three prerequisites fired.", True)
add(1, "B", "the truncated 'mo' identifies assembly of the model itself",
    "The excerpt ends at 'the assembled mo'; its completed referent is not visible.")

add(3, "A", "shuffled_H_floor recovers the raw value from the original layer",
    "The visible assignment is shuffled_H_floor: recov_shuf; the original layer is not named.")
add(3, "A", "wrong-layer null is recovered from the original layer",
    "The label explicitly says wrong_layer_H_null, assigned recov_wrong.")
add(3, "A", "recovery_minus_shuffled_floor subtracts the original layer's raw value",
    "The formula is recov_real - recov_shuf, not a stated original-layer raw value.", True)
add(3, "A", "recovery_minus_wrong_layer_null subtracts the correct layer's null",
    "The formula subtracts recov_wrong, labelled wrong_layer_H_null.", True)
add(3, "B", "wrong-layer null ignores all layer errors",
    "The visible field is recovery_raw_wrong_layer_H_null; no such error-free meaning is given.")
add(3, "B", "three_way_points is a list",
    "Only keyed assignment three_way_points[point_key] = three_way is shown; list type is unsupported.")
add(4, "B", "every shape element is at most 128",
    "The code requires positive integers; 128 is a divisibility constraint on shape[1].")
both(5, "the test is known to have run successfully",
     "The excerpt contains assertions, without a test outcome.")
add(5, "B", "the patched attribute is runners.EXPECTED_HELDOUT_BYTES",
    "The call patches runner, singular, at EXPECTED_HELDOUT_BYTES.")
both(6, "the snippet uses the Python type function to create a file path",
     "The visible fragment is the tail of .astype('<f4', copy=False).tofile(mapped_kqv).")
both(6, "the second hash checks a test's identity",
     "Both sha256_file calls check mapped_kq and mapped_kqv target files.")
both(7, "main raises an exception if OUT does not exist",
     "The visible condition is if os.path.exists(OUT): raise SystemExit(...).", True)
add(7, "A", "rec records the current thread's state",
    "The visible rec fields begin with stage and bundle; no thread state appears.")
add(7, "A", "rec includes a JSON object of self-test results",
    "The excerpt cuts off at bundle_sha25 and shows no such object.")
add(7, "B", "rec records the current thread's state",
    "Only stage and bundle fields are visible, with no thread state.")
both(9, "a ribald fetish was the work sold extensively in France",
     "The excerpt assigns the extensive sale to 'La Bible Amusante'.", True)
add(10, "A", "the narrator wonders who might currently be seated at the other table",
    "The long-nosed man has gone; the narrator asks who that man was.")
add(10, "B", "the man sits at a table inside the chateau",
    "The narrator watches the chateau from a cabaret and refers to the other table there.")
both(11, "the woman holds sugar-tongs on the varnished floor",
     "She drops the tongs onto the floor when offered sugar.")
both(11, "Anna is the person described as ill at ease",
     "The unnamed woman who nods and drops the tongs is ill at ease; Anna speaks afterward.", True)
add(11, "A", "Anna offers to pick up the tongs but is told to let Letty do it",
    "Anna herself says 'Letty will pick them up'; no one directs Anna otherwise.")
add(11, "B", "Anna suggests the situation may be embarrassing for her",
    "Anna says not to mind and that Letty will pick the tongs up; no self-embarrassment is stated.")
both(12, "the mother mouse is seen hiding the nest",
     "The excerpt reports a nest on the keyboard and the mother springing out; it does not show her hiding it.")
both(12, "the mother leaves behind the nest and its contents intact",
     "The excerpt explicitly says the nest and five young ones were destroyed.", True)
add(13, "A", "Bigner's age is listed",
    "Bigner, 495 is an index entry with a page number, not an age.")
add(13, "A", "Bigod's occupation is listed",
    "The index gives Roger Bigod with pages 257 and 365; no occupation appears.")
for claim, evidence in (
    ("Bigner is located in Birmingham", "Bigner and Birmingham are separate index entries."),
    ("Bigod is located on the Isle of Wight", "Isle of Wight modifies Binstead quarries, not Bigod."),
    ("Binstead is in Kirkham", "Kirkham modifies Bird-fair; Binstead quarries are on the Isle of Wight."),
    ("Birkenhead is in Birmingham", "Birkenhead and Birmingham are separate entries."),
    ("Bisham Abbey is in Birmingham", "Bisham Abbey and Birmingham are separate entries."),
    ("Bishops Hatfield is in Mendip Hills", "Mendip Hills modifies Black Down, not Bishops Hatfield."),
    ("Bisterne is in Birmingham", "Bisterne and Birmingham are separate entries."),
    ("Bigners is listed under Birmingham", "The excerpt lists Bigner as its own entry and never says Bigners."),
):
    add(13, "B", claim, evidence, True)
both(14, "the man is beating branches with a brush",
     "The visible excerpt starts mid-phrase at 'that he was beating the branches with'; the implement is absent.")
add(14, "A", "the duke does not understand the gesture",
    "The observers cannot quite understand the duke's Spanish; no gesture misunderstanding is stated.")
add(14, "A", "the duke sees another man and asks permission to approach",
    "The duke starts and approaches the observers, then begs pardon in Spanish; no second man or permission appears.", True)
add(14, "B", "the man wears a brown-study outfit",
    "'All in a brown study' describes a mental state, not clothing.", True)
add(14, "B", "the duke cannot understand their conversation",
    "The observers cannot quite understand the duke's Spanish, reversing who fails to understand.", True)
add(15, "A", "the passage contrasts abstinuerunt with abstinuere",
    "Only Abstinuerunt a vino and its translation are visible.")
add(15, "A", "Tiber-totallers are known for drinking wine",
    "The example translates as 'They abstained from wine'.", True)
add(15, "B", "Abstinuerunt a vino requires an accusative case",
    "The excerpt's accusative list begins later; a vino is not presented as requiring accusative.", True)
add(15, "B", "the preposition is ererga",
    "The visible list says erga, not ererga.")
both(16, "FFN means Flattened Pyramid Network for image recognition",
     "The excerpt names layer-1 FFN RMSNorm in a GigaChat engine result, with no image task or pyramid network.", True)
add(16, "A", "the model was trained under this frozen protocol",
    "The excerpt refers to a frozen cross-input diagnostic protocol, not model training.")
add(16, "A", "the result supports image classification and object detection",
    "No image applications appear in the visible excerpt.", True)
add(16, "A", "RMSNorm means Relative Positional Embedding Norm",
    "The text says RMSNorm and does not define it that way.")
add(16, "B", "a device execution is shown",
    "Only protocol and apparatus links and a result status are visible.")
add(16, "B", "the result proves conditions for device execution",
    "The excerpt gives a diagnostic status, not a device-execution certification.")
both(17, "the current layer itself passed all 32 checkpoints",
     "Exact reference l_out-0 makes all pass; the post-F16 current arm replays every failure.", True)
both(17, "there were no issues in the current arm",
     "The passage explicitly says the current arm replays every failure exactly.", True)
both(18, "the tensor verification happens during engine initialization",
     "The visible excerpt says fp32 arms verify tensors, without an initialization phase.")
add(18, "B", "Gate A's first failure came from the gate's state",
    "The visible diagnosis found two real bugs in the exporter.")
add(19, "A", "the predictor is treated as a simple arithmetic operation rather than a prediction task",
    "The excerpt only says a dtype-free convention flatters the predictor.")
add(19, "A", "all calculation values use 0.25 bytes per element",
    "The 0.25 B/element figure applies to W_g/W_u/W_d ternary weights; factors and codebooks differ.")
add(19, "B", "the expression s equals r(D+F)/(D F)",
    "The formula is explicitly labelled pred, not s.")
add(19, "B", "index codebooks and centroids use PQ codes of one byte",
    "The excerpt says codebooks/centroids fp16 and PQ codes one byte.")
add(20, "A", "e37_carved_nf.bin measures 30.75 and f.bin 89.83 tok/s",
    "The displayed order is f.bin against e37_carved_nf.bin, measured 30.75 vs 89.83.", True)
add(20, "B", "G-E45b reports the ratio without dispersion",
    "The gate requires dispersion or no report at all, with at least five pairs.", True)
add(21, "A", "the excerpt compares two versions of a code snippet",
    "It discusses an export flag default, a binary artifact and an R3 quality table.")
add(21, "A", "using --rule produced a performance improvement",
    "The binary predates habitual use of --rule; no such before/after improvement is shown.", True)
add(21, "A", "the exact performance difference is not specified",
    "The visible Qwen2.5-0.5B row explicitly says +0.461 above chance.")
add(21, "B", "the excerpt compares two code-snippet versions",
    "It discusses an export flag default and a binary artifact.")
add(21, "B", "the binary failed at lower rates than the default version",
    "No rate comparison to a default artifact is visible; the R3 statement concerns 1.5B/0.5B quality.", True)
add(22, "A", "curve is a dataset used to calculate digit-for-digit accuracy",
    "The visible command is donor_inventory.py curve and the measured quantity is tok/s.", True)
add(22, "A", "n_att is a number of iterations",
    "The excerpt only gives n_att=14 as a configuration value, not an iteration count.")
add(22, "A", "manual execution significantly improved accuracy over the tool",
    "The author re-derived the same 11.97 tok/s figure digit-for-digit, not an accuracy gain.", True)
add(22, "B", "11.97 tok/s is the lower bound",
    "The interval is [10.67 .. 12.68]; 11.97 is the central figure.", True)
add(22, "B", "the author re-derived the code",
    "The passage says the author re-derived the figure by hand, not code.")


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
                         "missing_detail": False, "ambiguous": []}
        rows.append(row)
    assert all(0 <= i < 24 and side in "AB" for i, side in FINDINGS)
    output = {"experiment": "METH-133-arm-blind-excerpt-verdict",
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
