"""Transcribe the frozen, arm-blind METH-57 excerpt review into JSON.

Only row numbers and A/B labels from the blind file appear here. Run this
before the unblinding score, and commit its output before scoring.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
BLIND = DIR / "meth57_blind_semantic_review.json"
OUT = DIR / "meth57_blind_semantic_verdict.json"


def finding(claim, evidence, severe=False):
    return {"claim": claim, "evidence": evidence, "severe": severe}


# A finding's evidence quotes the excerpt, including when that span
# contradicts the generated claim. Claims requiring outside information
# are unsupported under the frozen excerpt-only prompt.
F = finding
REVIEW = {
    0: {},
    1: {
        "A": {"ambiguous": ["'verifies that the SHA256 hash matches' may refer to the later verify_root assertion or the earlier manifest load."]},
        "B": {"ambiguous": ["The loop over manifest['files'] is cut off, so its later validation cannot be established from the excerpt."]},
    },
    2: {"B": {"unsupported": [
        F("Specific hooks are for only the first and last layers.", "'for li in range(L)' appears twice: hooks are made for every layer.", True),
        F("The function applies a masking layer to the output.", "'S.CTL.mode = \"off\" # no masking here' contradicts this operation."),
    ]}},
    3: {side: {"unsupported": [
        F("The function compares two implementations here.", "'Re-deriving it here would compare two implementations instead of using the rule' describes what is avoided."),
        F("The RMS measurements determine which implementation is better.", "The excerpt maps 'rms_in' and 'rms_h'; it gives no effectiveness comparison.", True),
    ]} for side in ("A", "B")},
    4: {
        "A": {"unsupported": [
            F("GradScaler makes the STE gradient the identity.", "'G-H0d the straight-through gradient is the identity' and separately 'G-H0f ... A GradScaler that declines a step' give no such cause.", True),
            F("A step function is always 1 and is applied to the step variable.", "'d(ste(w))/dw == 1' is a derivative; 'step == 1' refers to when the old test read masters.", True),
        ]},
        "B": {"unsupported": [
            F("GradScaler fails to update the gradient correctly.", "'A GradScaler that declines a step leaves the masters exactly where they were' does not describe an incorrect gradient."),
            F("The masters stay unchanged even if the step succeeded.", "The excerpt limits unchanged masters to a GradScaler that 'declines a step'.", True),
        ]},
    },
    5: {side: {"unsupported": [
        F("check_anchor takes only two parameters.", "'def check_anchor(gpu_run, anchor, sigma=S.SIGMA_SEED_BPB)' has three parameters, one optional.", True),
        F("ok is false exactly when the two point values are unequal.", "'Compare ... point ... p by p' and 'ok False means STOP' give no exact-equality condition; sigma is a parameter.", True),
    ], "ambiguous": ["The excerpt does not specify the type or content of report; the claimed explanatory string is unverified."]} for side in ("A", "B")},
    6: {
        "A": {"unsupported": [F("The loss is a squared difference between forward and backward predictions.", "'(_m0(_x0) - _t0).square().mean().backward()' compares two forward outputs; backward() follows the loss.")]},
        "B": {"unsupported": [
            F("The model is detached from the network.", "'_m0.router.detach().clone()' detaches a clone of one router parameter, not the model."),
            F("The gradient is computed to determine the step size.", "'AdamW([_m0.router], lr=1e-3, weight_decay=0.0)' specifies the learning rate; the gradient sum is recorded as _gr."),
        ], "ambiguous": ["Model initialization may be in preceding clipped code, but is not visible in the excerpt."]},
    },
    7: {
        "A": {"unsupported": [F("The code explicitly computes SHA-1.", "The excerpt calls 'git hash-object' and 'git rev-parse'; it names no hash algorithm.")],
              "ambiguous": ["'current working tree head' blurs the worktree blob and HEAD blob hashes."]},
        "B": {"unsupported": [F("The second command returns the head commit's hash.", "'git rev-parse HEAD:' + rel selects the path at HEAD, whereas the excerpt calls the result 'head_blob'.")]},
    },
    8: {
        "A": {"unsupported": [F("A polar star would need to be set at the equator.", "'if there were a star exactly at the pole it would only be necessary to set' is cut off before naming a setting or location.", True)]},
        "B": {"unsupported": [F("A polar star would need to be set at a specific latitude to be observed.", "'if there were a star exactly at the pole it would only be necessary to set' is cut off before saying what to set.")]},
    },
    9: {
        "A": {"unsupported": [
            F("Sam and Peter follow Ginger.", "'Sam and Peter followed 'im out ... Ginger, who was staggering arter them some distance behind' places Ginger behind them.", True),
            F("Bill follows Ginger.", "The excerpt says Ginger was 'staggering arter them'; it does not put Bill behind Ginger.", True),
        ], "ambiguous": ["The clipped opening 'g at Ginger very wicked' does not securely establish a narrator's character judgment."]},
        "B": {"unsupported": [F("Sam and Peter follow Ginger.", "'Ginger, who was staggering arter them some distance behind' places Ginger behind Sam and Peter.", True)],
              "ambiguous": ["The clipped opening 'g at Ginger very wicked' is incomplete."]},
    },
    10: {
        "A": {"unsupported": [
            F("Harry believes he knows the truth about what happened to Yuba Bill.", "Harry asks 'Did Yuba Bill know of it?' after Brice says he knew the lock had been forced.", True),
            F("Harry's heart sinks, yet he remains truthful.", "'Brice's heart sank, but he remained steadfast and truthful' assigns both to Brice.", True),
        ]},
        "B": {"unsupported": [
            F("A box was forced into a lock.", "Brice says 'the lock had been forced' and that he 'snapped it together again'.", True),
            F("Harry asks whether Brice knows about the box, and Brice says he knows it when he sees it.", "Harry asks 'do you mean to say YOU knew it?' and Brice says 'I knew it when I handed down the box'.", True),
        ], "ambiguous": ["Harry's skepticism is an interpretation of the question, not a directly stated feeling."]},
    },
    11: {side: {"unsupported": [F("David expresses current displeasure.", "'I should be if I thought you were serious' makes displeasure conditional.")]} for side in ("A", "B")},
    12: {
        "A": {"unsupported": [
            F("A father is speaking to his daughter.", "The narration calls the listener 'this fragile little wife of his'.", True),
            F("The daughter requested the Scotland trip and agrees to it.", "'Next Christmas Day ... we'll try to spend in bonnie Scotland' is his proposal; only roses coming to her cheeks are stated.", True),
        ]},
        "B": {"unsupported": [
            F("A father is speaking to his daughter.", "The narration calls the listener 'this fragile little wife of his'.", True),
            F("He is saying goodbye to his wife.", "The excerpt has a plan for 'Next Christmas Day' and no departure or farewell."),
        ]},
    },
    13: {side: {"unsupported": [F("The spherical carcass belongs to a ship.", "The excerpt discusses 'conflagrating powers of the spherical carcass' and bombardment, but no ship.")]} for side in ("A", "B")},
    14: {
        "A": {"unsupported": [F("COB offers the parting cup to MARIETJE.", "'COB. Give him just one, for a parting cup' names a male recipient, followed by MARIETJE's refusal.", True)]},
        "B": {"unsupported": [F("COB persuades MARIETJE to accept the parting cup.", "'COB. Give him just one' names a male recipient, and 'MARIETJE. [Angrily.] No! No!' refuses.", True)]},
    },
    15: {
        "A": {"unsupported": [
            F("Webb takes Amy out of her confinement.", "'during his enforced confinement so many opportunities to take Amy out fell naturally to Webb' makes Burt the confined person.", True),
            F("Webb is jealous and convinces himself his misgivings are absurd.", "'Burt was ever able to convince himself that his misgivings were absurd'; Webb is described as fraternal.", True),
        ]},
        "B": {"unsupported": [
            F("Webb was confined at home due to a health condition.", "'during his enforced confinement ... opportunities ... fell naturally to Webb' makes Burt the confined person and gives no cause.", True),
            F("Amy was also confined.", "The excerpt calls Amy 'the young girl' whom Webb had opportunities to take out; it gives no confinement."),
            F("Webb was jealous of Amy and his opportunities to take her out.", "'was almost jealous that ... opportunities to take Amy out fell naturally to Webb' describes Burt's jealousy, not Webb's.", True),
            F("Webb convinces himself his misgivings were absurd.", "'Burt was ever able to convince himself that his misgivings were absurd'.", True),
        ]},
    },
    16: {},
    17: {},
    18: {
        "A": {"ambiguous": ["The first sentence is grammatically tangled; its intended claim about the runner is uncertain."]},
        "B": {"unsupported": [F("A program has never been reproduced by the runner who produced it.", "'it has never been reproduced by a runner that was not the one that produced it' concerns a measurement and other runners.")],
              "ambiguous": ["'The program has never existed in the exporter' may mean the folded norm, which the excerpt says has not existed there."]},
    },
    19: {
        "A": {"unsupported": [
            F("The excerpt provides a model accuracy measurement.", "The visible table gives '+3.4% / exact / exact' and '49.96' without identifying any accuracy metric."),
            F("It reports a difference between a charged model and a lower charge level.", "The excerpt says a prediction is 'FASTER than the charged model'; it mentions no lower charge level.", True),
        ], "missing_detail": True, "missing_detail_evidence": "It omits the visible '3 HIT / 1 MISS / 2 held' and the registered 49.96 result."},
        "B": {},
    },
    20: {
        "A": {},
        "B": {"unsupported": [F("The two stated sizes disagree with the actual decimal-MB sizes.", "'The write-up's own \"12.4 MB\" and \"352 MB\" are the same figures in decimal MB. This one is right.'", True)]},
    },
    21: {},
    22: {
        "A": {"unsupported": [F("q is a vector and k is a matrix in the given inner product.", "'qᵀk' and 'projecting q,k' give no matrix k; the formula treats both as projected operands.")]},
        "B": {"unsupported": [
            F("The excerpt defines φ(x) = x^T q + (x^T q)²/2.", "The visible equation is '1 + qᵀk + (qᵀk)²/2 ≈ exp(qᵀk)', not this φ(x) formula.", True),
            F("The excerpt defines recurrence φ(x) = φ(x - q) + q^T x.", "The excerpt only states 'O(d³) recurrent state'; no recurrence equation appears.", True),
        ]},
    },
    23: {
        "A": {"unsupported": [
            F("The report is about performance of a co-activation carving algorithm on a single device.", "'What was measured' says 'D0 Part III closed FFN co-activation carving at 1.5B on a single num' and cuts off; no performance metric or device is specified."),
        ]},
        "B": {"ambiguous": ["'on a single num' merely repeats the truncated source; it does not assert what num means."]},
    },
}


def main():
    blind_bytes = BLIND.read_bytes()
    blind = json.loads(blind_bytes)
    assert len(blind["rows"]) == 24 and set(REVIEW) == set(range(24))
    rows = []
    for index, original in enumerate(blind["rows"]):
        row = {"source_id": original["source_id"]}
        for side in ("A", "B"):
            specifics = REVIEW[index].get(side, {})
            row[side] = {"unsupported": specifics.get("unsupported", []),
                         "missing_detail": specifics.get("missing_detail", False),
                         "ambiguous": specifics.get("ambiguous", [])}
            if "missing_detail_evidence" in specifics:
                row[side]["missing_detail_evidence"] = specifics["missing_detail_evidence"]
        rows.append(row)
    result = {"experiment": "METH-57-arm-blind-semantic-verdict",
              "blind_sha256": hashlib.sha256(blind_bytes).hexdigest(),
              "reviewer": "single agent, excerpt-only review",
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "verdict_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
