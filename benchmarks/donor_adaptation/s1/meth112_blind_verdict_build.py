"""Record METH-112 excerpt-only A/B findings before arm unblinding."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "d9585249eb4e243298ccafbf49583e2f0a3cdfcdeb5794aae0f1ff69db135468"
FINDINGS = {}
MISSING = {(0, "A"), (0, "B"), (3, "A"), (3, "B")}


def finding(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append(
        {"claim": claim, "evidence": evidence, "severe": severe})


finding(0, "A", "dst and s are strings being manipulated",
        "The excerpt uses float-pointer offsets and writes into dst, not string operations.", True)
finding(0, "A", "dst is concatenated with a 576U-length string",
        "The visible code copies STRAT01_L2KPX_TAIL floats into dst; no concatenation appears.")
finding(0, "A", "an offset from s is stored in positional",
        "The excerpt reads positional and writes dst; it shows no offset stored in positional.")
finding(0, "B", "a prefix and tail are appended to the string s",
        "s is used for control flags; the shown writes target dst, not a string s.", True)
finding(0, "B", "latent_control is set to -1 at all positions",
        "When latent_control is true, selected dst floats are negated; the field is not set.")
finding(0, "B", "positional_control is set to 7 at all positions",
        "The field is tested, not assigned; 7U is an index into dst.")
finding(1, "A", "the script performs gradient descent on a mass array",
        "The shown sequence collects mass with pc(xtr), chooses static_from_mass, and evaluates loss; no descent appears.")
finding(1, "A", "the static value was optimized by gradient descent",
        "pick comes from pc.static_from_mass after a no_grad collection, not optimization.")
finding(1, "B", "set_static(None) restores the mass to its original value",
        "The excerpt clears the static selection; it does not show restoration of a mass value.")
finding(2, "A", "dense random orth is block-random-orth under another name",
        "The list distinguishes block-random-orth from dense random orth.")
finding(2, "B", "there are four candidate test types",
        "The visible list names identity, permutation, block-Hadamard, block-random-orth and dense random orth.")
finding(4, "B", "the donor block is interleaved with a block of neurons",
        "The excerpt names an interleaved layout statistic and a donor block size, but no such block interleaving operation.")
finding(4, "B", "layout_stats calculates i.i.d. masks",
        "layout_stats is called on Bn to produce statistics; the note calls the mask synthetic, not a computed output here.")
finding(6, "A", "the --profile flag is applied identically to every arm",
        "The comment says HIGH_PRIORITY_CLASS is applied identically; --profile is merely in cmd.")
finding(6, "A", "the scheduler consequently lets background work preempt the benchmark",
        "The comment says high priority changes how often background work preempts it, not that it increases preemption.")
finding(6, "B", "--profile is the per-arm setting",
        "HIGH_PRIORITY_CLASS, not --profile, is identified as applied identically to every arm.")
finding(6, "B", "--profile means benchmarking on all available cores",
        "The command has a separate --threads value; no all-core meaning for --profile is shown.")
finding(7, "A", "the diagnostic is specifically a C++ program",
        "The excerpt is C-compatible header code and does not identify C++.")
finding(7, "A", "parsed is a parsed SHA256 hash",
        "parsed is compared with hashed, the byte count checked against STRAT01_EXPECTED_SIZE.", True)
finding(7, "A", "a parsed-count mismatch prints an error message here",
        "The visible parsed!=hashed branch goes directly to finish without a shown print.")
finding(7, "B", "hashed is the return value of strat01_sha256_file",
        "The call passes &hashed as an output; its return value is tested separately.")
finding(7, "B", "size and SHA mismatch must both occur before identity failure",
        "The source uses logical OR between hashed!=expected size and strcmp, not AND.", True)
for side in "AB":
    finding(8, side, "the excerpt reports a meeting between the rapporteur and President",
            "It shows a speaker addressing Mr President and congratulating Mr Harbour, not a meeting.")
finding(9, "A", "the speaker expresses hopelessness",
        "The speaker says much hard work is needed; hopelessness is not stated.")
for side in "AB":
    finding(9, side, "the excerpt calls for significant investment",
            "It names hard work, infrastructure and measures, but no investment amount or explicit investment call.")
finding(11, "A", "the speaker joined nearly two and a half years after the election",
        "The speaker joined just after the election, which occurred almost two and a half years ago.")
for side in "AB":
    finding(13, side, "companies are stakeholders in companies that profit from legislation",
            "The question concerns legislators who themselves sit on company boards or hold stakes.", True)
    finding(14, side, "the vote will take place at a specific time",
            "The visible excerpt ends after 'at' and gives no time.")
finding(15, "A", "the speaker is identified as a European Union official",
        "The 384-character excerpt does not identify the speaker's office.")
for side in "AB":
    finding(15, side, "the Agency itself is shown upholding rights or law",
            "The text says it may reassure citizens that bureaucrats and governments uphold rights while implementing EU law.")
    finding(17, side, "cache residency is a metric of model quality",
            "The excerpt separates a quality ceiling from temporal locality and cache hit rate.", True)
    finding(17, side, "the predictability regularizer has already mitigated the limitations",
            "The excerpt proposes a future experiment to test whether quality and locality improve.")
finding(19, "A", "passing this stop gate establishes a successful rollout",
        "The excerpt reports one frozen stop gate passed by one document, not overall rollout success.")
finding(20, "A", "A10B-K3 is a model used to analyze a text",
        "The excerpt identifies A10B-K3 as noise weights in a rate/numerical-equivalence cell.", True)
finding(20, "A", "this text corrects E26 and section 61.6",
        "It says E26 and section 61.6's correction support the shape-dependence warning.")
finding(20, "B", "A10B-K3 is trained on a dataset",
        "The excerpt explicitly calls its weights noise and says there is no 10B quality finding.", True)
finding(20, "B", "it is evaluated by identifying and correcting text errors",
        "Only rate and numerical equivalence are measured, never BPB or error correction.", True)
finding(20, "B", "identified text errors transfer elsewhere in the text",
        "The excerpt says nothing transfers to another shape; no text-error transfer appears.", True)
for side in "AB":
    finding(21, side, "the 4.0% rate spread is a 4.0% increase in the ffn witness",
            "The visible table assigns 4.0% to rate spread; ffn~ witness is a separate row.")
finding(21, "A", "BRIEF_E9_GLUE_PARALLEL is an experiment framework",
        "It is identified as a pre-measurement brief path, not a framework.")
finding(21, "B", "BRIEF_E9_GLUE_PARALLEL is a dataset",
        "It is identified as a pre-measurement brief path, not a dataset.")
for side in "AB":
    finding(22, side, "15/160 is the teacher-forced result",
            "The excerpt labels 115/160 teacher-forced; the preceding 15/160 belongs to the other rollout mode.")
    finding(22, side, "H0 is a statistical null hypothesis",
            "H0 names an earlier experiment that used rank 512, not a null hypothesis.")
finding(22, "A", "H0 fails to establish transfer in length",
        "The excerpt says 'larghezza', meaning width, and the 1/32 fraction.")
finding(22, "A", "E65 uses rank 512 for the post-hoc reduction",
        "The excerpt assigns fp32 redensified factors to E65 and rank 512 to H0.")


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
    output = {"experiment": "METH-112-arm-blind-semantic-verdict",
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
