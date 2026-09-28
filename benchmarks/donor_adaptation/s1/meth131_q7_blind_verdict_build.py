"""Commit excerpt-only findings for METH-131 before revealing arm identity."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "29e0e65e9da56bbb13fe70727a0e4d5dd4d27afc9b596655e763265d8eb6f3b6"
FINDINGS = {}
MISSING = {(10, "A"), (10, "B"), (23, "A"), (23, "B")}


def finding(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append(
        {"claim": claim, "evidence": evidence, "severe": severe})


for side in "AB":
    finding(2, side, "_mk_in takes the input tensor li",
            "The visible method uses li as a layer index in self.x[li]; args[0] is the input tensor.")
    finding(2, side, "_mk_in returns a dummy value to prevent unnecessary computations",
            "Its inner function stores args[0].detach() and has no explicit return or computation-suppression step.", True)
    finding(2, side, "_mk_mask calls the down projection layer after checking off mode",
            "_mk_mask supplies a pre-hook; the visible branch returns None and does not call down_proj.")
    finding(4, side, "the excerpt includes a placeholder for implementation details",
            "The shown text ends mid-string assignment; it contains no placeholder designation.")
    finding(6, side, "the excerpt is C++ code running on a GPU",
            "It is Python assignments of ROOT-relative paths; no C++ code execution or GPU appears.", True)
    finding(7, side, "the excerpt defines a function named ubprocess.Popen",
            "It begins mid-call at 'ubprocess.Popen'; no such function definition is shown.")
    finding(9, side, "Isolde describes a journey to Egypt",
            "The visible passage says she told Valerie of an adventure, without naming Egypt.", True)
    finding(10, side, "the speaker cannot earn enough to pay for his work",
            "The workers say they have received no wages for six weeks; they are not paying for their work.")
    finding(10, side, "the passage promotes hard work and perseverance",
            "The worker says unpaid work will not feed him and another proposes stopping work.", True)
    finding(12, side, "Edward tries to conquer Aquitaine",
            "Edward is fighting in Scotland; Philip seeks to disturb Edward's hold on Aquitaine.", True)
    finding(12, side, "Aquitaine is the kingdom of England",
            "The passage calls it part of Aquitaine held by the king of England, not the kingdom itself.")
    finding(12, side, "Philip is identified as Philip II of France",
            "The visible excerpt supplies only the name Philip, without regnal number or title.")
    finding(15, side, "James Claypole is the man refusing to profit from his estates",
            "Claypole is named as the author of a letter reporting another man's refusal.", True)
    finding(15, side, "the man sells sixteen thousand pounds of land or property",
            "The excerpt says he believes he is losing sixteen thousand pounds; it mentions no sale.")
    finding(17, side, "268,435,456 is the model's total parameter count",
            "That number is in the Charged attention column, not a total-parameter column.", True)
    finding(17, side, "805,306,368 is a number of charged tokens",
            "The value is in the selected + shared experts column, a parameter charge.")
    finding(17, side, "50 means 50 percent",
            "The header says 'GB/s needed at 50', referring to a token-rate target, not 50 percent.")
    finding(18, side, "nonfinite_microbatches: 0 means no microbatch was generated",
            "It reports zero nonfinite microbatches; the run had microbatches.", True)
    finding(18, side, "GradScaler failed to apply updates correctly and caused a dead loop",
            "The excerpt says GradScaler skips the first step by construction; the defect was an early gate check.")
    finding(21, side, "no expert read more than 48 tokens simultaneously",
            "The experiment has 48 expert touches per token, not a 48-token concurrency bound.", True)
    finding(22, side, "normalized router weights alone failed",
            "The opening says 'C normalized router weights alone pass.'", True)

finding(3, "A", "safe_open does not specify a framework",
        "The call explicitly passes framework='pt'.")
finding(3, "B", "the script writes original metadata to a file with a PT interpreter",
        "The visible operation opens a safetensors archive on CPU; no metadata write is shown.", True)
finding(5, "A", "the script uses TensorFlow 2.x",
        "Visible imports include torch and NumPy, with no TensorFlow reference.", True)
finding(5, "B", "the script reads data from e37_routers_E256.npz",
        "The visible comment says fit, score, write that path; no read is shown.")
finding(6, "A", "the protocol Markdown path is a model used by the snippet",
        "PROTOCOL is assigned a research Markdown file path, not a model.")
finding(6, "A", "ROOT names the operating system",
        "ROOT is a base path used with /; no operating system is named.")
finding(6, "B", "ROOT likely refers to a Linux distribution",
        "ROOT is a base path variable, not an identified distribution.")
finding(6, "B", "the engine path begins /marks/phase60",
        "The excerpt starts mid-string; it does not establish a leading slash or that path.")
finding(7, "A", "the function writes to standard error",
        "Popen captures stderr in a pipe, and the shown code decodes it on failure; no stderr write is shown.")
finding(11, "A", "Eugene visits his grandmother in the tent",
        "Wetherell enters Pogosa's tent and asks after her; Eugene interprets her reply.", True)
finding(11, "A", "Pogosa has been ill for several months",
        "The excerpt shows her groaning but states no duration of illness.")
finding(11, "A", "a broken back requires constant care",
        "A truncated quotation says 'She say her back broke'; no care requirement is described.")
finding(11, "B", "Kelley is a woman searching ten years for a lost mine",
        "The ten-year mine search is hypothetical; Kelley's gender and search are not established.", True)
finding(11, "B", "Pogosa is sleeping when Wetherell enters",
        "She lies in the tent, then shakes her head and groans; sleep is not stated.")
finding(16, "A", "insufficient memory causes the owner's 12-20 CPU hours",
        "The excerpt assigns CPU time to an uninformative experiment and gives no memory cause.")
finding(16, "A", "a curve is proposed instead of ΔBPB",
        "The proposed curve specifically plots ΔBPB against effective bits per weight.", True)
finding(16, "A", "the reported result is the lowest ΔBPB value",
        "The visible phrase ends at 'the lowest b'; it does not say lowest ΔBPB.")
finding(16, "B", "insufficient memory or disk space causes high CPU usage",
        "Neither memory nor disk shortage is mentioned in the excerpt.")
finding(16, "B", "the proposed curve relates CPU usage to effective bits per weight",
        "It plots ΔBPB against effective bits/weight; CPU time motivates it but is not an axis.", True)
finding(16, "B", "scales are included for both plot axes",
        "'Including scales' modifies effective bits/weight, not both axes.")
finding(19, "A", "poly is a library used by the function",
        "The excerpt names a poly runtime arm, not a separate library.")
finding(19, "B", "the program performs logarithms",
        "Only exponentiation-related softmax and silu calls are visible.")
finding(22, "A", "layer-2 sensitivity is not yet known",
        "The excerpt calls layer 2 'now-known sensitive'.")
finding(22, "B", "Q6 propagation failed through layer 2",
        "The excerpt says it was never propagated, not that a propagation run failed.")
finding(23, "A", "the excerpt evaluates hash function quality",
        "SHA-256 is only described as an identity gate for bit-identical output.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--blind",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    assert hashlib.sha256(args.blind.read_bytes()).hexdigest()==BLIND_SHA
    blind=json.loads(args.blind.read_text(encoding="utf-8"))
    assert len(blind["rows"])==24
    rows=[]
    for index,pair in enumerate(blind["rows"]):
        row={"source_id":pair["source_id"]}
        for side in "AB":
            claims=FINDINGS.get((index,side),[])
            row[side]={"unsupported":claims,
                       "severe_count":sum(bool(c["severe"]) for c in claims),
                       "missing_detail":(index,side) in MISSING,
                       "ambiguous":[]}
        rows.append(row)
    assert all(0<=i<24 and side in "AB" for i,side in FINDINGS)
    output={"experiment":"METH-131-arm-blind-excerpt-verdict",
            "blind_sha256":BLIND_SHA,
            "reviewer":"single agent, excerpt-only review",
            "rubric":"Count claims contradicted or unsupported by the visible excerpt; severe means a central entity, event or result is inverted; missing_detail means no concrete supported detail is supplied.",
            "rows":rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"blind_sha256":BLIND_SHA,
                      "verdict_sha256":hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "A_unsupported":sum(len(r["A"]["unsupported"]) for r in rows),
                      "B_unsupported":sum(len(r["B"]["unsupported"]) for r in rows),
                      "A_severe":sum(r["A"]["severe_count"] for r in rows),
                      "B_severe":sum(r["B"]["severe_count"] for r in rows)},indent=2))


if __name__=="__main__":
    main()
