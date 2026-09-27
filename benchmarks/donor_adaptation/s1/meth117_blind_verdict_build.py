"""Record METH-117 excerpt-only A/B claims before arm unblinding."""

import argparse
import hashlib
import json
from pathlib import Path


BLIND_SHA = "0b3f64f20be2380045e330d63b9d9b845540f71fabbd17aba0f9f3a408f02884"
FINDINGS = {}
MISSING = set()


def finding(index, side, claim, evidence, severe=False):
    FINDINGS.setdefault((index, side), []).append(
        {"claim": claim, "evidence": evidence, "severe": severe})


finding(0, "A", "the current file hash differs from the committed hash",
        "The excerpt only shows a conditional branch if they differ, not its outcome.")
finding(0, "A", "the file hash is compared to the HEAD commit hash",
        "The committed value comes from HEAD:rel, a file blob, not rev-parse HEAD.")
finding(0, "B", "git hash-object is called with -w",
        "The displayed command is git hash-object without -w.")
finding(0, "B", "the file hash is compared to the commit hash",
        "The displayed comparison is current versus HEAD:rel file hash.")
for side in "AB":
    finding(1, side, "validate_report performs the visible outputs.items checks",
            "Its definition begins after that loop, not before it.", True)
finding(2, "A", "index_url uses port 127.0.0.1",
        "127.0.0.1 is the host; the port is self.httpd.server_port.")
finding(2, "A", "mode is set to the literal http",
        "The getter returns self.httpd.mode and the setter assigns the supplied value.")
finding(2, "B", "the mode setter is not explicitly defined",
        "@mode.setter and def mode(self, value) are visible.")
for side in "AB":
    finding(3, side, "the listed keys' values and completion status are shown",
            "The excerpt shows required key names and only the start of a validation condition.")
    finding(4, side, "ARM expands to Application Runtime",
            "The excerpt uses ARM_NAMES but gives no expansion of ARM.")
finding(5, "B", "all invalid conditions raise ValueError",
        "Nonfinite values return source_nonfinite; only wrong shape/dtype raises ValueError here.")
finding(6, "A", "the loop checks whether each directory is a package",
        "It checks whether a code subdirectory exists before appending a package path.")
finding(6, "A", "it creates a new directory structure",
        "It appends (package name, existing code path) to trees; no directory creation appears.")
finding(6, "B", "it creates a new directory structure for packages",
        "The shown code checks paths and appends to trees without creating directories.")
finding(6, "B", "it checks whether a file is a package",
        "The only visible test is os.path.isdir(cd) for a code directory.")
finding(6, "B", "it has a function checking whether a file is another generation",
        "The excerpt contains a comment explaining a pinned relaytest bundle, not such a function.")
for side in "AB":
    finding(7, side, "the GGUF model path is a dataset",
            "DEFAULT_MODEL names a .gguf model file, not a dataset.", True)
    finding(8, side, "Burke's party arrived at Cooper's Creek after Wright set out",
            "The excerpt says no news of Burke's party came to Melbourne and a search expedition was formed.", True)
    finding(9, side, "Robert returned to the Holy Land",
            "The text says Robert was returning out of the Holy Land.")
    finding(9, side, "Anselme refused homage to Robert",
            "The dispute with Anselme precedes the separate mention of the king's brother Robert.", True)
finding(7, "A", "source_binding_report.jso is a dataset containing the model's binding information",
        "Only the path of DEFAULT_BINDING is visible; its contents and dataset status are not.")
finding(7, "B", "the GGUF dataset was created on September 21, 2026",
        "The date appears in a result-directory name, not a creation timestamp for the model.")
finding(7, "B", "the binding model comes from source_binding_report.jso",
        "The excerpt names a DEFAULT_BINDING path, not a model source relationship.")
finding(9, "A", "the king refused Robert's pilgrimage request",
        "No pilgrimage request or refusal appears in the 384-character excerpt.", True)
finding(9, "B", "the king wanted to appoint Robert as a bishop",
        "No such appointment appears; Anselme's separate consecration dispute is visible.", True)
finding(10, "A", "the woman is being treated unfairly",
        "She recalls grievances while trying to forgive her aunt, but unfair treatment is not established.")
finding(10, "A", "she must choose between two quoted options",
        "The same 'what shall I do?' is an internal question, not two presented choices.", True)
finding(10, "B", "a higher power calls the woman down",
        "The excerpt says she was called down without naming who called her.", True)
for side in "AB":
    finding(12, side, "this is a Bible passage with Jesus giving the listed instructions",
            "The excerpt is a question-and-answer text; no Jesus speech is shown.", True)
    finding(15, side, "the YMCA was established for recreational activities",
            "The passage shows meetings and the White Cross movement, not a recreational purpose.")
finding(12, "B", "Christ will receive the confessing person",
        "The excerpt poses that as a question and cuts off before its answer.", True)
finding(14, "A", "the speaker says women's minds are inherently weak",
        "The stated complaint is that women notice a clever man's weak points, not that their minds are weak.", True)
finding(14, "A", "the fool's virtues flatter the women's self-image",
        "The excerpt cuts off after 'their own'; it does not state self-image.")
finding(14, "B", "women's minds are weak points in clever men and strong points in fools",
        "The speaker says women notice men's respective weak and strong points, not that their minds are those points.", True)
finding(14, "B", "the speaker feels the most sympathy for the fool",
        "The excerpt assigns that sympathy to the women, not the speaker.", True)
for side in "AB":
    finding(16, side, "the passage evaluates PCA on a dataset",
            "It defines a subspace residual screening R2 candidates, not PCA dataset performance.", True)
    finding(16, side, "large residual means a dataset is unsuitable for PCA",
            "Large residual rules out R2's success for candidate phi, not PCA suitability.")
    finding(16, side, "the algorithm may need retraining with fewer SVDs",
            "The excerpt says the bound is learned for the cost of a few SVDs, not that fewer SVDs retrain a model.")
finding(17, "A", "arm C is constructed to return the same microseconds per expert as R",
        "Its memory traffic is designed near zero; equal timings are only a diagnostic possibility.")
finding(17, "B", "the zero-traffic negative control makes arm C untrustworthy",
        "The heading says this control makes arm C trustworthy.", True)
finding(18, "A", "the experiment used a T4 and a speed measurement",
        "The excerpt explicitly says neither was used.", True)
for side in "AB":
    finding(19, side, "the entire network has 35 neurons",
            "The 35-neuron number labels each of E=256 groups, not total network size.", True)
    finding(19, side, "D=1536 means a layer has 1536 parameters",
            "D is a width; it is not a parameter count.")
    finding(21, side, "the GGUF artifact was created using the cited commit",
            "The commit identifies the clean llama.cpp reference, not artifact creation.")
    finding(22, side, "lm_head.weight and model.embed_tokens.weight equal each other",
            "Each is separately compared with a namespaced duplicate; the excerpt does not compare head to embeddings.", True)
    finding(23, side, "the table is a dataset with 16 registers",
            "It is a benchmark table of S15 arms, including E256, E64 and E16, not registers.", True)
    finding(23, side, "GSZ is a contiguous register",
            "GSZ is a numeric group-size column or a dash; contiguous describes dense run access.")
    finding(23, side, "the E256 gate/up run is contiguous",
            "Its row shows 26,880 B; contiguous belongs to S15-DENSE.")
finding(19, "A", "layer 27 is a fully connected layer with 27 neurons",
        "27 appears in a list of layer indices; F=8960 is the FFN width.")
finding(19, "A", "the network is trained in this configuration",
        "The excerpt states CPU fp32/eager and no weight changes, not a training run.")
finding(19, "B", "a fully connected layer has 2148 parameters",
        "No 2148 value appears; the visible FFN width is F=8960.")
finding(19, "B", "the SHA-256 of labels is used to train the network",
        "The SHA identifies a labels file; the excerpt says no weight changes.")
finding(20, "A", "the falsifiable hypothesis is that degeneration is not free-running",
        "The changed coordinate explicitly measures 256-token greedy free-running trajectories.", True)
finding(20, "A", "the rollout primitives are unchanged for GigaChat execution",
        "STRAT-02 has original primitives but lacks the stated GigaChat GGUF/source-token-ID execution.")


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
    output = {"experiment": "METH-117-alpha075-arm-blind-external-verdict",
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
