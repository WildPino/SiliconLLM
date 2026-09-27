"""Transcribe METH-90 single-reviewer findings without donor/student labels."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
BLIND = DIR / "meth90_blind_semantic_review.json"
OUT = DIR / "meth90_blind_semantic_verdict.json"
BLIND_SHA = "30b8614e65582ebf5cab24d887332cd44f500e99fba00231ed55d4749164358d"


def f(claim, evidence, severe=False):
    return {"claim": claim, "evidence": evidence, "severe": severe}


# Indices and A/B labels only. The mapping to model arms is not consulted.
R = {
    0: {
        "A": {"unsupported": [
            f("The checkpoint was created using the M15.M13 model.",
              "'assert checkpoint[\"source_sha256\"] == M15.M13.MODEL_SHA' verifies an identity; it does not show how the checkpoint was created."),
            f("wrappers is a list of dictionaries.",
              "'for w, values in zip(wrappers, checkpoint[\"expert_state\"])' distinguishes wrapper objects from the saved value dictionaries."),
        ]},
    },
    1: {
        "A": {"unsupported": [
            f("The model comes from a GitHub repository.",
              "'hf_hub_download(M42.MODEL, \"model.safetensors\", revision=M42.REV, local_files_only=True)' names a Hugging Face Hub API, not GitHub.", True),
            f("The excerpt preprocesses the model.",
              "It selects a CUDA device and obtains a safetensors path; no preprocessing operation is visible."),
            f("The script starts a download.",
              "'local_files_only=True' requires an already local cached file."),
        ]},
        "B": {"unsupported": [
            f("The model comes from a GitHub repository.",
              "'hf_hub_download' is a Hugging Face Hub API; GitHub is not named.", True),
            f("The script sets the device to cuda:0.",
              "'torch.device(f\"cuda:{matches[0]}\")' uses a matched device index, not the literal 0.", True),
            f("It initializes the model for parallel processing.",
              "The excerpt obtains a source path and chooses one device; no model construction or parallel processing is shown."),
            f("It starts a download process.",
              "'local_files_only=True' rules out fetching a remote model here."),
        ]},
    },
    2: {
        "A": {"unsupported": [
            f("Terminating the child process is not allowed.",
              "'bounded._terminate_only_child(process)' is called when a process exists."),
            f("The process is not terminated.",
              "The excerpt explicitly calls '_terminate_only_child(process)'."),
            f("The error type is process rather than child.",
              "'error_type': type(exc).__name__ records the exception class name, not either process or child."),
        ]},
        "B": {"unsupported": [
            f("The parent process terminates.",
              "The visible termination call is '_terminate_only_child(process)', which targets its worker."),
            f("The parent never reaches the worker it created.",
              "The comment says 'Never reaches beyond the worker this parent created', a containment bound on termination."),
        ]},
    },
    3: {},
    4: {
        "A": {"unsupported": [
            f("Tracker.open initializes active and maximum to zero.",
              "'self.active = self.maximum = 0' is inside '__init__', before 'def open(self, path)'."),
            f("Tracker.open creates a list of shards for the path.",
              "The visible open body only assigns 'shard = path.name'; list creation occurs in '__init__' and is empty."),
        ]},
        "B": {"unsupported": [
            f("Tracker.open initializes active, maximum and opened.",
              "Those assignments appear in '__init__', before 'def open(self, path)'."),
            f("slice_refs is not defined in the excerpt.",
              "'self.slice_refs = []' appears in '__init__'."),
        ]},
    },
    5: {
        "A": {"unsupported": [
            f("down_proj computes a weighted sum that is passed through up.",
              "The excerpt says hooks sit on down_proj's INPUT, already computed as 'h = silu(gate x) * up x'; down_proj follows that input.", True),
            f("up multiplies the weighted sum by gate x and then applies sila.",
              "The visible order is 'h = silu(gate x) * up x'; SiLU acts on gate x before multiplication, not after up.", True),
        ]},
        "B": {"unsupported": [
            f("The intermediate uses Sigmoid (sila).",
              "The formula explicitly says 'silu(gate x)', the SiLU activation, not Sigmoid or 'sila'."),
        ]},
    },
    6: {
        "A": {"unsupported": [
            f("The script reads HEADER and ffn as two text files and combines them into one text file.",
              "'HEADER.read_text' and 'upstream.HEADER.read_text' read two headers; 'ffn[\"prefill8/ffn_gate-0\"]' is a mapping access, and no output file is shown.", True),
            f("kb_prior holds controls in the ffn file.",
              "'kb_prior.get(\"adjudication\").get(\"controls\")' is a separate mapping; no ffn file relationship is visible."),
        ]},
        "B": {"unsupported": [
            f("The script reads HEADER and ffn as text files and combines them into one string.",
              "It reads 'HEADER' and 'upstream.HEADER'; ffn is indexed as a mapping, and no concatenation is visible.", True),
            f("It extracts prior_kb_controls from HEADER.",
              "'prior_kb_controls' is computed from 'kb_prior.get(\"adjudication\")', not HEADER."),
        ]},
    },
    7: {
        "A": {"unsupported": [
            f("The function sums elements of an 8×1 x array.",
              "'_mm256_fmadd_ps(vf, _mm256_loadu_ps(x + i), acc)' multiplies converted int8 values by x and accumulates a dot product; the excerpt gives no 8×1 shape.", True),
            f("It converts x elements to 256-bit floating-point integers and stores the result back into x.",
              "'_mm256_cvtepi8_epi32(eight)' converts 'eight'; '_mm256_storeu_ps(lanes, acc)' stores into lanes, not x."),
        ]},
        "B": {"unsupported": [
            f("The function takes scalar e8 and e16 inputs.",
              "No e8/e16 variables appear; the visible inputs are vector loads and 'project(const float *basis, const float *x, float *out)'.", True),
        ]},
    },
    8: {
        "A": {"unsupported": [
            f("A poet praises the nightingale in this passage.",
              "'as the poets will have it sing' describes poets' doleful portrayal; the first-person narrator, not a poet, praises its 'con brio' song."),
        ]},
        "B": {"unsupported": [
            f("The nightingale is cooing.",
              "'cooing of the stock-dove' names the cooing bird; the Nightingale is discussed separately.", True),
            f("The author pities the nightingale's inability to sing melodically.",
              "The excerpt praises the Nightingale's 'varied phrases' and 'marvellous crescendo'; the clipped 'It is a pity to co' gives no such inability.", True),
        ]},
    },
    9: {
        "A": {"unsupported": [
            f("The woman criticizes her doctor's book and exercise regimen.",
              "She invokes 'her doctor's book' to say 'vig'rous exercise was the best physic', rather than criticizing it."),
            f("The doctor must chop hard.",
              "The text says 'he must chop hard' after she cites the doctor's book; it never identifies the chopper as the doctor."),
            f("She is looking for methods to cure her own illness.",
              "The visible cure is 'she cured 'em' and the chopping instruction applies to 'he'; her own illness is not stated."),
        ]},
        "B": {"unsupported": [
            f("The woman criticizes her doctor's book.",
              "She cites the book approvingly as saying vigorous exercise is the best physic."),
            f("She has already gained significant fitness.",
              "'figgering that she'd gained the first lap' refers to her plan after hearing the ax, not her physical fitness."),
        ]},
    },
    10: {
        "A": {"unsupported": [
            f("Illmarinen is washed up and made whole.",
              "'However, Sampo is washed up, and made whole' gives that event to Sampo, not Illmarinen.", True),
            f("Sampo is a person whose actions cause good days.",
              "The excerpt says Sampo is washed up and later possessed by the sons of Kalevala; it does not identify a person or make Sampo the speaker of 'Good days come'."),
        ]},
        "B": {"unsupported": [
            f("Illmarinen is a Buddhist monk.",
              "The Illmarinen/Sampo passage precedes a transition to 'Mr. Latham's ... view of Buddhism'; no monk identity is given.", True),
            f("He creates jewelry for his second wife.",
              "'He had previously made his second wife ... out of the same metals' describes making the wife, not jewelry."),
            f("Sampo is Illmarinen's second wife.",
              "'However, Sampo is washed up' follows a separate sentence about his second wife; no identity between them is stated.", True),
        ]},
    },
    11: {
        "A": {"unsupported": [
            f("The mill and church are being built.",
              "'A steam-engine has been put in the mill' and 'The church is restored' describe an existing mill and restored church; only the chapel is new."),
        ]},
        "B": {"unsupported": [
            f("The mill and steam engine are under construction.",
              "'A steam-engine has been put in the mill' states a completed installation, not construction."),
        ]},
    },
    12: {
        "A": {"unsupported": [
            f("Vassar considers plotting with Waldron to remove another man.",
              "'He wondered ... if she could be capable of plotting with Waldron to remove him' makes the woman the possible plotter and him the possible target.", True),
        ]},
        "B": {"unsupported": [
            f("Vassar considers manipulating Waldron into removing another man.",
              "The suspicion is that 'she' might plot with Waldron 'to remove him'; Vassar is not shown manipulating Waldron.", True),
            f("He looks into Waldron's brown eyes.",
              "'He looked into the depths of her brown eyes' refers to the woman, not Waldron.", True),
        ]},
    },
    13: {
        "A": {"unsupported": [
            f("The area has little or no vegetation.",
              "'very few remained of those which had grown here' still describes surviving trees; the 'waste land' is part of a historical panorama, not a vegetation census."),
        ]},
        "B": {"unsupported": [
            f("Most original inhabitants vanished, leaving a few survivors.",
              "'very few remained of those which had grown here' refers to forest growth, not human inhabitants."),
            f("The area has little or no vegetation.",
              "The excerpt describes remaining growth and a former forest; no current blanket lack of vegetation is established."),
        ]},
    },
    14: {
        "A": {"unsupported": [
            f("Lisette is over head and ears in love.",
              "'Poor Ludvig is over head and ears in love' attributes that state to Ludvig, not Lisette.", True),
            f("His compliments concern Lisette's lack of familiarity with him.",
              "'compliments such as Lisette is not accustomed to hear from him' says she is unaccustomed to his compliments, not to him."),
        ]},
        "B": {"unsupported": [
            f("Lisette is over head and ears in love with Ludvig.",
              "The passage explicitly says 'Poor Ludvig is over head and ears in love', not Lisette.", True),
        ]},
    },
    15: {
        "A": {"unsupported": [
            f("The passage describes Claude Monet.",
              "'One sees pictures of Matisse' and 'reminded of Manet' name different artists; Monet is absent.", True),
            f("The artist's style uses bright colors and focuses on nature.",
              "The excerpt contrasts Matisse pictures with 'inward vitality' or 'outer charm'; no colors or nature subject is stated."),
        ]},
        "B": {"unsupported": [
            f("The passage describes Claude Monet.",
              "It names Matisse and Manet, not Monet.", True),
            f("The passage establishes Manet as an Impressionist.",
              "'(How often one is reminded of Manet in this.)' makes a comparison but gives no style classification for Manet."),
        ]},
    },
    16: {
        "A": {"unsupported": [
            f("POSTHOC_TRANSFER_SIGNAL=false means the transfer is not performed or recorded.",
              "'POSTHOC_TRANSFER_SIGNAL=false, recorded in the H5 result' gives a recorded verdict; the question concerns whether two trained components transfer positively.", True),
        ]},
        "B": {"unsupported": [
            f("The H4 and H2I components transfer positively in the pretrained model.",
              "The excerpt asks '**Question:** do ... transfer positively' and opens with 'POSTHOC_TRANSFER_SIGNAL=false', so a positive result is not stated.", True),
        ]},
    },
    17: {
        "A": {"unsupported": [
            f("The excerpt describes an actual preprocessing step for model loading or quality measurement.",
              "'specifica prima di scaricare pesi o misurare qualità' calls it a specification before either action."),
            f("It gives a description of the tokenizer/scoring process.",
              "'[tokenizer/scoring](STRAT_02_STAGE0_TOKEN_SCORING.md)' is a link, not a description of that process."),
        ]},
    },
    18: {
        "A": {"unsupported": [
            f("Ath is the named experiment or kernel being compared.",
              "The excerpt begins midword at 'ath is'; it does not identify an experiment called Ath."),
        ]},
        "B": {"unsupported": [
            f("The fixed verdict cell refutes the 2.5× slowdown.",
              "The excerpt says the path is '~2.5× slower per weight' and therefore the speed lever 'is refuted'; the cell was fixed beforehand to prevent a posthoc choice."),
        ]},
    },
    19: {
        "A": {"unsupported": [
            f("GGUF expands to Generalized Generalized Unified Format.",
              "The excerpt uses 'GGUF' without an expansion; the invented expansion is unsupported.", True),
            f("The cited SHA-256 is the size hash of the accepted GGUF.",
              "'The accepted GGUF is 6,474,702,976 bytes with SHA-256 ...' gives separate size and content hash, not a hash of the size."),
        ]},
        "B": {"unsupported": [
            f("The adjudication and manifest are six bytes long.",
              "The number '6,474,702,976 bytes' refers to 'The accepted GGUF', not the two preceding documents, and is not six bytes.", True),
            f("The adjudication and manifest have the stated GGUF SHA-256.",
              "The excerpt gives two preceding hashes for those items and a separate 'accepted GGUF ... SHA-256 68a873...' hash.", True),
        ]},
    },
    20: {
        "A": {"unsupported": [
            f("The secondary threshold is 0.2 times the across-seed standard deviation.",
              "The excerpt says 'more than 2× the across-seed standard deviation' of COACT − D0C, not 0.2×.", True),
            f("E41 is the secondary cell.",
              "E41 is mentioned before a separately marked '**Secondary registered question**'; the excerpt does not identify it as that cell."),
        ]},
        "B": {"unsupported": [
            f("E41's unresolved cell is the COACT − D0C secondary question.",
              "The unresolved E41 cell is discussed before the separately marked secondary question; no link equates them."),
            f("The across-seed standard deviation of every arm is not provided in this specification.",
              "The excerpt says it is a required quantity and E41 did not have it; it does not say no arm's value is provided elsewhere."),
        ], "ambiguous": ["The excerpt begins mid-sentence, so the precise E41 primary cell wording is clipped."]},
    },
    21: {
        "A": {"unsupported": [
            f("The model was trained on the run-session log.",
              "'h1-qat-run-session-2.log' is listed under '**Results:**'; the trainer is separately named 's1/h1_qat.py'."),
        ]},
        "B": {"unsupported": [
            f("The training data is applied at both checkpoints.",
              "'G-H1 PASSES at both checkpoints, band TRAINING-HELPS' is an evaluation verdict, not a statement that training data is applied."),
        ]},
    },
    22: {
        "A": {"unsupported": [
            f("The recovery of real_H is −0.4827.",
              "'Its recovery +0.4827' explicitly gives a positive, not negative, value.", True),
            f("real_H and down_proj@50% are the two columns of the visible table.",
              "They appear in prose below a clipped numeric table; no such two-column table is visible."),
        ]},
        "B": {"unsupported": [
            f("C2 saturation is the treatment whose recovery is 0.4827.",
              "'real_H is the treatment, not a control. Its recovery +0.4827' attributes the recovery to real_H; C2 is labeled 'for reference'.", True),
            f("4.5129 is the recovery of C2 saturation from a reference value.",
              "'4.512897966...' appears in a clipped table with no column headers visible; its meaning as recovery is unsupported."),
            f("The visible table has only real_H and C2 saturation columns.",
              "The clipped row shows multiple numeric fields and the prose below mentions down_proj@50%; the claimed two-column structure is not supported."),
        ]},
    },
    23: {
        "A": {"unsupported": [
            f("FFN means Fast Forwarding Neural Network.",
              "The excerpt uses 'FFN' without this expansion; the invented term is unsupported.", True),
            f("The FFN predicts the output of a ternary set.",
              "The excerpt asks whether attention/other organs 'come out of the ternary set'; it gives no prediction role for FFN."),
        ]},
        "B": {"unsupported": [
            f("The FFN is not triggered by the absolute-Δ condition.",
              "'It does not trigger ATTENTION-BOUND' refers to an attention condition; the passage does not say FFN is not triggered."),
        ]},
    },
}


def main():
    raw = BLIND.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BLIND_SHA
    blind = json.loads(raw)
    assert len(blind["rows"]) == 24 and set(R) == set(range(24))
    rows = []
    for index, original in enumerate(blind["rows"]):
        row = {"source_id": original["source_id"]}
        for side in ("A", "B"):
            finding = R[index].get(side, {})
            row[side] = {"unsupported": finding.get("unsupported", []),
                         "severe_count": sum(c["severe"] for c in finding.get("unsupported", [])),
                         "missing_detail": finding.get("missing_detail", False),
                         "ambiguous": finding.get("ambiguous", [])}
        rows.append(row)
    output = {"experiment": "METH-90-arm-blind-semantic-verdict",
              "blind_sha256": BLIND_SHA,
              "reviewer": "single agent, excerpt-only review",
              "rows": rows}
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"rows": len(rows),
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
