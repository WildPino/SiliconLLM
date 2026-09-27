"""Transcribe the METH-62 single-reviewer, arm-blind excerpt verdict."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
BLIND = DIR / "meth62_blind_semantic_review.json"
OUT = DIR / "meth62_blind_semantic_verdict.json"
BLIND_SHA = "e9f0ebf528363406cdabaa6ce9aba0389b53bd18ce36bc66afce6f419641f3b2"


def f(claim, evidence, severe=False):
    return {"claim": claim, "evidence": evidence, "severe": severe}


# Row indices and A/B labels only: no donor/student mapping is used here.
# Each counted finding cites the excerpt's shortest useful contradicting
# or delimiting span. Ambiguous readings are listed but not counted.
R = {
    0: {
        "A": {"unsupported": [
            f("The script sets up a directory structure.", "'HERE = os.path.dirname(os.path.abspath(__file__))' computes a path; no directory is created."),
        ], "ambiguous": ["'part of a larger project' is generic context, not a checkable excerpt claim."]},
        "B": {"unsupported": [
            f("OCC_BAR and HERE are the two retired files.", "'OCC_BAR = 0.5' and 'HERE = os.path.dirname(...)' define variables; 'the two files below' are not named in the excerpt.", True),
            f("The script itself was produced by the runner without hard_gate.", "'They were produced by a runner that never set `hard_gate`' refers to the retired files, not this script."),
        ]},
    },
    1: {
        "A": {"unsupported": [
            f("Carving, pruning, basis, low-rank, reconstruction and sparsity are performance metrics.", "'has measured carving (D0/D0c), pruning (D1), basis (D2), low-rank (D3), reconstruction (D4) and sparsity (S1)' names interventions, not metrics."),
        ], "ambiguous": ["'each weight is multiplied by {-1,0,+1}' is loose wording for ternary weights."]},
        "B": {"unsupported": [
            f("The programme measured various types of weights called carving, pruning and basis.", "'has measured carving ... pruning ... basis ... all of them on fp32 weights' names techniques applied to weights."),
        ]},
    },
    2: {
        "A": {"unsupported": [
            f("The function takes config.eos_token_id, use_cache and output as inputs.", "The excerpt shows 'config.eos_token_id, use_cache=True)' and then 'continuation = output[...]', but no function signature."),
            f("It decodes continuation text into a list of 8-gram tokens.", "'continuation_text': tokenizer.decode(continuation, skip_special_tokens=False)' decodes token IDs to text; 'repeated_8gram_3x' is a separate field.", True),
        ]},
        "B": {"unsupported": [
            f("The function generates a list of dictionaries.", "'row[\"generation\"][arm] = {' assigns one dictionary entry; no list of dictionaries appears."),
            f("The dictionary contains a 3× repeated 8-gram token.", "The clipped field name is 'repeated_8gram_3x'; it does not store a particular repeated token."),
        ]},
    },
    3: {
        "A": {"unsupported": [
            f("qualified_runner is a file in the layer2.__file__ directory.", "'Path(layer2.__file__).resolve()' resolves the module file path itself, not a directory."),
        ]},
        "B": {"unsupported": [
            f("qualified_runner is a file in the layer2.__file__ directory.", "'Path(layer2.__file__).resolve()' resolves a file path, not its containing directory."),
        ], "ambiguous": ["The error may list 'recovery' among missing sources, but the excerpt does not identify which path is missing in any run."]},
    },
    4: {side: {"unsupported": [
        f("validate_report performs the shown path and payload checks.", "'return resolved' ends the preceding function; only then does 'def validate_report(root: Path, model: Path, sources: ...)' begin."),
    ]} for side in ("A", "B")},
    5: {
        "A": {"ambiguous": ["The protocol filename suggests a recovery test, but the excerpt only defines paths and hashes."]},
        "B": {"unsupported": [
            f("The excerpt provides a raw binary head SHA.", "It separately defines 'RAW_HEAD=...' and 'RAW_BINARY_SHA=...'; no 'raw binary head SHA' is identified."),
        ], "ambiguous": ["'location of the raw binary SHA' may loosely mean the constant's position in code."]},
    },
    6: {
        "A": {"unsupported": [
            f("adjudicate returns the shown tensors, caches, FFN and report tuple.", "'return tensors, caches, ffn, {\"report\": report, \"q8_census\": census}' appears before 'def adjudicate(...)'."),
            f("The Q8 census dictionary belongs to the shown adjudicate body.", "'census[key] = measured' is above 'def adjudicate(...)'; its body is clipped."),
        ]},
        "B": {"unsupported": [
            f("ffn and ref_ffn are functions.", "The signature annotates 'ffn: dict[str, np.ndarray]' and 'ref_ffn: dict[str, np.ndarray]'."),
            f("adjudicate returns the preceding tuple of tensors, caches and features.", "'return tensors, caches, ffn, ...' is before the 'def adjudicate(...)' line."),
            f("adjudicate first checks Q8 census and updates the candidate with measured data.", "'census[key] = measured' precedes 'def adjudicate(...)'; no body for adjudicate is visible."),
        ]},
    },
    7: {
        "A": {"unsupported": [
            f("The shapes of qA and qB are determined by counts of factors in r-th/B-th act-search.", "The excerpt calls 'H0.r3_actsearch(A.float(), ...)' and 'H0.r3_actsearch(B.float(), ...)' but states no such shape rule.", True),
            f("fh.write writes both matrix shapes.", "'fh.write(struct.pack(\"<ii\", MK_FACTORED, r))' writes a tag and r, not shapes for qA and qB.", True),
        ]},
        "B": {"unsupported": [
            f("A and B are passed through activation functions and concatenated.", "The excerpt calls 'r3_actsearch' twice and writes a tag; it shows no concatenation."),
            f("All dimensions of A, B, qA and qB equal r.", "Only 'A.shape[1] == r and B.shape[0] == r' is visible; qA/qB dimensions are not given.", True),
            f("The result matrices are written to the file here.", "'fh.write(struct.pack(\"<ii\", MK_FACTORED, r))' writes two integers; the comment says the reader later handles tagged matrices."),
        ]},
    },
    8: {
        "A": {"unsupported": [
            f("The narrator is sorry about the Queen of Bohemia's health.", "'the Court, which I am sorry to hear ... bring all to ruin again' is the cause of concern; the Queen is mentioned later.", True),
            f("The narrator attends dinner with the Queen.", "'he and I to the Wardrobe to dinner' precedes 'The Queen of Bohemia was here'; no dinner with her is stated.", True),
            f("They attend a concert.", "They went 'to the Opera, and saw \"The Witts\" again'; no concert is mentioned."),
            f("Craven was brought by the Queen and they met him.", "'The Queen of Bohemia was here, brought by my Lord Craven' reverses who brought whom.", True),
        ]},
        "B": {"ambiguous": ["'courtier' and 'friend' are plausible roles but not established for the unnamed dinner companion."]},
    },
    9: {
        "A": {"unsupported": [
            f("Minnie's pet monkey is the gentleman.", "'A gentleman, returning from India, brought a monkey, which he presented to his wife' distinguishes man and monkey.", True),
            f("The gentleman named the monkey Sprite.", "'She called it Sprite' assigns the name to his wife.", True),
        ]},
        "B": {},
    },
    10: {
        "A": {"ambiguous": ["'journey to the Episcopal church in both places' interprets a clipped sentence following church visits in Glasgow and London."]},
        "B": {"unsupported": [
            f("She considered music a form of prayer.", "The excerpt says she took 'her prayer book and go for ‘music’ to the Episcopal'; it does not equate music with prayer.", True),
        ]},
    },
    11: {
        "A": {"unsupported": [
            f("The palace is explicitly identified as from the Roman Empire.", "The excerpt calls it 'the now ruined Palace of the Cæsars' and does not name the Roman Empire."),
            f("The Thermæ is a similar structure from the same era.", "Only 'this palace was probably as an architectural object inferior to the Thermæ' is stated."),
        ]},
        "B": {"unsupported": [
            f("The Palace of the Cæsars is one of the engineering works described.", "'aqueducts and engineering works ... and ... Palace of the Cæsars' lists them separately."),
            f("The Thermæ is a ruined palace.", "Only the Palace of the Cæsars is called 'now ruined'; the Thermæ is the comparison object.", True),
        ]},
    },
    12: {
        "A": {},
        "B": {"unsupported": [
            f("The horseman is trapped in a deep ravine and cannot escape.", "The excerpt starts with him crawling back to the horse and then riding west; no ravine or entrapment appears."),
            f("The horse found the horseman.", "'He crawled back to his horse and found him' gives the opposite action.", True),
            f("The horseman says he himself will go no further.", "'If he goes this twenty miles ... he will go no more' refers to the horse.", True),
            f("He has never taken the Sun Dance Trail before.", "'We must make for our old beat, the Sun Dance Trail' calls it an old route.", True),
        ]},
    },
    13: {
        "A": {"unsupported": [
            f("General Custer is leading the regiment.", "'General Custer rode up alongside of me' and 'The regiment left the column' do not place him in command of its movement."),
            f("The regiment reaches Appomattox Station within the excerpt.", "The excerpt only states a plan 'to Appomattox Station' and that the regiment accelerated, ending at 'until'.", True),
        ]},
        "B": {"ambiguous": ["Custer's 'Go in' encouragement occurs beside the plan to seize trains and pike, but the quote itself does not name those objectives."]},
    },
    14: {
        "A": {"unsupported": [
            f("Miranda was thrilled.", "The excerpt ends at 'Miranda had t' before any reaction is described.", True),
        ]},
        "B": {"unsupported": [
            f("Miranda was delighted to see the guest.", "The guest is only 'expected'; the excerpt ends at 'Miranda had t' before an arrival or reaction.", True),
        ]},
    },
    15: {
        "A": {"unsupported": [
            f("The woman asks whether she will leave the man alone.", "She says 'You will not leave me without a word, Mr. Raymond', asking him not to leave her silently.", True),
            f("The woman is surprised to find the man there.", "She asks him 'Were you so very much astonished to find me here?', making his surprise the question.", True),
        ]},
        "B": {"unsupported": [
            f("The man says 'Were you so very much astonished to find me here?'.", "'Then, as I came slowly forward: \"Were you so very much astonished to find me here?\"' continues the woman's speech; the man's answer follows.", True),
            f("The man tries to recall the man's intentions.", "His reply is 'I do not know--I did not expect--'; no intentions are described."),
        ]},
    },
    16: {side: {"unsupported": [
        f("The source's vector-dot right operand stays F32 in the dot.", "'F32 right operand to F16 before an F16×F16 vector dot'; only 'project reconstruction currently retains that operand as F32'.", True),
    ]} for side in ("A", "B")},
    17: {
        "A": {"unsupported": [
            f("Untying and ternarizing removes the output head.", "'the head is a dense GEMV over 151,936 rows' and 'head 17.42 → 4.65 ms' show it remains.", True),
            f("The optimization reorganizes the embedding table into a row lookup.", "'The embedding table stays fp32 because it is a row lookup' describes its existing role, not a reorganization."),
            f("Packing two trits per byte is the successful speedup.", "The section says 'Two optimisations, and only one worked' and labels 'Untie and ternarize the head — worked'.", True),
        ]},
        "B": {},
    },
    18: {
        "A": {"unsupported": [
            f("RoPE is a RoPE-based Encoder-Decoder architecture.", "The excerpt says 'RoPE only on 64-d key/query positional component'; it provides no such expansion."),
            f("YaRN is a Yen-Ren Network architecture.", "The excerpt only says 'YaRN values listed'; this named expansion is absent."),
            f("YaRN is used for values here.", "'RoPE only on 64-d key/query positional component; YaRN values listed' does not say YaRN acts on value vectors.", True),
            f("The model reduces size by separating full K/V projections.", "'MLA compressed KV path is absent from both local engines' and 'Separate full K and V projections only' give no such goal or result.", True),
        ]},
        "B": {"unsupported": [
            f("attn_k_b and attn_v_b are derived from themselves.", "'K-B and V-B derived from attn_k_b / attn_v_b' distinguishes derived tensors from sources."),
            f("An absent key-value pair is a model feature.", "'Absent' is a table entry for the MLA compressed KV path, not a key-value pair."),
            f("The local engines use a compressed KV path.", "'MLA compressed KV path is absent from both local engines' directly contradicts this.", True),
            f("YaRN is YaRNet.", "The excerpt says 'YaRN values listed' and gives no expansion to YaRNet."),
        ], "ambiguous": ["n_kv_a_norm may normalize a KV intermediate, but the visible fragment does not specify its exact operation."]},
    },
    19: {
        "A": {"unsupported": [
            f("attn+head falls from 1.367 to 0.278096 BPB.", "'+0.278096 BPB' is the cost of the same-rank lm_head experiment; the 'attn+head falls from 1.367' sentence is cut before a destination value.", True),
        ]},
        "B": {"unsupported": [
            f("The lm_head rank occupies 51–56% of the speed budget.", "'The head is 545 M on the Coder-7B = 51–56% of the entire 50 tok/s budget' attributes the share to head payload, not rank.", True),
            f("The attn+head rank is 1.367.", "'attn+head falls from 1.367' gives a truncated metric value, not a rank.", True),
        ]},
    },
    20: {
        "A": {"unsupported": [
            f("E32 moved the programme further than E33.", "'(E32, E33) moved the programme further than this one did' compares both with the current probe, not each other.", True),
            f("The 21,069,824 B attention figure belongs to E32 and 67,239,936 B belongs to E33.", "'At T10's width, with the FFN at zero ... attention, per layer ... 21,069,824 B; head ... 67,239,936 B' assigns them to organs, not E32/E33.", True),
            f("21,069,824 B is the budget ceiling for the smallest attention.", "The excerpt labels it 'attention, per layer, moved'; the budget line is clipped after 'moved budget at'.", True),
        ]},
        "B": {"unsupported": [
            f("E32 moved the programme further than E33.", "'(E32, E33) moved the programme further than this one did' does not compare E32 to E33.", True),
            f("E32 and E33 each have the stated attention/head moved-byte figures.", "The figures appear under 'At T10's width ... attention, per layer ... head, moved', with no E32/E33 attribution.", True),
            f("67,239,936 B is attention moved per head.", "'head, moved: 67,239,936 B' is a separate model head, not attention per-head bytes.", True),
        ]},
    },
    21: {
        "A": {},
        "B": {"unsupported": [
            f("The model has 28,672 experts.", "'d_ffn = 28672' and 'E = 28672 / 128 = 224 experts' distinguish FFN width from expert count.", True),
            f("Attention active fraction is 3.57%.", "'At 3.57% FFN active, 12.5% attention' gives different fractions.", True),
        ]},
    },
    22: {
        "A": {"unsupported": [
            f("Fixed row-column pairing is a 3D matrix issue from graphics/data analysis.", "'Fixed row-column pairing: concatenating up-projection matrix columns' names a packing approach, not a 3D matrix field."),
            f("The paper reports a quantified failure magnitude.", "'No numeric bundling-degree or redundancy percentage ... [X] for a quantified failure magnitude' says the value was not surfaced.", True),
        ]},
        "B": {"unsupported": [
            f("The authors observed a quantified failure magnitude.", "'No numeric bundling-degree or redundancy percentage was surfaced ... [X] for a quantified failure magnitude' states the opposite.", True),
        ]},
    },
    23: {
        "A": {"unsupported": [
            f("Document §11.1 table vs JSON has 120 rows.", "'d0_model.log ... all 120' is separate; 'document §11.1 ... 23 parsed rows × 7 fields' has 161 checks.", True),
            f("Document §11.6 table vs JSON has 7 rows.", "'document §11.6 ... 28 × 7 ... 196' gives 28 rows and 7 fields.", True),
            f("Document §11.8 table vs JSON has 5 rows.", "'document §11.8 ... 25 × 5 ... 125' gives 25 rows and 5 fields.", True),
        ]},
        "B": {"unsupported": [
            f("All listed document-section row counts are rows of d0_model.log.", "'d0_model.log fidelity rows (all 120)' is one row; the §11.1/§11.6/§11.8 rows compare document tables with JSON."),
        ], "missing_detail": True,
        "missing_detail_evidence": "The excerpt offers concrete counts such as '23 parsed rows × 7 fields' and '196'; B states none of them."},
    },
}


def main():
    blind_bytes = BLIND.read_bytes()
    assert hashlib.sha256(blind_bytes).hexdigest() == BLIND_SHA
    blind = json.loads(blind_bytes)
    assert len(blind["rows"]) == 24 and set(R) == set(range(24))
    rows = []
    for index, original in enumerate(blind["rows"]):
        row = {"source_id": original["source_id"]}
        for side in ("A", "B"):
            details = R[index].get(side, {})
            row[side] = {"unsupported": details.get("unsupported", []),
                         "missing_detail": details.get("missing_detail", False),
                         "ambiguous": details.get("ambiguous", [])}
            if "missing_detail_evidence" in details:
                row[side]["missing_detail_evidence"] = details["missing_detail_evidence"]
        rows.append(row)
    result = {"experiment": "METH-62-arm-blind-semantic-verdict",
              "blind_sha256": BLIND_SHA,
              "reviewer": "single agent, excerpt-only review",
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"rows": len(rows),
                      "verdict_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
