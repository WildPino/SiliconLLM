"""Transcribe the METH-74 single-reviewer verdict without arm labels."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
BLIND = DIR / "meth74_blind_semantic_review.json"
OUT = DIR / "meth74_blind_semantic_verdict.json"
BLIND_SHA = "d4599a0bbd27f6adfebbed69dcb7ac6faada8717145e111134ef3e34f6cec1d9"


def f(claim, evidence, severe=False):
    return {"claim": claim, "evidence": evidence, "severe": severe}


# Only row indices and A/B labels are used. A finding is counted when
# the response asserts more than the visible 384-character excerpt.
R = {
    0: {
        "A": [f("The visible time measurement is for restoration.",
                "'tq = time.time()' precedes 'apply_arm'; the shown field starts 'quantize_seconds'.")],
        "B": [f("The model is restored using the T2b code path.",
                "'restore, s2 = apply_arm(...)' receives a restore handle; no restore call appears.")],
    },
    1: {
        "A": [f("The read-only stream over a buffer is not a measurement.",
                "'a spec-sheet number is not a measurement. This is that something: a read-only stream' says the stream is the measure.")],
        "B": [f("The read-only stream over a buffer is not a measurement.",
                "'a spec-sheet number is not a measurement. This is that something: a read-only stream' says the stream is the measure.")],
    },
    2: {
        "A": [
            f("max_relative_l2_error is an absolute actual-versus-expected error.",
              "'max((x[relative_l2_error] for x in body_stats), default=0)' takes a maximum of relative L2 error fields."),
            f("max_abs_error compares actual and expected absolute error values.",
              "'max((x[max_abs_error] for x in body_stats), default=0)' takes a maximum of existing per-body fields."),
        ],
        "B": [f("The maximum absolute error equals zero.",
                "'default=0' is the fallback for an empty body_stats collection, not the observed maximum.", True)],
    },
    3: {
        "B": [
            f("The loop iterates over a dictionary of Git objects.",
              "'for item in sources.values()' refers to source entries, not Git objects."),
            f("It checks for a file named relative in the root directory.",
              "'for path in relative' and 'str(path)' check each relative path, not a file named relative."),
        ],
    },
    4: {
        "A": [f("The number of bytes is appended to a list named nb.",
                "'byts.append(nb)' appends nb to byts, not to nb.")],
        "B": [f("The tensor has dimensions ids_all, byts and offsets.",
                "'ids = torch.tensor(ids_all, dtype=torch.long)' constructs ids from ids_all alone.")],
    },
    5: {
        "A": [
            f("masked_objective is nested inside set_experts.",
              "'def masked_objective(...)' is at the same indentation as 'def set_experts(...)'."),
            f("The function calculates log-softmax of teacher tokens.",
              "'teacher_lp = F.log_softmax(teacher_logits.float(), dim=-1)' uses logits, not tokens."),
        ],
        "B": [f("The masks are for teacher logits and teacher tokens.",
                "The signature names 'ce_mask, kl_mask'; the visible lines apply log-softmax to logits and NLL to targets.")],
    },
    6: {
        "A": [f("Saved teacher rows contain prompt_ids.",
                "'prompt = prompts[saved[train_row]]'; prompt_ids are then read from prompt, not saved.")],
        "B": [f("Concatenating prompt and continuation forms a full prompt.",
                "'full = prompt[prompt_ids] + saved[continuation_ids]' adds continuation IDs to the prompt.")],
    },
    7: {
        "A": [f("The visible code defines a Python class Ice.",
                "The excerpt begins with a clipped method ending 'ice(self, key)' and never shows a class named Ice.")],
        "B": [f("The visible code defines a Python class Ice.",
                "The excerpt begins with a clipped method ending 'ice(self, key)' and never shows a class named Ice.")],
    },
    8: {
        "B": [
            f("Bathing in salt water is a way to believe in the millennium.",
              "'human nature is by no means altered by bathing every morning in salt water' rejects that implication.", True),
            f("The same point applies to the office environment.",
              "The excerpt ends at 'And there are many office'; it gives no office-environment claim."),
        ],
    },
    9: {
        "A": [
            f("Frankie explains the bear attack.",
              "'It's true,' affirmed Professor Scotch identifies the speaker.", True),
            f("Bears are attracted to mosquitoes and attack them.",
              "'bears, lured by hunger, will come down into the lowlands, where mosquitoes will attack them' reverses the cause and attacker.", True),
        ],
        "B": [
            f("Frankie responds that the mosquito story is true.",
              "'It's true,' affirmed Professor Scotch identifies the speaker.", True),
            f("Barney asks what the problem is.",
              "Barney asks 'Pwhat's thot?' and 'Kill a bear?' but no problem question appears."),
        ],
    },
    10: {
        "B": [f("A battle itself is depicted in the excerpt.",
                "'prepared to leave at once for the battlefield' and 'boat was loading' describe preparations, not battle.")],
    },
    11: {
        "A": [
            f("A person named Dr. Nostrums makes the argument.",
              "'Nostrums--Druggist's Arguments--Use of Proprietary Medicines...' is a chapter heading, not a named doctor.", True),
            f("The medical pretender claims dangerous diagnoses.",
              "'The Medical Pretender--Dangerous Diagnosis Graft' lists topics; no claim by that figure is shown."),
        ],
        "B": [f("The excerpt warns of the dangers of hygienic living.",
                "'Disease Prevention Rather than Cure--Hygienic Living' lists hygienic living as a topic, not a danger.")],
    },
    12: {
        "A": [
            f("Arne is a young man.",
              "'Arne was startled' gives no age or sex for Arne."),
            f("His mother and the clergyman's lady are sitting with him.",
              "'till your mother comes' and 'the Clergyman's lady will have finished' place them elsewhere."),
        ],
        "B": [
            f("Arne is the clergyman's daughter.",
              "'Arne was startled' and the separate mention of 'the Clergyman's lady' give no kinship.", True),
            f("The scene takes place in a church.",
              "'Let's sit down here' gives no church setting."),
            f("Arne is sitting with her mother.",
              "'till your mother comes' says the mother has not arrived."),
        ],
    },
    13: {
        "A": [
            f("The elders preside over the scene.",
              "'the youngest man plainly presided' says the opposite."),
            f("Gilly is not present.",
              "'Gilly's tired, honest eyes' and his spoken 'Impossible' place him in the scene.", True),
        ],
        "B": [f("The elders are the young man's parents.",
                "'his elders' states age relation, not parentage.")],
    },
    14: {
        "A": [
            f("The person sought is the menagerie's manager.",
              "'that little menagerie you boys found last night' never identifies a manager.", True),
            f("Sam and Carl both make the bet.",
              "'Then I'll bet he knows where the tunnel is!' is Carl's line alone."),
        ],
        "B": [
            f("The person sought is the menagerie's manager.",
              "'that little menagerie you boys found last night' never identifies a manager.", True),
            f("Sam and Carl both make the bet.",
              "'Then I'll bet he knows where the tunnel is!' is Carl's line alone."),
        ],
    },
    15: {
        "B": [f("A series of announcements and cheers occurs.",
                "'admonitory shouts' and Tracy's name 'almost' taking on a personal cheer do not establish announcements or actual cheers.")],
    },
    16: {
        "A": [
            f("The no-extrapolation rule appears in a code snippet.",
              "The excerpt presents it as a quoted section: 'No extrapolation. Whatever wins on the FFN...'."),
            f("qwen_export.py applies the no-extrapolation rule to model matrices.",
              "'qwen_export.py applies the chosen rule' refers to a conversion rule, not the warning against extrapolation."),
        ],
        "B": [f("The no-extrapolation rule appears in a code snippet.",
                "The excerpt presents it as a quoted section: 'No extrapolation. Whatever wins on the FFN...'.")],
    },
    17: {
        "A": [f("The rule is being applied incorrectly at the larger scale.",
                "'Running that same audit one scale up turns up the reverse' is clipped before any outcome or correctness claim.")],
    },
    18: {
        "A": [
            f("FFN means Fast Fourier Transform.", "The excerpt labels an FFN organ; no Fast Fourier Transform appears.", True),
            f("G-Z1 and G-Z2a are two FFN decompositions.",
              "'G-Z1 | the FFN decomposed' differs from 'G-Z2a | rel L2 of --mvacc 2 / 4 against --mvacc 1'."),
            f("The second decomposition has higher accuracy.",
              "The two rows report timer sum/FFN and relative L2, not a comparative accuracy."),
        ],
        "B": [
            f("FFN means Fast Fourier Transform.", "The excerpt labels an FFN organ; no Fast Fourier Transform appears.", True),
            f("The --mvacc options are two FFN decompositions.",
              "'G-Z2a | rel L2 of --mvacc 2 / 4 against --mvacc 1' is a matvec accumulation comparison."),
        ],
    },
    19: {
        "B": [f("Predicted 24.0 is 24.0 ms and equals 17.0 GB/s.",
                "'35.37 ms = 28.3 tok/s, against a predicted 24.0' gives 24.0 as token/s; '17.0 GB/s' is a separate FFN rate.", True)],
    },
    20: {
        "A": [f("The commit hash is a protocol ID.",
                "'completed from qualified producer commit 37c639...' identifies a commit, not the protocol ID.", True)],
        "B": [
            f("The protocol itself completed from the producer commit.",
              "'The sole scientific diagnostic authorized by the frozen protocol ... completed' names the diagnostic as completing."),
            f("The layer-1 expert authorizes the diagnostic.",
              "'authorized by the frozen protocol' identifies the authorizing source."),
        ],
    },
    21: {
        "A": [f("The process streams data from p0.bin to test_gen_p0.",
                "'30.46 GB fp32 model being streamed off disk' identifies model weights as the streamed data, not the prompt and output paths.", True)],
        "B": [f("The process runs on a private server.",
                "'from a different checkout (SiliconLLM_private)' describes a checkout, not a server.")],
    },
}


def main():
    assert hashlib.sha256(BLIND.read_bytes()).hexdigest() == BLIND_SHA
    blind = json.loads(BLIND.read_text(encoding="utf-8"))
    assert len(blind["rows"]) == 24
    rows = []
    for index, original in enumerate(blind["rows"]):
        row = {"source_id": original["source_id"]}
        for side in ("A", "B"):
            row[side] = {"unsupported": R.get(index, {}).get(side, []),
                         "missing_detail": False, "ambiguous": []}
        rows.append(row)
    result = {"experiment": "METH-74-arm-blind-semantic-verdict",
              "blind_sha256": BLIND_SHA,
              "reviewer": "single agent, excerpt-only review",
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"rows": len(rows),
                      "verdict_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
