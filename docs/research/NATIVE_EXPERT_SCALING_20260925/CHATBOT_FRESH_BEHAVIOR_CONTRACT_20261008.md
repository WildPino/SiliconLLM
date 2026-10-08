# Fresh chatbot behavior: prewritten acceptance contract

8 October 2026. Freeze before the first whole-output learner/endpoint answer.
No endpoint model call, tokenization or answer has occurred at this contract.
Final quality and >=50 accepted batch1 IDs/s must use the SAME native artifact.
This contract does not replace the additional family/useful-n/physical-memory
requirements of the full goal.

## Reserved endpoint and excluded own-history dialogs

Use all64 cases in immutable `chatbot_global_endpoint_reserved_v1.json`:
8 each arithmetic/code/instructions/factual_text/translation/multilingual/
extraction/history. Some are partly templated; this is a finite limited-domain
assay, not broad benchmark generality. Max prompt256/context512/new64, source
BOTH EOS151645/151643, deterministic greedy complete vocabulary. Do not discard
length/invalid/unscorable cases. Use the source as a quality comparator, not
answers for another training pass or checkpoint selection.

Additionally use all16 prewritten two-turn dialogs in
`chatbot_whole_fresh_dialogs_v1.json`. Each model receives its **own first-turn
assistant text** in the second-turn prompt, separately for source/student/C.
Both turns scored; no source assistant answer replaces student history. Max
new64 per turn/context512. All32 turn outputs retained, including EOS/length
and invalid histories; prior messages remain original user facts.
No tools, retrieval or special tool templates in this plain-text scope.

## Prewritten rubric and counting

Score each task/turn as one pass/fail. Evaluate complete unedited visible text,
not cherry-picked tokens; a truncated incomplete answer fails its task. An
ambiguous/unscorable result is counted as a failure, with the reason retained.
Retain blinded paired texts and a per-case explanation to separate numerical
IDs/format checks from semantic judgments; do not infer agreement from strings
alone. Rubric is fixed before source/candidate outputs:

- Arithmetic: correct final integer from the prompt operations, consistent
  calculation/no contradictory numerical claim. Merely echoing input fails.
- Code: a valid Python function returning all matching zero-based indices,
  handles empty/no-match/duplicates and supplied tiny example correctly. Check
  AST/behavior in an isolated evaluator only after inspecting generated code;
  no unrestricted execution of arbitrary output. State any unsupported evaluator
  construct, count unresolved behavior as fail. Formatting prose is allowed.
- Instructions: exactly two numbered sentences, each about the requested folder
  preparation, with no extra numbered items or unrelated instruction.
- Factual text: answer supported by given text; correct requested fact/number/
  relationship, no incompatible or invented fact.
- Translation: preserve the complete supplied meaning in the requested language;
  no additions changing actors, quantities, negation or temporal relation.
- Multilingual: follow requested language and requested content/count constraints;
  intelligible and factually consistent with the supplied prompt.
- Extraction: requested fields/values exactly grounded in supplied record and
  required format; missing, contradictory or fabricated field fails.
- History/continuation: resolve relevant earlier messages correctly; continue the
  supplied assistant prefix coherently for continuation cases, without repeating
  the whole prefix or treating it as a new user instruction.
- New two-turn dialogs: all rubric facts in their authored reference must be
  preserved; correct second-turn answer, no conflicting invention. First-turn
  summary must retain requested entities/quantities/order while following its
  sentence/list constraint. References are independently authored prompt facts,
  not source-generated answers.

Track separately blank/invalid outputs, incoherent/repetitive text, unsupported
factual claims, wrong EOS/role/control-token behavior, context overflow and
failure to use own history. No unsupported-claim count inferred from perplexity.

## All required finite acceptance gates

1. All80 cases/96 turn outputs attempted for both models, canonical IDs/roles/
   BOTH EOS verified and exact artifact/config/model hashes retained. For native
   quality and rate, the recorded deployed artifact hash must match exactly.
2. On64 reserved task cases, student/native passes >=48 and fails no more than
   source+2 overall; each category loses at most1 pass versus source. A weak
   source cannot excuse an absolute failure of the48/64 floor.
3. On16 own-history dialogs, **both turns** pass in >=12 dialogs, and complete
   dialog failures exceed source by at most1. Track all32 turn failures too.
4. Unsupported factual claims, blank/invalid outputs and incoherent/repetitive
   outputs each no more than the source count+1 across96 outputs; no wrong
   role/EOS policy or context overflow. No exclusion of difficult cases.
5. Fresh source histories/tokenization/outputs and candidate answers excluded
   from fitting and selection. Exactly one fixed final learned candidate, no
   choosing an intermediate checkpoint by endpoint results. DEV metrics are
   consumed development and cannot satisfy these fresh behavioral gates.
6. On the same native artifact, >=50 accepted generated batch1 IDs/s end-to-end
   on the complete declared request mix, and no quality-invalid output counted
   as accepted useful prose. Include bothEOS/control filtering/canonical history,
   tokenizer/head/cache/router/normalizer/time and report cold/first request and
   context/thread/hardware conditions. Physical DRAM and useful large n/LUT mass
   need their own measured gate.100/s remains stretch.

Torch student behavior may be a preliminary candidate screen; it does not
admit native arithmetic or SAME-artifact quality/rate. Export rounding and C
operators require their own comparisons and fresh native actual observations.
Report uncertainty of the finite small sample and correlated templates; no
universal chatbot/family/large-donor claim from this contract alone.
