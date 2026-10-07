# Qwen canonical chatbot input stage: qualified

7 October 2026. The compact pretrained-chatbot pipeline remains INCOMPLETE.
Source Qwen2.5-0.5B-Instruct revision7ae557604adf67be50417f59c2c2f167def9a775.
The [prospective protocol](CHATBOT_INTERACTION_PROTOCOL_20261007.md), eight literal
text goldens and four stop goldens were frozen BEFORE any token observations.
Successful worker/launcher/binding repair3 freeze:
`d92f9c141e4b3297bdc1568c9bc6cb37906ba382`.

## Actual result and reusable stage

[Raw result](chatbot_qwen_interaction_repair3_20261007.json), SHA256
`f82a7e88689d06f4934799aae76485402168a7137ecd865d63703bef33994644`;
[through-exit witness](chatbot_qwen_interaction_repair3_20261007.terminal.json), SHA256
`66bb8bfe0d7916a212e5c0fe340331970c5bd595e4dabb31001876b15f41ed77`;
[worker log](chatbot_qwen_interaction_repair3_20261007.worker.log), SHA256
`9f2cc405ad40d7e10841258b46df5abf0ae19c8c9f8e56a0721d8fd41d57c9b4`.
Executor initial3ef9b8/session91348, final018b6c: **exit0**. All four worker
correctness gates and all five launcher binding/resource gates PASS.
Decision: `PLAIN_QWEN_INTERACTION_QUALIFIED_PIPELINE_NOT_QUALIFIED`.

| Frozen conversation | Prompt IDs |
| --- | ---: |
| Default system, assistant generation |30|
| Explicit system, sealed history |18|
| Multiple turns, assistant generation |27|
| Assistant continuation, trailing space |35|
| Unicode NFC and multiline |33|
| Later system message |34|
| Literal spaces/CRLF, no generation |30|
| Default system, sealed assistant |33|

Every rendered string equals its prewritten literal golden. Every ID vector
equals all four paths: canonical HF chat API, HF encode of rendered text,
original Rust tokenizer JSON, and independently implemented Python byte-level
BPE using source vocabulary/merges/normalizer/regex. The Python witness derives
IDs from source data; expectations were not copied from observed HF output.
All decoded strings equal NFC of the rendered string, including the decomposed
Unicode fixture. Rendering itself preserves literal spacing/Unicode bytes.
Special IDs151643/151644/151645 match source; no automatic BOS is inserted.
Canonical empty conversation and simultaneous generation/continuation reject.

All four stops PASS: both source EOS151645 and151643, maximum length and
exhausted generated input. Only NEW generated IDs enter stopping; history
`im_end` markers are not generation stops. Emitted EOS counts as an accepted
ID; displayed text skips special tokens. Declared comparison policy is greedy,
max_new_tokens128; original sampled generation defaults remain separately stored.

Reusable [adapter and independent BPE](../../../benchmarks/native_expert_scaling/chatbot_interaction.py)
and [prewritten goldens](../../../benchmarks/native_expert_scaling/chatbot_interaction_goldens.json)
now supply canonical plain system/user/assistant serialization, stopping and
actual golden prompt IDs. Transformers5.13.1/tokenizers0.22.2/Jinja3.1.6 are the
actual pinned local runtime, distinct from historical Transformers4.43.1 in
the donor config. No Torch/TF/JAX/model import, tensor-weight reads, LLM/native
calls, downloads or T4. Tool/function/multimodal/developer-role contracts and
native adapter integration are NOT qualified here.

## Actual resources and administrative closure

Successful launcher19412/create1791379237.6558712, CPU11;
worker22008/create1791379241.1510732, creationFILETIME134358528411510733, CPU10.
Actual worker exit0. Held Windows OS process handle supplies peak working set
**256540672B through actual exit**, sampled AFTER exit. Launcher last peak
snapshot30904320B; conservative sum287444992B. Whole monitored launcher elapsed
**14.093s**, against prospectively frozen120s/512MiB. Source/tokenizer/runtime
trees, empty isolated cache and three foreign tracked SHA remain unchanged.
Launcher final tiny report/stdout tail is outside its last peak snapshot;
the worker's final JSON serialization IS included in its through-exit peak.

[Typed Windows closure](chatbot_interaction_windows_terminal_20261007.json)
uses seven known PID+creation instances from all four launches, typed UTC
intervals and Event1000 positive controls179810/179791. All seven instances
closed, all queries available, zero relevant OS faults; administrative
executor1bfd2f exit0. This does not erase the failed Python assertions below.

Runtime junction/hardlink setup: actual session97514, final093b7f exit0,
58.944449s administrative PowerShell time. Its OS peak was not retained and
is not charged as part of the separately bounded scientific launch. No package
installation or modification; existing source packages are exposed through
a new restricted import view. Setup must not be rerun in the existing namespace.

## Retained first faults and repairs

1. Metadata inspection using PowerShell ConvertFrom-Json without -AsHashtable
   cannot read case-distinct BPE vocabulary keys. Its printed null fields were
   INVALID; the corrected metadata read establishes actual NFC/ByteLevel config
   before fixture freeze, without tokenization.
2. Original launcher174284 exit1, no worker/tokenization: Get-Item.Length is0
   for HF snapshot symlinks. [First launcher fault](chatbot_qwen_interaction_20261007.launcher_failure.json)
   is preserved. [Repair1](CHATBOT_INTERACTION_REPAIR1_20261007.md) binds physical
   OpenRead lengths, keeping source content SHA unchanged; metadata correction
   e0e168 exit0. Original freeze7fb1a38, corrected freeze102d3c3.
3. Repair1 actual worker20168/launcher22052, final3e2d6b exit1: first literal
   render PASS; comparison wrongly uses canonical BatchEncoding against lists.
   [Worker fault](chatbot_qwen_interaction_repair1_20261007.failure.json),
   [launcher fault](chatbot_qwen_interaction_repair1_20261007.launcher_failure.json)
   and log retained; first ID arrays were lost. Through-exit worker peak253243392B.
4. [Repair2](CHATBOT_INTERACTION_REPAIR2_20261007.md), freezefe698052,
   preserves pending vectors BEFORE assertion. Actual executora428a1 exit1,
   launcher30228/worker2712: failed first case computed again; diagnostic JSON
   also fails because BatchEncoding is not serializable. No worker failure JSON
   exists; [launcher trace](chatbot_qwen_interaction_repair2_20261007.launcher_failure.json)
   and log retain actual exit1 and through-exit peak253370368B.
5. [Repair3](CHATBOT_INTERACTION_REPAIR3_20261007.md) corrects ONLY the API
   container, explicitly return_dict=False/list of Python ints. Bound local
   primary implementation shows return_dict=True became the default. Previous
   "token-ID mismatch" assertions are INVALID evidence of actual differing
   IDs; lost arrays remain unknown. No source/tokenizer/golden/threshold change.
   The failed first case is computed a third time, then the remaining seven
   cases and four stop cases execute for the first time and all PASS.

Failed-prefix repeats are disclosed: first case three executions and remaining
seven once =10 canonical render/fixture executions,40 four-path encoding
observations, plus final special-ID/negative probes. No completed fixture,
source-response/native/model science was replayed. Preserve all bindings and
fault namespaces; do not re-execute the successful stage for another audit.

## Decision toward the full pipeline

The canonical conversation stage is AVAILABLE and qualified in its declared
plain-text scope. Proceed to a finite complete compact Qwen weight conversion,
using the already executed [complete cost ledger](CHATBOT_OPERATOR_CENSUS_RESULT_20261007.md).
Old METH125 supplies only256 positions per layer on augmented-model trajectories;
it is not broad original-donor training support. METH231's source-derivative
fixed nonlinear prior and METH214's saved compact ranking recipe remain closed.
No inference, compact capacity, whole chatbot quality+accepted50, useful larger
n, physical DRAM or second-family/~100B evidence is produced by this input stage.
