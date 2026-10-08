# Prospective whole-transfer cohort and FIRST wire/ID audit

8 October2026. Freeze code/manifests/first auditor before NEW teacher queries.
Goal ACTIVE: complete pretrained CHATBOT -> compact conditional C -> fresh
dialogue/own-history/tasks AND accepted>=50 SAME artifact/useful n/RAM/LUT
winner+mass/physical DRAM/actual other families/~10B/~100B. This stage supplies
data for ALL24 initialization and global output loss; it is not a compact model.

## Uncertainty, prior reuse and decision

The [actual ALL24 initializer](CHATBOT_WHOLE_INITIALIZER_RESULT_20261008.md)
stopped before layer1 source features:parent3 FIT32/9 cases,novel DEV3<4.
The original48-case complete recipe remains CLOSED. A NEW transfer cohort
provides broader states and complete final donor distributions for the whole
learner. This directly serves initialization plus final-output supervision;
no additional local response-coefficient experiment or old-case capture.
Success/FIRST byte-ID adoption permits new-cohort support assessment and actual
complete assembly. It does not demonstrate support at EVERY layer, knowledge
transfer, generative quality, convenient physical engine cost or useful large n.

Reuse pinned Qwen2.5-0.5B-Instruct snapshot7ae557604adf67be50417f59c2c2f167def9a775,
actual model BF16 archive SHA fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe,
source config/tokenizer/gen-config/canonical API/BOTH EOS and known restricted
runtime. No source old48 case/fixture, compact field, J/H, old source features,
optimizer or reserved endpoint query. Original source load/forward mechanics
are reused; no old donor-adaptation learner resumes.

## Exact manifests and limited scope

chatbot_global_cohort_cases.py fixes eight categories:arithmetic,code,
instructions,factual_text,translation,multilingual,extraction,history.
Each has20 FIT +5 development cases:160/40,total200. Concrete texts/numbers/
mode/role/history are fixed in chatbot_global_transfer_cases_v1.json before
source queries, not selected using routing outcomes. Some prompts are templated;
do not call this broad knowledge coverage or an IID sampled benchmark.
History includes multi-turn and assistant continuation. Other categories use
plain user messages with source's canonical default system.

chatbot_global_endpoint_reserved_v1.json fixes64 separate concrete endpoint
messages (eight/category); no encoding/forward/answers on these in this stage.
Exact message+mode disjointness from original48 and transfer manifest is checked
before launch and independently on saved data. Domains intentionally overlap;
exact disjointness is not a domain-generalization or semantic leakage proof.
Development is calibration/selection data, never a final fresh-quality sample.
Endpoint protocols/criteria and any task truth need fixing before later queries;
these64 texts alone are not an evaluation or an exhaustive goal gate.

## One-pass original source and supervision positions

Tokenize ALL200 NEW transfer cases BEFORE loading the model, using qualified
serialize()/local HF tokenizer, add_special_tokens=False. Fixed prompt<=128,
context<=256, max_new_tokens16, EOS151645 AND151643. Save rendered UTF8/exact
U32LE prompt IDs and hashes. A token-cap/input fault stops before source queries;
no text modification after observing teacher outcomes.

Load original BF16/eager Qwen once; freeze all parameters/check original MLP
class/zero FFN biases/tied head/source config. Register ALL24 pre-MLP hooks that
save x only. First call feeds the full prompt, subsequent calls only the previous
new ID with original cache. logits_to_keep=1 returns full vocabulary BF16 logits
at the last evolving input position. Do not silently round F32 logits or retain
only topK. Greedy first-ID argmax, generated-only BOTH EOS, maximum16 decisions.
No source-query repetition on old contexts, no source FFN y/J/H targets.

For decision step s, teacher label corresponds to complete source prefix
prompt+generated[:s], history position len(prompt)+s-1. Keep original x for
every prompt/new input row in ALL24 layers. Teacher labels supervise generated
decision positions only, not every prompt token. Future whole learner must
use matching contexts/masks; teacher-forced KL cannot prove own-history quality.

## Wires, preservation and resource envelope

Flat per-case x BF16 file:24-byte <8s4I header QWGX0001,24,896,2,0;
payload C-order [input rows,24,896],little U16 BF16 words,43008B/row.
Flat per-case logits BF16 file:24-byte QWGL0001,151936,2,0,0;
one303872B full-vocabulary vector per decision. Forward journal records exact
input IDs/next ID/history position/x and label offsets/bytes/payload hashes.
Append+flush after each complete forward; fsync after each conversation;
per-case metadata contains complete frames/role text/generated stop/payload
hashes. Main result contains compact receipts, avoiding a huge duplicated frame
report. On any first fault retain complete cases, journaled prefixes and pending
state; no completed query replay or retrospective whole producer promotion.

Tensor upper bound:200*(128+15)=28600 x rows ->1230028800B; <=3200 logits ->
972390400B; total2202419200B. Output directory cap3GiB including all headers/
metadata/journals/tokenized inputs. Per-case metadata<=64KiB/main report<=2MiB.
Collector worker900s/family1500s/per conversation60s,OS family16GiB,
Torch GPU allocated8GiB/reserved9GiB,log4MiB. CPU numerical6/OpenBLAS1,
worker affinity0..10/launcher11; local RTX306012GB only, no T4/new resources.
Isolated Python3.12.10 -I -S -B/UTF8/empty cache, known safe tree-hashed roots,
Torch2.6.0+cu124/Transformers5.13.1/Tokenizers0.22.2/NumPy2.4.6; Arrow/datasets
absent, subprocess forbidden, offline HF, TF32 disabled/deterministic algorithms,
CUBLAS workspace :4096:8. Preserve foreign work/exact publisher exception.

New dedicated launcher retains generic held Windows worker handle/FILETIME/
actual exit/OS peak through exit/conservative family and beforeafter inputs,
runtime and foreign seals. Its job explicitly permits1..3200 NEW original
full forwards for collection, zero for audit; the old directional zero-forward
contract is unchanged. Worker GPU maxima precede final CPU serialization;
launcher final receipt/stdout tail remain outside last memory snapshot.

## FIRST independent adoption fixed before source observations

Auditor CPU-only NumPy/regex/psutil, no Torch/HF/model. Independently reconstruct
ALL200 NEW rendered strings from previously qualified plain source grammar,
and ALL prompt IDs using source-JSON IndependentBPE; no old fixture execution.
Reserved64/old48 are read for exact text exclusion only, never encoded/queried.

For ALL complete saved conversations, independently verify metadata/case order/
role text/IDs, journal identity/forward sequence/byte offsets/typed headers,
finite ALL24 x and full151936 labels, exact x row totals, prompt+previous new ID
input relation and label history positions. Decode BF16 words to exact F32
representatives, NumPy first argmax for ALL labels must equal saved next IDs;
independent BOTH EOS/no subsequent input/16 bound and exact EOF. This verifies
retained wire/ID/winner consistency and provenance, not a second independent
source-model execution or source logit arithmetic fidelity.

Complete200 successful producer requires full output/resource/procedure receipts.
If producer fails, the audit may adopt only its complete-case prefix with an
explicit INCOMPLETE cohort/producer admission FALSE; pending conversation is
excluded, its data retained. No full cohort/support/quality promotion follows.
Do not restart because a tool observation times out; check the live handle.

Audit worker300s/family600s/OS4GiB/output1MiB/log4MiB, same isolated source-
bound environment, only seven NumPy/regex/psutil runtime roots. Its binding
includes actual producer raw/terminal/typed outputs plus source/case manifests.
Faults remain before any numbered unchanged-science repair; never rerun a
completed auditor namespace. Administrative typed UTC/FILETIME/Event1000 exit
closure follows both processes without numerical/scientific replay.

## Next actual conversion stages

Reuse old0/1/12 routing and0/12 source-derived warm initial state BYTE. Assess
NEW cohort per-layer support under frozen >=8 distinct FIT/>=2 FIT cases/>=4
novel DEV, retaining maps/mass. NEW geometry where absent and NEW source-row
selection where uninitialized require a separate frozen stage/FIRST audit;
no old geometry/features/copies replay. If support still fails, consider a new
balanced/nonuniform representation, not threshold lowering or targeted sampling.
Then actual ALL24 original BF16 core installation/resident optimizer peak,
ONE finite whole-output fit, export/new C operator/catalog/canonical chat/BOTH
EOS, fresh own-history/tasks AND accepted50 SAME artifact/useful n/DRAM/scales.
All these remain unachieved; source data alone cannot substitute for the goal.
