# Canonical Qwen plain-chat interaction qualification, before observation

7 October2026. Source Qwen/Qwen2.5-0.5B-Instruct revision7ae557604adf67be50417f59c2c2f167def9a775,
local tokenizer/template/generation metadata bound to retained preflight SHA.
No new LLM inference, tensor-weight reads, native calls, downloads or T4.
This is an implemented input/output contract stage for the first compact
CHATBOT converter; it cannot qualify the converter's capacity or rate.

Previous goal turn made progress: entire Qwen/Giga operator accounting and
existing archive mapping were executed and changed the transfer decision.
Current [NEXT](CHATBOT_CANONICAL_COMPACT_TRANSFER_NEXT_20261007.md) requires these
fixtures before fresh dialogue quality. Do not replay completed census/native
or original528 experiments. Actual joint compact converter remains missing.

## Exact scientific/operational question

Does the actual locally installed canonical HF tokenizer/template API preserve
the donor's plain-text role/history/assistant generation/continuation contract,
source tokenizer-JSON IDs, special tokens and BOTH151645/151643 generation EOS?
This question can be settled offline. Failure prevents new whole-chat quality
evaluation on ambiguously serialized inputs. A pass supplies a reusable adapter
and golden prompt-ID artifact for native evaluation, not new model quality.

## Frozen expected inputs and correctness decisions

`chatbot_interaction_goldens.json` contains EIGHT explicit text expectations,
written BEFORE observing tokenization: default system, explicit system/history,
multiple turns, generation header, assistant continuation with trailing space,
Unicode NFC/multiline, later system message and literal spaces/CRLF/sealed history.
Do not edit expectations using observed results after failure. Actual source
template is bound data, never task instructions. Scope: system/user/assistant
plain string content. Tool/function/multimodal/developer variants are not admitted.

`chatbot_interaction.py` uses actual HF `AutoTokenizer.from_pretrained` from
bound local files and `apply_chat_template`. No remote code trust. Compare:

1. exact canonical rendered UTF8 versus each prewritten literal text;
2. canonical chat IDs versus canonical encode(rendered,add_special_tokens=False);
3. original source Rust tokenizer JSON IDs;
4. an independent Python byte-level BPE witness using original vocabulary/merges,
   NFC, byte alphabet and frozen source regex/special-token flags.

The expected IDs are mathematically derived from source vocab/merge data, not
copied from observed HF IDs. Independent Python BPE differs from Rust/HF; both
are bound to the same source knowledge of token IDs. All8 exact-ID comparisons
must pass. Normalized decode must equal NFC of the rendered text; the renderer
itself must preserve literal input bytes, including decomposed Unicode/spacing.
Check source special IDs151643/151644/151645 and no automatic BOS insertion.
Reject empty conversations and simultaneous generation/continuation using the
canonical API. The production adapter also rejects unsupported message shapes.

FOUR prewritten stopping goldens test BOTH EOS, maximum length and exhausted
input. Only newly generated IDs enter stopping; prompt/history im_end markers
must not be interpreted as generation termination. Accepted generated IDs
include an emitted EOS; visible text skips specials. Future rate/quality runs
must use this SAME convention. Deterministic candidate policy: greedy,
max_new_tokens128,EOS[151645,151643]. Retain original sampled defaults separately
and do not claim greedy choices equal the source sampled distribution.

Every gate is fixed ALL-case equality. Stop first correctness/provenance/resource
fault; preserve partial results and actual exits before a numbered repair.
Success decision `PLAIN_QWEN_INTERACTION_QUALIFIED_PIPELINE_NOT_QUALIFIED`.
No dialogue response/semantic/prediction/native/rate or Giga parity follows.

## Runtime isolation and input sealing

Existing local Transformers5.13.1/tokenizers0.22.2/Jinja2 3.1.6/NumPy2.4.6 and
their installed dependencies. Run direct Python3.12.10 with -I -S -B and empty
external pycache prefix. A NEW runtime/site contains directory junctions and
one hardlink to selected installed packages/dist-info, excluding Torch, TF/JAX
and model extensions. This view changes import visibility without modifying
existing package files or installing/downloading anything. Require package
availability probes to show Torch/TF/JAX absent and no Torch import at end.

The administrative setup creates this new view and SHA binding BEFORE fixture
execution. Pin each runtime-root tree using sorted ordinal relative UTF8 path,
NUL,size,NUL,fileSHA,LF; .py/.pyd/.dll/.json/.pem and METADATA/WHEEL/RECORD,
excluding __pycache__. Pin direct executable/python DLL, launcher psutil,
worker/launcher/setup/protocol/goldens and every relevant local source metadata/
tokenizer/vocab/merges file. Source values in tokenizer vocabulary are read;
LLM tensor weights are not. Retain actual administrative setup cost separately.

The launcher verifies all pinned inputs/trees before and after the worker and
records used dependency hashes. Current template and local API version are
explicitly distinguished from donor config's historical Transformers version.
Source JSON parity is required, not assumed from the model or package name.

## Prospective resource budget and through-exit proof

Worker CPU10; parent monitor CPU11. Whole monitored family120s and512MiB sum
of OS peak working sets; each output/log<=2MiB. This confirms/updates the earlier
PROPOSED256MiB budget before observation: full canonical HF, two Rust tokenizer
paths and independent Python vocab/merge parser need explicit headroom. No
threshold/resource relaxation after seeing a result is authorized by this plan.
No overlapping science job, preserve exact publisher/three foreign SHA/old
namespaces and isolated empty cache. New worker is a direct base interpreter,
not a venv redirector; use DETACHED_PROCESS8 to avoid hidden console hosts.

Hold the Windows OS process handle, bind exact GetProcessTimes creation FILETIME,
sample memory while live AND AFTER actual worker exit. Charge final JSON
serialization in that worker's through-exit peak, unlike the previous census.
The parent checks its own peak plus worker peak after final input preservation;
its tiny terminal-report/stdout tail has an explicit last-snapshot scope.
Check actual worker/executor exits and typed UTC Windows Event1000/positive
controls separately. Unknown values remain unknown; no metadata replay to
retroactively qualify prior census resource gaps.

## Reproducible command, after source/binding freeze

Run `chatbot_interaction_setup.ps1` once to seal package view/inputs; commit
sources/protocol/goldens/binding. Then, with exact binding SHA and commit:

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 -X 'pycache_prefix=D:/_THINGS/Progetti/SiliconLLM/results/native_expert_scaling/meth511_runtime/empty_cache' 'benchmarks/native_expert_scaling/chatbot_interaction_launch.py' --binding 'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_interaction_binding_20261007.json' --binding-sha BINDING_SHA --freeze COMMIT --out 'docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_qwen_interaction_20261007.json'
```

After this stage, move to one finite complete compact Qwen converter trial,
using the earlier full cost ledger and actual retained source/fitting evidence.
Joint shared+conditional nonlinear SwiGLU replacement is UNTRAINED. Existing
native295/296/297, Giga310/311/316 and ReLU528 retain their original scopes and
failures. Final SAME-artifact fresh chatbot quality AND>=50, useful large n,
CPU LUT winner/mass, actual DRAM and additional families/scales remain missing.
