# Actual original-engine entry, persistent chat and complete raw decode cost

9 October 2026. ENGINE_CHAT_APPARATUS_PASS; goal ACTIVE/INCOMPLETE.
[Protocol](CHATBOT_HYBRID_ENGINE_PROBE_PROTOCOL_20261009.md), freeze
54011fcfde457918e61ed5dc27735f8b0f2906bf,
[30-input binding](chatbot_hybrid_engine_probe_binding_20261009.json)
SHA29519ab7545563fa8bd65e300e83e9268b5ba1d1e0441c473c76ec573c658cc3,
[actual result](chatbot_hybrid_engine_probe_result_20261009.json)
SHAd4c7b1079d2969a44c47efc82eaafa030307312932f39af36ad46e653aecccd2,
[through-exit receipt](chatbot_hybrid_engine_probe_result_20261009.terminal.json).

## What is now executable

The actual phase60/engine.c compilation selects SILICON_FALCON_TERNARY_CHAT.
Only its first dispatch branch changed; default E4 implementation and all11
original extracted matrix/LUT/AQ63 body hashes remain unchanged. The branch
includes the existing compact target operator implementation, packed-only loader
and new binary incremental chat interface. This is an actual executable path,
not evidence that it is the identical D256/L6/V1024 original model.

Target remains D512/L12/SSM10/SWA2/E72/k8/h128/V65537, F32 controls/readout,
signed SiLU, AQ63 ternary expert matrices with original LUT kernels. Actual
retained 8-update packed artifact191d1d20:425,210,736B/84,934,656B pair codes,
9,117,696B state; no master/unpacked expert reference banks. Checkpoint286 and
new common/private variant were not executed/exported by this probe.

The plain-role client uses source Rust tokenizer0.22.2 and original Jinja3.1.6
chat template with BOS17/BOTH EOS11/228, matching both consumed FIT packet IDs.
Torch/Transformers/donor weights/GPU are not loaded. Every emitted ID, including
the last, is consumed once and runs its complete next head. The unused final
head is charged. Generation replies return after the whole turn; token display
streaming and tool use are not implemented.

## Five actual requests and measured one-core rates

Ryzen53600X, CPU affinity0, batch1, one thread, no fast math, FTZ off.
One native process; normal EOS enabled without output filtering.

| Request | Prompt IDs | Reused IDs | Generated | Full heads | Prefill s | Decode s | Raw decode IDs/s | Raw request IDs/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full fit_history_00 |124|0|64|65|2.142446|1.391072|46.0077|18.1200|
| First half, no generation |62|0|0|1|0.544738|0|N/A|N/A|
| Append remaining half |124|62|64|65|0.663408|1.308917|48.8954|32.5038|
| Actual own-history followup |209|188|32|33|0.202240|0.687777|46.5267|35.9551|
| Changed instruction history |23|0|32|33|0.229869|0.543880|58.8365|40.9731|

All484 core advances/197 full V65537 head calls/192 emitted IDs are counted.
Five final next-logit rows are saved; intermediate197 head rows are not saved.
The two64-ID executions are the preregistered segmentation control, not two
independent quality samples. Every generated response hit its fixed length cap;
NONE emitted EOS. Both EOS contracts are checked but actual EOS stopping has not
been observed in these replies.

Full versus split prefill:64/64 emitted IDs and ALL65537 final next-logit F32
bits agree. The followup reused all188 consumed IDs and added21 prompt IDs.
Changing history reset state and reused0 IDs. Canonical decode/encode prefix
agreement was observed on this actual reply, not assumed for arbitrary text.
SWA window128 eviction is exercised by the longer trajectories.

Native aggregate counters:core4.503656s, MLP1.336686s, head1.816190s,
all forwards7.663331s. Counts differ by component because most prefill steps
omit readout; aggregate head fraction is not a steady decode cost share.
Model load0.252742s. First request including that load gives16.9100 raw IDs/s;
neither compile time nor a physical cold-cache flush is included in that ratio.
Raw request rates include prefill/pipe transfer, not a source-quality acceptance
filter. Retain all four rates rather than choosing the favorable short request.

## Quality, resources and decision

The old8-update outputs are degenerate/repetitive; these consumed cases do not
establish useful chatbot preservation. Existing19/32 strict native RMS failures
remain. Checkpoint286 has better saved prefix quality but still fails absolute
criteria; its own-history/native quality is not measured here. Accepted50,
physical DRAM, structured routing/mass and useful large n are not admitted.

Launcher25264/worker27436/compiler driver32228/native30872 all exit0; unified
tool session9664 closed. Family22.000s; worker20.500s; compile7.438s/native
process12.265s. Held worker peak84,344,832B; native439,779,328B through exit;
compiler driver4,788,224B. Compiler descendants are name-guarded, their individual
peak memories are not separately measured. 15 files/1,614,805B output, all fixed
resource/input gates pass. No fault/source/model/optimizer/GPU call, no T4.
Three foreign tracked hashes remain unchanged; process inventory is empty.

The target now has a real engine entry and a complete one-core decode envelope
around46-59 raw IDs/s in these contexts. This supports further conversion work
and exposes prefill/readout/state costs; it does not prove useful50 or n-independent
latency. Do not expand the shared-function variant or run month-scale training
before selecting broader data and controlled stages under the
[conversion roadmap](CHATBOT_ENGINE_CONVERSION_ROADMAP_20261009.md).

## Reuse commands

The compile command and all actual paths are in the result/receipt. The reusable
plain-chat client is `benchmarks/native_expert_scaling/chatbot_hybrid_engine_chat.py`:

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/chatbot_hybrid_engine_chat.py --engine results/native_expert_scaling/chatbot_hybrid_engine_probe_20261009/hybrid_engine.exe --model results/native_expert_scaling/chatbot_hybrid_export_20261008/model.bin --source results/native_expert_scaling/falcon_1p5b_source_repair1_20261008 --max-new 64
```

This launches the retained unqualified8-update artifact. `/exit` closes the
session. Larger/different packed target contracts need their own version.
