# METH-219: native FFN emulator gate stops before quality

The [frozen protocol](METH_219_NATIVE_ARITHMETIC_PROTOCOL_20261001.md)
and runner were committed at `8c8efef`; `c74b758` corrects only the
prior result filename after a binding failure before export/inference.
The second invocation exports and audits all 144 METH-211 FFN code/scale
segments (323,592,216 bytes), then compares the existing METH-182 C
executable to the proposed FP32 GPU W8A8 emulator on all 384 existing
METH-125 token/layer states.

Median relative L2 is **4.095281e-7**, but the worst is **0.006069825**,
above the frozen <=0.0005 worst-case limit. The emulator gate fails;
the run stops before any of the five full-model quality arms, prompt
shortlists, generation, tasks or fresh source consumption. No W8A32 or
W8A8 full-model conclusion follows. Do not relax the parity tolerance or
infer native quality from the rejected emulator.

[Component record](meth219_native_arithmetic_development_component.json)
contains the exact binary, segment, executable, source and native-output
hashes and timing, SHA256 `b00676aa0b8c162386c3f9e38d9acd6a8f4afddd520bd947b2a2d52e5225e5f1`.
The FFN binary hash `d8f578ab42b99258339c2cfe0ee846e01537e5f4f49cae68c9bcdba7112b5613`
and native output hash match METH-182 exactly: these weights and this
kernel are not a new cost candidate. METH-182 already rejects its
<=10 ms/token feasibility gate. [Terminal failure](meth219_native_arithmetic_development_component.failure.json)
records the assertion after 34.734 seconds (imports excluded). Six-thread
native passes take 13.388 / 13.244 / 12.081 ms per 24-layer FFN token;
these are component timings on old states, not accepted end-to-end rate.
No T4 was used. Local component files are retained for diagnosis.

Next action: freeze an intermediate-state C/emulator comparison on the
same saved FFN bytes and states. Test whether small FP accumulation
differences cross hidden int8 rounding boundaries; this is a hypothesis,
not a diagnosed cause. The exact C outputs must first be reproducible.
METH-214 BF16 quality remains failed, METH-218 routing is not promoted,
and useful large-n capacity plus full native quality/rate remain open.
