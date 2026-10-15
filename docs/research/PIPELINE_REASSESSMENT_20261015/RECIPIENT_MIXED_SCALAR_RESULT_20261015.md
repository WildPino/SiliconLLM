# Complete mixed recipient: exact decoded parity, failed cost envelope

15 October 2026. One continuation frozen at `ea96e58`, following the
[scalar-reference repair protocol](RECIPIENT_MIXED_SCALAR_REPAIR_PROTOCOL_20261015.md).
**Decoded numerical PASS / complete cost envelope FAIL.** The original
[failed reference](RECIPIENT_MIXED_BASELINE_RESULT_20261015.md) and its
[first-divergence diagnostic](RECIPIENT_MIXED_DIAGNOSTIC_RESULT_20261015.md)
remain retained. This is not BF16-donor quality or useful large-model admission.

## Qualified artifact and actual work

Reuse the exact original executable and source-derived packet, with all 24
Qwen Instruct layers, full FFN, causal attention/KV and 151936-row head.
No native operation, packet byte or gate changed. The new reference computes
the exponential using scalar binary64 math rounded to F32, after a stored
check reproduced all 933888 native SiLU values at 192 sites bit for bit.

All 291 packet roles again pass independent derivation from 290 BF16 source
tensors. The new probe is byte-identical to the original native probe. At
positions 0,7,15,31, every normalized hidden-state and full-vocabulary logit
value matches the new explicit reference **exactly** (relative L2 and maximum
absolute difference both zero); all four top1 values agree. The reference
processes the complete 32-token prefix and all activation quantizers.

Packet: 630039552 bytes, SHA256
`8bb77b499d76031a128cf8044889a7a772953e96dbe4c65b670e518ae95d00cb`.
Executable SHA256:
`df497c03349f752c582e95b0f4d61590c65e26625db276d9d6094c86b1614d42`.
The original C Git blob at `4b78ac7` binds the executable; later trace guards
are not part of its compiled release path. Retained continuation directory:
`results/native_expert_scaling/recipient_mixed_scalar_20261015`.

## All 15 cost processes

Six physical workers, unchanged ACTIVE/infinite waiting, three fresh processes
per case. All exited zero. Each case emits and actually forwards 64 own-history
tokens; total 960 charged forwards and 960 emitted IDs. IDs repeat exactly
within each case. No early EOS occurs. This is not a semantic or looping gate.

| Case | Three native forwards/s | Median | Worst ms/forward | Rate spread |
| --- | --- | ---: | ---: | ---: |
| English explanation | 40.896 /45.607 /33.380 | 40.896 | 29.958 | 1.3663 |
| Italian planning | 50.175 /58.507 /38.112 | 50.175 | 26.238 | 1.5351 |
| Python coding | 35.185 /47.627 /42.008 | 42.008 | 28.421 | 1.3536 |
| Context 512, cost only | 41.040 /36.958 /48.535 | 41.040 | 27.058 | 1.3133 |
| Context 2048, cost only | 37.552 /22.970 /35.729 | 35.729 | 43.534 | 1.6348 |

Every case group fails both frozen criteria: worst process mean <=18 ms and
max/min rate spread <=1.10. Some individual processes exceed 50 forwards/s;
they do not qualify the envelope. Per-step/native-forward accounting includes
the final output token, and excludes invented forwards for EOS. Load, packet
verification and prefill remain separate. These are raw native forward rates,
not useful accepted chatbot rates or final end-to-end throughput admission.

The continuation took 251.469 s after initial input verification, with
4059166 output bytes before result serialization. Largest observed child RSS
was 4146196480 bytes. Numerical audit child took 32.219 s, its internal work
28.890 s. No download, re-export, recompile, training or T4 allocation occurred.

## Decision and limits

Close this unchanged full-width W4/W8 recipient encoding/kernel envelope.
No thread/wait grid or months-long recovery is justified for it. Compared
with the [all-I8 baseline](RECIPIENT_INT8_BASELINE_RESULT_20261015.md), the
analytical matrix payload drops from 495785472 to 357484032 bytes per forward,
but this complete runtime does not become fast enough. Formats, activation
quantization, integer accumulation, grouped decode, attention and scheduling
all contribute; the test does not isolate one cause or establish a universal
CPU lower bound. Logical payload is not PMU-measured DRAM.

The selected donor-memory program therefore needs a substantially cheaper
online core before its long joint recovery. The analytical next cost candidate
is the pretrained 135M core from [triage](RECIPIENT_CORE_TRIAGE_20261015.md),
only as an online component for a very-large-donor transfer falsifier. Its
weak initial competence is an explicit risk, not a new small-model goal.
Pin/acquire its source and qualify a complete C artifact before bank training.
General usefulness, donor-gap recovery, no-hit behavior, shuffled/removed banks
and increasing distinct capacity at matched active work must still be tested.
Months of T4 usage remain acceptable; no long run is currently waiting.

Result SHA256:
`1cabb4d329af9a62f4b48e2734f1cfbdfcc868b7760e47a39a96a891655ea42c`.
Audit SHA256:
`aee03b981e47a5337f1645eba679d44a5b8628f2781b02961bebbd89bbfea2ff`.
The [independent retained readback](recipient_mixed_scalar_readback_20261015.json)
passes, reconciling all 17 commands, binary IDs, step times, request/input
hashes, numerical gates and cost arithmetic without importing producer parsers
or executing a model. Every owned command identity is absent. See the
[closure](recipient_mixed_scalar_closure_20261015.json),
[full result](recipient_mixed_scalar_result_20261015.json) and
[numerical audit](recipient_mixed_scalar_audit_20261015.json).
Closure SHA256:
`a6d3c60053d160f34331df92e78e141475f78ce1d8f1de1280e9f7e5939081fd`.
The overall goal remains active/incomplete.
