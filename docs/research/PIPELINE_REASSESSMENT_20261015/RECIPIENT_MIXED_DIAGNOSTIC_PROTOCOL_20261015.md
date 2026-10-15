# Mixed-recipient numerical failure: first-divergence diagnostic

15 October 2026. Prospective diagnostic after the retained attempt frozen at
`4b78ac7`. This is not a timing retry or a change to the numerical gate.

## Observed failure

`results/native_expert_scaling/recipient_mixed_baseline_20261015` retains export,
compilation, probe, audit, owned process and terminal receipts. Every one of
291 packet roles passed independent derivation from the 290 BF16 tensors.
Position 0 matches (norm relative L2 3.79e-8, full logits 1.14e-9); position 7
fails (norm 0.08260, logits 0.09218, native top1 11 vs reference 271).
Positions 15/31 also exceed the frozen 1e-3 tolerance. No timing case ran.
The archive is 630039552 bytes, SHA256
`8bb77b499d76031a128cf8044889a7a772953e96dbe4c65b670e518ae95d00cb`.
The failed probe SHA256 is
`9dfdc44494987db9116608d70261af9e6790f8b5856a0214260077dbf8090d12`.

No definite codec, geometry or KV-index bug has been identified. A working
hypothesis is that floating-point primitive or reduction differences cross
dynamic I8 rounding boundaries and propagate through subsequent layers/KV.
This hypothesis must be distinguished from an actual implementation error.

## Instrumented replay and reference comparison

Keep the same archive, catalog, manifest and complete 32-token probe request.
Add C instrumentation behind `RC_DIAGNOSTIC`; the release forward remains
unchanged when that macro is absent. Only positions 0 through 7 produce a
binary trace, each record containing little-endian u32 position, layer, tag,
element count and that many F32 values. Context layer 24 denotes the head.
For projection organs 1=Q, 2=K, 3=V, 4=O, 5=gate, 6=up, 7=down and 8=head,
tags organ*10, +1 and +2 are activation codes widened to F32, activation
scales and projection output. Additional tags expose norms, rotary Q/K,
biased V, attention, FFN activation and residuals where available.

The independently implemented reference is the exact frozen auditor module.
A subclass records matching projection inputs/outputs without changing their
calculation. Find the first differing activation code, scale and output in
execution order and report the preceding differences. Do not infer a new
numerical admission from a diagnostic tolerance or a matching top1 alone.

Before interpreting the trace, require the instrumented QWCT probe to be
byte-identical to the original failed probe and reference complete-logit
hashes to reproduce the retained reference at all four saved positions.
If instrumentation changes either trajectory, retain and stop that diagnostic.
Original failure evidence stays immutable. Bind original inputs; the only
intentional original-input edit is the C trace guards, with the original C
recoverable at commit `4b78ac7`. Bind all diagnostic code and protocol too.

No source reconversion, fitting, model download, timing campaign, relaxed
numerical threshold or T4 work is authorized by this diagnostic. Run each
child at most 900 seconds and 8 GiB observed RSS, the whole diagnostic at
1800 seconds and 2 GiB outputs. Use exclusive logs/binding/receipts and owned
PID/create-time; terminate only the owned process on cap. No automatic retry.
The objective remains active because this inquiry is bounded below one hour.
