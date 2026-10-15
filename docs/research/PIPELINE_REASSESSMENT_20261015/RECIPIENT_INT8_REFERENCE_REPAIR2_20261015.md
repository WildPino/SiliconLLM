# Recipient baseline: CPU rotary-buffer repair and sealed artifact adoption

15 October2026. The repair1 export and compilation completed, and the C
decoder produced all32 full vocabulary rows. The reference auditor then
exited1 in15.984s with `Cannot copy out of meta tensor; no data!`. No reference
logits or cost result were admitted. All failed receipts remain unchanged in
`results/native_expert_scaling/recipient_int8_baseline_repair1_20261015`.

The meta-initialized HF model's persistent decoded parameters were assigned,
but its nonpersistent RoPE frequency buffer was still meta. The repaired
auditor reconstructs that parameter-free module on CPU from the pinned source
configuration and verifies every model parameter/buffer is materialized on CPU
before execution. It changes no stored coefficient, quantizer or comparison
threshold and never uses original source weights as a reference fallback.

Prepare a new exclusive repair2 namespace. Reuse the already exported archive
and native executable, with explicit byte/hash adoption and original receipt
provenance. The exporter sidecar's full archive hash must agree. Check all
previous input pins; only the controller/adjudicator changes stated here may
differ. Both historical faults are preserved; no exporter or compiler replay.

A fresh native32-token probe must exactly match the retained repair1 probe
before the repaired CPU reference runs. This inexpensive replay binds the
currently sealed executable and archive to the adopted numerical output;
it changes no data or candidate. Three process repetitions of the original
fixed cost schedule can proceed only after full reference parity passes.

All source/runtime pins, owned child exits, source conversion, decoded-wire
gates, contexts,18ms allowance and1.10 stability requirement remain. The
separate stored readback reduces the retained timing/IDs/worker receipts; it
does not substitute for the reference model, a fresh donor-quality assessment
or a measured large-memory pipeline. No T4 training is launched.
