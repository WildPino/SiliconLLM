# Execution ordering: full hashes before exhaustive preflight

V2 table preparation passed every finite word/nonfinite fallback/FP guard
control. Keep its binding, receipt, adapter and all generated artifacts.
Before any paired timing, add a frozen execution entrypoint that hashes ALL
declared inputs before invoking V2's exhaustive preflight, then invokes the
unchanged paired/replay implementation and checks all pins again on exit.
The inherited harness's own before/after hashes remain; no waiver.

Separate execution binding appends only the new entrypoint/document and V2
successful preparation descriptors. No compilation, table, parameter, frame,
gate, FP, timing order or resource change. Old bindings/receipts preserved.
Use native_bf_activation_lut_execution.py prepare/run/audit/close, each with
exclusive outputs and own PID/create receipt. Per phase2700s/OS2GiB/E32MiB,
CPU affinity0, no GPU/source forward/RESERVED/delegation. The extra hash passes
are qualification overhead, never inference time or accepted-model TPS.
