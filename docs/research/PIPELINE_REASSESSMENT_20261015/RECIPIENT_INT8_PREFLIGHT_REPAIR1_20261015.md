# Recipient baseline: isolated import repair

15 October2026. The first frozen launch at5b52aab stopped in3.641s with
`ModuleNotFoundError: synth_export`. The isolated runpy bootstrap had inserted
the selected dependency view but not the exporter's own directory. Ordinary
script execution supplies that directory; runpy under `-I -S` did not.

The exporter failed during module imports, before model construction, source
weight conversion or any compilation, numerical comparison or timing. The
original binding, command/owned-process/terminal receipts, logs and failure
remain in `results/native_expert_scaling/recipient_int8_baseline_20261015`.
The terminal records exit1; the controller also exited1. No result was admitted.

The repair adds the explicit exporter parent directory to that bootstrap.
It changes no source, quantizer, packet format, C arithmetic, prompts, gates,
resources or schedule. Freeze the repair and prepare a new exclusive namespace
`recipient_int8_baseline_repair1_20261015`. Do not rerun the original directory
or replace the first fault. No numerical trial is repeated by this correction.
