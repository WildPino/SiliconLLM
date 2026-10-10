# Cached-state capture repair1: verify the flag after mixer initialization

10 October2026. Exact scientific protocol/gates/limits are inherited from
[original protocol](SOURCE_CACHED_FINAL_PROTOCOL_20261010.md). Original script,
binding/freeze/log/terminal and overlap rejection remain immutable.

## First fault and correction

Session42559 TERMINAL1; launcher25876/worker28136 gone. Held71.625s.
The original worker accessed code.is_fast_path_available before constructing
the model. Pinned modeling_falcon_h1.py defines that module global inside
FalconH1Mixer.__init__ (lines455ff), rather than at module import. Thus the
AttributeError occurs before any model instance, source history, generation,
head contraction, state capture, cache update or parameter check. No completed
observation exists to replay/adopt. No namespace directory/result was created;
worker log/terminal are the authoritative fault records. OS499,113,984B +
launcher29,802,496B was below the8GiB cap; this was not a resource failure.

New source_cached_final_capture_repair1.py moves ONLY that flag assertion to
immediately after model construction, matching the previously qualified source
capture tools. Original source operations/runtime/weights/tiled SSD installation,
hooks, actual parameter checking, four exact gates and600s/120s resource caps
remain unchanged. No setting the global by hand or bypassing the check.

Bind new script/protocol AND original protocol/fault terminal/log before use;
use NEW repair1 binding/namespace/result. Full consumed old code stays intact.
No source observation has yet occurred. Failure is a startup-mechanism defect,
not evidence about compact capacity or paired hidden-state alignment.
