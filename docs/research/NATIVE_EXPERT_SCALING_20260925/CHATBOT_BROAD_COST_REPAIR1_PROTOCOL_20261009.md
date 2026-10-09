# Broad learner cost repair1: use actual gradient modules

9 October2026. Original cost preflight FAILED before reaching the long case.
Freeze d026658a34bf82d16bf54cd2ebcd966e9481174e,binding
79c8d3a71b0cb45b5089a59cbeb444be35e27eccb8bee6e90c95a02eff6f0bde.
Worker8016/launcher26184 exit1/session95297 closed. FirstfaultAssertionError()
at21.063s while inspecting short FIT58 gradients:the name-prefix collector
expected norm parameters starting with'norm';actual modules are input_norm
and ff_norm. A short forward/backward WAS computed,global211 finite-gradient
assertion passed before the bad group check,but no durable per-case timing/
loss/gradient record was written. These missing observations remain unavailable;
do not infer timings or claim completion. Long case/model-unchanged comparison
were not reached. Optimizer updates/source/native calls0.

Actual family28.703s/held OS5,535,723,520B;worker GPU allocated4,412,008,448B,
reserved4,766,826,496B,within original caps. This is an apparatus fault,not an
observed long-case memory failure. firstfault/workspace extents/binding/log/
launcher receipt/original worker byte archive are retained.

Repair uses explicit layer.core/layer.banks/layer.input_norm+layer.ff_norm
parameter groups,matching the already retained original recovery collector.
Persist forward/backward cost/loss BEFORE inspection assertions so another
reporting fault cannot lose completed metrics. Model/data/selection/RNG/loss/
QAT/gradient/state criteria and900s/16GiB/11GiB allocated/12GiB reserved caps
are unchanged. New namespace/binding/freeze;bind all original fault bytes.

Re-executing the short training forward/backward is explicitly necessary because
its process/tensors are gone and no usable cost/gradient record exists;it is
not a new independent sample. No old286 update,old source generation,old160
baseline or completed cost case is replayed. Report original partial work
separately;repair observes two cases with0 optimizer steps. Firstfault stops
remain;no cap/tolerance relaxation. See [original protocol](CHATBOT_BROAD_COST_PROTOCOL_20261009.md).
