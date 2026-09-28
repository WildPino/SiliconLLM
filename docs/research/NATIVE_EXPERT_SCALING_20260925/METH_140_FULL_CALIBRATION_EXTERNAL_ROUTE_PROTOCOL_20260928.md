# METH-140: full training calibration, source-held-out route screen

METH-139's half-draw calibration preserved initial BF16 logits and
covered 7,299–10,897 grandchildren per layer, but missed one
candidate/control load-ratio gate (1.311 versus 1.25) and one
within-parent share gate (26.94% versus 25%). Those viewed validation
draws must not be a promotion test for a revised sidecar. METH-140
tests whether estimating the **same conditional-decile mechanism**
from all 256 training draws generalizes to separate source-held-out
documents. It does not change the METH-139 decision.

Bind the METH-126 exact BF16 bank, METH-135 projection sidecar,
METH-107 centered child checkpoint and METH-136 draw recipe by their
existing hashes. For each selected E1280 child `c`, route to one of
ten grandchildren using projection dimension `c mod 32` and nine
ordered linear empirical decile thresholds. Fit thresholds on all
256 METH-136 raw/chat draws; children with fewer than ten samples use
the layer-wide deciles for the same projection dimension. Thresholds
are float32 with search-to-the-right ties. Export a new versioned
sidecar and verify every byte after readback. Exact source-child ID,
four gates and clone-factor full-model BF16 logits must be preserved.

Use two existing 24-source manifests **only for routing load**:
METH-121 SHA-256
`7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366`
and METH-133 SHA-256
`ec6f839a2c9b1803dc8d5658aafa8fd9191b59fa2607961850f5a8666bc35674`.
Their sources/fragments were excluded from METH-136 training. They
were viewed in earlier **quality** experiments, so this is a routing
generalization screen and cannot support a new quality claim. Do not
fit or adjust thresholds using either manifest. For each document,
run its fixed token IDs in nonoverlapping windows of at most 512
tokens, resetting context at each window. Count the four selected
child/grandchild IDs at every routed position. Evaluate the two
manifests independently, not pooled to hide a failure.

On **each** manifest and in **every** layer require: candidate
maximum/mean slot load at most 1.25× the E1280 control's, at least
4,000 selected grandchildren, and no single grandchild receiving
more than 25% of a source child's selections when that child has
at least 250 selections. Require exact `grandchild_id // 10 ==
source_child_id` and exact cloned BF16 logits on the eight METH-121
parity prompts. Save per-layer top IDs, coverage, skew and worst
within-parent share. If any gate fails, do not promote the sidecar
to native cost or B training. If all pass, preregister a native C
cost test and only then a learned E12800 candidate against the matched
METH-136 continued E1280 control. External quality must use new
sources not among the routing-screen manifests.

Use only the local RTX 3060. Stop at 15 minutes, 10.5 GiB allocated
GPU memory, 20 GiB process RSS or 1 GB new disk. No T4.
