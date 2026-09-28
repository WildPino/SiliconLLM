# METH-139: traffic-calibrated ten-way third-tier routing screen

METH-136's seeded third-tier argmax selected 7,996–11,488 slots per
layer during training, but failed the frozen maximum/mean load limit.
METH-138 final-state replay found 287.349× worst grandchild load versus
56.417× for its continued E1280 control; one hot child sent 50.8% of
its traffic to one grandchild. This experiment tests a new **routing
mechanism**, not a relaxed verdict on METH-136. The original E1280
router, gates, shared A and centered BF16 B bank stay fixed. No B is
trained in this screen.

Bind the METH-126 exact BF16 bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`
and METH-135 projection/keys sidecar SHA-256
`692db3dd1a647debb051ef66612263b5e16ef27c73978e9abd3beb6b3229abc9`.
Use METH-136's deterministic 256 raw/chat draws, seed 136136, and the
unchanged centered E1280 teacher. The new third tier reuses METH-135's
32-dimensional projection but does not use its argmax keys. For a
selected E1280 child `c`, take projected dimension `c mod 32` as a
scalar and choose one of ten grandchildren by comparing it with nine
ordered thresholds specific to that child. The source E1280 child and
its four gate weights remain exactly the same. At initialization all
ten grandchildren copy that child's BF16 B, so the full-model function
must be unchanged.

Collect scalar/child pairs on updates 1–128 only. For each layer and
child with at least ten calibration selections, set thresholds to
linear empirical deciles (10%, ..., 90%) of its scalar samples. For
fewer than ten, use the corresponding layer-wide deciles for its
`c mod 32` projection dimension; preserve deterministic tie handling
with search-to-the-right. Serialize the 24 projection tensors and all
thresholds in a versioned binary sidecar, verify full readback and
SHA-256. Freeze it before scoring updates 129–256.

On the untouched 129–256 routing inputs, compare candidate and E1280
control load histograms per layer. Require: (1) every selected
grandchild ID divided by ten equals its original child ID, (2) all
thresholds finite and nondecreasing, (3) candidate maximum/mean load
at most 1.25 times the control's maximum/mean in **every** layer,
(4) at least 5,000 selected candidate slots per layer, and (5) among
source children with at least 500 validation selections, no one
grandchild takes more than 25% of its parent's selections. Save every
layer's coverage, skew, worst within-parent share and top IDs. These
are load-routing gates only; they do not establish learned useful
experts, held-out quality, native C cost, a quality-valid LUT, or
full-model accepted-token rate.

If this screen passes, preregister native C route cost and a new
E12800 B-only training rung. A relative skew gate is used because the
unchanged E1280 control itself exceeds 50× in two layers; the new
third tier must avoid adding substantial concentration rather than
claiming to repair those first-two-tier hotspots. Reuse of the
calibration inputs for later B training is permitted and must be
disclosed. External quality data must exclude them and previous
audit sources/fragments.

Run on the local RTX 3060 only. Stop at 15 minutes, 10.5 GiB peak
allocated GPU memory, 20 GiB process RSS or 1 GB new disk. No T4.
