# METH-204: shared learned content keys with per-parent load bias

**Uncertainty.** METH-175's tenfold E1280→E12800 hash-route training
balances traffic but METH-179/201/202/203 show little useful
child-specific function. METH-184's fixed content-score/hash mixture
fails load, especially on repeated chat tuples. Test a new route
mechanism before allocating another E12800 B bank: learn nine
content-dependent keys per layer from the E1280 parent's real pre-MLP
states, with per-parent calibration biases. This screen is not
specialist training or a quality result.

**Inputs.** Bind the exact METH-126 centered E1280 BF16 bank, METH-135
rank-32 projection, METH-175 training draw manifest and structural
table, METH-150 24 source-separated documents, and donor/child
checkpoints through their recorded SHA-256s. Verify eight-prompt
donor/control BF16 logit parity before collecting states. Use the
METH-175 first 1,024 ordered raw/chat draws for fitting, draws
1,025–1,280 as a distinct replayed training-sequence check, and
METH-150 documents at 128- and 512-token windows for source/context
transfer. The METH-150 sources are already viewed and cannot promote
model quality. Keep E1280 parent IDs, gates, model weights and
structural shared routing fixed. Structural tokens follow the exact
METH-175 table/special-token path and are excluded from content gates.

**Fitting rule.** For each layer, project each content pre-MLP state
with the pinned METH-135 32×896 matrix and L2-normalize its 32-vector.
Use a seeded sample of at most 65,536 fit vectors (`204204+layer`)
and 16 iterations of spherical k-means with nine centroids. Initialize
from nine positions of a seeded permutation; replace an empty centroid
with its initialization vector. After fixing the nine unit centroids,
fit a 1,280×9 per-parent additive bias by 30 full-pass soft-count
updates at temperature 0.05: add `0.05*log((N_p/9+1)/(soft_count_pj+1))`,
then center each parent's nine biases. Hard inference chooses the
lowest-index argmax of cosine score plus that saved bias; it never
uses labels, source IDs or stochastic noise. Store the projection,
centroids and biases with an exact readback hash. Fit only on the
first 1,024 draws; freeze before the reserved and source screens.

**Gates.** Evaluate six cells: fit raw, fit chat, reserved raw,
reserved chat, source-separated 128 and source-separated 512. Require
every layer to meet: `9*max_grandchild_count/max_parent_count ≤1.25`,
no content child above 25% of a parent with ≥250 content selections,
≥4,000 distinct content grandchildren, mean selected standardized
cosine score minus the mean of all nine standardized scores ≥0.05,
and ≥15% agreement with un-biased score argmax. The source document
cell is scored as one set per window size, not split raw/chat. Report
all per-layer gates, traffic, bias magnitudes, structural counts, and
router storage/ideal addressed bytes. The screen passes only if all
six cells pass. If either fit cell fails, stop before reserved and
source; if either reserved cell fails, stop before source. A failure stops this
route before B training. A pass licenses a native CPU route-cost
probe, followed by a short matched B-training pilot and fresh quality.
Do not infer useful added experts from route health alone.

**Budget and stop.** Local RTX 3060, six host threads, ≤45 minutes,
≤20 GiB process RSS, ≤10.5 GiB GPU allocation, ≤1 GB new stored
artifact/result. Stop on binding/parity, nonfinite, shape, readback,
budget or heldout-gate failure. No T4. Retain a failure record and
partial cell metrics if stopped.
