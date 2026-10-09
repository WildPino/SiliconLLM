# Source SSD temporary-storage qualification

9 October2026. Preregistered before new variant observation. This addresses
the actual [source capture GPU fault](CHATBOT_BROAD_CAPTURE_FIRST_FAILURE_20261009.md),
not a new learner architecture. Original failure/caps/packets remain immutable.

Change only temporary storage for three independent source SSD contractions:
G=sum_N(C*B),states=sum_C(B_decay*x),and output=sum_N(C*previous_states).
Tile along the EXISTING chunk-index dimension and concatenate reduced outputs.
Each individual reduction still sees all original inputs in the same order;
all cross-chunk recurrence/decay operations remain original. Source chunk size128,
weights/BF16/F32 arithmetic/attention/cache/multipliers remain unchanged.
The native DLL/source files are not edited;worker-local method is generated from
the bound original method with exact checked substitutions and saved as text.

The algebraic contraction identity is exact in real arithmetic. CUDA reduction
kernel choices can depend on shapes,so bit equality is a measured gate,not assumed.
ONE new local family<=300s/8GiB OS/10GiB GPU allocated/11GiB reserved/64MiB output,
15s reserve. Source BF16/eager/no optional kernels/TF32off/seed0/CPU6 unchanged.

On the TWO already saved original trajectories,force exactly the saved preceding
output IDs through original cached one-token steps after full prompt prefill.
Use original logits_to_keep1 and complete65537 vocabulary. This is a new source
forward experiment for a changed storage schedule,NOT a regeneration/new label
capture or a source-sized deployment candidate. Source generation calls0.
Persist every new whole trajectory packet before comparison. Require ALL151
BF16 logit rows bit-identical to saved original packets AND all151 greedy IDs
equal,finite/BF16-lossless rows,unchanged source inputs and fixed resource caps.
Report coordinate disagreements even if the semantic/numerical gate fails.
Stop on first apparatus/resource fault;no tolerance/cap relaxation or retries.

TILED_SOURCE_TRAJECTORIES_PASS admits this storage variant only on the measured
trajectories. General contraction algebra is reusable;unseen bit equality or
quality is not thereby proved. Only after PASS freeze a numbered capture repair,
bind these receipts and TWO original packets,reuse them without source replay,
and generate the remaining46 under the same fixed48 cases/source contract.
Final engine LUT/ternary/SSM geometry stays unchanged;no learner/optimizer/native/
T4 calls or quality/accepted50/large-n admission in this qualification.
