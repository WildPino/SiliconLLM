# METH-216: changed parent-specific leaf keys in reused child coordinates

Freeze before fitting. Original BF16 centered E1280 source, exact METH-126
factor bank, METH-107 learned child projection, METH-175 draws and METH-172
structural table. Require exact eight-prompt teacher/control logit parity.
Do not use the rejected compact core. The METH-211 combined ledger is
only a hypothetical layout; METH-214 blocks its quality promotion.

Normalize a separate copy of the original rank-32 child feature. Fit nine
unit keys independently for each of 1280 E1280 parents, shared across
text/ChatML. Existing E1280 selection and factor functions remain unchanged
during this route-only screen. Diagnostic feature capture recomputes the
projection; eventual native reuse and cost are separate required checks.

Raw fit uses draws [0:1024]+[1280:3328], 3072 sequences; ChatML fit uses
[0:1024], 1024 sequences. Exclude old reserve [1024:1280] and new reserve
[3328:3840] from key/bias fitting. Within each layer/parent uniformly sample
at most 256 pooled fit selections with NumPy seed 216216+layer. Initialize
keys by farthest-point selection starting from that sample's first point,
with seeded unit Gaussian fallback for empty/exhausted distinct points.
Run 16 spherical Lloyd updates with empty clusters retaining initial keys.
Fit two zero-initialized bias banks using all fit selections of the relevant
mode, 100 updates at each temperature 0.05/0.01/0.002/0.0004, same +1
count smoothing and per-parent zero-mean bias as METH-206. No tuning/retry.

Use the observed first token ==151644 causal mode, persisting across windows,
and the unchanged structural slot. Raw/chat are evaluated separately.
Every layer in every evaluated cell must satisfy max-load ratio <=1.25,
hot-parent (>=250 selections) worst child share <=25%, >=4000 content
children covered, mean standardized selected score advantage >=0.05 and
agreement with unbiased local-key argmax >=15%. Require exact artifact
readback and finite keys/biases. Fit failure stops all reserved/source
screens. If fit passes, score old reserved raw/chat; if both pass, score
new 512-draw raw/chat reserve; only then source-separated METH-121 documents
at windows 128 and 512. Every prior nonfit corpus is previously consumed
development evidence; no new-source quality claim follows.

Store FP32 projection/local keys/two mode biases, recording selected-address
ledger (114048 new weight bytes/token) and physical hashes. This gate does
not add/train expert B, prove useful specialists, claim native constant
cost, or establish 10B/100B transfer. Route pass permits a separately
frozen real CPU router/LUT test and content-coupled specialist training.
Local RTX 3060, six host threads, <=45 minutes, 20 GiB RSS, 10.5 GiB
allocated GPU, <1 GB new disk. Save partial cells on failure. No T4.
