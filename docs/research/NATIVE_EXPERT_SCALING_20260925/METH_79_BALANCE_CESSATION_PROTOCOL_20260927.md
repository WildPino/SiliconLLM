# METH-79: controlled cessation of router balance after update 64

**Motive.** METH-78 measures a 62–498× larger router-gradient norm
from the weighted axis-balance term than from CE on two training-source
examples per E. METH-75 continued that balance objective through update
256 and lost donor-position agreement. Test whether removing balance
after the already trained update-64 checkpoint improves retention while
preserving learned E1280 coverage. This is a causal intervention on one
training term, not an explanation assumed in advance.

**Arms and controlled comparison.** Resume the exact METH-71 update-64
E128 (8×16) and E1280 (32×40) checkpoints, including optimizer and RNG
states. Run updates 65–256 with the exact METH-75 raw/chat draws,
two raw/two chat microbatches, CE, raw KL weight 2, chat KL weight 4,
factor LR 1e-4, router LR 1e-5, and global-norm clip 1.0. Set only
the axis-balance coefficient to zero. Preserve product-key top-four,
rank-8 factors and BF16 donor. The existing METH-75 balanced update-256
checkpoints are fixed comparators on the same new prompts; report both
arms and the unchanged update-64 baseline. Do not alter the rule after
viewing outcomes.

**Fresh development.** Before either training run, freeze 24 prompt rows
with seed 797979 from the same bound corpus, excluding all prior prompt
development sets through METH-75, METH-70 sampled raw rows, and all
METH-71/75 raw draws through update 256. Both E arms and both fixed
balanced comparators use this same set. Record row IDs and the manifest
hash before inference. The old METH-71 and METH-75 development scores
remain historical and are not confirmation data.

**Terminal gate.** Require each cessation arm to reach ≥95% donor top-1
on fresh prompts, decline ≤1.0 percentage point from its own frozen
update-64 baseline on those prompts, raw held-out ΔBPB≤+0.05,
≥64/640 changed B slots per layer at E128/E1280, and exact routed
top-four versus exhaustive pair scores on evaluated prompts. Require
≥96/640 selected experts per layer and maximum/mean load ≤32×/64×.
For a retention benefit specifically attributable to cessation at E1280,
require its fresh-prompt top-1 to exceed the matched METH-75 balanced
update-256 checkpoint by ≥0.5 percentage point. To maintain quality
as expert count grows 10×, E1280 top-1 must be no more than 0.5 point
below E128. Report numerical values regardless of gates. Passing only
permits a new source-disjoint automatic and blind semantic audit; it
does not reverse METH-72/74 or establish a native LUT/full-engine rate.

Use the local RTX 3060; stop at ≤10.5 GiB peak allocated GPU,
≤20 GiB process RSS or ≤15 minutes per arm including checkpoint save
and hash. Preserve failures. No T4.
