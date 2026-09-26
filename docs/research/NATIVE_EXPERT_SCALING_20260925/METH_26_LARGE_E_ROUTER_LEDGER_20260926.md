# METH-26: large-E routing obligation for the quality-screened Qwen geometry

This is **shape arithmetic**, not a 10B/100B run. METH-25 establishes
a small quality-screened starting point: Qwen2.5-0.5B core plus 24
layers of E128 rank-8 residual experts with top-4 routing. Each added
expert contributes `24 × (2 × 8 × 896 + 896) = 365,568` distinct
factor/router parameters. The output factors and router currently use
fp32; head/body R8 compression changes none of these counts. The
current E128 adapter file is 187,177,472 bytes. These experts are
distinctly trained, but the experiment does not establish quality at
larger E.

| Hypothetical added-capacity target | E needed | Exhaustive fp32 router bytes/token | Exhaustive int8 router codes/token | Ideal int8 read time at 40 GB/s |
|---|---:|---:|---:|---:|
| ~10B expert/router parameters | 27,355 | 2,352,967,680 | 588,241,920 | 14.71 ms |
| ~100B expert/router parameters | 273,547 | 23,529,418,752 | 5,882,354,688 | 147.06 ms |

Both E values assume **the small current geometry stays unchanged**;
they do not identify a useful 10B/100B model. The calculation uses
`24 × E × 896 × bytes_per_router_weight` addressed router bytes each
token and omits row scales, candidate selection, selected experts,
core, head and memory-system overhead. At the favorable 40 GB/s
yardstick, even a one-byte exhaustive router for the 100B case
consumes over seven entire 20 ms budgets. The selected fp32 factors
alone are 5,505,024 bytes/token with top-4, independent of E; this
does not rescue exhaustive routing. Full fp32 factors would also make
the stored expert pool impractically large for many users' RAM.

**Decision:** any 10×/100B expert-count claim for this family requires
an index/hierarchy that reads a bounded or sublinear fraction of
router rows, a packed-only factor bank, and a route-recall/quality
audit as E grows. NES-03's exact-match int8 shortlist still scans
all E rows; its E1280 success cannot be extrapolated to E273,547.
The next native experiment must measure CPU router selection and LUT
expert time at large E independently of language quality, then test
the chosen routing scheme on *distinctly learned* expanded experts.
