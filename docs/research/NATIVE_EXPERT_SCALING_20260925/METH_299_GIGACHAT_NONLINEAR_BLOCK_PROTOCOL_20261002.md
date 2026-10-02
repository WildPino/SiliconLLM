# METH-299: real GigaChat nonlinear32-channel block preservation

## New variable, prior stops and prospective decision

298 rejects a global linear rank192 representation, even with full input
covariance. Test a different conditional geometry: preserve original
SwiGLU channels in contiguous32-channel blocks within each pretrained
GigaChat macro expert. 185's individual norm-ranked channel omission failed
on dense Qwen; this sparse11.480B family, narrower1280-feature experts,
32-channel blocks and direct residual-aware greedy selection are unmeasured.
This is not a rerun/regrade of185 or298.

Reuse the real source-BF16 GGUF down inputs captured/verified298. Their
nonlinear activation values are already available; no new model inference,
download,GPU/T4. Both previously used calibration domains are consumed,
not new full-model heldout quality. Source/tensor/capture bindings remain
identical. Freeze script and this protocol BEFORE any omission score.

## Exact rule, source fidelity and static control

For each of nine layer1/13/25 × expert0/32/63 down matrices W[1536,1280],
use EVERY real captured F32 post-SwiGLU input z from BOTH domains:
22,549 initial-domain and15,977 distinct-domain expert states. Original
BF16 W is converted exactly to FP64 for this necessary mechanism screen.
Recheck every raw BF16 tensor SHA against180; capture result SHAs:

- Fit `57cc46da176b6be16953ecfadb21c173d85bb0b8bb6dcc026ddba709fb9c0331`.
- Test `124610e4ddf83f2f8f5d58bc7465deaebc91183c2452614522cdb1ef4db8f4fa`.
- Prior298 `b0b9a4773b02dc6c309504f2eddc9a8dcc623d115d70c751df9d392219d00223`.

Partition channel IDs into40 fixed consecutive32-channel tiles. For each
state calculate all actual tile output vectors `v_j = W[:,tile_j] z[tile_j]`.
Full reference is independent `y = W z`; sum of ALL40 tiles must match
relative L2<=1e-12. The tolerance admits FP64 summation-order roundoff,
not source-model approximation. Every input/output/norm must be finite;
positive reference norm, exact geometry/indices/counts, unique selection.

Greedy selector starts residual r=y and selects unselected tile maximizing
`2 r dot v_j - ||v_j||^2`, then subtracts its ORIGINAL vector, without
rescaling,refitting,bias or intercept. Stable lowest tile-ID ties. Observe
exactly5,10,20 selected tiles (160,320,640 of1280 source channels), fixed
before scores. It sees the full activation and full target output; this
is a nondeployable oracle diagnostic, NOT an optimal combinatorial subset.
Unlike contribution-norm ordering, it accounts for cancellation with the
current residual. No cheap selector or sparse execution is assumed.

Static comparator selects same number of tiles by original down-column
Frobenius norm, lowest tile-ID ties, fixed for all states of that expert.
No fit/evaluation feedback alters tile partition or rule. Keep full per-state
relative output L2,source coordinates,norms and all18 expert/domain summaries.
Percentiles use NumPy linear interpolation. Batch64 bounds tile workspace.

## Frozen fidelity gates and stop decision

For each retained count independently, require ALL of:

- Pooled relative L2 median<=1% and p95<=5% on EACH domain.
- EVERY sampled expert's median<=1% and p95<=5% on EACH domain.
- Pooled greedy median<=half the static median on EACH domain.

Smallest passing count licenses ONLY a separately frozen,costed input-only
selector and nonlinear/full-model experiment. Failure at all<=20 tiles
closes this greedy32-block omission rule at <=half width. Do not increase
count,relax error,exclude experts or choose a new algorithm using these
scores under this protocol. It does not reject every possible subset,
trained compensation or another geometry. Original185/298/297 stops stay.

## Capacity, physical cost and limitations

64 macro experts ×40 blocks would have2560 partition labels/layer,using
the SAME original source parameters. This partitions real pretrained
functions; it adds no new learned capacity and does not establish that
labels represent independent useful experts. A real larger donor could
provide more source macro experts; no100B result follows from this assay.

Hypothetical packed32-channel gate/up/down tile contains147,456 original
elements:294,912 BF16 bytes or about82,944 uniform-Q4_K bytes, above48KB.
Actual existing GGUF down columns are not this packed layout; row gathers,
block scales/alignments,overfetch and DRAM locality need real measurement.
Twenty selected blocks halve expert channels mathematically; with actual
Q4 routed payload uniformly halved and all other organs unchanged,the
addressed-byte calculation is838.141MB/token,still above560MB allotment.
No direct full export follows. MLA/head/shared treatment remains necessary.

The measured oracle computes all gate/up source channels and ALL down tile
outputs. It saves no active bytes or native time; a hypothetical budget
cannot be added to another artifact's rate. Cheap input-only prediction,
CPU LUT/router costs at larger useful n and source-relative whole quality
must all be established before any50tok/s claim.

## Resources and command

Local CPU,six BLAS/host threads;expected several minutes including capture
parse,hard stop15minutes/12GiB parent RSS/50MB result. No model job or native
performance benchmark overlap. Preserve failure/partial rows before any
narrow apparatus repair. Helper298/build libraries/protocol remain frozen.

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth299_gigachat_block_selection.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth299_gigachat_block_selection_result.json
```
