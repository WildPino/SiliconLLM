# Wider trained recurrence:fixed state96 at sites0/12/23

13 October2026. Freeze before observation. Reuse exact FIT dual96 state maps
from complete state-projection analysis,no channel projection/no refit/rank/
DEV choice. All3072 trained channels/48 scalar head decays retained. C AVX F32
state updates/reads and B/C96 projections,explicit expf per head,sourcefullskip.
F64 source gate/postgate combined RMS->BF16->original engine AVX F32 dotf output
projection/BF16 output. Original engine.c immutable/primitive bodies checked.

ALL48 histories at ALL3 sites,144 new native scans. Actual source x/B/C/delta/
gate supplied;source hidden/gen/conv/all-layer/causal encoder not qualified.
Both norm modes per fixed site. State3072x96=1,179,648 F32bytes versus147,456 for
channel384x96. This8x count is not assumed8x total latency;actual stages timed
only AFTER full independent numerical qualification.

Save every B/C96/latent3072/finalstate3072x96/rawF32scores/normalized BF16 inputs/
BF16 output vector. Same numerical gates:projectedB/C1e-6,scan/finalstate1e-4,
smooth normalized/rawscores/BF16 output1e-4,scalar metric1e-8. ALL raw F32 score
words bitexact separate scalar eight-FMA reference. Full independent CPU states
EVERY timestep,export bits/source weights/maps,all norms/scores/outputs,
metrics/labels/domains/gates and before-after source/result/output hashes.
Complete all controls even numerical or quality failure;retain false flags.

Original local DEV gates,both modes/per site:mean/centered<=.1,
worstcase/worstdomain<=.2. Prior state-only measurements used different F32
normalization/matmul:report scalar metric discrepancies with1e-3 flag,not vector
precision equivalence (prior vectors at0/12 unavailable). Old precision failures
remain failed. Site23 stored axis-composition state-only arrays also bound.

Compile45s/DLL,prepare900s/qualification900s/audit1200s,OS6GiB,
NEW immutable E namespace<=6GiB including all3 parameter packets/run arrays.
Native single thread/affinity0;independent audit BLAS/CPU0..5. Offline/noGPU/T4/
RESERVED/source model forwards/optimizer. Exact compiler/runtime/source binding
before launch. If actual work requires hours,follow user's announce/launch/PAUSE.

Cost decision:if wider trained state is a small share of budget,retain useful
functions where quality is better rather than enforce an arbitrary384 bottleneck.
Generator/readout/compact CAUSAL hidden and conditional ternary bank still need
transfer. No generic donor runtime/usability/n/DRAM/multiple-family/same50 claim.
