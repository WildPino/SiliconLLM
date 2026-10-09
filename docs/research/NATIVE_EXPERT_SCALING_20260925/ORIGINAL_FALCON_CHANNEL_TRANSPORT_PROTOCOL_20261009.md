# Output-aware channel/head and denominator assay, before observation

9 October2026. Goal ACTIVE/INCOMPLETE. This follows complete rank96 recurrent
transport, but keeps actual source state256 in this separate assay. Original
engine/LUT/ternary endpoint and actual Adam25 stay unchanged. No source-model
forward, reply/label generation, optimizer, native, RESERVED or T4 call.

## Uncertainty and decision

Rank96 can retain local source output better with dual read/write bases, but
that evidence keeps all3072 x/gate channels/48 heads/full norm. To price a warm
original-core extension, determine whether384 source channels can preserve the
actual block output, and whether its full-vector denominator can be replaced.
Use retained full y/gate/output at sites0/12/23/ALL48 full histories only.
24 FIT histories select/calibrate;24 DEV histories evaluate. No DEV selection
of channels or calibration;12 domains/two cases per split. Input/runtimes/code/
protocol/criteria/limits bound in raw bytes and committed before execution.

Compare EXACT384 active channels:

- all48_heads_x8:8 actual channels per every source64-channel head;
- selected6_heads_x64:6 complete source heads,384 channels.

These are source-local restrictions. Full source B/C generators/state256/delta/
gate/input representations stay; no compressed state/P256/depth/FFN combination.
Oracle full-denominator outputs cannot provisionally admit a deployable layout.

## FIT output-aware greedy selection

u=y*SiLU(gate),d=sqrt(mean3072(u^2)+sourceepsilon),
F=BF16(u*rsqrt(mean3072(u^2)+sourceepsilon)*source_norm_weight).
Source out_proj matrix O is BF162048x3072. Reference R is stored BF16 source output.
All coefficient tensors are read directly from pinned safetensors, no model load.
Full F*O^T must reconstruct captured output to relative RMS<=1e-4 on all144 cases.

For fixed unit coefficients of a subset S, least-square target error is:

```text
G_ij = mean_equal_FIT_cases(mean_positions(F_i F_j)) * dot(O_i,O_j)
b_i  = mean_equal_FIT_cases(mean_positions(F_i dot(O_i,R)))
E(S) = target_energy - 2 sum_S b_i + sum_(i,j in S) G_ij
gain(i | S) = 2*(b_i - sum_S G_ij) - G_ii
```

Contractions use F32 CUDA/TF32 disabled; case means accumulated in CPU F64;
G is symmetrized by(G+G^T)/2 in F64. Fixed-unit greedy is a heuristic, not an
optimal subset or rank384 theorem. Choose maximum gain, lowest index breaks
exact ties. all48x8 imposes capacity8 per head until384 selections, even if final
gains are negative. Whole-head variant sums G and b over each64-channel group and
greedily selects6 groups. Sort actual channel IDs for restricted matmul.
No post-observation threshold/mode/channel-budget changes or extra fit.

Save initial b,diag(G),384 selected G rows,F64 gains/order and48x48 head Gram/
head b/order/gains,target energy/constant denominator in numeric-only NPZ, plus
selected IDs/calibration/receipts. This fits32MiB across3 sites. Independently
replay selected greedy steps from SAVED rows/groups, with exact order/gain checks.
Verify E(S) against direct F32 restricted matmul on all24 FIT cases; absolute
difference/target_energy<=1e-4. This checks moment accounting without claiming
rounding-exact optimality. Complete selection is durable before DEV evaluation.

## Three denominator modes

1. full_source_denominator:restrict original full normalized/rounded F to S;
   no calibration. Diagnostic uses omitted source channels to obtain d.
2. subset_RMS_calibrated:use u_S/sqrt(mean384(u_S^2)+epsilon)*norm_weight_S,
   round BF16 and restricted BF16 out_proj. Runtime subset norm is a proposed,
   UNIMPLEMENTED original-core extension.
3. constant_RMS_calibrated:replace d by D0=sqrt(equalFITcase mean(d^2));
   round BF16 normalized features/restricted BF16 out_proj. Omits runtime norm.

For modes2/3 fit ONE nonnegative scalar per site/layout/mode on FIT only:
alpha=max(0,equalcase E[dot(pred,R)]/equalcase E[norm(pred)^2]). Apply alpha in
F32 AFTER BF16 out_proj. This is the measured operation; no exact folded-weight
implementation is claimed. Cache FIT restricted outputs in CPU BF16; do not
replay them after fitting scalars. DEV restricted outputs are computed once.
Preserve source gate/nonlinear function: their target realization is still missing.

Full/centered/label-position output relative RMS:double metric accumulation,
reference denominator clamp1e-12;center each history/output channel separately.
Save case/domain equal means/worst,selected u energy fraction(min/mean/max),
implied full-denominator relative RMS(ds/alpha orD0/alpha;null ifalpha0).
Full-denominator diagnostic is explicitly optimistic. Changed subset normalization
does not mathematically recover omitted energy without the measured approximation.

## Predeclared engineering budget and next

For each layout/deployable mode, EVERY assayed site's DEV output mean<=.10,
case worst<=.20,domain mean worst<=.20,and centered-output mean<=.10.
Provisional selection only if all pass. Among passing combinations minimize
maximum site DEV mean,then sum of site means,then lexical layout/mode.
This chooses a target-representation candidate after held-out evaluation;it does
not fit channels on DEV or admit chatbot quality. Fresh future evaluation still
required. If no deployable combination passes, report CHANNEL_HEAD_REPRESENTATION_
BUDGET_FAIL and revise mixed features/gate realization/depth before longer dose.
Do not infer general capacity impossibility or relax criteria after observation.

Even a local PASS needs combined dual96+channel+P256+operator checks, a real warm
learner/export/native variant and fresh chatbot/quality+>=50 same-artifact evidence.
Original all-history numeric FAIL and previous whole-quality FAILs stay.

## Cost, resources and interruption

One family1800s/reserve90s,conservative held worker+launcher OS6GiB,
GPU4GiB allocated/5GiB reserved allocator peaks,output namespace32MiB/log4MiB.
No other owned Python/compiler/native workload;publisher exception exact command
only. Launcher affinity11/worker0..5,Torch threads6/inter-op1/NumPy1;deterministic
algorithms/TF32off/CUBLAS :4096:8. Pre/post all input byte hashes;source unchanged.
Capture held PID/create_time/exit/OS through exit,worker GPU/time,namespace extents.
First fault/resources/completed selections/metrics saved;never overwrite outputs
or restart merely on timeout. Reuse durable selections/metrics in a separately
bound missing-work completion if necessary. No unrecorded retries/refits.

```text
original_falcon_channel_transport.py --bind --out <new binding JSON>
original_falcon_channel_transport.py --launch --binding <binding> --binding-sha <raw SHA> --freeze <commit> --directory <new namespace> --out <new result JSON>
```

Python312 -I -S -B -X utf8 from repository root. Code and binding must be frozen
before execution. Principal runtime binaries,not full DLL tree,are fingerprinted.
