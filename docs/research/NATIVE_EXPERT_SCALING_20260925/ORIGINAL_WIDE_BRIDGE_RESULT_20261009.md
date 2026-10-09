# Actual wider original-engine learner: update26 complete, numerical gate FAIL

9 October 2026. Goal ACTIVE/INCOMPLETE. All owned jobs terminal. No T4 allocation.
This is progress toward pretrained chatbot conversion into the original LUT,
ternary, AQ63, dReLU, SSM/SWA engine. It is not a useful converted chatbot.

## Decision

The frozen [protocol](ORIGINAL_WIDE_BRIDGE_PROTOCOL_20261009.md) executed once:
actual Adam25 masters/moments/RNG -> DN1024/DT48 transport -> one whole FIT
Adam25->26 update -> durable checkpoint -> original native packed export.
Engineering transport/update/export/resource checks PASS. Whole-history GPU/C
numerical gate FAIL. Actual native quality remains poor.

Stored inspection also identifies a defect in the initialization: both factors
of the 32 added delta coordinates are zero. These coordinates cannot activate
under the frozen deterministic optimizer. Do not describe this candidate as
having 48 active learned delta directions, or continue an unchanged long dose.

## What actually ran

D256/N96/L6/n1152/k8/HID_E128/full V65537. Five SSM sites DN1024/DT48 and one
original SWA site. Same original native consumer as the qualified
[cost envelope](ORIGINAL_NATIVE_ENVELOPE_RESULT_20261009.md); no change to the
original computational kernels in this experiment. Bank masters are actual
source-informed masters, not a dequantized replacement bank.

92 master tensors, 721,008,128 coefficients, including unchanged-count
679,477,248 bank coefficients. 45 enlarged / 47 unchanged tensors. All primary
master/moment coordinates preserved bitwise; new moments zero; master replicas
explicitly initialized; all old Adam counters25 and CPU/CUDA RNG restored.
Initial direct master export SHA equals the earlier native fixture exactly.
Thus its retained3352 native before rows/routes were adopted without replay.

One new FIT history `broad_fit_smol_magpie_ultra_022`:1507 input positions,
256 complete cached source distributions. Whole core and streamed bank adjoints;
lr5e-5/betas .9,.999/eps1e-8/no decay/foreachFalse/global gradient clip1.
All92 gradients/parameters/moments finite, all steps26. Gradient norm23.002519,
clip coefficient.043473499. Forward43.734s/backward120.610s/optimizer plus finite
inspection7.235s. Exposure unions[688,851,954,1009,954,923];12056 selected pairs
per site. Full CPU bank gradient/Adam residency is priced, not only top8.

Five new out_proj half-channel blocks receive nonzero gradients; all131072
added coefficients per block change,655360 total. Their L2 norms .00977..00979.
This proves new read-coordinate activity. It does not prove independent new
history information, a causal width advantage, or broad quality recovery.

Actual26 model/optimizer/RNG/ancestry/transport ledger is durable:

| Artifact | Bytes | Raw SHA256 |
|---|---:|---|
| initial_wide.packed | 520029440 | a06e2619fb5bbe2a96690ed5dba03bc2345bc03799d48485b134630f56daf423 |
| candidate_wide26.pt | 8652209850 | 4fabd24f63bd2361ce170c1a25f9a4365d94eb45117376ae14fcebc179b28156 |
| candidate_wide26.packed | 520029440 | 866c582648157d962feb79e60b47c824dd403d44c6eb4f913151ca771431f5a8 |

The original Adam25 checkpoint remains immutable. New geometry/new update
lineage is explicit; inherited24 FIT IDs and this new FIT ID are separate.

## Actual native source-relative endpoint

One native process evaluated three fixed forced histories,3352 positions and
219680024 F32 head values. Before rows are the exact identical initial export.
Each case uses256 cached source labels; this is not own-history generation.

| Case | KL before | KL after | Argmax disagreements before -> after |
|---|---:|---:|---:|
| DEV self_oss036,357 positions | 9.157763 | 9.059255 | 248/256 -> 248/256 |
| FIT magpie022,1507 positions | 6.655792 | 6.312887 | 239/256 -> 240/256 |
| DEV magpie036,1488 positions | 9.327919 | 9.164405 | 252/256 -> 251/256 |

KL declines on these three cases. No matched narrow-core update was performed,
so that change cannot be attributed to width. Post-update disagreement rates
96.875% /93.75% /98.046875% remain incompatible with useful donor preservation.
No broad DEV/domain gate or fresh chatbot admission is inferred.

All actual native head values finite; unique valid IDs in0..1151; normalized
nonnegative route masses with defect<=1.789e-7. All18432 updated LUT integer
witnesses equal independent master quantizer expectations bytewise. Native
after unions[408,419,458,482,467,467] /[686,845,951,1004,948,911] /
[664,821,907,962,925,870]. These are union exposure, not useful capacity evidence.

## GPU/C numerical failure and persistence gap

1507 complete updated FIT GPU head rows versus native rows:
26 exceed relative RMS1e-4; worst .07098787967 at position1251; first82.
One greedy mismatch at339. Six failed rows are supervised positions:
1251,1288,1324,1343,1352,1362. GPU/native supervised KL delta .00109662396;
GPU KL6.655792 ->6.313984, native after6.312887. This delta is small relative
to the native donor loss, but it does not qualify the numerical implementation.
No numerical cause or harmlessness is established by head fields alone.

Worker observed four ID-mismatch calls and maximum mass delta .00141611695.
Raw GPU routes were retained in worker memory only, despite the protocol's
persistence wording. Consequently these two counts cannot be independently
recomputed from disk. Native routes and GPU/native heads were persisted and
independently rechecked. Correct future observation code to persist raw GPU
routes, plus the intermediates needed to locate first divergence.

## A zero-lock in the added delta coordinates

Let U = x_proj[16:48,:], V = dt_proj[:,16:48], x = post-convolution activation.
The new contribution to pre-softplus delta is z=VUx. For upstream derivative g:

```
dV = g (U x)^T
dU = (V^T g) x^T
```

Both factors and both Adam moment fields are initialized zero. Therefore both
gradients are identically zero for every history; with this frozen graph,
zero moments and deterministic optimizer they stay zero. Softplus and recurrent
loss supply g but cannot change this product rule. This statement requires no
statistical experiment. Existing16 directions can still learn and DN1024
channels can become distinct; it is specifically the added32 delta factors
that are locked. The factor rank upper bound remains16, not48.

[Stored algebra audit](original_wide_delta_zero_lock_20261009.json) verifies
all ten initial/updated packed U/V blocks have zero nonzero coefficients.
A small independent algebra witness verifies that seeding U alone, keeping
V=0, preserves the real-arithmetic function while allowing nonzero dV.
It is not an observed model gradient or a converted chatbot improvement.

Correction requires an explicit fork of actual26, preserving old state and
unchanged moments: nonzero bounded independent U rows, V zero. Prefer directions
supported by existing FIT activations/source functions; verify rank and scale.
Real native pre-update output identity and actual new V gradients/changes must
then be measured. Do not modify frozen initialization code or rewrite history.

## Resources, provenance and stored verification

Freeze1520c44eb9ad1aac316e62d704ad8e9ba5de9202;41-input binding
ed6bd8b9f0fe0f9e6cdefc913af4816b6988629745cf3357c0f2a61a860bdede.
Raw [result](original_wide_bridge_result_20261009.json) SHA
6f91350744c47fc28e65829a34931549aefb822d542b8521e27d92febe640dd3.
[Terminal](original_wide_bridge_result_20261009.terminal.json)/
[log](original_wide_bridge_result_20261009.worker.log) retained.

Launcher29524/worker28108/create_time1791566251.4117844/exit0;
session25642 CLOSED. Held family487.641s; worker result452.641s.
Native17712/create_time1791566654.1404717/exit0, held46.875s.
OS worker peak13544849408B + launcher30265344B + direct native528060416B
=14103175168B conservative sum of separate peaks, not a simultaneous peak.
GPU allocator allocated2262381056B/reserved2971664384B, not total device residency.
17 outputs10967978123B. Caps3600s/reserve300s/OS32GiB/GPU10/11GiB/output12GiB/log4MiB PASS.
No source calls, RESERVED queries, or T4. No accepted speed/DRAM/general n admission.
Parity mode includes full-logit IO; raw per-request timings are not a speed campaign.
Previous fixture speed is not speed certification of this updated useful model.

[Independent stored audit](original_wide_bridge_stored_adjudication_20261009.json)
rehashes all41 inputs/17 outputs, verifies extents/counters/changed export/integer
identity; recomputes all six native teacher metrics/routes and1507 head comparisons.
KL aggregate delta0; row-metric delta3.053e-16. Its26.610s is script elapsed only,
not a separately held process-family resource measurement. No checkpoint
deserialization or independent recheck of unpersisted GPU routes is claimed.

## Next and goal-level judgment

[Exact next](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md): locate first GPU/C
divergence with persisted intermediate/route witnesses; unlock the32 zero delta
directions in an explicit function-preserving fork; qualify actual new-direction
gradients/export/native outputs. Then select a finite joint history-function
recovery pilot with whole native DEV/own-history criteria and plateau stops.

The original engine remains the endpoint. Local N256->96 state transport passed;
linear384 channel decoders failed; whole original-shaped quality failed. None
proves a universal capacity impossibility. Added width/copying functions does
not supply source information automatically. Useful original native chatbot
and>=50 accepted IDs/s on the same artifact, useful large n/structured CPU routing,
physical DRAM and other families/~10B/~100B remain missing. Goal ACTIVE/INCOMPLETE.
