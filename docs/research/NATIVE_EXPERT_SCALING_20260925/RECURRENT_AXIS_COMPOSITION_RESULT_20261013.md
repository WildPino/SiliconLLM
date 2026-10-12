# Fixed channel384/state96 composition: site23 local pass

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / SITE23 LOCAL PASS.**
Goal ACTIVE/INCOMPLETE;all owned sessions closed. No full chatbot or speed claim.

## What changed and why

The site23 recurrent-output adaptive384 channel basis had independently passed
local output budgets with full skip/source state256. A separately fitted dual96
state basis had a small local source-output error. Neither separate result
certified composition. This experiment uses the exact fixed FIT maps,without
refitting,gradient updates or DEV selection,on all48 source histories.

Source S_h is64x256. Reduced state T_h=Wc_h^T S_h Ws isr_hx96;
B'=Ws^T B,C'=Vs^T C,x'_h=Wc_h^T x_h. Scalar head decay permits the same
linear axis projection to evolve recurrently. Decode Vc_h T_h C',add full
D_h*x_h,then apply source gate/full combined RMS and trained out projection.
This approximate readout need not equal the source:both omitted subspaces and
their interaction contribute error. Ranks total384,all48 source heads retained.

New actual F32 SSD chunk128 scans:3072x96 state-only and384x96 joint,all48
histories;96 scan/history records. Source B/C projections are F32. Old384x256
control latents/outputs are reused with hashes and all outputs reproduced.
Output contractions use promoted F64 with BF16 normalized input/output. This
does not admit equivalence to the previous all-F32 decoder or a native engine.

## Held-out results, unchanged local gates

Equal-case output relative RMS against actual captured BF16 source mamba output;
24 DEV histories,12 domains/two cases each. These are output errors,not accuracy.

| Fixed arm | Own norm mean | Worst case | Centered mean | Worst domain mean |
|---|---:|---:|---:|---:|
| channel384/state256 | .063570856 | .087007061 | .067569235 | .077655411 |
| channel3072/state96 | .024903825 | .064048382 | .028022403 | .038258789 |
| channel384/state96 | **.068812658** | **.096949374** | **.073770292** | **.083198809** |

| Fixed arm | Source-denominator oracle mean | Worst case | Centered mean | Worst domain mean |
|---|---:|---:|---:|---:|
| channel384/state256 | .064564370 | .087046713 | .068569344 | .079567195 |
| channel3072/state96 | .026243932 | .064235484 | .029521418 | .039213615 |
| channel384/state96 | **.070423051** | **.097636514** | **.075507058** | **.085286841** |

All three fixed arms pass all four local gates in BOTH modes:
mean/centered<=.1,worstcase/worstdomain<=.2. Joint own-norm mean increases by
.005241802 versus the accepted channel-only control;state compression consumes
part of its error budget. FIT own-norm means:.057580173/.020349897/.061378421
for channel-only/state-only/joint. No DEV-based arm/rank/candidate reselection.
Sites0/12,layer composition,probabilities and autonomous conversation remain open.

## Error interaction and temporal qualification

Stored raw actual output errors satisfy e_joint=e_channel+e_state+e_interaction;
all component energies/three pair inner products and total-energy identity
independently qualified. Mean DEV raw relative RMS versus source y:
channel.090313338,state.034333396,joint.097957753,interaction.004335079.
These are per-case norm ratios before gate/norm/out,not additive quality errors
or a universal interaction bound. Actual projection commutation,all48 histories,
qualifies within original1e-4 scan tolerance;DEV maximum2.750e-7.

Full independent CPU direct recurrence updates EVERY3072x96 and384x96 state
coordinate at EVERY timestep. All77,922,432 scan values checked,maxrelative
5.64528e-7/source-only and1.38719e-7/joint. All12,987,072 B/C/input projection
values verified,maxrelative1.91528e-7. All277,057,536 BF16 output values
(six outputs per case,including prior baseline) independently reproduced,
all vector differences EXACTLY0. Metrics/domain/gates/energies maxscalarrelative
2.511e-14;basis identities and before/after input/output hashes PASS.

## Cost and limits toward engine.c

State coordinates3072*256=786,432 ->384*96=36,864,ratio21.3333.
F32 state bytes3,145,728 ->147,456 (3MiB ->144KiB) per operator,plusconv/generator/
decode buffers. Original engineDN512/N96 has49,152 state coordinates. These
counts establish state-memory reduction,not equal runtime speedup.

Still retained:source hidden/input functions,3072 x and gate channels,48 delta
functions,fullsource B/C generation,3584 source convolution channels,width4,
full3072 skip/gated norm and2048x3072 output projection. A compact closed causal
hidden/generator is NOT supplied. Source-projected recurrence fits original
engine state algebra in part,but source width/gate geometry does not automatically
fit its cheap active computation. [Native cost and nonlinear transfer next](COMPOSED_OPERATOR_NATIVE_COST_NEXT_20261013.md)
derives the dynamic gated readout constraint and exact retained weight counts.

Producer62.7561s/terminal62.7752,OSpeak1,350,414,336B,
CUDAallocated1,032,772,096/reserved1,459,617,792B. Audit100.9771s/
terminal100.9863,OSpeak939,237,376B. New immutable arrays733,172,288B on E.
Preparation41.1101s. All frozen caps unchanged;timing is not a clean rate claim.
No source forward/native call/optimizer update/new FIT basis/RESERVED query/T4.

## Reproduction and custody

Producer source/protocolc109fa2;binding SHA256
`7a89930b161c05d7c23a2ff5e8e86f470de7b3c85c17d1988958a1cfd51240f3`;
launch64ddc82. Audit source/protocoldce8d2d,launch481d97b.
Prepare56515/capture52306/audit67194 CLOSED0.
Original engine.c and three foreign tracked hashes preserved exactly.

- [Producer](../../../benchmarks/native_expert_scaling/recurrent_axis_composition.py), [protocol](RECURRENT_AXIS_COMPOSITION_PROTOCOL_20261013.md), [storage](recurrent_axis_composition_storage_20261013.json), [binding](recurrent_axis_composition_binding_20261013.json), [launch](recurrent_axis_composition_launch_20261013.json)
- [Result](recurrent_axis_composition_result_20261013.json), [terminal](recurrent_axis_composition_result_20261013.terminal.json), [independent audit](../../../benchmarks/native_expert_scaling/recurrent_axis_composition_audit.py), [audit protocol](RECURRENT_AXIS_COMPOSITION_AUDIT_PROTOCOL_20261013.md), [audit launch](recurrent_axis_composition_audit_launch_20261013.json)
- [Adjudication](recurrent_axis_composition_adjudication_20261013.json), [audit terminal](recurrent_axis_composition_adjudication_20261013.terminal.json), [session closure](recurrent_axis_composition_session_closure_20261013.json)

## Decision

Admit this fixed source-injected site23 composition as a LOCAL operator with
stated budgets. Next implement a separate batch1 C qualification/cost pilot
counting all retained functions before proposing a cheap engine extension.
No further unchanged short gradient epoch at failed site12 is selected.
Required goal remains useful causal pretrained CHATBOT->SSM/SWA/ternary engine,
sealed donor-relative/own-history quality,useful n/IDs+masses/physical expert
DRAM,multiple families/scales and>=50 useful tokens/s on the SAME artifact.
