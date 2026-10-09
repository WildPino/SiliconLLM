# FFN F32 arithmetic positive control

9 October2026. Freeze before observations. The complete ternary/AQ63 package
fails DEVKL2.09967/disagreement42.9549%,screen1/16(original14)/history0/4.
Its intervention also changed FFN internal BF16 arithmetic toF32. Before
calibration/recovery,isolate that numerical confound using original weights.

## Single intervention

Pinned original Falcon1.5B/all411 parameter objects unchanged,source cached
schedule/qualified SSD helper/non-FFN organs/BOS17/bothEOS11,228 unchanged.
Worker-local FFN forward uses original BF16 coefficient VALUES converted toF32
for each projection,F32 linears/original gate multiplier/SiLU/product/down
multiplier,andonlyfinal BF16 cast. No ternarization or AQ63. This preserves
the real-arithmetic function,not a bit-equivalence claim about finite arithmetic.
No source-file or weight modification;no training/export/native/T4.

Repeat ONLY the preflight metadata-shortest/longest FIT trajectories against
their original saved274 BF16 rows;new numerical variable,not a confirmation
replay. Preserve per-label KL_F64/IDs/allchanged logits. Also generate original
known16 screen prompts/64-ID caps/canonical template,save scores/IDs/text.
Original teacher14/16 andquantized1/16 reused,not regenerated. No new4 followups:
this narrow control prices the F32 confound,not whole source interaction.

## Fixed decision and cost

Each FITcaseKL<=.01 andcombined differing-IDfraction<=.05,screen>=13/16,
everycategory>=2/4,blank<=1,no non-EOS special leak andvalidEOS/cap stop.
These bounded consumed-FIT/known-screen criteria do not admit fresh quality.
Complete both trajectories andall16 prompts unless a resource/finite fault.
If PASS,F32 alone does not explain the observed ternary-package damage in
this scope;price activation-aware ternary calibration before compression.
If FAIL,retain loss/outputs andresolve arithmetic boundaries before assigning
the package loss to trits/AQ63 or starting long recovery. Neither outcome
separates ternaryweight fromAQ63 effects or proves their interactions additive.

Actual complete control8534 new forced rows inabout1476s implies274 about47s;
worst16*64 generated IDs at its observed~.177s/ID about181s,plusload/IO/guards.
This is a conservative price estimate,not a measured unquantized speed.
Fix420s family/30s reserve,8GiBOS/10GiBallocated/11GiBreserved/512MiB outputs/
4MiBlog/sixcores/no overlap. Original FFN BF16weights remain resident instead
ofint8codes,adding about679MB tothat component;temporaryF32 projection remains.
Retain firstfault andcompleted results;no automatic replay for a favorable result.
