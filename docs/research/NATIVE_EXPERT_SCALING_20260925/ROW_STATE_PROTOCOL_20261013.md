# Frozen contextual preconv-row ABI and all24 source-only exports

Extend admitted token_state.c into row_state.c by factoring only row selection:
token API selects F0 row by actual uint32 ID;new rs_block accepts contiguous
BF16 preconv/scaled rows[n,6704] with caller uint32 provenance tags. SAME
post-selection arithmetic body retained literally,including conv/state/own
norm/C I16 quantizer/accepted integer readout. Persistent shared TokenState,
tags stored as last-ID metadata;tags do not imply a semantic expert selector.
Before ANY state write,reject nonfinite BF16 row (exponent255) with rc4.
Existing token API also uses guard;all admitted original rows finite. No dtype
inference from raw pointers:caller must enforce BF16 uint16 row ownership.

Source-only ALL24 exports from canonical model.safetensors and saved FIT
basis records,verified stored-algebra parent. Ws/Vs=actual W/V transpose cast
F32,Alog/D/gamma actual canonical values F32,conv weights/bias/delta F64.
Verify W/V shapes256x96,finite,and max|W^T V-I96|<=1e-8 before export.
Actual O2048x3072,p6 scale=rowmax/364,zero explicit0,q=RNE(actualF64ratio)
clip+-364/int16. No responses used to select precision or coefficients.
Independent auditor reconstructs every coefficient/scale/manual parity RNE
and every source/core/conv/basis identity. Source full checkpoint/runtime/
parents/basis/parameters/compiler/actual DLL/protected foreign pins before/after.

Qualification stages fixed before observations:
1. All48 actual initial-layer histories22547 positions:new row ABI full
   block versus EVERY one-token continuation bit-exact;ALL arrays/finalstate/
   ring/tag/count byte-equal COMPLETE admitted original token-state archive.
   Store full new arrays;independent auditor reads every word against prior
   qualified archive. Continuation assertion is a producer receipt distinct
   from the independent all-word/archive test. Bad-row initial-state immutability
   control likewise a checked producer receipt,not a second native execution.
2. ALL24 parameter sets receive first16 rows of first existing FIT initial-
   layer history as ARBITRARY operator inputs. These are not the actual
   later-layer hidden trajectory and cannot support source-local/whole-model
   quality admission. Store all field/B'/C'/latent/final state/ring/norm/I16/
   scale/I64/raw/BF16/tag/count outputs. Independent F64 projection/scan/
   own-normalization/exact bounded-integer contractions/manual BF16 and row
   field equations verify math for actual coefficients. Every word decoded.

Numeric thresholds inherited:fields and B/C relative<=1e-6,F64 scan/final/
own norm<=1e-4,exact integer/F64 raw/BF16 bits,full initial-layer archive
bits and continuation bits required. Current single-field delta min0/maxNone.
Source parameter export and RNE equality exact;dual identity<=1e-8. Complete
ALL records/audit even on numeric failure. No gate loosening or all24 fidelity
extrapolation from arbitrary control. Source input contextfulness remains.

Prepare/run900s/audit1200s/OS6GiB/compiler child45s,new ordered E
row_state_20261013<=3GiB (~2.14GB:24 parameter packets,48 full histories,
24 arbitrary16-row controls,DLL/log). F0/source/basis/old arrays referenced,
not copied beyond new qualification outputs. CPU0 native calls/independent
BLAS0..5,no GPU/T4/RESERVED/source MODEL forwards/optimizer. No timings.
Ready generic ABI/verified exports enable the proposed full-width source
quality bridge;actual later21 input data/full chatbot/performance/useful n/
final engine CPU deployment still unverified. Original goal unchanged.
