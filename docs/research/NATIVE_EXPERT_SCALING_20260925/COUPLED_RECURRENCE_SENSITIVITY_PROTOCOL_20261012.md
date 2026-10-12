# Frozen coupled recurrence/output sensitivity diagnostic

All48 source histories/sites0/12/23,both frozen recurrent384 layouts. No fit,
scan,model forward or native call. Ideal projector e=(VW^T-I)(y-D*x),using
promoted actualF32 maps;actual-scan discrepancy separately recorded.
Unrounded F64 source operator F*(g*y)/sqrt(mean((g*y)^2)+eps),F=O diag(gamma).
Full Jacobian finite difference and head/time interactions per selected plan.
BF16 quantized operator is distinct and has no smooth derivative admission.

Save full F64 finite-output difference and full J_y e for BOTH layouts/all
positions,plus every-position/head energies,rho and raw statistics. FIT-only
mean no-radial and full-Jacobian64x64 Grams perhead;reuse prior FIT GY_rec for
independent-time trace proxies. No DEV Gram fitting. Arithmetic controls:
shortest FIT history fixed before observation,positions0/mid/last,all48 heads,
explicit2048x64 J_h Gram versus closed form(relative1e-9).18 derivative controls
(3sites*2layouts*3steps),central differences at1e-3/1e-4/1e-5,relative tolerances
1e-4/1e-5/1e-6. Full FIT GramPSD relative negative<=1e-10.

Predeclared heuristic next-action flags computed on FIT ONLY per site/layout:
mean finite remainder relative>.25;paired/independent-time no-radial energy
ratio outside[.8,1.25];absolute mean cross-head energy fraction>.25 =>coupled
FINITE output objective next. If none,radial change in paired head energies>.25
=>corrected-GM fit;otherwise other geometry/cost study. These are objective
selection heuristics,not capacity bounds/model-quality gates. Record all flags;
do not treat small tangent error as a bound on finite quality.

Independent CPU audit uses radial/tangent RMS action,explicit head J_h vectors,
full source contractions,all stored vectors/statistics,all mean Grams/proxies/
controls/aggregates/decisions and hashes. Vector relative1e-9,scalar relative
1e-8,Gramrelative1e-9. Full histories/positions,not sampled substitutes;explicit
finite-difference controls are additional qualifications. Fixed principal
runtime/GPU metadata and source/binding/code/operands before launch.

Producer1800s/audit1800s,OS6GiB,GPUallocated4GiB/reserved5GiB,output6GiB.
New immutable arrays in authorized E:\SiliconLLM\results\native_expert_scaling\
coupled_recurrence_sensitivity_20261012;storage receipt and all extents in repo.
No T4/new source data/RESERVED/longalign/clean timing/native/chatbot/n/50 admission.
