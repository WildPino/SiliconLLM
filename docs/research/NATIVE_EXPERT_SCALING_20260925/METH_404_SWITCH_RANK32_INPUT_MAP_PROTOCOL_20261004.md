# M404: ONE fixed restricted input alignment before source functions

Freeze before fitting/prediction errors.403 measured all12 raw coordinate identity
failures with near-zero median cosine. New variable: a learned restricted input
map, not an identity interface. Reuse403 exact paired traces/tokenizer/outputs and
its prospectively declared18 development/6 validation book split. All consumed
362 books; no claim of new held-out model quality or unobserved source vectors.

Per bank: center source256 development X and128 Y, F64 SVD X, ONE rank32 affine
principal-component regression. Require singular32/singular1>=1e-6. Prediction
Ymean + ((X-Xmean) @ basis768x32) @ coefficient32x768. Fit only18 development
books. No full768D identification, ridge/rank/hyperparameter search or validation
selection. Export F32 factors/means; diagnose F32 inference alongside F64 fit.
Report development input energy as descriptive only, never a promotion gate.

Input eligibility on each bank's six validation books requires ALL:

- F32 total squared error<=0.5 times prediction by development Ymean alone;
- median row L2 error / own128 row norm<=0.25;
- 95th row relative error<=0.50;
- F32-versus-F64 prediction relative L2<=1e-5 and all predictions finite.

These are local screening bounds chosen before map outcomes, not inherited whole
quality tolerances. EACH bank must pass. Failure closes this fixed rank32 interface
before new selector/model work; it does not close all learned transfer methods.
Pass licenses only actual selected128 function response and output-alignment
tests, not a384 model or capacity gain. Mean-only and identity controls retained.

Fresh exact SHA every paired trace and committed403 raw/controller/helper. No
source weight or native model inference needed. BLAS1, <=2min CPU /1GiB checkedRSS,
no network/GPU/T4/training of a language model. Twelve factors2,433,024 F32 bytes,
1,179,648 active coefficients per all12 banks (not measured engine cost). Bound
all bytes/NPZ hashes and first failure. Original output/function transformation
has not been tested. No accepted-rate/DRAM/other-family claims.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth404_switch_rank32_input_map.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth404_switch_rank32_input_map_result.json
```
