# Complete source-feature and changing-decoder derivative audit

12 October2026. CPU1200s/OS6GiB,no child/Torch/GPU replay/model/new data.
Freeze completed report/terminal/binding and all consumed/output extents before/
after. All8808 actual source rows/all26424 decoder-control rows,not sampled.

Source BF16 head/gamma/multiplier independently read;source/current decoder
scores sum256-feature blocks and log partitions combine4096-word blocks by
logaddexp. Full q source moments use4096-vocabulary expectations. Decoder
feature gradients use direct (p-q) A expectations with those different sums,
not producer CUDA pA-m reuse. All complete losses/direct losses/IDs/entropy/
radius/weights/source-self fields verified at array relative1e-9/scalar1e-8.
Probability mass1e-11,source entropy/self1e-8;input/control shapes/dtypes exact.

Full2048-dimensional RMS-Jacobian action verified via independent eigenspaces:
unit=y/||y||,radial=unit*(unit^T grad_u),tangent=grad_u-radial;
grad_y=tangent/r+radial*epsilon/r³. This applies EVERY component and includes
radial epsilon contribution,without claiming dense Jacobian matrix allocation.
Zero y uses zero radial and identity/r tangent. No diagonal/identity RMS shortcut.
All54116352 feature and54116352 y-gradient words plus18038784 moment words checked.

Producer internal GPU compiled-versus-direct gradient comparison maxima are bound
frozen-code observations;original threshold flags rederived from those records,
not independently duplicated exact CUDA arithmetic. Both stored full gradients
and direct/compiled loss arrays are independently qualified against actual full
CPU source probabilities. No GPU replay/optimization/native quantized coefficient
admission or causal/future-state/model/useful-n/same useful artifact50 proof.
Full before/after hashes/terminal/PID/creation on failure;no partial deadline
admission. Complete source-function certificate then return to causal native pilot.
