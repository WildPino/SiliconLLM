# Frozen source-head2048 feature compilation and decoder controls

12 October2026. All4422 FIT/4386 DEV original source label states/full65537 words,
actual BF16 source head/gamma/LM multiplier/sourceepsilon1e-5. F64 operator q,
not new generation/cached BF16 logits. Require complete previous oracle audit.
Source-head moments m=A_source^T q,H(q),source IDs/weights stored. No optimization
or causal model/native quantized export. Fixed source-head column space only.

Compile GPUF64 deterministicTorch2.6,TF32 off,workspace4096:8,physical GPU0 UUID/
driver/runtime bytes bound,CPU6threads/affinity0..5,batch64. Validate source self:
logZ(Au)-m^T u-H(q)≈0 and qA-m≈0. Source entropy agrees with qualified old full
source q<=1e-8. Head/source matrix2048 shape is qualified by full CPU audit rather
than assuming previous257-column timing/precision applies.

Three REAL existing controls:original broad decoder,chosen coupled-correction
decoder,and final oracle source reconstruction Q*tF32+mu_perp. Each decoded y
uses actual source RMSr=sqrt(mean(y²)+1e-5),u=y/r. Compare compiled loss
logZ(Au)-m^T u-Hq with full q*(logq-logp);compare compiled gradient pA-m to
full (p-q)A. Store complete2048 gradients both in u and y;
grad_y=grad_u/r-y*(y^T grad_u)/(2048*r³),all radii/IDs/entropy/loss fields retained.
Frozen absolute loss1e-8,gradient1e-8,Frobenius relative norm(delta)/max(1,norm)
1e-9,self1e-8. Record false flags and complete all data before independent audit.
Control source-derived F64 probabilities differ from native quantized current C;
do not reuse their metrics as actual native quantized quality.

Capture900s/OS6GiB/GPUallocated+reserved2GiB/output2GiB. Source-moment payload
144,310,272B;all three full gradient controls explain additional outputs. Fresh
namespace/result/terminal PID/creation even on failure;complete before/after
consumed extents,known publishers untouched,timing admission=false. No children.
CPU audit1200s/OS6GiB:every moment/source probability/3 control loss+gradients/
explicit full RMS-Jacobian action/ID/radius/summary with different reductions.
No GPU replay/optimization/source forward/T4/RESERVED/longalign. Admission after
independent PASS only. No scalar synthetic matching tests as sole evidence.
Next concrete causal forward/backward/native gate and donor-relative pilot;
moment compilation is not the converter or n/DRAM/families/same useful artifact50.
