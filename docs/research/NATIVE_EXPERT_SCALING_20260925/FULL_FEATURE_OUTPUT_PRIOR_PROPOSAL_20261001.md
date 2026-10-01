# Proposal: source value/derivative priors in the complete nonlinear output basis

**Status: proposed; projection qualification and conditional fitting unexecuted.**
METH-234 qualifies all4864 original source features, row-Q8 projections
and SiLU lookup at8.522ms/24 layers with.01944% scoped source-function error.
It licenses function transfer in that shape,not expert-count usefulness.
METH-231's H2048/affine geometry remains rejected. Its repeated-input
diagnosis shows why data-only centered slope fitting can copy parent weights.

## Changed output-only prior mechanism

Freeze the complete METH-234 gate/up maps and shared lookup. For a center c,
define phi(c)=lookup(Gq c)*Uq c,H4864, and its piecewise analytic Jacobian
A=J_phi(c),shape4864x896. LUT slope within an interpolation interval is
16*(table[i+1]-table[i]); outside branches it is0 or1. Explicitly preserve
the rounded-position>=512 boundary. Qualify this full analytic derivative
against autograd of the actual row-scaled/LUT feature function.

Original source BF16-weight/FP32 FFN supplies f_source(c),J_source(c).
Let B0 be decoded row-Q8 source down weights initially; for eventual child
priors B0 may be the actually stored fitted parent readout. Define residual
R=J_source(c)-B0*A. Seek an output-only coefficient change satisfying
deltaB*A=R,with minimum Frobenius norm. If A is full column rank, thin QR
in FP64 A=Q*T gives

`deltaB = solve(T.transpose(), R.transpose()).transpose() * Q.transpose()`.

This transfers source slope through all existing nonlinear features;
there is no896x896 affine branch or larger active feature count. Rank,
conditioning, roundoff and coefficient growth may prevent useful priors;
the formula alone is not evidence that they succeed.

Encode B0+deltaB with the actual row-Q8 codec; choose FP32 bias to conserve
the source value at c under the **row-scale-after-reduction** operator.
Report both unrounded source-Jacobian reconstruction and stored Jacobian
distortion. Output quantization can change the continuous projected prior;
do not call the stored derivative exact or use BF16 coefficients while
pricing int8 traffic. No random/hash perturbation to force distinctness.

## Next exact prerequisite: METH-235

Freeze/implement a bounded layer12 projection qualification on all16
existing METH-227 parent centers,using no validation labels/new sources.
Bind METH-234 input matrices/table, METH-227 routes, actual original source
and METH-222 fit inputs. Check feature/autograd derivative, FP64 QR rank/
conditioning and unrounded source-value/gradient reconstruction; encode,
read back and evaluate every actual row-Q8 prior. Inspect fit-input source
function error and distinct coefficient hashes as controls. Fix numerical,
conditioning/coefficient-growth and prior-fidelity gates plus local budget
in the protocol **before** execution. No such projection currently exists.

A pass licenses a separately frozen full-feature anchored output fitting
comparison with E16/E160/rotated controls, using the same full-input keys
and actual BF16 donor-output fit targets. Child priors must carry source
information when support repeats; quantify distinct effective functions
and the physical bank, then validate complete function error and useful
count gain. Do not recycle the H2048 scores as evidence for this new basis.

One selected output function costs104.767MB across24 layers; shared gate/up
cost210.125MB and the single lookup2052bytes. Resident E16/E160/E1600 shapes
remain1.886/16.973/167.838GB before other organs/routes. Those banks do not
exist. Larger resident n requires actual native route/LUT/DRAM timing and
quality on useful learned functions; a component fixture cannot prove it.
Ultimately require full held-out prediction/generation/tasks and>=50
accepted tokens/s on the same stored artifact, plus a real second family/
approximately10B variant. GigaChat's donor-adaptation assets are reusable,
but sparse feature maps/top4/shared FFNs need separately priced transfer.
