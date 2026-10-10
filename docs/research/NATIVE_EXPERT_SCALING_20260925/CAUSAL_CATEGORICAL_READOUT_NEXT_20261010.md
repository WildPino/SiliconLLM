# Next executable converter branch: shared categorical native readout

10 October2026. Prospective implementation; no new head fit/optimizer/native
history yet. Reuse [qualified approximate native coordinates](CAUSAL_READOUT_COORDINATES_RESULT_20261010.md)
and frozen new paired A=aK. Head-image72 probe exposes feasible code room;
actual27 old causal states and output coordinate system differ from source phi.

## One convex problem that changes an original C artifact

For fixed native normalized features f_j, source teacher q_j, A65537x255:
p_J=softmax(AJ f_j), J255x256. Categorical KL is convex in J. Teacher dependence
is only m_j=A^T q_j and entropy constant c_j=sum q_j log q_j. Gradient is
sum_j w_j[A^T p_j-m_j] f_j^T. No source/core/expert/route weights updated here.
Fit only24 FIT/4422 labels, equal case weights w_j=1/(24 m_case). DEV never
selects head/gain/optimizer/initialization. Use every existing label, not just72
optimized oracle codes. Old decoded features are approximate; propagate their
per-label uncertainty to the learned head and record whitening amplification.

Numerically whiten FIT noncentered C=sum w f f^T. Eig support >=1e-12 lambda_max,
zero/discarded energy bounded and complete numeric qualification before descent.
With W_r=C_support^-1/2 and t_j=W_r f_j, E[t t^T]=I_r. Optimize Theta255xr,
J=Theta W_r. FIT least-squares warm Theta0=sum w (phi_j/a) t_j^T reuses already
paired full FIT source coordinates, not a new head/encoder sweep. Its Frobenius
norm <= RMS(phi/a)<=8 in exact arithmetic; this gives a principled warm point
inside a prospective Frobenius radius16. This radius is a NEW shared-parameter
domain, not the old radius16 per-label code domain. Do not conflate the bounds.

Proposed one32-update full-gradient accelerated projected descent with
backtracking (project Theta to Frobenius ball16), checkpoints/best feasible
upper and actual tangent lower point. For ANY tangent point Y, lower=
max(0,F(Y)-<g,Y>-16||g||_F). Extrapolated Y may lie outside the parameter ball;
only upper points require feasibility. Preserve aggregate and every case/domain
loss; a global lower concerns the shared bounded map, not all causal encoders.
Freeze exact extrapolation, L/backtracking/gap tolerances, counts, gates and caps
with implementation before observing objective values. No extra initialization,
scale arm, optimizer restart or indefinite unchanged dose.

## Cost and lifecycle to freeze before launch

Previous72-label fixed-head probe2688 objective/gradient evaluations in153s
implies roughly3.5s per4422-label full evaluation (forecast, not guarantee).
32 accelerated updates with roughly5 backtracks each could cost~10min plus
teacher-moment/initialization/hash work. Proposed1200s/120s reserve, local GPU
allocated4GiB/reserved5GiB,OS4GiB,chunk64/128 labels and output<=1GiB; complete
stored audit600s. Refine actual caps once code/bytes/gradient instrumented.
No T4 required for this branch. Teacher full-V probabilities/moments training-only.
Count actual full/partial evaluations and every label/matrix pass, not32 as if
only32 objective calls. Durable state/checkpoints before later assessment.
Mechanical failures preserve partial gradients/actual coordinates/counters;
repair code/namespace without completed update replay. Complete audit on failure.

## Offline fusion and final authority

Export H_new=A Theta W_r as ordinary F32 Vx256 head in a NEW original packed
artifact. Old embedding/core/SSM/SWA/ternary functions/router/final norm fields
remain exactly fixed and separately checked. All multiplication by Theta/W_r is
fused OFFLINE: no added runtime matrix/inversion/carrier or donor read. One fitted
head changes effective code domain; no old fixed-head floor can be reused blindly.

Independent audit must verify moment/entropy inputs, covariance/eigensystem/
whitening/LS initializer, analytic gradient witnesses, stored bound points,
all shared-map scores/losses/decisions/checkpoints/export fields/hashes/resources.
It must not replay descent or history. Then run unchanged packed C consumer on
this new artifact for FIT/DEV and own-history generation/tasks. Full numerical
comparison with proxy and rounding sensitivity required, not inferred from old
heads. Routes/core trajectories under fixed teacher inputs should remain identical;
new head has no feedback into those forced prefixes. If uncertainty amplification
matters, use a qualified native normalized-state observer in the next new
assessment rather than pretending the reconstructed state was exact.

Proposed native full48 forced histories cap900s/output8GiB plus600s complete
stored audit, exact output bytes and prior native cost to be resolved before
launch. Generative task cost/reuse/profile separately frozen, no quality-triggered
audit omission. Native timings on incorrect answers remain raw rates, not50
accepted tokens/s. Quality and speed must ultimately pass on this SAME artefact.

The final objective remains useful pretrained chatbot transfer to original
conditional capacity, useful n/CPU ID+mass/physical DRAM and multiple families.
A readout-only improvement does not substitute for learning the causal/functions
path when those features are inadequate. Result determines the next correction.
