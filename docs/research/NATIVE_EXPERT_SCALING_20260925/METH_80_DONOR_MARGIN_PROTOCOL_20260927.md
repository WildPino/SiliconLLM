# METH-80: donor-margin retention during E128/E1280 continuation

**Uncertainty and prior evidence.** METH-79's removal of the large
axis-balance router gradient modestly improves donor-position top-1,
yet E1280 still falls 1.548 points over updates 65–256. The final
no-balance total gradient is below the norm-1 clip, so continued
CE/KL adaptation can move teacher decisions even without balance.
Test one fixed decision-retention term on confident teacher positions.
This targets donor fidelity while the independent factor bank keeps
training; it does not presume semantic or route-utility success.

**Matched arms and sole changed variable.** Resume the exact METH-71
update-64 E128 (8×16) and E1280 (32×40) checkpoints, optimizer and
RNG states. Run updates 65–256 on the exact METH-75/79 training draws
with the METH-79 zero-balance rule, original CE, raw KL weight 2,
chat KL weight 4, factor LR 1e-4, router LR 1e-5 and norm-1 clip.
Add the following margin term to each raw and full-chat microbatch.
No other model, routing, precision or optimizer change is allowed.
The fixed METH-79 update-256 checkpoints are comparators on the same
new prompts, alongside the update-64 parent.

For each teacher logit vector, find its largest and second-largest
logits and keep the position when their difference is at least 0.2.
For each kept position, find the largest student logit **other than**
the teacher's top token. Apply
`relu(student_other_max - student_teacher_top + 0.2)`. Average over
kept positions, with zero loss if there are none. Add this mean with
coefficient 2 on raw microbatches and 4 on full-chat microbatches.
The 0.2 confidence/target margins restrict the intervention to donor
decisions with a visible logit gap. Report selected/violating position
counts, unweighted margin loss, CE/KL and clip norm for every update.
The teacher logits are detached; only expert factors and product keys
train. This term is applied before the common four-microbatch average.

**Fresh evidence and gates.** Freeze 24 seed-808080 prompts before
training, excluding all previous development sets through METH-79,
METH-70 sampled raw rows and every METH-71/75/79/80 raw draw through
update 256. Both E arms, fixed METH-79 comparators and update-64
parents use the same new prompts. At update 256 each candidate must
reach ≥95% donor top-1, decline ≤1.0 percentage point versus its
own update-64 parent, raw held-out ΔBPB≤+0.05, minimum changed B
slots/layer ≥64 at E128 or ≥640 at E1280, exact product-key top-four
versus exhaustive pair scores on evaluated prompts, minimum selected
slots/layer ≥96/640, and worst maximum/mean load ≤32×/64×.
Require E1280 top-1 ≥ the METH-79 E1280 comparator plus 0.5 point,
and E1280 no more than 0.5 point below the new E128 candidate.
For the E128 control, require no worse than its METH-79 comparator
by more than 0.5 point. Report all values regardless of pass/fail.

Passing would only permit a newly frozen source-disjoint automatic,
task, route-pair-utility and arm-blind semantic audit. The earlier
METH-72/74 failures remain historical. This experiment cannot prove
full native `engine.c` parity, CPU LUT cost or ≥50 accepted tok/s.
Local RTX 3060 only; stop at 10.5 GiB peak allocated GPU, 20 GiB RSS,
15 minutes per arm including checkpoint save/hash, or nonfinite loss.
Preserve a failed outcome. No T4.
