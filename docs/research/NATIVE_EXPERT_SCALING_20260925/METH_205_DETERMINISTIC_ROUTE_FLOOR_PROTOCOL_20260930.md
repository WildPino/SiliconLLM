# METH-205: exact-state floor and soft/hard load mismatch

METH-204 fails deterministic content load after fitting shared keys and
soft-count parent biases. Before another route fit, distinguish an
input-identical-state lower bound from a calibration/argmax mismatch.
Bind its exact router artifact/result and repeat the same first 1,024
METH-175 raw/chat draws, E1280 model, rank-32 projected normalized FP32
states, structural bypass and eight-prompt BF16 parity. Reconcile the
replayed raw/chat content metrics with METH-204 before interpreting them.

For raw, chat and their pooled fit set, group each layer's content
occurrences by selected E1280 parent. For each parent with >=250
occurrences, count identical normalized FP32 32-vector byte strings.
Report the largest identical group / parent count, hard child maximum
share, and soft child maximum share at METH-204's saved temperature
0.05. Report all hot-parent rows and counts of parents above 25% for
each measure. No approximate similarity, source label, target or new
data enters the comparison.

If an identical-vector group exceeds 25%, any deterministic selector
whose entire input is this q-vector and parent ID must assign that
group to one child and cannot meet the existing hot-parent gate in
that cell. This is a bound for these exact projected inputs, not for
full hidden states, added context, stochastic or stateful routers.
If no raw/chat cell has such a floor, continue with a hard-load
parent-specific calibration candidate. Otherwise change the input or
treatment of recurrent content first. Soft shares diagnose the fit
surrogate; they do not license deployment or quality promotion.

Local RTX 3060, six host threads, <=20 minutes, <=20 GiB RSS,
<=10.5 GiB GPU allocation and <=1 GB result disk. Stop on binding,
parity, nonfinite, baseline-reconciliation or budget failure. Keep
failure records. No T4, new quality text or B training.
