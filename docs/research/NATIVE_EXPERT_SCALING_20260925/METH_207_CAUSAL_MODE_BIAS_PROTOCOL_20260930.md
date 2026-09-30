# METH-207: two bias banks selected by a causal prefix mode

METH-206 annealing fixes pooled hard load but separate raw/chat fit
gates still fail. Test whether separately calibrated biases close
that gap without changing the saved shared keys, projection, inputs,
structural path, temperature schedule or five per-cell gates.

A bounded fit-only prefix audit found all 1,024 METH-175 chat draws
start with token 151644; zero of the 1,024 raw draws contains it.
This motivates the frozen serving rule: initialize mode from the
observed first token, with mode 1 iff it equals 151644, else mode 0.
Persist the mode through the sequence, including independent model
windows. Later ChatML markers do not change it. No source, dataset,
future token, target or held-out metric selects the serving bias.
An input continuation needs its prefix mode supplied as persistent
state; without that state the rule initializes from its first token.
This is a Qwen ChatML prerequisite, not a universal family rule.

Use the METH-204 collection and teacher/control eight-prompt BF16
parity apparatus. Assert the fit mode counts above and replay all
saved METH-204 baseline metrics to 1e-6 per layer. Freeze its keys
and initialize both bias banks from its original parent biases.
Apply 100 soft-count updates independently in each fit mode at
temperatures 0.05, 0.01, 0.002, 0.0004, with the METH-206 update
and centering. Save every calibration-stage hard-load summary.
Export FP32 projection, shared keys and 24x2x1280x9 biases, then
verify exact readback and unchanged projection/keys. Mode selection
is derived per token from the observed leading prefix before route
evaluation, independently of cell names. Log mode counts in every cell.

Retain all original gates in every evaluated cell/layer: load ratio
<=1.25, hot-parent child share <=25% for parents with >=250 selections,
coverage >=4,000, selected standardized content-score advantage
>=0.05 and unbiased argmax agreement >=15%. Stop before the 256
reserved draws if either fit cell fails; stop before the two METH-150
source-document width cells (128/512) if either reserved cell fails.
Only all six passing cells license native CPU route cost, followed
by matched specialist training and untouched model quality.

The doubled bias store is 2,211,840 bytes and 22,118,400 bytes at
tenfold parents. One selected mode reads the same ideal addressed
projection/key/four-parent-bias payload as METH-204: 2,783,616
bytes/token. These are layout counts, not measured DRAM traffic
or native routing time. No claim of useful trained specialists.

Local RTX 3060, six host threads, <=30 minutes, <=20 GiB RSS,
<=10.5 GiB allocated GPU memory, <=1 GB result/router. Stop on
binding/parity/reconciliation/nonfinite/readback/budget failure.
Preserve failure and apparatus outputs. No T4 or B training.

Runner: `benchmarks/native_expert_scaling/meth207_causal_mode_bias.py`.
Bind METH-204 router/result and METH-206 result hashes in code;
all existing model/draw/table/manifest bindings remain unchanged.
