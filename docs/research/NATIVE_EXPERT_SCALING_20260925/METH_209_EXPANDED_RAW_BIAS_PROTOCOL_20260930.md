# METH-209: triple independent raw bias-calibration support

METH-208's conditional empirical fit bootstrap places the six
METH-207 reserved-raw failures beyond its 99th failure-count percentile.
It does not include bias-estimation uncertainty. Test a fixed increase
in raw calibration support; do not tune keys, thresholds or annealing.

Retain METH-207 projection and shared keys byte-identically. Its
ChatML bias already passes fit and old reserved cells and stays
byte-identical; reuse those two recorded cell results by artifact
identity, explicitly marking them as reused evidence. The underlying
control model/states do not change in this route-only screen.
The diagnostic maximum-bias field covers both stored banks and is
recomputed for the new artifact; reused routing metrics/gates are unchanged.

The raw fit consists of original METH-175 draw positions 0:1024
followed by 1280:3328, totaling 3,072 source-distinct sequences.
Exclude all old reserved positions 1024:1280 (256) and new reserved
positions 3328:3840 (512). Assert complete disjoint partition.
Every draw remains bound to the original manifest and SHA. The new
512 pairs are unopened for this calibration path, previously used
in METH-175 training; they are not fresh model-quality data.

Reconcile METH-207 raw metrics on the original first 1,024 fit
sequences. Initialize raw biases from METH-204, exactly as METH-207,
and apply the same 100 updates at each temperature .05/.01/.002/.0004.
No other fit variable changes. Save every calibration-stage summary,
export both bias banks, and verify exact readback and ChatML identity.
Retain causal leading-token mode, structural path, width 512 and
eight-prompt BF16 teacher/control parity. Reconcile the old raw
reserved metrics with the prior bank before evaluating the new one.

All original per-cell/layer gates remain: max-load <=1.25, >=250
hot-parent share <=25%, coverage >=4,000, standardized score advantage
>=0.05 and unbiased argmax agreement >=15%. Stop before old reserved
if fit fails, before new reserved if either old reserved cell fails,
and before source if either new reserved cell fails. Then evaluate
the same 24 METH-150 source documents at widths 128 and 512. Every
one of the eight cells, including reused old chat cells, must pass
before native route cost or specialist training is licensed. No
failure is reclassified by a larger pooled sample.

The router store and ideal addressed bytes are unchanged from
METH-207. Report actual capture/calibration resources separately.
Local RTX 3060, six host threads, <=30 minutes, <=20 GiB RSS,
<=10.5 GiB allocated GPU, <1 GB combined result/router. Stop on
binding/parity/reconciliation/nonfinite/readback/budget failure.
Preserve failure records. No T4, new B training or new quality text.

Runner: `benchmarks/native_expert_scaling/meth209_expanded_raw_bias.py`.
