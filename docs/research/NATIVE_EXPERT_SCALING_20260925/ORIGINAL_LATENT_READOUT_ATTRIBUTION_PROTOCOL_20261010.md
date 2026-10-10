# Stored target/readout attribution: executable control protocol

10 October2026. PROSPECTIVE, UNEXECUTED. Implements the already proposed
[direction](ORIGINAL_LATENT_READOUT_ATTRIBUTION_NEXT_20261010.md) only after
independent actual51 adjudication and its held receipt are complete.

## Uncertainty and decision

The normalized fixed projected donor boundaries did not recover varying
trajectory or chatbot tasks with auxiliary weight1. Is the final projected
target even compatible with the actual learned decoder? Keep the same24 DEV
histories/4386 labels, source h24P projection and actual A51/B51 packed heads.
No fit, map search, new labels, source/student histories or native execution.

For each already labelled position, compute in F64:

    z = retained h24P
    a = z / sqrt(mean(z*z) + 1e-5) * actual51 final_norm
    logits = a @ actual51 head.T

Compare the complete65537-way distribution to the retained BF16 donor logits
and the final native metrics. Case and label weighting/domain stratification
remain separate. Measure KL, argmax disagreement, donor entropy/uniform KL.
Store all new logits losslessly in F64. Stable max-subtract log-sum-exp and
independent logaddexp reductions must agree per label within1e-10 absolute.
Save one durable case result per arm. No decoder fitting or DEV selection.

Interpretation fixed by the prior proposal:

- BOTH injected caseKL<=0.50*native and disagreement<=0.80*native:
  existing head can read this target substantially better; prioritize reaching
  compatible source-history functions with the internal path.
- BOTH injected caseKL>=0.80*native and fail absolute/domain screens:
  investigate target/decoder coupling, distinguishing coordinate mismatch from
  rank loss before a width or conditional-readout change.
- Otherwise preserve mixed/domain outcomes; no unique bottleneck identified.

These diagnostic branches do not qualify a chatbot or claim a capacity ceiling.
This control is not a learned optimum, causal history recovery, C precision,
own-history behavior, useful speed, large-n scaling or DRAM measurement.
Original quality gates and all failures are retained.

## Inputs and custody

Require completed actual51 result/terminal and missing-only independent audit
result/held receipt, all matched by raw SHA. Bind original history binding,
all24 final target arrays/full-V BF16 labels, both final packs, relevant code,
Python and NumPy native runtime, this protocol and prior proposal.
Windows peak-memory helper code, psutil native extension and Python runtime DLL
are included prospectively in the same binding before the first contraction.
The fixed projection, target record IDs/positions and full head fields are not swept.
Hash consumed inputs before/after with held launcher. Full history arrays are
read only to index existing labels; no model is instantiated.

## Storage correction and finite cost before any contraction

The earlier proposal priced~2.3GB score output, which corresponds to F32.
Lossless F64 output for2*4386*65537 entries is4,599,124,512B (~4.28GiB).
Use a prospective5GiB output cap so the independent stored audit can reconstruct
the exact control without F32 rounding. This changes storage only; there are
still exactly two heads,24 DEV cases per head and the same interpretation.
No candidate result was observed when correcting this arithmetic.

CPU six threads, no GPU/optimizer/native/source calls or other owned benchmark.
Expected2–5min compute plus input/output IO; finite900s held cap,4GiB OS union,
5GiB outputs,8MiB log. Use bounded label chunks, release maps per case and save
F64 scores incrementally. Principal BLAS native files are explicitly bound;
no retroactive closure of previous full-runtime gaps.

Stop on deadline, nonfinite arrays, size/hash mismatch, precision disagreement,
unqualified parent, unexpected subprocess or output cap. Preserve first fault
and completed cases; a NEW missing-only continuation may adopt durable cases.
Independent stored attribution audit recomputes metrics from saved F64 logits
and confirms decision arithmetic; it is outside the held contraction family.
Hold that audit separately through exit, cap600s/4GiB OS/8MiB log/2MiB output,
no model or head contractions. Bind audit code/result/binding before/after;
its resource receipt does not retroactively qualify any earlier family.
No T4 or new runtime engine operator is introduced by this control.
