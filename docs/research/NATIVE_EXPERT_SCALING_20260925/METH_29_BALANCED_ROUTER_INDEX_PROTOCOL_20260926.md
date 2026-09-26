# METH-29: balanced coarse index for the trained E128 router

METH-26 shows an exhaustive one-byte router would read 5.882 GB/token
at the hypothetical E273,547/100B point in the current Qwen L24/D896
geometry. NES-03's int8 shortlist still scans all rows. This cell
tests a different changed coordinate: **post-hoc balanced grouping**
of the existing E128 trained router rows, followed by coarse group
scoring and exact rescoring of a bounded candidate set. It asks if
the actual trained routes have enough geometric structure for that
index to retain the exact top-4. No E1280/100B learned model, C
kernel, language-quality change or CPU speed result is claimed here.

Bind the Qwen2.5-0.5B source SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, packed R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`,
and factor-0.50 E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
Reconstruct the 24-prompt manifest from METH-27, SHA-256
`94abd7544d4f887e476286774b2ce51bbc0c5c68f079907b88b3984589d62ec6`.
Those prompts have been used for generation and top-1 screening, so
this is a **routing diagnostic**, not an independent quality gate.

For each of the 24 layer routers, recursively split its 128 fp32
rows into two equal halves by the leading right-singular direction
of the centered rows, with expert ID as deterministic tie-break.
Continue to 16 leaves of eight expert IDs. Store every leaf assignment
and its fp32 mean row. This uses only frozen router weights, no prompt
activations, labels or expert outputs to fit the index. Score all 16
means for an input. Two frozen arms choose the best **4 leaves (32
candidate rows)** or **8 leaves (64 candidate rows)** by mean dot
product; rescore only those candidates with the original fp32 row
weights and return top-4 among them. Compare against exhaustive
fp32 top-4 from the same hidden input at every layer, before any
route replacement. Keep the model on its exact original route so
later layer inputs are the true composition trajectory.

Capture all 256-token positions of 8 code, 8 prose and 8 technical
prompts, 6,144 positions × 24 layers = 147,456 input-layer cases.
Report exact top-4 ID inclusion, full-set match, gate-probability mass
of missed exact experts, per-layer/category and pooled counts,
route-stream SHA-256, grouping IDs and matrix row-dot equivalents.
The diagnostic gate for a candidate budget passes only if all three
hold: **≥99.9% exact top-4 ID inclusion**, **≥99% full top-4 set
match**, and **≥99% ID inclusion in each category**. These are
apparatus/route gates, not language-quality gates. A pass permits
quality testing with route replacement and a CPU implementation;
a fail rejects this *post-hoc grouping rule* and motivates a jointly
trained hierarchy or another index, without refuting sublinear
routing generally. Report 16+32=48 or 16+64=80 row-dot equivalents
versus 128 exhaustive; no CPU time or extrapolated 100B rate is
allowed from GPU timing.

One local RTX 3060 diagnostic; stop at 20 minutes wall time,
10.5 GiB allocated GPU memory or 20 GiB RSS. No T4 requested.
