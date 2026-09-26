# METH-19: independent document gate for the half-amplitude E128 bank

**Prospective document audit.** METH-18 found that multiplying the
trained E128 residual output factors by 0.50 reduced METH-17's code
loss, but the 60 diagnostic documents influenced that choice. METH-19
tests the fixed 0.50 candidate on a newly selected, disjoint set.
This is a document gate only; even a pass does not establish useful
generation, task quality, low-bit core quality or native C speed.

## Frozen selection and source binding

The [selector/scorer](../../../benchmarks/donor_adaptation/s1/meth19_half_residual_independent_audit.py)
ran once in `--prepare` mode without loading a model or scoring text.
Its [manifest](meth19_half_residual_independent_manifest.json) has
SHA-256
`1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70`.
It contains **56 documents and 229,316 selected UTF-8 bytes**:
24 code, 24 prose, 8 general technical. Deterministic source rank
seed is `meth19-19019`.

- Code spans come from the same pre-METH-15 Git commit
  `45afab6847cfd328c105f73256441f36748dc323`, but exclude every
  source file selected for METH-17. All 50 remaining eligible files
  passed the tested overlap screen; 24 were selected by rank.
- Prose and technical spans come from
  `strat01_gigachat_fresh_v2/heldout.jsonl`, SHA-256
  `04687034c1054e0985e24efa87ba37a3fb11031de5b2880e5b8ac1742f80ec6e`.
  Of 32 candidates per category, 31 prose and 8 technical passed
  the screen; 24 and 8 respectively were selected. No external
  source ID overlaps METH-17's selected IDs.
- The screen checks the first, middle and final 256 UTF-8 bytes of
  each candidate span against both prior Qwen corpus files and all
  60 METH-17 selected text spans. The METH-17 selector and manifest
  are reconstructed and hash checked before preparing or scoring.
  The screen does not prove absence of shorter overlap, paraphrase,
  or pretrained-donor exposure. The external source was collected
  for a separate sparse-donor line, not used to train METH-15/16.

The Qwen calibration/heldout SHA-256 values and METH-17 manifest hash
are pinned in the runner. The donor is Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, safetensors SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The tokenizer fingerprint is `4efeeb9382a77a06`. The METH-16
update-1024 E128/top-4 checkpoint SHA-256 is
`a9e74c9ceeb8a91fd04f9b9e35a981965db9ab824d2d25ea728dd940ea5365a4`.

## Fixed arms, scoring and decision

For each document, score the same model with residual experts
disabled (donor), output factors multiplied by **0.50**, unchanged
output factors **1.00**, and 0.50 factors with independently
permuted router rows per layer (NumPy seed 1919). The final arm is
a destructive route null, not a deployment candidate. Other weights,
IDs, and scoring conditions are identical. Use RTX 3060 BF16/SDPA,
donor EOS prefix, 512-token target strides, up to 512 preceding
tokens, absolute document positions, and score every text token.
Pool paired nats over UTF-8 bytes for BPB. Report every document,
category and pooled donor/half/full/permuted loss.

Bootstrap documents within each category, retaining 24/24/8 counts,
20,000 draws with seed 191919. The one-sided 95% upper bound is the
95th percentile of paired half-minus-donor BPB. The document gate
passes only if all these prospective conditions hold:

- Pooled half-minus-donor point delta ≤ **+0.01 BPB** and one-sided
  95% upper bound ≤ **+0.02 BPB**.
- Code half-minus-donor delta ≤ **+0.01 BPB**; prose and technical
  deltas each ≤ **+0.03 BPB**.
- Pooled permuted-minus-trained half-route gap ≥ **+0.002 BPB**,
  showing route/expert correspondence on this held-out set.

If the document gate fails, do not export this fixed candidate;
diagnose or train a distinct repair using data outside this audit.
If it passes, proceed to held-out generation and tasks, then seek a
quality-preserving low-bit shared core and exact native export of the
same candidate before any joint quality/rate claim. No scalar may be
retuned on these 56 documents and re-tested here as if fresh.

One local RTX 3060 run; stop at 15 minutes wall time, 10.5 GiB peak
allocated GPU memory or 20 GiB process RSS. No T4 requested.
