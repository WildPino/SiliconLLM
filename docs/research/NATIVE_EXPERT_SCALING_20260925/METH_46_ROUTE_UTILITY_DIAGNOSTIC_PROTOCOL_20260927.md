# METH-46: useful expert routing before another retention recipe

**Uncertainty.** METH-45's E128 Instruct continuation improves raw BPB
yet fails its chat retention stop at update 512. Before changing the loss,
measure whether the conditional router actually contributes to the raw
gain while the update-256 checkpoint still passes its chat development
gate. If the route itself is not useful, further tuning of this E128
adapter is unlikely to justify a much larger expert bank. This is a
development diagnostic on previously viewed data, not promotion.

Bind Instruct donor SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-45 run-result SHA-256
`a48deede189f1ff0e91bf2a0db9876fbd246fc53c24dc70010f935ba84a0d636`,
update-256/512 checkpoint SHA-256
`3d5835eae07384ed6709897212c9d0e31734403f049b0aa254622bafe0b57c4c` /
`c988d26fa0a122db292cd56b253317ae6c0c8ce5fff9c1aedaf004ca2b8fb6c2`,
the METH-15 heldout raw ID hash
`c5345782fc0b08ef3212403878a33960ae0da236c999272e79e418f29c277d3c`,
and METH-44's 24-prompt development manifest SHA-256
`703abca3f83a983d9380ba84c1a79744d15881d638174b064cf38a47453f47ba`.
Do not read the METH-45 frozen external set or PIQA outcomes.

For each saved checkpoint, load frozen BF16 Instruct donor plus its actual
E128/top-4/rank-8 factors. On the fixed raw heldout slice, measure intact
student BPB; then independently permute router rows in every layer using
NumPy seed 451616 without changing factors and measure BPB again. Restore
routers after the control. Report permuted-minus-intact BPB. Also tally
selected expert IDs per layer on the raw heldout inputs and the viewed
METH-44 chat prompts; report minimum distinct experts used, maximum
load-to-mean ratio and normalized load entropy. Tally exactly the same
top-4 rule used by the wrapper. Report output-factor norm distribution
to distinguish changed slots from substantial stored factors.

The decision threshold is route utility ≥+0.002 BPB at update 256, with
its already measured 95.298% chat-development agreement. If met, the
next retention experiment may keep this conditional geometry and focus
on controlling donor drift. If not met, deprioritize this residual
geometry as a large-E parent and revisit the transformation. Update-512
measurements are explanatory only because chat retention failed there.
Local RTX 3060 only; ≤5 minutes, ≤10.5 GiB allocated GPU, ≤20 GiB RSS.
Stop on any hash mismatch or budget breach. No T4 or fresh external
quality result is authorized by this diagnostic.
