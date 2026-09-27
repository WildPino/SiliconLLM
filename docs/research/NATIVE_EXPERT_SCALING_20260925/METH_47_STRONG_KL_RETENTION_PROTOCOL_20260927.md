# METH-47: stronger donor trust region for useful E128 routing

**Uncertainty.** METH-45's fixed full-chat recipe learns useful
conditional routing at update 256 (METH-46 permuted-minus-intact
+0.020330 BPB) but loses chat donor agreement by update 512 while raw
BPB improves. Test whether a stronger teacher KL constraint and a
smaller update step keep the same conditional geometry useful without
chat drift. This is one prespecified recipe, not a parameter sweep.
Failure closes this objective/step size combination, not conditional
pretrained transfer generally.

Bind the exact METH-44 checkpoint SHA-256
`4fade15804f5e2297ff60cafacf870e5ae11926ba886786a524110dea94f16da`,
Instruct weights SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-43 teacher result SHA-256
`f42f678280a9086a0fc2ec6e57a712e33fdea8f5cb62bdac7f53c9f28a5f1dd7`,
and original h0 raw train IDs SHA-256
`9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da`.
The [new 24-prompt development manifest](meth47_retention_dev_manifest.json),
SHA-256 `0195852ce7a46af5a2d9442b38e687a19393a943168b61a8f8cdb0c66ec78a60`,
uses NumPy seed 474747 and excludes all 256 METH-43 chat-training rows
and all 24 METH-44 development rows. Exclude both development sets from
raw training samples. This same-corpus set is a stop screen, not an
external promotion result. Do not view METH-45's frozen external results
unless the METH-47 terminal gate passes.

**Single continuation.** Resume all 24 E128/top-4/rank-8 layers from
METH-44 update 16, including AdamW moments and RNG states. Set factor
LR `1e-4` and router LR `1e-5` from update 17 onward, weight decay zero,
clip norm 1.0. Keep two independently sampled raw 128-ID windows and
two saved full-prompt teacher-chat rows per update, with the same sampling
RNG stream and teacher responses. Raw objective becomes student CE plus
`2.0 ×` frozen donor→student KL across all 127 targets. Chat objective
is assistant-response CE plus `4.0 ×` donor→student KL across every
full-sequence next-token position. Divide the four microbatch objectives
by four. The frozen donor runs with experts disabled under `no_grad`.
No base or donor parameter may update.

Before training verify the resumed raw BPB and METH-44 checkpoint hashes,
and require ≥95% donor top-1 agreement on the new development prompts.
Train to update 512, saving optimizer/RNG checkpoints and scoring only
the fixed raw heldout slice and new development prompts at 256 and 512.
Stop if raw student-minus-donor BPB exceeds +0.05, new-development
top-1 is <95%, loss/gradients become nonfinite or a resource bound is
breached. At update 512 also require ≥64 changed output slots per layer
and permuted-router-minus-intact raw BPB ≥+0.002 under the METH-46
independent-row permutation with seed 451616. No earlier checkpoint
may be chosen after seeing a later failure. Local RTX 3060 only;
≤20 minutes, ≤10.5 GiB allocated GPU, ≤20 GiB RSS. No T4.

Passing this bounded continuation licenses the already frozen METH-45
external document/chat manifest and full PIQA paired audit on this
new candidate. That audit must apply its same document, prompt,
generation and task gates before packed export. A pass here does not
prove independent large-E capacity, bounded CPU routing or ≥50 accepted
tok/s in `benchmarks/phase60/engine.c`.
