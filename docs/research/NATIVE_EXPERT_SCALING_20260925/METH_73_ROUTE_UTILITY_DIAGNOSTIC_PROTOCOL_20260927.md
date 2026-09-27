# METH-73: diagnose learned route utility after METH-72's failed gate

METH-72's E1280 automatic audit passes document, prompt, generation,
PIQA and changed-slot criteria but fails its fixed pair-utility gate:
one factor-pair permutation worsens four-window raw BPB by +0.001743,
below +0.002. Its `stop_instruct_parent_promotion` decision stands.

Use the exact METH-71 E1280 checkpoint, donor source and METH-72
24-source external manifest already bound by SHA-256. First score the
trained factors on all 24 documents. Then, with product keys fixed,
permute A and B together independently within each layer using eight
preselected NumPy seeds 730000 through 730007. Restore the original
factors after each permutation. Record pooled and code/prose/technical
BPB for every permutation, its difference from trained BPB, median,
range, and the count of positive differences. Also repeat the exact
METH-72 four-window raw held-out measure with those eight seeds.
Require exact checkpoint and manifest hashes, exact restoration after
every permutation, and finite scores. This is a **diagnostic** about
signal strength and sample variability; it cannot replace the failed
METH-72 single-permutation gate or establish semantic quality.

Local RTX 3060 only, GPU peak ≤10.5 GiB, RSS ≤20 GiB and ≤20 minutes.
Stop with partial results on a budget or nonfinite failure. No T4.
