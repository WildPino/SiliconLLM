# METH-138: diagnose the failed METH-136 E12800 route-load gate

METH-136 trained both matched arms for 256 updates but stopped at its
frozen E12800 maximum/mean route-load limit of 50. Its final progress
record kept coverage but not the route histogram, so the exact
train-time maximum/mean value is unavailable. The 256-update continued
E1280 control recorded a maximum of 56.417 in layer 17 and 55.39 in
layer 1. The E12800 artifact was fully written before the failed gate.
METH-136 remains rejected regardless of this diagnostic.

Bind the original METH-126 exact bank and METH-135 third-tier sidecar
by their METH-136 hashes, the exported control bank SHA-256
`d5d9e6b3753ab55ecbb20a8f60d1ba11a6d9bdd41f66bb34909207b9a29ba299`
and candidate bank SHA-256
`61570f9efb291fcd43e5bb83c4f0ac18b7fbeb8e44e4ce80791efbd80b56808a`.
Verify binary headers and sizes. Recompute BF16-distinct rows against
the original source child and the candidate's post-rounding sibling
mean error. Bind the partial report and final progress by their hashes
in the result.

Replay all 256 METH-136 raw/chat input draws on each **final exported**
bank, with the donor and all routing parameters unchanged. Record every
layer's coverage, maximum/mean route load, and top five hot slot IDs;
also aggregate the E12800 counts to their 1,280 source-child IDs.
Compare these final-state histograms with each other and with the
continued-control train-time skew. This replay is a diagnostic proxy:
METH-136 accumulated its route counts while B changed at every update,
so do not label replayed values as the exact failed-gate value or use
them to retroactively pass METH-136. Record elapsed time and transfer
volume. The result may guide a separately frozen router redesign.

Use only the local RTX 3060. Stop at 20 minutes, 10.5 GiB allocated
GPU memory or 30 GiB process RSS. No T4 and no quality inference.
