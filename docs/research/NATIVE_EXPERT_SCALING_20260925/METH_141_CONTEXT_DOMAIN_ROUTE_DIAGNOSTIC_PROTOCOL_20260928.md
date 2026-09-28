# METH-141: isolate context length and source shift in quantile routing

METH-140 fitted child-specific scalar deciles on short raw/chat training
draws and failed route-load gates on separate document windows of up
to 512 tokens. Its sidecar remains rejected. This diagnostic holds
that exact sidecar fixed and varies only source and window length to
see which shift predicts the failure. It cannot promote METH-140.

Bind METH-140 sidecar SHA-256
`a581272ed84c154b660a4fd9a7c108f0dcced394e0d763ddfdd366aa92eadc5f`,
METH-126 bank, METH-107 centered child, and the same METH-136 training
draws by their hashes. Bind METH-121 and METH-133 manifests by the
hashes in METH-140's protocol. Use the unchanged centered E1280 donor
function and quantile clone factors; require exact grandchild-to-child
mapping and eight-prompt BF16 parity before reading load results.

Measure six cells with identical routing code:

- **Training source:** the 256 pinned H0 raw rows selected by the
  METH-136 draws. Run either each draw's original 128-token window or
  its full 512-token row, resetting context per window.
- **METH-121 and METH-133 documents separately:** run all 24 fixed
  document ID lists in nonoverlapping windows of at most 128 or 512
  tokens, resetting context per window.

For each cell and layer, save token count, selected-child and
grandchild coverage, maximum/mean load, candidate/control ratio,
top five slots and worst grandchild share among parents selected at
least 250 times. For short final windows, include every token; do not
pad or drop content. The H0 short cell repeats calibration inputs and
is a positive control, not held-out evidence. The four external
cells were already viewed for routing in METH-140. Interpret a gap
between 128 and 512 within a source as context-length sensitivity,
and a gap between sources at the same length as source shift. Do not
change thresholds, gates or artifacts based on this diagnostic.

Run on the local RTX 3060 only. Stop at 15 minutes, 10.5 GiB peak
allocated GPU memory, 20 GiB process RSS or 1 GB new disk. No T4 and
no new model-quality claims.
