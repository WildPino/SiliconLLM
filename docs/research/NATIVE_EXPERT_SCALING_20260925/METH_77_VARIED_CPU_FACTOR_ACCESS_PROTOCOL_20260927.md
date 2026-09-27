# METH-77: varied-token CPU factor access for learned E128/E1280

METH-76 measures exact learned component parity and hot repeated-input
cost; each loop revisits the same expert rows. Test whether the
E1280 factor bank remains affordable when real prompt positions
exercise a wider working set. METH-72/74 quality failures remain
binding, and this study is a performance diagnostic only.

Use the exact METH-71 update-64 E128/E1280 checkpoints and the same
METH-76 versioned native banks, verified by their SHA-256. Use all
24 frozen METH-72 external prompts. Select 11 evenly spaced
positions from each of the first 16 prompts and 10 from each of
the remaining 8, for 256 positions total. Capture each selected
pre-FFN BF16 hidden state at every one of 24 layers from the
actual respective E128 or E1280 model. Save a versioned,
token-major vector file with dimensions, source hashes and exact
byte length. The positions and all vector hashes are fixed before
the native timing run.

The single-thread C reader must verify file/bank dimensions and
score finite routes/residuals. On the 256 varied positions, report
per-layer unique selected experts and estimated unique selected
factor bytes. Run five timed repetitions of the complete 24-layer
route pass and a separate selected-factor residual pass, preserving
the same token order. Report each repetition and median, C process
RSS and checksums. Compare E1280/E128 route and residual medians;
do not replace the METH-76 parity fixtures or claim cold DRAM
latency from a finite repeated vector set. Report exact physical
bank bytes and nominal selected-factor bytes per token.

Local RTX 3060 for vector capture, ≤10.5 GiB peak allocated GPU,
≤20 GiB RSS and ≤15 minutes per export; native runs ≤2 minutes
per arm. No T4. This remains a component-only diagnostic, not
LUT-coded factors, `engine.c` integration or accepted token rate.
