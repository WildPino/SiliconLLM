# METH-346: same HEAD A16 executable complete CPU cost PASS

Freeze0bcdfbc; authoritative exec64814 exit0. Raw SHA
`1d32e982f8392632f4e1745f3fec91b3cb59884aa3f679a085a5307eb1432fcf`.
MAIN37.297s/max checked combined RSS927,735,808B. ALL five unchanged341
cost/identity gates PASS; fresh target/manifest/runtime/binary hashes exact.
EXACT347 executable, no recompile; all eight engineering and four long
bridges match347. All warm/primary/profile outputs repeat EXACT347.

| Source tokens | Threads | Median decode ms / forced position | Full incl prefill ms / position | Decode max/min |
| --- | --- | --- | --- | --- |
| 9 | 6 PRIMARY | 6.426 | 7.452 | 1.0174 |
| 64 | 6 PRIMARY | 7.207 | 15.372 | 1.0037 |
| 9 | 1 comparison | 8.459 | 10.181 | 1.0106 |
| 64 | 1 comparison | 9.278 | 23.417 | 1.3063 |

Same32 forced positions, one warmup/three measured repeats/fixed order;
head logical weight reads unchanged, total129,017,344 matrix B/position.
Full encoder/crossKV/router/expert/head cost charged.64-source PRIMARY
prefill is about261ms (full minus decode); per-position figure amortizes over32
positions. It does NOT prove >=50 end-to-end on shorter naturally generated
responses. Filesystem/RAM primed by hashes/qualification, actual route unions
retained; logical addressed bytes are not measured physical DRAM. No causal
head speedup claim against unpaired341 or cold-cache/n-scaling claim.

PASS licenses NEW original-primary prediction on same profile. Keep343 W8A8
quality FAIL and347 consumed composition separate. Free generation/task/
accepted end-to-end throughput (including prefill), useful larger-n/LUT and
cross-family/~100B still open. Short reconstruction may need further exact
prefill optimization if actual generation exposes the amortization bottleneck.
Reproduction in [346 protocol](METH_346_SWITCH_HEAD_A16_COST_PROTOCOL_20261003.md).
