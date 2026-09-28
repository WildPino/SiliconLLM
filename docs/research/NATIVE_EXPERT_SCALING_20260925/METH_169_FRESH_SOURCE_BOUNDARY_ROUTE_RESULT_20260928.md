# METH-169: fresh-source boundary route fails the composed hot-parent gate

**Decision:** reject the fixed METH-168 route for the proposed long matched E1,280/E12,800 training. The genuinely unused PG19 shard-11 source passes all four individual cells, but composition with the 3,840 training pairs crosses the frozen >=250-selection hot-parent boundary and exceeds its 25% grandchild-share cap. No matched learning, native METH-170 timing or reserved shard-9 quality evaluation follows this failed candidate.

The manifest SHA-256 is `8b2e3a545f33dbd31283fd173b9044e2c87bd55463e168d6b2f8c513cde4e468`: 512 chat rows, 512 raw rows and 24 distinct document rows. The local greedy BF16 teacher merge SHA-256 is `b051d9cc45bac62bf55d599ea5f0a34ec08d3dbe015483d7818d59997f5140ce`: 512 responses, 30,948 continuation tokens, 501 EOS terminations and three triple-repeat 8-grams, in 1,315.453 s with 1.014 GB peak allocated GPU memory. Eight-prompt teacher/control BF16 parity passes in the route run.

| Cell | Input tokens | Minimum content coverage/layer | Maximum nine-way load ratio | Worst hot-parent share | Structural traffic maximum |
|---|---:|---:|---:|---:|---:|
| Raw | 65,024 | 7,169 | 1.126 | 19.06% | 0.12% |
| Chat | 111,686 | 7,610 | 1.085 | 21.56% | 29.83% |
| Documents, width 128 | 24,576 | 5,713 | 1.221 | 17.87% | 0.16% |
| Documents, width 512 | 24,576 | 5,361 | 1.143 | 18.82% | 0.04% |
| New raw+chat pool | 176,710 | 8,545 | 1.082 | 21.33% | 18.90% |
| Training+new raw/chat pool | 1,506,806 | 11,030 | 1.025 | **27.15% FAIL** | 18.98% |

The composed pool's minimum active content median is 63; maximum fraction of content slots below 32 selections is 38.91%. All other frozen gates pass, including the prospectively corrected chat structural cap of 35% and <=1% newly structural positions per new-source cell. The failed layer is 3/source child 215/content local 3: 82 of 302 parent selections (27.1523%). The training sample contributes 62 of 244 parent selections and the new-source raw/chat sample 20 of 58. Each subset is below the >=250 hot-parent definition; their union is not. This is a different parent from the METH-165 failure and shows why subset passes cannot be composed into a large-expert route claim.

The full 24-layer all/content histograms for every cell, old-route comparison and both pools are preserved in `meth169_fresh_boundary_route_result.zip`, 6,020,952 bytes, SHA-256 `f3dc86e2b3ff00f2bff643b7cd9662a3deb3f7367fadb9fbf60114636df594f6`. Its single JSON member is 99,391,344 bytes, SHA-256 `1cf6bad51c2625fee83201242347697b1998a4ca11d48cda5f31715931bfb792`; the member was decompressed and verified byte-for-byte. The source-ordered RTX 3060 route run took 288.782 s, with 2.945 GB peak allocated GPU memory and 4.702 GB RSS. METH-171 now freezes exact-trace attribution of the new hot parent. This result does not measure learned expert usefulness, compact LUT cost or full `engine.c` accepted-token rate.
