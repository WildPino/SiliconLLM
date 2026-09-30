# METH-183: learned E1280 child-route alignment diagnostic

**Uncertainty and prior evidence.** METH-121–123 found a small but gated
quality and semantic improvement for a mean-preserving, learned ten-child
expansion of the pretrained Qwen2.5-0.5B-Instruct E128 adapter. Copying each
parent B to all children removes that improvement, but this establishes a
benefit from distinct B rows, not that the learned router matches contexts
to the right rows. METH-179 found no such alignment for a later hash-routed
E12,800 bank. The two banks and routes differ, so that failure cannot decide
the E1280 case. This test determines whether learned child-score routing is
a useful ingredient to preserve when increasing expert count again.

**Frozen inputs and comparison.** Bind the exact METH-121 24-document
manifest SHA-256 `7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366`,
the METH-107 checkpoint SHA-256
`15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`,
the METH-56 E128 checkpoint, the pinned BF16 Qwen donor and tokenizer, and
the exact METH-122 mean-centering operation. Keep donor core, E128 parent
route, top four scores, A/B factors and all token windows unchanged. Compare
the exact learned local child ID with all nine nonzero cyclic shifts
`(local + shift) mod 10`. These are bijections within each parent and leave
the total per-child route counts unchanged. Compute document negative log
likelihood with the same METH-17 scorer and verify that the shift-zero
per-document nats reproduce the stored METH-122 exact arm. No weights are
trained or updated.

**Frozen analysis and decision.** Report pooled and per-category BPB, each
shift minus exact, and a 10,000-draw paired bootstrap over the 24 sources
(seed `183183`) for mean-shift-minus-exact BPB. A route-alignment diagnostic
passes only if the exact route beats every shifted route in pooled BPB and
the bootstrap fifth percentile of mean-shift-minus-exact is positive.
Otherwise do not treat learned key selection as demonstrated functional
specialization in this bank. Either result is a mechanism diagnostic on
previously consumed audit sources, not a new held-out quality result, a
10B transfer result, a native CPU timing or a reason by itself to enlarge
the bank. A positive result would justify testing a content-coupled,
load-controlled third-tier training mechanism; a negative one would direct
the next method step toward changing the learning signal.

**Budget and stop.** Local RTX 3060 only, no T4; six host threads, at most
20 minutes wall time, 10.5 GiB allocated GPU memory, 20 GiB process RSS and
1 GB new disk. Stop on binding/parity failure or exceeded budget, preserve
the failed stage and partial document rows. The implementation and exact
commands are recorded with the result after execution.
