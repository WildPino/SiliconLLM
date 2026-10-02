# METH-295: actual native held-out prediction PASS

Original freeze `00613ac`; pre-native CUDA-cleanup apparatus failure at
174.219s is [preserved](METH_295_NATIVE_PRIMARY_APPARATUS_FAILURE_20261002.md).
Repair freeze `b2f3cd6`, session91076 exits0,1771.578s; actual CPU1765.328s,
peak childRSS1,362,563,072bytes. Exact saved valid GPU controls reused in a
new process that never initializes CUDA. Same input bytes/operators/margins.
No native observation occurred in the failed first launch.

All11 fixed gates pass on ALL24 new292 sources,8/category:29,395 document
tokens and4,264 prompt positions,93 independently rebuilt windows including
21 nonzero-offset windows,23,040 additional prefill rows. Original276
1,329,447,260byte/725-field archive loaded directly in actual phase60 C,
no fallback, original285 arithmetic profile. All window positions/targets/
counts/EOF verified;93 first full-head NLL oracles have maximum absolute
error6.785683126508957e-13 against NumPy float64, below1e-9.

| Category | Donor BPB | BF16 E1280 BPB | Actual native BPB | Native minus E1280 | Native donor-top1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled | 1.24367193 | 1.23842220 | 1.23886670 | +.00044450 | 96.0600% |
| Code | .84612447 | .84286168 | .84319631 | +.00033463 | 95.0681% |
| Prose | 1.16788109 | 1.16112619 | 1.16172591 | +.00059972 | 97.0808% |
| Technical | 1.71701024 | 1.71127873 | 1.71167787 | +.00039914 | 96.1788% |

Native pooled BPB is-.00480523 against donor, within+.01 against EACH
control; all category margins+.02 pass. Pooled donor-top1 agreement
improves+.867730percentage points over E1280's95.1923%; all category
nonregression limits pass. Independent stdlib recomputation from every
saved source nats/byte count and token-choice pair reproduces all reported
summary values and quality gates exactly.

Descriptive10,000-source bootstrap/seed295295 BPB difference p05/p95:
versus donor[-.00552288,-.00410729],versus E1280[+.00026593,+.00061665].
Native slightly degrades E1280 BPB within the fixed margin; do not claim
zero loss. Fresh GPU276 descriptive BPB1.23877445/agreement95.5206%; native
minus this GPU BPB+.00009224. All GPU scalars exactly reproduce independent
canonical M17 calls. No CUDA numerical-orbit equivalence follows.

[Raw result](meth295_native_primary_prediction_repair1_result.json) SHA
`82a86f078409783859b8199e89d6186b53883917351baefd6787b46e4fc7d476`;
[protocol](METH_295_NATIVE_PRIMARY_PREDICTION_PROTOCOL_20261002.md).
Raw retains all per-source/per-token rows, executable/source/entry/bundle/
native/log hashes, actual loader report, compile command, resource readings
and prior failure/complete GPU partial hashes. Total first+repair elapsed
1945.797s, including valid GPU controls173.766s; no T4/downloads.

## Decision and unchanged limits

Freeze native generation on these SAME prompts, then anonymous semantics,
full1838 PIQA, actual CPU K64 and accepted>=50 batch1decode on SAME artifact.
This is the first direct donor-relative held-out quality pass of the actual
current C execution profile under292's prospectively declared method
revision. Original285/288/2895% failures remain failures. Archive remains
diagnostic/unpromoted. These teacher-forced full-head assay timings are
NOT accepted decoding throughput. No useful tenfold new capacity, arbitrary
RAM-only n, real large-pool DRAM/LUT cost or family/10B/100B transfer proof.
