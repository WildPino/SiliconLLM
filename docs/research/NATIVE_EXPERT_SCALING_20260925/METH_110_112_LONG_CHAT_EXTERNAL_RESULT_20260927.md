# METH-110–112: long-chat E1280 passes automatic audit, fails blind semantics

**Decision.** The frozen METH-107 long-chat E1280 checkpoint passes every METH-111 automatic external gate on 24 new documents and the full 1,838-item PIQA task. The committed arm-blind METH-112 verdict finds more unsupported and severe claims than E128, so the joint quality gate **fails**. Do not promote this checkpoint as donor-relative quality-preserving. Neither the factor LUT nor full native model throughput has been measured for it.

The [protocol](METH_110_112_LONG_CHAT_EXTERNAL_PROTOCOL_20260927.md), [manifest](meth110_long_chat_external_manifest.json), [runner](../../../benchmarks/donor_adaptation/s1/meth111_long_chat_external_audit.py) and [automatic result](meth111_long_chat_external_audit_result.json) bind the donor and METH-56/METH-107 checkpoint hashes. METH-110 selected eight code, eight prose and eight general technical sources, excluding previous source IDs and overlapping text fragments through METH-108. The prose comes from eight disjoint 1-MiB windows of a pinned Europarl English concatenation, a different corpus from development's Simple English Wikipedia. These windows are not independent documents or a broad domain sample.

| External automatic metric | E128 | E1280 | Frozen gate |
|---|---:|---:|---|
| Pooled document BPB | 1.128143 | 1.127693 | E1280−E128 ≤+0.01: pass |
| Donor prompt top-1 agreement | 96.902% | 95.960% | Loss ≤1 point: pass, loss 0.942 point |
| Greedy EOS / 24 | 19 | 19 | E1280 ≥E128−2: pass |
| Repeated 8-gram three times / 24 | 0 | 1 | E1280 ≤E128+1: pass |
| PIQA correct / 1,838 | 1,292 | 1,287 | Delta ≥−2 points: pass, −0.272 point |

All per-category document, prompt and generation gates pass. PIQA has 16 E128-correct/E1280-wrong items and 11 reverse swaps; the paired bootstrap lower fifth percentile is −0.00762, above the frozen −0.05 bound. PIQA reuses the public task from earlier audits and serves as a regression check, not a new independent task sample. The GPU audit took 1,038.11 s on RTX 3060, peaked at 4.258 GB allocated GPU memory and used 3.537 GB RSS, inside the 30-minute and memory stops.

The [anonymous A/B pairs](meth112_long_chat_blind_pairs.json) contained only each 384-character excerpt and the two continuations. The [verdict builder](../../../benchmarks/donor_adaptation/s1/meth112_blind_verdict_build.py) and [verdict](meth112_long_chat_blind_verdict.json) were committed at `c6c0a27` before the [unblinding scorer](../../../benchmarks/donor_adaptation/s1/meth112_long_chat_blind_semantic.py) was run. The [unblinded score](meth112_long_chat_unblinded_score.json) is:

| Excerpt-only finding | E128 | E1280 | Gate |
|---|---:|---:|---|
| Unsupported claims | 27 | 28 | **Fail** |
| Severe unsupported claims | 5 | 7 | **Fail** |
| Answers missing a supported specific detail | 2 | 2 | Pass |

The largest severe regression occurs on the technical `A10B-K3` excerpt: E1280 describes noise-weight throughput evidence as trained text-error correction and transfer, contradicting the visible caveats. This finding was recorded while arms were hidden. A single agent judged short excerpts, so the counts are not a population estimate; the predeclared zero-regression semantic gate still rejects promotion. This external set has now been viewed and cannot be used to tune and re-test the same checkpoint.

Long-response teacher supervision improved the development EOS count and preserved many distinct child factors, but did not solve grounded technical-detail fidelity. The METH-104 C router evidence remains valid for the unchanged METH-107 router weights, yet it measures only hot-input FP32 route cost. The next method must improve grounded numerical and technical details using training/development sources separate from METH-110, then test on another new external set. It also needs learned-factor CPU/LUT traffic and full `engine.c` inference on the same artifact before any throughput claim.
