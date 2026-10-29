# Wall map of Ling-mini-2.0 and Ling-flash-2.0: result

**Status:** RESULT, 29 October 2026. **Protocol:** `WALL_MAP_PROTOCOL_20261029.md` (with its amendment).
**Numbers:** `benchmarks/wallmap/results/` (`w0_byte_map`, `w0_desk`, `w0_family_map`, `w0_ref_read_2`).

## The answer

- **The first wall is the ternary kernel's compute rate, not the bytes.**
  - The operative kernel processes 46 G weights/s (E28). At 0.25 B/weight that is 11.5 GB/s, a third of the RAM wall.
  - So an all-ternary Ling-flash would run at ~7.8 tok/s, as fast as today's Q4_K_M in ik_llama.cpp (7.88).
- **The second wall is the active weights.** Ling-flash reads 5.5 G weights per token. With routed experts at 50% and
  attention at 20% of its bytes, even a kernel at the RAM wall caps all-ternary flash at 19 tok/s.
- **5× llama.cpp on Ling-flash is not reached by any per-weight form, even with an infinitely fast kernel.**
  - 5× the measured reference is 30.8 tok/s.
  - The best modelled form gives 26.6 (experts at 0.8 bit, the rest ternary): 4.3×.
  - Adding the two structural levers modelled (half the experts, a 32K vocabulary) gives 27.8: 4.5×.
  - Reaching 5× needs 1.7× fewer weights read per token with a kernel at the RAM wall, or 4.8× fewer with today's
    kernel.
- **Size independence:** flash/mini measures 0.35 (llama.cpp) and 0.32 (ik_llama.cpp) today. Per-weight conversion
  keeps it at 0.23-0.30, because flash has 5× mini's active parameters.

## 1. Where the bytes are (SOURCE-DERIVED from the GGUF headers)

MB read per decoded token at context 2048. Today = Q4_K_M as stored, F16 KV; the count reproduces ROADMAP §3's 867.3
MB for Ling-mini. nibble = ternary at 0.25 B/weight (built in engine.c), with a ring KV (1 layer in 8) and its
recurrent state.

| Organ | mini today | flash today | flash share | flash, nibble |
| --- | ---: | ---: | ---: | ---: |
| Routed experts (8 of 256) | 288 | 1,885 | 50% | 780 |
| Attention weights | 134 | 766 | 20% | 302 |
| Head (157,184 × d) | 264 | 528 | 14% | 161 |
| Shared expert | 36 | 236 | 6% | 98 |
| KV cache / ring state | 84 | 134 | 4% | 9 + 117 |
| Router (F32) | 40 | 130 | 3% | 35 (int8) |
| Dense FFN (layer 0) | 20 | 73 | 2% | 28 |
| Embedding row, norms | 0.4 | 1 | 0% | 1 |
| **Total** | **867** | **3,754** | | **1,531** |

At context 32,768 the flash KV grows to ~2.1 GB and becomes the second organ. A ring KV removes it, but its state
costs more than the q8_0 KV it replaces at 2048 (117 against 62 MB).

## 2. Reference speeds (MEASURED, lower bounds)

Same Q4_K_M files, 6 threads, depth 2048, best cell of each engine (`w0_ref_read_2.json`). Medians of 2 processes, with
min..max, after a correctness check that passed on both files (every cell within 3.4% of the reference perplexity; a
planted wrong model moved it +14.4%).

| | llama.cpp | ik_llama.cpp | ik / llama | desk, Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| Ling-mini | 17.49 (one process) | 24.71 [24.54..24.88] | 1.41 | 33.6 |
| Ling-flash | 6.15 [6.07..6.22] | 7.88 [7.87..7.88] | 1.28 | 8.2 |
| flash / mini | 0.35 | 0.32 | | 0.25 |

**Why lower bounds.** Run 1 and run 2 stopped VOID: llama.cpp processes read more foreign CPU occupancy than ik's on
the same file (flash: 7-8% against 2-3%). On Ling-flash this is not paging: the same figure appears without mmap
(`w0_ref_diag_loadmode.json`). VOID and OK processes of the same cell agree within 3% (llama.cpp mini 17.17-17.92
against 17.49). Contention can only slow a process. A clean run would raise these figures, and with them the 5× bar,
so no conclusion here weakens.

**Predictions** (registered before the dry runs, which were seen before the run):
- llama.cpp mini 13.5-15.5: **miss** (17.49).
- llama.cpp flash 3.0-4.5: **miss** (6.15). llama.cpp runs Ling-flash at 75% of the desk model, not at 37-55%.
- ik over llama.cpp ≥ 1.3× on both files: **miss on flash** (1.28; 1.41 on mini).
- flash/mini 0.20-0.30: **miss** (0.32).

## 3. Desk ceiling against the reference (DESK-MODEL estimate, context 2048)

Each organ costs max(bytes / 32 GB/s, weights / kernel rate), plus 2.70 ms per 20 layers. Kernel rates: packed 46.16 G
weights/s; `--lutblk` 61.64 (its quality licence was revoked by E32). Neither rate is measured at flash's shapes.

| Ling-flash scenario | tok/s, bytes only | × llama.cpp | tok/s, packed | × llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| Likely: experts at 2 bits post-hoc, int8 router, q8_0 KV | 12.2 | 2.0 | 8.0 | 1.3 |
| Every organ ternary (nibble) | 19.2 | 3.1 | 7.8 | 1.3 |
| Experts at 0.8 bit, the rest nibble | 26.6 | 4.3 | 7.8 | 1.3 |
| + 4 experts per token and a 32K vocabulary (structural) | 27.8 | 4.5 | 12.0 | 2.0 |

Against ik_llama.cpp, divide by 7.88 instead of 6.15: the all-ternary row is 2.4× bytes only and 1.0× with the packed
kernel. 50 tok/s (8×) needs under ~500 MB and ~516 G weights/s per token: out of reach of every modelled form.

## 4. Literature per organ (read in the papers) and the repo's priors

| Organ | Literature | Repo |
| --- | --- | --- |
| Routed experts | QMoE: Switch (masked LM), no training: 2-bit +1.7%, ternary +6.7% relative C4 loss; the 0.8 bit form is the ternary model entropy-coded, with a GPU decoder | E12: post-hoc ternary at chance. E37: a 1.17% keep is at chance; a random router equals the trained one |
| Attention, shared, dense | BitNet b1.58 2B4T: ternary from scratch on 4T tokens, MMLU 53.2. BitDistill: Qwen3 0.6-4B ternary with 10B tokens, task fine-tuning only | E65: post-hoc rank 1/32 costs 52% of the dense-to-chance gap. H1: trained carve 0.9626 vs 1.0966. H4: trained rank-48 q/o, BPB 2.80 -> 1.154 |
| Head | BitNet b1.58 keeps the embedding in full precision; no ternary head found | A5: not low-rank (rank 512 holds 44% of the squared norm) |
| KV | LoLCATs: 40M tokens, −11 MMLU points (8B, 70B). Mamba in the Llama: 50% attention, MMLU 68.12 -> 62.13. Ring-linear (same producer): 600B / 1T tokens, >98% in most categories | none |
| Router | — | exactly BF16-representable on Ling-mini |

None of the conversions that kept quality close was cheap: hundreds of billions of tokens (Ring-linear), or a
multi-point loss (Mamba in the Llama, LoLCATs). No post-hoc result on a generative MoE of this size was found.

## 5. The cheapest test that could answer "impossible"

| Wall | Test | Disproved if | Cost |
| --- | --- | --- | --- |
| **Ternary kernel rate** | A ternary GEMV at Ling-flash's expert and attention shapes (1024 × 4096, 4096 × 4096) on the 3600X, 6 threads, the best known layouts | under 73 G weights/s, the top of Q4_K's rate (A1: 35-41 GB/s at 0.5625 B/weight = 62-73): ternary would then not beat Q4 on this box | CPU only, ~1 week |
| Routed experts | Ternarize the experts of one Ling-mini layer, train them alone on dumped inputs, swap the layer in; KL and top-1 against BF16 | one layer costs more than 1/19 of the quality budget | ~2 h CPU + ~4 GPU-h |
| Attention, shared, dense | The same on one layer's q/k/v/o and shared expert | as above | ~2 GPU-h |
| Head | Post-hoc head at 4, 3, 2 bits, the rest as stored; KL and top-1 | top-1 agreement falls below 99% at 4 bits | ~1 h CPU |
| Router, KV | int8 router post-hoc; KV not a wall at 2048 | — | ~0.5 h CPU |

## 6. Proposed thresholds (measured at depth 2048 on the 3600X, against the Q4_K_M originals)

- **Jump:** Ling-flash converted ≥ 3× llama.cpp, i.e. ≥ 18.5 tok/s (2.3× ik_llama.cpp).
  - This is the all-ternary ceiling (19.2). It requires the kernel at the RAM wall.
  - 5× (≥ 30.8) stays the ambition. It requires the kernel at the wall and, on top of it, reading ≥ 1.7× fewer
    weights per token.
- **Size independence:** Ling-flash converted at least as fast as Ling-mini original in the faster engine (≥ 24.7
  tok/s): the 100B runs like today's 16B. The flash/mini ratio is reported beside it (0.32 today).
- **Quality:** at most 5 points lost on average on a fixed panel registered before the first conversion. Each
  conversion reports top-1 agreement with the original and its trivial control.

## 7. Recommendation

**Test the ternary kernel first, at Ling-flash's shapes, on the CPU.** Every route of the form goes through it. With
today's kernel, no amount of training makes a ternary Ling-flash faster than ik_llama.cpp on the original.
- Pass at ≥ 128 G weights/s (the RAM wall at 0.25 B/weight).
- Fail under 73 G weights/s, the top of what Q4_K already does. The per-weight route would then be closed on this box, and
  only reading fewer weights per token would remain.

**What would disprove it:** an organ test that fails first and cheaply. Example: one ternary Ling-mini expert layer,
trained, still costs more than its share of the quality budget. That would close the route whatever the kernel does, and it costs only ~4 GPU-h. If it is run, it goes first.

**Independent review:** done before the protocol push; its 13 objections and the answers are in the protocol.
