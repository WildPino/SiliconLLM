# LUT-Kernel Weight-Packing Prior Art (Q2 of the 2026-08-29 Researcher brief)

**Question as put to me.** Our ternary kernel packs two trits into one full byte as
`(w0+1)*3+(w1+1)` — a value in `[0,8]` — and decodes with `_mm256_shuffle_epi8`
against a 16-entry LUT, so the upper nibble of every weight byte is structurally
zero. `BRIEF_P1_NIBBLE_PACKING.md` proposes packing two such codes per byte
(4 bits/weight to 2 bits/weight) with the `pshufb` count per weight unchanged.
**Do published LUT-based low-bit CPU kernels already do this?**

**Answer, stated up front: yes, unambiguously, in at least two independent published
systems, and one of them uses our exact base-3 pair alphabet.** Our byte-per-2-trit
index is a defect against the published state of the art, not a design choice with a
literature gap behind it. Per the brief this is a perfectly good outcome — the fix is
worth the same either way — and **no novelty claim of any kind is available here.**
A third finding is that P1's own "this is the pshufb optimum" claim is **false**.

Tagging follows the convention already in use in `ACTIVATION_SPARSITY_PRIOR_ART.md`:
**[T]** = read in the paper's own table (table number given), **[A]** = prose or
figure only, **[X]** = could not verify. Source-code findings are tagged **[S]** and
carry the file, the symbol, and the date fetched, because a repository is not a paper
and must not be cited as one.

---

## 1. bitnet.cpp TL1 — two ternary weights into one 4-bit index. This is P1, published.

**Source.** Jinheng Wang, Hansong Zhou, Ting Song, Shaoguang Mao, Shuming Ma, Hongyu
Wang, Yan Xia, Furu Wei — *1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58
Inference on CPUs*, arXiv:2410.16144, **v2 HTML read at `arxiv.org/html/2410.16144v2`**
(fetched 2026-08-29).

**[T] Table 2**, caption verbatim: *"TL1 Kernel transforms every two full-precision
weights into 4-bit index and performs LUT computation."* The table body, transcribed
verbatim, is the mapping from a weight pair to its code:

| w0 | w1 | Pack |
|---|---|---|
| -1 | -1 | 0000 |
| -1 | 0 | 0001 |
| -1 | 1 | 0010 |
| 0 | -1 | 0011 |
| 0 | 0 | 0100 |
| 0 | 1 | 0101 |
| 1 | -1 | 0110 |
| 1 | 0 | 0111 |
| 1 | 1 | 1000 |

**Read that table against our engine.** `(w0+1)*3+(w1+1)` maps `(-1,-1)` to 0,
`(-1,0)` to 1, `(-1,1)` to 2, `(0,-1)` to 3, `(0,0)` to 4, and so on to `(1,1)` at 8.
**The bitnet.cpp column is the same function, and their code 4 (`0100`) is our neutral
code 4.** This is not a similar idea; it is the identical alphabet, published, with the
codes stored four bits wide rather than eight.

Accompanying prose **[A]**: *"TL1 Kernel preprocesses every two full-precision weights
by packing them into 4-bit index (see Table 2), and pre-computes their corresponding
activations into 3^2 = 9 values. The index-value pairs are stored in a lookup table to
perform LUT computation. GEMV processing is performed using an int16 LUT and
accumulation through addition."*

**Bits per weight: 4 bits per 2 weights = 2.0 bits/weight**, versus our 4.0. Derived by
me from the Table 2 caption's own words ("every two ... into 4-bit index"); the paper
does not print a bits-per-weight figure for TL1 as such — **that arithmetic is mine, and
it is one division.**

**Difference from our kernel worth noting: their accumulation is int16, ours is int8.**
That is a separate axis (accumulator width and overflow headroom) and P1 does not touch
it; I flag it only so the Builder is not surprised that the published kernel differs
from ours in a second place.

## 2. bitnet.cpp TL2 — three weights into five bits. This falsifies P1 section 1's optimality claim.

**[T] Table 3**, caption verbatim: *"TL2 Kernel compresses every three full-precision
weights into a 1-bit sign (0 or 1) and a 4-bit index."* Rows transcribed as rendered
(the table is elided with "..." in the source itself, so this is a partial
transcription and I mark it as such):

| w0 | w1 | w2 | Pack (sign + index) |
|---|---|---|---|
| -1 | -1 | -1 | 1 1101 |
| -1 | -1 | 0 | 1 1100 |
| -1 | -1 | 1 | 1 1011 |
| -1 | 0 | -1 | 1 1010 |
| ... | ... | ... | ... |
| 0 | 0 | 0 | 0 0000 |
| ... | ... | ... | ... |
| 1 | 0 | 1 | 0 1010 |
| 1 | 1 | -1 | 0 1011 |
| 1 | 1 | 0 | 0 1100 |
| 1 | 1 | 1 | 0 1101 |

Accompanying prose **[A]**: *"TL2 Kernel is similar to TL1. The major difference is that
it compresses every three weights into a 5-bit index, while TL1 compresses every two
weights into a 4-bit index. Therefore, TL2 achieves a higher compression ratio than
TL1. We recommend using it in environments with limited memory or bandwidth, since it
employs LUT and reduces model size by 1/6 compared to TL1 Kernel, thereby lowering
bandwidth requirements."*

**5 bits per 3 weights = 1.667 bits/weight.** Their own stated ratio to TL1 is 1/6
smaller, and 2.0 x 5/6 = 1.667 — the paper's claim and my arithmetic agree, which is a
weak internal cross-check but a real one.

**Why this matters to P1 specifically.** `BRIEF_P1_NIBBLE_PACKING.md` section 1 states:

> *"three trits do not fit a nibble (27 > 16), so 2 trits/nibble is the pshufb optimum
> — there is no denser packing that keeps a single-instruction decode. This brief
> therefore claims the last free doubling on this axis, not a step along a ladder."*

**That is false, and TL2 is the counterexample.** The 27 states of a trit triple are
split into a 4-bit magnitude index (still a 16-entry `pshufb` table) plus a 1-bit sign
handled outside the table. The decode is no longer *one* instruction, but the table
lookup is still a single `pshufb` against 16 entries — which is the property P1 argued
was the binding constraint. **P1 is a step along a ladder, and the next rung is
published.** I make no claim about whether TL2's sign-handling pays on Zen2; that is a
measurement, and it is the Builder's. I claim only that the optimality assertion in the
brief does not survive contact with Table 3.

## 3. bitnet.cpp implementation — the proposed inner loop, verbatim

**[S] Source.** `microsoft/BitNet`, `include/bitnet-lut-kernels.h`, `main` branch,
fetched from `raw.githubusercontent.com` on 2026-08-29 (75,116 bytes). Function
signature at line 412:

```c
template<int batch_size, int K2>
inline int32_t two_tbl_impl3200_8640(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
```

`a` is the packed weight-index array (`uint8_t*`); `lut` is the precomputed activation
table (`int8_t*`). The loads and the split, lines 421-423 and 439-444:

```c
        __m256i vec_as[KK / 2];
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
...
                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);
```

Note the operand order: in `_mm256_shuffle_epi8(table, index)` the second argument is
the index, and here it is the nibble extracted from `a` — so `a` is unambiguously the
weight stream and `lut` the table, despite the variable name `vec_a`.

**Compare to `BRIEF_P1_NIBBLE_PACKING.md` section 1's proposed per-byte step table:**
one load instead of two; `lo = b & 0x0F` (`vec_v_bot`); `hi = (b >> 4) & 0x0F` via
`_mm256_srli_epi16` then mask (`vec_v_top`, and the brief correctly anticipated that
AVX2 has no byte shift); then a `pshufb` per nibble. **The brief's proposed inner loop
and the shipped bitnet.cpp inner loop are the same loop.** `KK/2` 32-byte loads for
`KK` index-pairs is the halving P1 predicts.

I did not benchmark this and make no speed claim from it. What the source establishes is
existence and shape, not rate.

## 4. T-MAC — stores uint4 indices, unpacks to uint8 for the shuffle

**Source.** Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, Mao
Yang — *T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge*,
arXiv:2407.00088, **v2 HTML read at `arxiv.org/html/2407.00088v2`** (fetched
2026-08-29).

**Mechanism [A], section 3.1.** T-MAC is **bit-serial**: an n-bit weight matrix is
decomposed into n one-bit matrices, each multiplied against the activation via table
lookup, and the partial results summed. Group size is `g = 4` one-bit weights per index,
giving a `[1, 2^g] = [1,16]`-entry table. Verbatim: *"given g=4, for an activation
(A1,A2,A3,A4) of shape [1,4] and 1-bit weights of shape [4,M], the activation will be
precomputed online into a LUT of shape [1,16], containing elements from -A1-A2-A3-A4 to
A1+A2+A3+A4. By grouping every 4 weights together, the weight vector of 0000 will
lookup -A1-A2-A3-A4 and 0101 will lookup -A1+A2-A3+A4."*

**The storage sentence, which is the answer to the brief's question [A], section 3.1**,
verbatim:

> *"Taking the group size g = 4, the tile size of the index matrix W_i[K_tk, M_tm] =
> [4, 32], and the bit width as b = 4. ... For T-MAC on the left side, **the 32 uint4
> indices are first unpacked into uint8 bytes (blue) to ensure compatibility with the
> hardware data type and instructions.** Subsequently, the uint8 indices are utilized to
> look up the table."*

**T-MAC stores its LUT indices as `uint4` — nibble-packed — and unpacks to `uint8` in
register purely because `tbl`/`pshufb` consumes byte lanes.** That is precisely the
trade P1 proposes: pay a shift and a mask in register, halve the bytes on the stream.
It is stated in prose and shown in Figure 3, **not in a table** — I did not find a
bits-per-weight table in this paper and tag the storage claim **[A]**, not [T].
The paper's section 4 also lists *"Efficiant table look-up by TBL/PSHUF"* as a named
implementation technique, confirming the instruction family is the same as ours.

A note on the lookup rate, because it is the one place our kernel is *not* behind. For
2-bit weights T-MAC needs 2 bit-planes times 1 lookup per 4 weights = **0.5 lookups per
weight**; our nibble-packed proposal is 2 lookups per byte of 4 trits = **0.5 lookups
per weight**. Equal. **This arithmetic is mine, from their g and b, not a number the
paper prints — treat it as a derivation to be checked, not a result.**

Also relevant to the D5 line: T-MAC's section 2.4 reports that LUT-GEMM kernels on
**A100 GPU** were *"2.34x, 1.87x, and 1.75x longer than dequantization-based kernels in
BitBLAS"* for W_INT4/W_INT2/W_INT1 mpGEMV **[A]** — i.e. the LUT approach is a CPU
argument, and the same authors report it losing on GPU. That is consistent with our own
framing and worth having on the record.

## 5. llama.cpp TQ1_0 / TQ2_0 — sub-byte, but not via pshufb

**[S] Source.** `ggml-org/llama.cpp`, `ggml/src/ggml-common.h`, `master`, fetched
2026-08-29 (135,128 bytes), lines 274-288, transcribed verbatim including the comments:

```c
// 1.6875 bpw
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K/64]; // 4 elements per byte
    ggml_half d;
} block_tq1_0;

// 2.0625 bpw
typedef struct {
    uint8_t qs[QK_K/4]; // 2 bits per element
    ggml_half d;
} block_tq2_0;
```

**TQ1_0 = 1.6875 bits/weight, by base-3 packing of 5 trits per byte** (3^5 = 243 < 256)
— the scheme D5 section 12.5 costed as "5-trit". **TQ2_0 = 2.0625 bits/weight, 2 bits
per element**, the extra 0.0625 being the `ggml_half` scale amortised over `QK_K` = 256
weights. Both figures are comments written by the implementers in the struct
definitions, not a benchmark result.

**Neither decodes with `pshufb`.** [S] `ggml/src/ggml-cpu/arch/x86/quants.c` (fetched
2026-08-29, 197,407 bytes): TQ2_0's AVX2 path extracts its 2-bit fields with plain
shifts, lines 1530-1532:

```c
            __m256i qx1 = _mm256_srli_epi16(qx0, 2);
            __m256i qx2 = _mm256_srli_epi16(qx0, 4);
            __m256i qx3 = _mm256_srli_epi16(qx0, 6);
```

TQ1_0's AVX2 path extracts base-3 digits with adds, shifts, masks and `avg`, lines
1398-1420, comments verbatim *"8-bit multiplies with shifts, masks and adds"* and
*"Multiply by 3 and get the top 2 bits"*:

```c
            __m256i qx1 = _mm256_add_epi8(qx0, _mm256_add_epi8(qx0, qx0)); // 1 * 3
            __m256i qx2 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx0, 3), _mm256_set1_epi8(-8)), qx0); // 1 * 9
            ...
            qx0 = _mm256_avg_epu8(qx0, _mm256_avg_epu8(qx0, _mm256_setzero_si256()));
```

**A correction to our own cost model falls out of this, and it cuts against a premise
of the D5 5-trit analysis.** `BRIEF_P1_NIBBLE_PACKING.md` section 0.1 says base-3
packing *"needs a divide-by-3 decoder and cannot use `pshufb` at all — so the cost side
always looked heavy."* The `cannot use pshufb` half is correct. **The divide half is
not: llama.cpp extracts all five base-3 digits with roughly five cheap integer ops per
vector and no division anywhere.** Whether that changes the 5-trit verdict is a
measurement question and not mine to answer — but the cost side of 5-trit was priced
against a divide that the reference implementation does not perform, and that pricing
should be redone before 5-trit is dismissed again.

---

## 6. What this does and does not settle

**Settled, from primary sources.**

1. Nibble-packing two ternary codes per byte behind a 16-entry `pshufb` table is
   published practice, in bitnet.cpp's TL1 (**[T] Table 2**) and in shipped source
   (**[S]**), using our exact `(w0,w1)` to `[0,8]` alphabet.
2. T-MAC independently stores LUT indices as `uint4` and unpacks to `uint8` for the
   shuffle (**[A]**, section 3.1 prose plus Figure 3).
3. llama.cpp's ternary formats are sub-byte at 1.6875 and 2.0625 bpw (**[S]**), by
   base-3 and by 2-bit fields respectively, neither via `pshufb`.
4. **P1's claim that two trits per nibble is the `pshufb` optimum is false** —
   bitnet.cpp TL2 reaches 1.667 bpw with a 4-bit table index plus a sign bit
   (**[T] Table 3**).

**Not settled, and not claimed.**

- **No speed claim.** I benchmarked nothing. Every rate quoted here belongs to its
  paper's own hardware, none of which is Zen2.
- **[X]** I did not find, in any of the three, a stated bytes-streamed-per-matvec
  measurement of the kind P1's control C4 requires. Their compression ratios are
  storage-format arithmetic, not instrumented traffic. C4 remains ours to run.
- **[X]** No source here addresses P1 section 2's padding-neutrality trap (code 0
  meaning `(-1,-1)`, not zero) or the odd-`T` tail. Those are properties of our layout
  and the literature does not relieve us of deriving them.
- The standing law applies unchanged: **a gap is confirmed only for the literature
  actually searched.** Here no gap was even sought and none is asserted — the finding
  runs the other way.
