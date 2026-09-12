# BRIEF E43 — is the VOCABULARY a lever, and what is a token actually worth?

**Pre-registered. Pushed before the runner exists and before any vocabulary has been trained.**
Nothing in the probe may contradict this file; anything it has to change is an addendum with its
own date, in the E40/E41/E42 style.

**Scope: SPEED and UNITS only. Not one BPB is measured here, and §8 says why that is the half
that decides.**

---

## 0. Why this, and why now

E40 §50.6 left the floor decomposed and one term untouched:

> `R128`'s 0.218 G base: **head 134,217,728 (61.5%)**, attention 67,108,864 (30.8%), router
> 16,777,216 (7.7%). **The untied head is now bigger than all the attention put together** and
> has never been probed on this axis; `V = 32768` is a choice, not a law.

Three facts make this the next probe rather than a footnote.

1. **The head is charged in full on every token and it is not multiplied by `L`.**
   `synth_export.active_weights` ends in `+ V * D` — one term, outside the layer loop, 61.5% of
   the floor. Every other lever this programme has pulled (NKV 8→2, rank 512→128) attacked a
   term that was already smaller than this one.
2. **`V` converts base into FFN at the best exchange rate in the programme.** At fixed
   9,999,220,736 parameters, a weight removed from the head is a weight added to `F`, and `F` is
   charged at `k/E` — 1.2% at `k = 3`. The NKV lever, which the ledger calls "the cheapest large
   lever in the programme", worked by exactly this mechanism on a term a third the size.
3. **`V` also moves the unit the goal is written in.** The standing goal says *50 token/s*. A
   token is not a fixed amount of text: halving `V` makes every token cover fewer bytes. So `V`
   moves the numerator and the denominator of the headline number in opposite directions, and
   **nobody in this programme has ever measured the denominator.** That is the
   charged-bytes-vs-moved-bytes error one level up, and it is sitting under the target itself.

E43 is a speed probe, and it is run now because the quality road is where it is: post-hoc
conversion is closed by three independent measurements (E37, E38, E40), and the one live road is
training into the format, which is blocked on the T4 session in `COMMUNICATION.md` APERTO 0.
**`V` is a design parameter of that training.** If H1 is going to be trained, E43 says what
vocabulary to train it at, and it costs no GPU and no user time to find out.

### 0.1 What E42 changed about how gates get written here

E42's null-arm control went VOID because I fixed a numeric tolerance (`0.010`) on an axis whose
dispersion had never been measured (it was `0.099`). The repair is applied in this brief as a
rule, not a promise:

> **Every gate below is either ORDINAL (direction only, no tolerance) or uses a tolerance this
> programme has already measured on that exact axis.** No gate in E43 invents a number.

`G-E43A` uses `±5%`, which is the session-comparability bar E39 and E40 both ran on and which
has fired correctly. `G-E43B` and `G-E43C` are exact equalities. `G-E43D` is direction-only.

---

## 1. The question

**At a fixed 9,999,220,736 parameters on this box, how does the vocabulary size change the rate —
and once a token is priced in bytes, which `V` actually delivers the most text per second?**

Secondary, and the reason the probe has a units half at all: **how much of the goal's "50 tok/s"
is a property of the model and how much is a property of the tokenizer?**

---

## 2. Setup, inherited unchanged

- Box: Ryzen 5 3600X, 6c/12t, `--threads 6`, L3 32768 KB. `BW-CEIL = 36.30 GB/s` (E30).
- Engine: `donor_engine_e26.exe`, the binary E39 and E40 timed on.
- Shape family: E40's `A10B-R128` — `D=4096, L=16, NH=32, NKV=2, HD=128`, untied, `rank 128`,
  `--carve 256`, `--codes mixed --seed 1234`. **Only `V` and `F` move.**
- Parameter count: **exactly 9,999,220,736** on every arm, `F % 256 == 0`. This forces
  `V ≡ 32768 (mod 6144)`; `V = 4096`, `16384` and `65536` are **not reachable** on this grid and
  are not used.
- Corpus: the frozen `density/corpus/` pair, `calib.txt` (140,933,631 B, sha `10d4d281…`) and
  `heldout.txt` (140,596,742 B, sha `f46b0310…`).
- Eval span: the frozen slice `("heldout", 24, 512, 1234)` — **its byte offsets**, giving
  **51,870 bytes** of heldout text. Under Qwen2.5's own `V = 151936` tokenizer that span reads
  **4.22945205479452 bytes/token**, and that is the anchor `G-E43C` reproduces.
- **Vocabularies are trained on `calib.txt` and measured on `heldout.txt`. Never the same half.**
- No GPU. No gradient. No donor weights. Synthetic ternary weights, speed only.

---

## 3. The arms

### 3.1 Timed arms — four, three of them new builds

| arm | `V` | `F` | total params | base (`k=0`) | charged `k=3` | head % of base |
|---|---|---|---|---|---|---|
| `V2048` | 2,048 | 50,432 | 9,999,220,736 | 92,274,688 | 208,470,016 | **9.1%** |
| `V8192` | 8,192 | 50,176 | 9,999,220,736 | 117,440,512 | 233,046,016 | **28.6%** |
| **`V32768`** | **32,768** | **49,152** | 9,999,220,736 | **218,103,808** | **331,350,016** | **61.5%** |
| `V131072` | 131,072 | 45,056 | 9,999,220,736 | 620,756,992 | 724,566,016 | **86.5%** |

`V32768` **is** E40's `A10B-R128`. The existing `D:/_ktmp/e40/e40_r128.bin` is re-timed, not
rebuilt, so the control is the same bytes on disk that produced 112.73 / 106.44 tok/s.

The four arms span **64× in `V`** and bracket where the product is expected to peak (§7).

### 3.2 Vocabulary grid — eight points, all cheap, no export needed

`bytes/token` on the frozen span is measured at every reachable `V` in
**{2048, 8192, 14336, 20480, 26624, 32768, 51200, 131072}**, plus Qwen2.5's own tokenizer as the
external reference point. Only four of these get a 10 B export; the other four cost minutes and
give the curve its shape between the timed points.

### 3.3 The null arm

At `V = 2048` and `V = 32768` only, a second BPE is trained on **character-shuffled `calib.txt`**
(same characters, same multiset, word and line structure destroyed, valid UTF-8 preserved,
shuffle seed `43043`). Two points, not eight, to keep the cost bounded. This is `G-E43D`.

---

## 4. How the vocabularies are built, fixed here before any of them exists

- `tokenizers` BPE, byte-level pre-tokenizer and byte-level decoder, no normaliser, no special
  tokens, `min_frequency = 2`, initial alphabet = the full 256 bytes so **every vocabulary can
  represent every byte** and no text is ever unencodable.
- Trained on the **whole** of `calib.txt`, one pass, `vocab_size = V` exactly.
- Trainer seed is not a parameter of `tokenizers`' BPE (the merge order is deterministic given
  the corpus and the counts); the shuffled null's seed `43043` is the only randomness declared.
- Each trained vocabulary is written to `engine/results/e43_vocab/bpe_V{V}.json` and hashed. The
  hash goes in the results file so any number here can be traced to the exact vocabulary.
- `bytes/token` is measured as **the 51,870 frozen bytes divided by the number of tokens that
  vocabulary needs to encode them**, span by span, summed. No special tokens, no BOS.

---

## 5. The verdict cell and the bands, named before the run

**The cell is `BPS_peak / BPS(V=32768)` at `k = 3`**, where
`BPS(V) = tok/s(V) × bytes/token(V)` — bytes of heldout text emitted per second — and the peak is
taken over the four **timed** arms only. `k = 3` because that is the configuration E40 headlined
(112.73 tok/s at ten billion with the FFN alive), i.e. the one that is actually shippable.

| cell | band |
|---|---|
| `≥ 1.50` | `VOCABULARY-IS-A-BIG-LEVER` |
| `1.15 – 1.50` | `VOCABULARY-IS-A-LEVER` |
| `1.05 – 1.15` | `VOCABULARY-IS-A-SMALL-LEVER` |
| `< 1.05` | `VOCABULARY-IS-NOT-A-LEVER` |

**Second cell, ordinal, no tolerance:** is `V = 32768` **below**, **at**, or **above** the peak of
the measured `BPS(V)` curve? "At" means it is itself the argmax of the four timed arms.

Both cells are reported from **run 1**. E36's run-2 rule governs: run 1 is the registered
measurement, run 2 may not promote it, and if the two disagree beyond dispersion the cell is
unresolvable.

---

## 6. The gates

| gate | what it demands | consequence if it fails |
|---|---|---|
| **`G-E43A`** *(planted control, known positive)* | `V32768` is byte-identical to `e40_r128.bin`, and its re-timed `k=3` rate is within **±5%** of E40 run 1's **112.73 tok/s** | **No rate in E43 counts.** The sessions are not comparable and the whole speed half is void |
| **`G-E43B`** *(arithmetic independence)* | a closed form written in the runner **without** calling `synth_export` equals `active_weights` for every (arm, `k`) **exactly**, and `total_weights` reads **exactly 9,999,220,736** on every arm's header | **That arm is dropped.** It is not the object the brief describes |
| **`G-E43C`** *(the units instrument's known positive)* | re-encoding the frozen span from its own offsets with **Qwen2.5's own tokenizer** reproduces `total_scored_bytes = 51870` and `bytes/token = 4.22945205479452` **exactly** | **The whole bytes/token column is void**, and with it both cells — the instrument cannot read the slice it claims to read |
| **`G-E43D`** *(null arm, ORDINAL)* | at both `V = 2048` and `V = 32768`, the character-shuffled BPE reads **strictly fewer bytes/token** on the heldout span than the corpus-trained BPE at the same `V` | **The bytes/token column is void.** A vocabulary learned from destroyed structure that ties or wins means the trainer is not learning corpus structure |

`G-E43D` is deliberately **direction-only**. The dispersion of `bytes/token` across vocabulary
training has never been measured on this harness, so per §0.1 no number may be attached to it.

**A gate that fires is doing its job and is not re-run to a pass** (E40 addendum A, standing).

---

## 7. Predictions — fixed here, before the runner exists

1. **All four gates fire.**
2. **Cell ≈ `1.25`, band `VOCABULARY-IS-A-LEVER`. Peak at `V = 8192`. `V = 32768` is ABOVE the
   peak** — i.e. the vocabulary this programme inherited is already on the losing side, and the
   lever points *down*, not up.
3. `bytes/token` = **3.12 / 3.56 / 4.05 / 4.58** at `V` = 2048 / 8192 / 32768 / 131072.
   **This is registered as the weakest prediction in the probe.** E41 and E42 each missed every
   magnitude I registered, in both directions, and this one rests on a log-`V` rule of thumb I
   have not measured on this corpus.
4. **The measured small-`V` arms come in BELOW the charged-weight extrapolation, by 5–20%.**
   E40's own ladder shows `base × floor` is not constant — 35.2, 41.2, 40.0 G·tok/s across
   `R128`, `NKV2`, `R512` — and the head is the single largest contiguous streaming run in the
   token. Taking it away should cost some of the streaming efficiency it was providing. Straight
   `1/charged` scaling from 112.73 predicts **179.2 / 160.3 / 112.73 / 51.6 tok/s**; I expect the
   first two to read lower and the last to read at or slightly above.
5. **The floor at `V = 2048` exceeds 350 tok/s** (straight scaling says 381).

---

## 8. What E43 will NOT be able to claim, and the half that decides

- **Nothing about quality. Not one BPB.** No arm here has trained weights. Whether a 10 B model
  with an 8,192-token vocabulary is as good per *byte* as one with 32,768 is a **training**
  question, it needs the GPU, and E43 cannot see one bit of it.
- **This is the half that decides, and it is not measured.** If a smaller vocabulary costs real
  BPB, the bytes/s win is bought with quality and the trade has to be priced — and this
  programme has just spent two probes (E41, E42) establishing that it cannot resolve BPB
  differences below ~0.3 on this harness at three seeds. E43's honest output is therefore a
  **design input to H1**, not a recommendation to ship.
- **Nothing about retrofitting a donor.** `V` here is a design parameter of a model that does not
  exist yet. Changing a trained donor's vocabulary means replacing its head and its embedding —
  that is not a conversion, it is training, and it belongs to the T4 road.
- **Nothing about `D` or `L`**, still the untouched axes of E40 §7.
- **No change to the standing T4 ask.** E43 cannot strengthen or weaken APERTO 0. At most it
  tells H1 which `V` to be trained at, and it will say so in exactly those words.
- **Every absolute tok/s here carries ±5%. The ratios do not** — and both cells are ratios, on
  purpose.

---

# ADDENDUM A — run 1's `G-E43A` went VOID, and I found an ORDER CONFOUND in my own plan

**Written and pushed BEFORE the re-run, and before any re-run number exists.** Run 1 is
`results/e43_vocabulary.json`, committed with this addendum.

## A.1 What happened

`G-E43A` read **79.27 tok/s** on the control at `k=3` against E40 run 1's **112.73** — **−29.7%**
on a **±5%** bar. **VOID.** The registered consequence applies in full: **no rate from run 1
counts**, the two cells it printed (`1.3425`, `VOCABULARY-IS-A-LEVER`, control **above** the peak)
carry **no registered force**, and they are reported only as a record of what the arithmetic did.

The gate was not mis-specified — **it caught exactly what it was built to catch.** The box was
verified at **5.8% busy** immediately before launch, with the control reading **115.02 (+2.0%,
spread 8.0%)** across five reps. It then degraded monotonically *during* the run:

| rep | box busy | control `V32768 k=3` | vs 112.73 |
|---|---|---|---|
| 1 | **8.2%** | **115.82** | **+2.7%** |
| 2 | **12.8%** | **114.20** | **+1.3%** |
| 3 | 19.7% | 38.26 | −66.1% |
| 4 | 42.8% | 75.06 | −33.4% |
| 5 | 45.0% | 53.01 | −53.0% |

**Reps 1 and 2 are clean and their control sits inside the bar. They are NOT used.** Choosing the
reps that pass, with a threshold picked after seeing which ones do, is the same error that voided
E42 — a bar fixed after the data. Run 1 is void as a whole.

## A.2 The confound, which is mine and is worse than the VOID

Run 1 timed `plan = [(arm, k) for arm in order for k in KS]` with **reps outermost**, which
spreads drift across arms *between* reps. It does **not** spread it *within* a rep: inside every
rep the arms are timed in **ascending `V`** — `V2048` first, `V131072` last. So a box load that
rises monotonically **penalises arms in proportion to their position in the plan**, and the plan
position is ordered by `V`.

**That bias points in exactly the direction of the registered prediction.** A drifting box would
manufacture "small `V` is faster" out of nothing. Run 1's box drifted 8.2% → 45.0%, so run 1
cannot distinguish the effect from the artifact even if `G-E43A` had fired.

E40 already had the antidote and used it — it ran `--order reversed` as a second session. **I ran
only `forward`.** That is an execution gap on my side, not a design gap, and it is registered here
rather than quietly fixed.

## A.3 What the re-run is allowed to be, fixed here

1. **Exactly ONE re-run attempt.** Two sessions, `--order forward` and `--order reversed`. If
   `G-E43A` goes VOID again, **E43 has no speed half and no cells, permanently**, and the probe
   reports the units half alone. I am not running this until it passes.
2. **Box verified before AND after each session**, with the control probed independently
   beforehand. A session launched on an unverified box does not count.
3. **Rep validity, fixed NOW and not after the data:** a rep whose recorded box reading is
   **≥ 25% busy** is discarded before any arm statistic is computed. 25% is not a number I chose
   from E43's readings — it is just above the **22.5%** maximum that E39's and E40's own accepted
   sessions actually ran at, a measured range, per §0.1. **If fewer than 3 reps survive in a
   session, that session is VOID** (three reps is this programme's standing minimum for a rate).
4. **Both orders must agree** on the second cell — which side of the peak `V = 32768` falls on —
   or the cell is **unresolvable** and is reported as such. The first cell is reported from
   `forward` with `reversed` alongside; E36's run-2 rule governs any disagreement beyond
   dispersion.
5. **Nothing else moves.** Same arms, same files on disk, same engine, same `k` grid, same
   bands, same predictions. The vocabulary half is finished and is not re-measured.

## A.4 What does NOT change

The units half stands: `G-E43B`, `G-E43C` and `G-E43D` all fired and none of them depends on the
stopwatch. The measured `bytes/token` column is unaffected by anything in this addendum. And the
scope in §8 is unchanged — **still not one BPB, and E43 still cannot move the T4 ask.**
