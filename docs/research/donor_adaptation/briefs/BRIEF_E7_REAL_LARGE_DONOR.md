# E7 — a real large donor, end to end: the largest trained weights on this disk

**Pre-registered. Pushed before the export was run.** Opened by INDEX item **0-bis**.

---

## 1. Why this exists

E6 showed a donor generating text and could only do it at **0.5 B and 1.5 B**. Every measurement
at target scale in E3, E4 and E5 was taken on **`T10`, a synthetic shape file** — correct
dimensions, untrained weights. So the programme's headline currently reads:

> "10 B at 50 tok/s" has a **measured speed** and **no measured model**.

That is the gap this probe attacks, with the largest **real** weights available offline.

## 2. The donor

`Qwen/Qwen2.5-Coder-7B`, already in the HF cache (14.5 GB of bf16 safetensors). Nothing else
locally is bigger: `Qwen3-8B`, `Qwen3-30B-A3B` and `Qwen3-Next-80B-A3B` are **config stubs of 1 MB**,
not weights.

| | Coder-7B | `T10` (synthetic) |
|---|---|---|
| `d_model` / `d_ffn` | 3584 / 18944 | 4096 / 14336 |
| layers | 28 | 48 |
| heads (q/kv) | 28 / 4 | 32 / 8 |
| head_dim | 128 | 128 |
| vocab | 152064 | 32768 |
| tied embeddings | **no** | no |
| **active weights/token** | **≈ 7.07 B** | 10.603 B |

**7.07 B is 67% of the target size.** E7 does not claim 10 B. It converts "no measured model at
scale" into "a measured model at 7.07 B", which is the largest step available without a download.

## 3. What gets built

| artefact | export | why |
|---|---|---|
| `qwen25-coder7b_f32.bin` (~30 GB) | `--quant fp32 --fold layers` | the arm that can be checked against PyTorch. The largest file this runtime has ever been handed — E1 found a 32-bit `ftell` bug that made everything over 2 GB unloadable, and 30 GB is a new size class |
| `qwen25-coder7b_p.bin` (~5.7 GB) | `--quant packed --rule R0 --head-ternary --fold layers` | the **deliverable format**, and the arm the speed number comes from |

**`--rule R0`, not E1's R3.** R3 needs 32 calibration sequences through a 7.6 B fp32 model on CPU,
which is an hour of arithmetic that cannot change a **speed** result: the packed format is a fixed
2 trits/byte and the kernel scans it densely, so throughput is rule-independent. The quality of the
ternary conversion is not what E7 measures — E6 §2 already showed what it is.

## 4. The gates

| gate | test | threshold, fixed now |
|---|---|---|
| **G-L** | the engine consumes **exactly** the file, and its header matches `config.json` | exact byte count; every one of D, F, L, NH, NKV, HD, V |
| **G-P** | `--logits` from the fp32 arm vs PyTorch fp32, first 8 positions of a prompt | **rel L2 ≤ 1e-4** and **top-1 agreement 1.0000** — E1's circle at 5× the size it was gated at (E1 read 2.8e-06 / 1.5e-05 at 0.5–1.5 B) |
| **G-G** | `--generate` on the fp32 arm vs PyTorch greedy, E6's five frozen prompts, 32 tokens | **≥ 90% of 160 positions**, any divergence at a REF top-2 gap < 1e-2 |
| **G-C** | planted control: the packed arm must **not** pass G-G | else G-G is measuring prompt predictability and E7 is void |
| **G-S** | `--bench` on the packed arm, contexts 300 and 800, **≥ 3 repetitions**, dispersion reported | see §5 — a **pre-registered band**, not a free reading |

## 5. The prediction, written before the run

E3 measured `T10` at **3.090 tok/s** @300: 313.880 ms of weight path + 10.576 ms of `f`. Its
delivered weight rate is **32.8 G-weights/s**. Carrying that rate to Coder-7B's 7.07 B:

    weights   7.07 / 10.603 x 313.880  =  209.3 ms
    f         10.576 x (28x28)/(48x32) =    5.4 ms      (layers x heads, E4's engine is faster still)
    total                                 214.7 ms  ->  4.66 tok/s

**Pre-registered band at 300 tokens of context: 4.0 – 5.4 tok/s.** Outside it, either the
weight-rate model or the measurement is wrong, and E7 says which before it says anything else.

**This band is also the point.** If it holds, a *real* 7 B donor runs about **11× short of
50 tok/s**, on an engine whose attention organ has already been optimised 6.5× (E4) and taken apart
(E5). It would say, in measured units on trained weights, that the wall is the **weight path** and
nothing in the attention organ can move it.

## 6. What E7 will not claim

- **Not 10 B.** 7.07 B active is 67% of the target and the brief says so in §2. The remaining step
  needs a download the disk has not got.
- **Not a quality result.** R0 ternary at 7 B will not write language; E6 §2 established what the
  ternary conversion costs and E7 adds nothing to it.
- **Not that the fp32 arm is a deliverable.** A 30 GB fp32 file is a measuring instrument.
