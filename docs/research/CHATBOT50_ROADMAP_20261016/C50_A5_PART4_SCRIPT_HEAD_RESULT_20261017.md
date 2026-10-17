# C50-A5 part 4 result (exploratory, FIT): dropping non-Latin-script rows makes the head exact for Latin text, but at 166.6 MB it does not solve the budget

17 October 2026. Under the [part 4 protocol](C50_A5_PART4_SCRIPT_HEAD_PROTOCOL_20261017.md).
This part decides nothing: A5 has failed and the next step is the owner's choice
(COMMUNICATION item 3). No DEV or RESERVED data were read.

## Measured (`results/c50_a5_script_head_fit.json`; oracle control 1.0 on 7,971 positions)

**Kept set** (from `tokenizer.json` alone):

| Class | Rows |
| --- | --- |
| kept-script tokens | 96,667 |
| UTF-8 fragments | 2,211 |
| added tokens | 262 |
| **N_keep** | **99,140** |
| dropped-script tokens | 58,013 |
| padding rows | 31 |

**Exact restricted scan:** 99,140 × 1,680 B = **166.6 MB/token**, against 264.1 MB for the
full head.

**FIT top-1 recall of the exact restricted scan:**

| Category | Recall |
| --- | --- |
| code | 1.0 |
| technical_general | 0.8822 |
| prose | 1.0 |
| **all** | **0.9596** (3,219 misses) |

**Every miss is Chinese:**
- The most frequent missed tokens are `。` (97), `：` (39) and `，` (23), then common Chinese
  words.
- They fall in 5 technical documents from `kubernetes_website/content/zh-cn/…`.
- FIT12 has six documents with ≥ 1% dropped-script characters: five Chinese and one
  Korean (`content/ko/…`, 42% Hangul). The Korean document has no misses.

**Characterisation after reading** (not a selection; `results/c50_a5_script_head_fit_latin_docs.json`).
It uses the 30 FIT12 documents with < 1% dropped-script characters, 65,652 positions:

| Arm | K = 4,096 | 16,384 | 32,768 | MB/token at K = 32,768 |
| --- | --- | --- | --- | --- |
| exact restricted scan | 1.0 | 1.0 | 1.0 | 166.6 (no K) |
| r512 3.5-bit preview over kept rows + exact rerank | 0.98248 | 0.99796 | **0.99933** | **81.5** |
| the same preview over all rows (part 1b selection) | 0.98053 | 0.99740 | 0.99913 | 94.5 |

## Registered prediction scored

| Clause | Predicted | Measured | Scored |
| --- | --- | --- | --- |
| N_keep | 60k-100k | 99,140 | right (at the edge) |
| FIT recall | ≥ 0.9999 | 0.9596 | **wrong** |
| Exact restricted scan | 100-170 MB | 166.6 MB | right |
| Combined arm | below 70 MB at 0.999 | 81.5 MB at 0.99933 (Latin documents) | **wrong** |

- **Why the recall clause failed:** the protocol called FIT code plus English technical
  text plus English prose. FIT's technical category contains Kubernetes localisations
  (zh-cn, ko, id). I had not checked this.

## Reading (each sentence names its row)

- **The rule is exact for Latin-script text.** The exact restricted scan never misses on
  the 65,652 positions of Latin-script documents (first characterisation row).
- **It removes Chinese, Japanese and Korean output completely.** On Chinese documents the
  model does predict Chinese (the technical_general recall of 0.8822). Restricting the
  head is therefore a product-scope decision, which belongs to the owner. It is not a
  free byte saving.
- **It does not fix the budget.**
  - The exact restricted head costs 166.6 MB, 72 MB more than the failed preview
    (first table).
  - Combined with the preview, it saves 13 MB at slightly higher recall (second and
    third characterisation rows).
  - Every arm is far from the ledger's ~24 MB.
- **What does not move:** the A5 decision (COMMUNICATION item 3) is unchanged, and so is
  its arithmetic.
  - With the combined arm, P4 would be 461.0 − 23.8 + 81.5 = 518.7 MB, about 54.9 desk
    tok/s.
  - 50 tok/s would still need ≥ 0.91 of desk.
