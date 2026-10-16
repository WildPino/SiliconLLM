# C50-A0 result: pinned donor screen

16 October 2026. Cell A0 of the [CHATBOT50 roadmap](ROADMAP.md) §5. Metadata
only: no tensor payload was requested and no weights were acquired.

**Verdict: the Phase-A donor is `inclusionAI/Ling-mini-2.0` at revision
`a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab`**, by the decision rule fixed in the
roadmap. Its tokenizer is byte-for-byte the same parsed object as Ling-flash-2.0's
(the Phase-C teacher), and the Ling-3.x flash models share every ordinary token
id with it.

## Protocol and deviation

The decision rule was fixed in ROADMAP §5 before the screen ran. The roadmap
was committed and pushed (`23bfa20`) after the screen had run, not before:
a deviation from programme rule 5 (push before run), recorded here. The rule
text in §5 is unchanged from the version written before the screen.

## Instrument

- [`benchmarks/chatbot50/a0_donor_screen.py`](../../../benchmarks/chatbot50/a0_donor_screen.py):
  pins each repository revision through the Hugging Face API (the API
  `sha` must equal the pinned revision), fetches the small text files
  (config, generation config, tokenizer files, chat template), reads every
  safetensors shard header with HTTP range requests, sums coefficients per organ
  and prices one decode token with the policies and constants of
  [`byte_budget_ledger.py`](../../../benchmarks/chatbot50/byte_budget_ledger.py).
- Output: [`c50_a0_donor_screen_20261016.json`](../../../benchmarks/chatbot50/results/c50_a0_donor_screen_20261016.json)
  and the printed log [`.txt`](../../../benchmarks/chatbot50/results/c50_a0_donor_screen_20261016.txt).
- Cross-check: the header-derived Ling-mini-2.0 ledger (P0 826.0 MB, P2 544.5 MB,
  P4 461.0 MB per token at 2048) equals the hand-entered ledger row taken from
  METH407, so the generic organ classifier reproduces the earlier count.

## Candidate search (2026-10-16)

Hugging Face API searches by name (`A1B`, `A1.5B`, `A0.6B`, `A0.8B`, `A1.4B`,
`A1.3B`, `A1.2B`, `A1.7B`, `moe`, `Ling-3.0`, `Ling-3.1`, `Ling-mini`,
`Ling-tiny`, `Ring-mini`) and listings by author (inclusionAI, arcee-ai, yandex,
utter-project, Edge0, LiquidAI, ibm-granite, allenai). Instruct MoE checkpoints
with <=1.5B active parameters found, beyond the roadmap's three:

| Repository | Why it is or is not screened |
| --- | --- |
| `inclusionAI/Ling-3.0-tiny` (Aug 2026) | Screened: MIT, instruct, 7.89B stored, ~1.3B active (card), hybrid 3:1 KDA/MLA attention |
| `utter-project/EuroMoE-2.6B-A0.6B-Instruct-2512` | Screened: Apache-2.0, instruct, 2.61B stored, 0.6B active |
| `Edge0/Edge0-8B-A1B-preview` | Not screened: ships only an int4 MLX base plus adapters, derived from Ling-3.0-tiny-base; no source-precision checkpoint to measure quality against |
| `yandex/AliceAI-T5-35B-A0.6B` | Not screened: encoder-decoder (`is_encoder_decoder: true`), not a decoder chatbot; no licence in the card |
| Pruned/merged community variants (Qwen3-30B-A1.5B "High-Speed", Huihui-MoE, Qwen3.8-1.0B-A0.6B) | Not screened: not producer checkpoints with producer-reported scores |

## Ledger at context 2048 (DESK-MODEL, header-verified)

| Donor @ revision | License | Stored | P0 MB | P2 MB | P2 desk tok/s | P4 MB | P4 desk tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ling-mini-2.0 @ a810f64 | MIT | 16.256B | 826.0 | 544.5 | 52.6 | 461.0 | 60.9 |
| Ling-3.0-tiny @ 9a98e35 | MIT | 7.893B | 771.1 | 589.5 | 49.0 | 522.8 | 54.5 |
| EuroMoE-2.6B-A0.6B-Instruct-2512 @ 96e1982 | Apache-2.0 | 2.612B | 369.4 | 256.1 | 100.0 | 212.9 | 115.6 |
| Trinity-Nano-Preview @ b576b32 | OpenMDW-1.1 | 6.120B | 655.1 | 458.6 | 61.2 | 399.3 | 69.1 |
| LFM2.5-8B-A1B @ 5dd2260 | LFM Open License 1.0 | 8.468B | 1,046.7 | 838.9 | 35.4 | 707.3 | 41.5 |

Ling-3.0-tiny is smaller in storage but not cheaper per token: its KDA layers
carry five 2048x1536 projections each (attention/mixer 216 MB at Q4_K versus
118 MB for Ling-mini) plus an F32 recurrent state of 37.7 MB read and written per
token (18 layers x 16 x 128 x 128; the key/value head width of 128 is an
ASSUMPTION from `head_dim`, to be checked against the modelling code if this
donor is ever used).

## Decision rule applied

Rule (ROADMAP §5, A0): the Phase-A donor is the MIT/Apache instruct checkpoint
with the lowest P2 bytes at 2048 among those whose producer-reported scores are
not lower than Ling-mini's on every benchmark the two cards share, with at least
two shared benchmarks required; default Ling-mini-2.0.

Ling-mini-2.0's card (rev a810f64) reports LiveCodeBench (2408-2505) 34.8,
CodeForces 59.5, AIME25 47.0, HMMT25 35.8, MMLU-Pro 65.1, HLE 6.0 (bar-chart
labels in the card's figure).

| Candidate | MIT/Apache | Shared benchmarks with Ling-mini's card | Eligible |
| --- | --- | --- | --- |
| Ling-3.0-tiny | yes | HLE only (9.30, thinking mode). Its card reports HMMT-Feb26, not HMMT25; GPQA, IFBench and the AA indices are not on Ling-mini's card | **no** (one shared benchmark); its P2 bytes are also higher than Ling-mini's |
| EuroMoE-2.6B-A0.6B-Instruct-2512 | yes | none (the card reports no benchmark) | **no** |
| Trinity-Nano-Preview | no (OpenMDW-1.1) | not evaluated | **no** |
| LFM2.5-8B-A1B | no (LFM Open License 1.0) | not evaluated | **no** |

No candidate is eligible, so the default holds: **Ling-mini-2.0**.

Third-party scores (for example the Edge0 card's MMLU-Pro 65.8 for Ling-3.0-tiny
in fp16) are not producer-reported and were not used.

## Ling-mini-2.0 chat facts (rev a810f64)

- Licence MIT, not gated. Files: config, generation config, tokenizer.json,
  tokenizer_config.json (holds the chat template), special_tokens_map.json,
  4 safetensors shards (14,813 tensors, 16,255,643,392 coefficients).
- `eos_token` `<|role_end|>` = 156895 (config); generation config stops on
  [156892, 156895]; `pad_token` `<|endoftext|>` = 156892; `bos_token`
  `<|startoftext|>` (no `bos_token_id` in config). Template roles
  `<role>SYSTEM</role>`, `<role>HUMAN</role>`, `<role>ASSISTANT</role>`, `<role>OBSERVATION</role>`, with a
  thinking option defaulting to off.
- No MTP layer in the released chat checkpoint (`num_nextn_predict_layers: 0`).

## Tokenizer identity with the family (needed for Phase C)

Hashes are over parsed content (vocabulary, merges normalised to "a b" strings,
added tokens, pre-tokenizer, normaliser, and the whole file in canonical JSON).

| Model @ revision | Vocab | Merges | Pre-tok./normaliser | Added tokens | Whole parsed file |
| --- | --- | --- | --- | --- | --- |
| Ling-flash-2.0 @ 96f9b7a | same | same | same | same | **same** (only the file bytes differ) |
| Ling-3.0-flash @ fcc60a5 | same | same | same | 9 renamed | differs |
| Ling-3.1-flash @ a4ccd6b | same | same | same | 9 renamed | differs |
| Ling-3.0-tiny @ 9a98e35 | same | same | same | 9 renamed | differs |

The nine renamed added tokens are ids 156900-156908: Ling-mini's
`<|reserved_token_5|>` .. `<|reserved_token_13|>` become `<|fim_start|>`,
`<|fim_hole|>`, `<|fim_end|>`, `<think>`, `</think>`, `<arg_key>`, `</arg_key>`,
`<arg_value>`, `</arg_value>` in Ling 3.x. Every other id maps to the same
string, so ordinary text tokenizes identically.

**Consequence for Phase C (recorded, not decided):** Ling-flash-2.0 is an
exact-tokenizer teacher. Ling-3.0-flash (127.5B stored per the API) and
Ling-3.1-flash are id-compatible teachers apart from nine special tokens,
which widens the C1 options.

## Phase-B notes (no decision)

EuroMoE-2.6B-A0.6B is the cheapest per token (256 MB at P2, 100 desk tok/s) but
its card reports no quality; Trinity-Nano-Preview stays the cheapest
second-family candidate with reported scores (licence OpenMDW-1.1).
Ling-3.0-tiny would exercise KDA and MLA operators.
