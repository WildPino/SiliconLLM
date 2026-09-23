# STRAT-01 GigaChat 3.1 Rung-2C attention cross-input result

**Verdict:** `QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT`.

The layer-1 causal-attention plus V-B operator passes on the exact captured
reference boundary. The admitted C query residual alone and the admitted C KV
residual alone each make `kqv_out-1` fail. This is a valid zero-donor result;
do not rewrite attention/V-B or repeat the Rung-2C producers.

## Frozen-arm result

All arms are compared with the captured reference `kqv_out-1.full` under the
unchanged Rung-2C limits, NRMSE `<= 0.002` and normalized maximum `<= 0.01`.

| arm | NRMSE | normalized maximum | result |
|---|---:|---:|---|
| C query / C KV | `0.0061361676` | `0.0030491103` | FAIL; exact native replay |
| reference query / reference KV | `0.0003458783` | `0.0011237641` | **PASS** |
| C query / reference KV | `0.0021686789` | `0.0028514977` | **FAIL** |
| reference query / C KV | `0.0059545735` | `0.0036754120` | **FAIL** |

The query-only breach is narrow in aggregate but real under the frozen gate;
its largest token NRMSE is token 2 at `0.0052349683`. The KV-only arm is much
larger and reproduces the dominant native-error pattern: tokens 3 and 4 reach
NRMSE `0.0088885575` and `0.0081652233` respectively. These token-local values
are descriptive; the preregistered aggregate gates determine the verdict.

## Identity and causal controls

- C/C output is byte-identical to the immutable Rung-2C C target, SHA-256
  `7e6d44661f21dee62b449b44e1939b59d8ad0799c4ccf6779d77a01a48e3ee41`.
- The frozen C/reference metrics reproduce within `1e-12`.
- Reference and C `Vcur-1` are byte-exact prefixes of their respective
  `Kcur-1` rows.
- One-byte mutations of all six Q/K/V inputs are refused.
- Mixed-arm label swapping is detected.
- The token-7 reference-query negation rejects at NRMSE `0.8918076435`.
- The reference KV row-0/row-7 swap rejects at NRMSE `1.0939986236`.
- The accepted layer-1 V-B descriptor is Q4_K `[512,192,32]`, offset
  `326098944`, file offset `332201856`, span `1769472` bytes.
- Donor graph executions: `0`.

## Canonical evidence

The scientific invocation ran from HEAD
`2490aa994f0f5c48baed7af8c375afbd6c5ad639`. Its immutable directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_cross_input_20260923/`

| record | SHA-256 |
|---|---|
| adjudication | `97e71cd07575ef4ed79fd00be13634226d37474592b9528c5a92e1ba2131cf61` |
| run manifest | `94393a7d996b1b77996f2631817003c0e1031596d566ed49568457e698f542e1` |
| reference/reference output | `4ef78f758402092f0e4597ed39a42f73108444c5807a40aa85b422437fdf02ec` |
| C-query/reference-KV output | `396483de5d84f92a531911b31b70a0632585aa931ec959ff538248cedded053b` |
| reference-query/C-KV output | `576837918d4025d8ffb747c41cef77e997107bd12e4743368983fc71175f9809` |

## Consequence and next coordinate

The exact-reference arm rules out a standalone defect in the composed
attention/V-B operator at this boundary. The two mixed failures show that the
individually in-gate `Qcur-1` and `Kcur-1` residuals are not jointly required:
each is sufficient by itself, with KV the dominant contribution. Therefore:

1. do not change attention scoring, cache F16 semantics, softmax, or V-B;
2. do not relax the Rung-2C gate merely because Q/K/V passed individually;
3. do not attribute the failure to routing or experts;
4. localize how the accepted block-0 terminal residual propagates through the
   layer-1 attention normalization and Q/KV projections.

The next admissible cell is a zero-producer layer-1-start cross-input
diagnostic. It must feed captured reference versus C `l_out-0` into the
existing layer-1 pre-attention builder and inspect `attn_norm-1`, direct Q,
KV-A, `Qcur-1`, and `Kcur-1` before reusing the already validated
attention/V-B boundary. This separates inherited block-0 error amplification
from layer-1 projection semantics without rerunning either donor producer.

This result does not repair or promote Rung 2C and makes no claim about output
projection, MoE experts, later layers, tokenizer, logits, generation quality,
RAM, or rate. It does not update `SPEED_LEDGER.md`.
