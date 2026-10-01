# METH-225: nonlinear common improves, but fails the fixed accuracy screen

The [protocol](METH_225_NONLINEAR_COMMON_DISTILLATION_PROTOCOL_20261001.md)
and runner were frozen at `07c07a0`. Session22782 completed all4096
updates and exited0. No checkpoint selection, validation-time fitting,
schedule/width retry or new donor inference was performed.

## Stored-function outcome

| Fixed check | Measurement | Outcome |
| --- | ---: | --- |
| Initialized validation output SSE / teacher energy | .5769681512 | Control |
| Final validation output SSE / teacher energy | **.3729760963** | **Fail <=.10** |
| Final / initialized SSE | .646441 | Pass <=.90 |
| Paired128-window gain P05/P95,10,000 draws | .201937 / .206029 | Pass P05>0 |
| Every checkpoint tensor readback / decoded final forward | Exact | Pass |
| Trained layer12 native relative L2 median / worst,16 states | 2.858789e-7 / 3.414963e-7 | Pass |
| Other23 fixture layers' C outputs | Bit-exact to METH-224 | Pass |
| Mixed24-layer component timing median,six threads | **3.199365 ms/token** | Pass <=10ms |

Timing passes:3.199365,3.119704,3.239729ms/token. Only layer12 is trained;
the other23 are source-subset fixtures. This is not a trained24-layer
model or full accepted throughput. The validation windows were already
consumed development data from METH-222, not independent quality.

**Decision: stop this fixed common-distillation recipe before conditional
descendants.** The intermediate accuracy gate fails. No regularized child
bank, full causal model, generation or task is run. Do not relax either
the intermediate .10 gate or complete conditional-function .01 gate.
The improvement and numerical/cost pass do not establish transferred
useful capacity, useful increased n, or multi-family applicability.

This result rejects the fixed recipe, not every nonlinear common width,
optimizer, joint conditional model or function representation. The final
training-log block's mean loss=.309167 spans changing weights and is not
an exact final fit evaluation; it cannot prove a final fit/validation gap
or an irreducible capacity limit.

## Reproducibility and cost

[Raw result](meth225_nonlinear_common_distillation_result.json), SHA256
`90535d6d8ffcb9cfb08b45badfb1b093a103cc9fb427c1e65c86bc1686318f42`,
retains all128 initialized/final validation rows, updates, bound inputs,
bootstrap, all96 native segment checks and every numerical/timing value.
Report hashes refer to Windows CRLF working-tree bytes; `.gitattributes`
pins that serialization for the bound METH-222 through225 result reports,
so a checkout with another `core.autocrlf` setting retains those hashes.

- Initialized4,133,000-byte checkpoint:
  `846b8f33a8e5d77c58e6711da994a8a8091df256fe6367fa1f9c1a47c5c39f36`.
- Final4,133,000-byte checkpoint:
  `91af92127ac1511cf69013bc09429d9940e16e3be72e5c6f8191ed5f61898f2d`.
- Mixed native99,176,472-byte fixture:
  `ebb36809f3e7a190c2aadd47aa58a1432e79a9eca7c2f817767d9184116499be`.
- Native output:
  `d8d27c261465ec3626b1ce2626fe50710e1091ca8d948c56c900d87cc6976c79`.

These ignored local binaries/checkpoints remain in
`results/native_expert_scaling/` under the frozen command's paths.
Runtime43.156s after imports,ending RSS1.574GB,CUDA peak2.170GB on local
RTX3060; no T4. The checkpoint/fixture/check output total108,818,748bytes.
The partial training log is preserved. No project job remains active.

## Changed mechanism to investigate

The [conditional donor-tangent proposal](CONDITIONAL_DONOR_TANGENT_PROPOSAL_20261001.md)
moves the complete local function into the selected expert and derives
its coefficients from the pretrained function itself. It has no trained
common prerequisite and does not continue this failed recipe. Qualify
its native full-input affine operator and source-derivative apparatus
before freezing an E16/E160 test. It remains proposed; no result licenses it.
