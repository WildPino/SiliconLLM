# Mixed-recipient first divergence: quantization amplifies a primitive difference

15 October 2026. Completed the [frozen diagnostic](RECIPIENT_MIXED_DIAGNOSTIC_PROTOCOL_20261015.md)
at commit `1d35abe`; controller terminal exit 0. No cost or training ran.
The original [numerical rejection](RECIPIENT_MIXED_BASELINE_RESULT_20261015.md)
remains valid under its original reference.

## Identical trajectories, observed divergence

The instrumented native probe is byte-identical to the original failed probe;
all four complete reference-logit hashes also reproduce the failed audit.
This qualifies the trace as an observation of those exact trajectories.
Retained directory:
`results/native_expert_scaling/recipient_mixed_diagnostic_20261015`.
Native trace SHA256:
`bb965c40a21648b677f6b10f5ee780d15c07cc03a5ea47df4f759baa732c5db8`.
Reference trace SHA256:
`870dd12dd34f1c48c7e40f7036a62089fc65273c0584d4395796c617a2d2c296`.

The first F32 difference is position 0, layer 0, after SiLU*up. At that site
gate and up inputs are bit-identical. The first scale differences are also
at layer 0, with relative L2 1.58e-8. All activation codes throughout position
0 remain identical; its largest compared relative error is 9.32e-8.

The first activation-code disagreement occurs at position 1, layer 21,
down-projection input element 2805. Native/reference inputs differ by only
2.6077e-7, but lie on opposite sides of a half-integer quantization threshold:

| Value | Native | Original reference |
| --- | ---: | ---: |
| Input | -0.09306909143924713 | -0.09306883066892624 |
| Group scale | 0.006418542470782995 | 0.006418542470782995 |
| Signed activation code | -15 | -14 |

The down output's relative error becomes 0.00013017; the next layer Q output
already reaches 0.00294513. Later code mismatches and KV propagation grow.
This demonstrates amplification by a discrete quantizer. It does not establish
that the entire discrepancy has one primitive cause, or prove a codec bug.

## Stored primitive check and bounded next action

At the first SiLU site, recomputing the stable scalar formula with Python
binary64 `math.exp`, rounded to F32 at every prescribed F32 operation,
matches all 4864 native activations exactly. The original NumPy F32-exp path
differs at 814 elements. This check consumes stored gate/up vectors only.

The next [prospective scalar-reference repair](RECIPIENT_MIXED_SCALAR_REPAIR_PROTOCOL_20261015.md)
first extends that primitive check to all 192 recorded native SiLU sites.
Only if those native outputs are reproduced may a wrapper replace the
reference exponential primitive, keeping the original graph, executable,
packet and 1e-3/top1 gates. This is one documented repair, not an automatic
retry or permission to relax the threshold. Numerical qualification and the
complete cost envelope remain pending; the overall goal stays active.
