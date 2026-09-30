# METH-216: local leaf keys miss four raw share gates

Protocol/runner frozen at `b2fe0df`. Original BF16 E1280 teacher/control
logits match exactly on eight prompts; projection matrices equal the
original child buffers. Fit: 3072 raw / 1024 ChatML sequences,
parent-local spherical keys and separate annealed causal-mode biases.
The METH-214 rejected compact core is not used.

| Fit metric, all 24 layers | Raw | ChatML |
| --- | ---: | ---: |
| Layers failing max-load ratio | 0 | 0 |
| Worst max-load ratio, limit 1.25 | 1.003802 | 1.005573 |
| Layers failing hot-parent share | 4 | 0 |
| Worst hot-parent share, limit 25% | 30.163% | 11.811% |
| Minimum child coverage, limit 4000 | 10797 | 9884 |
| Minimum standardized score advantage, limit 0.05 | 1.006424 | 0.956334 |
| Minimum unbiased argmax agreement, limit 15% | 52.886% | 52.423% |

Every other fit gate passes. Raw share fails at zero-based layers
12/16/17/20: maxima 30.163/28.811/26.706/29.874%. All ChatML fit gates
pass. Frozen decision: old/new reserved and source screens stay closed.
No B specialists are trained; no native timing is licensed.

The changed representation stores feasible keys and retains fit content
signal, but fails per-parent concentration. Diagnose these specific raw
parents before retrying: exact-state multiplicity, final-temperature
soft/hard counts, score ties/margins and support. METH-205 used different
projection coordinates and only 1024 raw draws; its recurrence finding
cannot substitute for this new diagnosis. Do not blindly add iterations
or train against this failed router.

RTX 3060: 732.453 s, 2.945 GB peak allocated GPU, 4.434 GB end RSS;
largest logged RSS 11.118 GB. Router: 40,354,576 bytes, all readback
exact, SHA256 `c92d603885513ec30d2eaee36975760df243dffb30521830524e32512ec3397e`.
Post-run read-only key norms: 0.999999821–1.000000119, all finite.
Keys: 35,389,440 bytes; two mode biases: 2,211,840; diagnostic projection
copy: 2,752,512. Hypothetical native feature reuse addresses 114048 extra
selected weight bytes/token. Reuse was not implemented/timed here;
larger-parent selection, 100B quality and constant full-token cost remain open.

[Raw result](meth216_local_child_keys_result.json) SHA256:
`b2669d8ad250cc6e980717c94ec95641f8469e04437c1937ec9f643351000f0b`.
Handle 5284 completed successfully; no process is running.

Terminology correction: the protocol calls the source projection
"learned". It is the frozen existing projection in the trained E1280
checkpoint: METH-95 registers it as a buffer; METH-99/107 train child B
only. This does not assert learned projection coefficients or change
any source bytes, fit rule or result.
