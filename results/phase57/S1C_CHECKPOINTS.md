# S1c checkpoints — the structured-`x_proj` evidence (release asset manifest)

The nine checkpoints backing the **adopted** low-rank `x_proj` result (`docs/PHASE64_TRAINING_PLAN.md` §12). They are
too large for git (278 MiB) and live as a **release asset**, per the backup rule: an artifact that is local-only and
backs a claim we keep is one hardware failure away from an unfalsifiable claim.

Release tag: **`checkpoints-s1c-v1`** · asset: `s1c_checkpoints_v1.zip` (278 MiB)

## What they are

A paired 3-seed A/B from the Inventor side-lab (`benchmarks/in_research/s1c_structured_xproj.py`): the SSM `x_proj`
projection replaced by a rank-*r* factorization, against a dense control, at the ~8.3 M sandbox scale (`D256 N96 L6
swa@5 dt16`, V=1024), 4000 steps, seeds 0/1/2.

Each file holds `model` (state_dict), `cfg` (hyperparameters + final `bpb`), and `top1_slice` (a 5120-token argmax
slice, the determinism witness). **No paths, usernames, or environment data** — checked before publication.

## The numbers, re-derived from the checkpoints themselves

Not copied from the report — read back out of the `cfg['bpb']` field of each file:

| seed | control (dense) | r=26 | Δ | r=52 | Δ |
|---|---|---|---|---|---|
| 0 | 0.875690 | 0.868289 | **−0.007401** | 0.879224 | +0.003534 |
| 1 | 0.881138 | 0.868807 | **−0.012331** | 0.872087 | −0.009051 |
| 2 | 0.878410 | 0.869596 | **−0.008814** | 0.872889 | −0.005520 |

- **r=26 mean Δ = −0.009515, winning 3/3 paired seeds** — reproduces the adopted claim (−0.0095) exactly.
- **σ_seed on the control = 0.002724** — reproduces the quoted 0.0027.
- r=52 mean Δ = −0.003679 ≈ 1.35 σ_seed: **not a claim**, consistent with the "neutral" reading it was given.

Read the scope with the claim: this is a **sandbox-scale, 4000-step** result. It is adopted as a *candidate*, and it
is re-tested as a declared A/B arm at the S0 boundary (`docs/PHASE64_RUNG1_PREREG.md` §4.2), where a null is the
pre-registered expected outcome of a regularization benefit fading with scale.

## Verify

```sh
sha256sum -c S1C_CHECKPOINTS.sha256
```

```
202070bb910728cc8523b0fff3c43860a8e363fd92e6d124ac57cea9dfaed648  s1c_ctl.pt
f6dac2c01a5fb2d7733159838464869a059b38a30e4862bcaeada8c8d731bee0  s1c_ctl_s1.pt
ccd54a79dd23b46a0248b7b88db20a3969c3dbe3e0c10886a4278320f0b9798e  s1c_ctl_s2.pt
f01ec92c4d36c903c551f30af2a9e453786029cd52b56c2246bbbba746e616bc  s1c_r26.pt
fdaf6dd281735eceff96231061673c8a721241f33676094d1b0c236e822fad72  s1c_r26_s1.pt
944e77295f739f6da49c1d187153790d021c806b942e3b8cded5efd50cfd6284  s1c_r26_s2.pt
484819000dc5580e9899d18ffa24a8c38daf1d30422faf94620881e6508226f2  s1c_r52.pt
e5c4c0ba7b9daa6c48c35b3b2657cf3a9fae7f85c9b2e26f6c575211ec77cc6c  s1c_r52_s1.pt
e810bff7d76ac5c577337ef192b221923838e42ed2c69761316b2a013a27db67  s1c_r52_s2.pt
```

Re-derive the table above from the assets alone:

```python
import torch, statistics
b = lambda f: torch.load(f, map_location="cpu", weights_only=False)["cfg"]["bpb"]
d = [b(f"s1c_r26{s}.pt") - b(f"s1c_ctl{s}.pt") for s in ("", "_s1", "_s2")]
print(statistics.mean(d), sum(1 for x in d if x < 0), "/3")   # -0.009515 3 /3
```
