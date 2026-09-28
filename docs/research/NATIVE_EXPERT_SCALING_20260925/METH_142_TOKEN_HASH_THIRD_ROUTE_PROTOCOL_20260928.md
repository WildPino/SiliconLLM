# METH-142: calibration-free ten-way context-hash route-load screen

METH-140/141 show that fitted scalar deciles shift badly across source
and context length. This experiment tests a deterministic ten-way
partition that has no learned or fitted thresholds. It leaves the
pretrained donor, E1280 geometry, first two routers, four gates and
source-child B factors unchanged. The route is a *development load
screen* on already viewed sources; it does not establish useful
specialization or quality.

For token ID `t` at zero-based position `p` within the current
nonoverlapping input window, let `u` be the preceding token ID in that
window (zero at `p=0`), `c` the selected E1280 child ID and `l` the
zero-based layer. Compute unsigned 64-bit arithmetic modulo `2^64`:

```
z = 142142 ^ (t*C1) ^ (u*C2) ^ (p*C3) ^ (c*C4) ^ (l*C5)
z = z + C1
z = (z ^ (z >> 30)) * C2
z = (z ^ (z >> 27)) * C3
z = z ^ (z >> 31)
grandchild = 10*c + (z % 10)

C1 = 0x9E3779B97F4A7C15
C2 = 0xBF58476D1CE4E5B9
C3 = 0x94D049BB133111EB
C4 = 0xD6E8FEB86659FD93
C5 = 0xA5A3564E27F8862D
```

Bind the METH-126 exact BF16 bank and METH-107 centered checkpoint
by their existing hashes. Check E1280 control BF16 parity against its
centered teacher on the eight METH-121 prompts. Capture the original
child IDs and apply only the hash rule offline; require every hashed
ID divided by ten to equal that child. Cloned B/A would then preserve
the selected factor algebraically, but this screen does **not** run a
hash-routed full model or assert its logits.

Use the same six fixed cells as METH-141: 256 H0 raw training rows at
their 128-token METH-136 draw windows and full 512-token rows, and
the 24 METH-121 and 24 METH-133 documents in nonoverlapping windows
of at most 128 or 512 tokens. Reset `p` and `u` at every window. Do
not fit the hash or choose its constants after seeing these outputs.
For each cell and layer save token count, coverage, maximum/mean
load, candidate/control ratio, top IDs and worst within-parent
grandchild share for parents selected at least 250 times.

Require in **all six cells and every layer** candidate/control
maximum-to-mean load ratio <=1.25, at least 4,000 selected
grandchildren, and worst hot-parent share <=25%. If any gate fails,
stop this hash design. If all pass, preregister a native CPU router
and full-model clone-parity implementation, then test learned B and
new-source quality. Uniform traffic alone cannot prove that random
partitions learn useful distinct experts. The current six cells were
already viewed for routing in METH-141, so even a pass needs a fresh
source-held-out load check before promotion.

Run locally on the RTX 3060. Stop at 15 minutes, 10.5 GiB allocated
GPU memory, 20 GiB process RSS or 1 GB new disk. No T4.
