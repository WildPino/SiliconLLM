# H2I Phase B -- one continuous T4 session

Scientific protocol: `BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md`, addendum B. This bundle is
weight-only R8. Do not enable activation int8/LUT, change k/layers, resume H1, or choose a
checkpoint by its progress BPB.

Upload this directory as one Kaggle dataset. Use a T4 GPU, copy the dataset contents into the
working directory if Kaggle mounted them read-only, then run from the directory containing
`MANIFEST.json`:

```bash
python3 h2i_qat.py \
  --manifest MANIFEST.json \
  --factors h0_trained3.npz \
  --labels labels_E256.npz \
  --stats h1_actstats.npz \
  --routers e37_routers_E256.npz \
  --phase-a h2i_applied_8L.json \
  --train h1_train.npz \
  --heldout h1_heldout.npz \
  --calib h1_calib.npz \
  --out /kaggle/working/h2i_trained_s1.npz \
  --steps 4000 --bs 2 --accum 8 \
  --lr 2e-4 --router-lr 3e-4 --aux 0.01 \
  --every 250 --seed 4242 --max-hours 11.0
```

Expected startup:

- manifest and all input hashes pass before model loading;
- R8 parity, STE-gradient, hard/soft k=E, real k16=560 and live-carve controls fire;
- E37 routers are installed exactly;
- first applied update moves `L03.gate`, `L24.down`, and `L03.router` within 40 attempts.

Stop and return the log if any control refuses, any non-finite microbatch appears, or no update
is applied within 40 attempts. Otherwise let the trainer stop itself at 11 hours. Return:

- `h2i_trained_s1.npz` and `h2i_trained_s1.json`;
- `h2i_trained_s1.checkpoint.npz` and `.json` if present;
- the complete stdout/stderr log.

The GPU BPBs are progress diagnostics. Do not select the best checkpoint. The final artifact,
whatever its curve says, is adjudicated once by `h2i_eval.py` on CPU fp32.
