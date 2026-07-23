#!/usr/bin/env python3
"""Recover the DECIDING metric offline, because the uploaded package could not compute it.

WHAT WENT WRONG. The data packages uploaded to Kaggle were a generation OLDER than the local ones: they
carried 6 files, not 7, missing p62_s0.u16. The trainer treats P62 as optional (the MVE packages had no
code-val), so with the file absent it printed no [DECIDING] line and still wrote STAGE1-DONE. Both arms
therefore completed cleanly and produced only the tail val -- the one number the prereg forbids as a
decider. The failure is the house pattern: a healthy run answering a different question than the one that
decides.

WHY OFFLINE RECOVERY IS LEGITIMATE, not a patch over the hole. The P62 BPB is a deterministic function of
(trained weights, p62 stream, tokenizer). The stage-C exit checkpoint IS those weights -- stage C is the
only stage these arms ran, so the saved final .pt is exactly the model whose P62 the trainer would have
printed. This recomputes it with the identical code path (bpb_on + the planted byte invariant), so the
number is the one that should have appeared at stage exit, not an approximation of it. It is declared as
recovered-offline because the uploaded artifact was deficient -- a deviation on the record, not a silent
substitution.

Run: python benchmarks/phase64/data/ws3_recover_p62.py
"""
import io, json, math, os, sys, zipfile
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase64", "mve"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
from mve_model import MVEStudent, S0            # noqa: E402
from cartography import Bpe                      # noqa: E402

RES = os.path.join(ROOT, "kaggle_rung1", "results")
# Third field: the tail-val BPB Kaggle actually PRINTED for this arm. It is the planted control on the
# recovery itself -- if the offline pipeline reproduces a number Kaggle emitted, using the same weights
# and the same code, then the P62 number it also produces (which Kaggle could not emit) is trustworthy by
# the same machinery. A recovered decider nobody can cross-check is an assumption.
ARMS = [("arm1_V2048", 2048, "account_1/arm1_V2048", 1.1452),
        ("arm2_V4096", 4096, "account_2/arm2_V4096", 1.1328)]
SEQ = 512
BATCH = 8
EVAL_TOK = 200_000        # mve_train --eval-tok default; the cap the stage-exit tail val used


def load_extracted(dirpath):
    """torch.save writes a zip; Kaggle handed it back EXTRACTED (data.pkl + data/ + version). Re-zip in
    memory under a single top-level folder -- the exact layout torch.load expects -- and load that."""
    buf = io.BytesIO()
    root = os.path.basename(dirpath.rstrip("/\\"))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for base, _, files in os.walk(dirpath):
            for fn in files:
                full = os.path.join(base, fn)
                rel = os.path.relpath(full, dirpath).replace("\\", "/")
                z.write(full, f"{root}/{rel}")
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def bpb_on(model, stream, V, el, dev, cap=0):
    """The trainer's bpb_on, transcribed verbatim so the number is the one stage-exit would have printed:
    cap=0 = whole stream, full windows batched, ragged remainder alone, bytes from exp_len."""
    model.eval(); bits = 0.0; nb = 0
    with torch.no_grad():
        n = (min(cap, len(stream) - 1) if cap else len(stream) - 1)
        W = SEQ; full = n // W
        for i in range(0, full, BATCH):
            k = min(BATCH, full - i)
            st = [(i + j) * W for j in range(k)]
            x = torch.from_numpy(np.stack([stream[p0:p0+W] for p0 in st])).to(dev)
            y = torch.from_numpy(np.stack([stream[p0+1:p0+1+W] for p0 in st])).to(dev)
            lg, _ = model(x, y)
            bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
            nb += int(el[y.reshape(-1).cpu()].sum())
        rem = n - full * W
        if rem > 0:
            p0 = full * W
            x = torch.from_numpy(stream[p0:p0+rem][None]).to(dev)
            y = torch.from_numpy(stream[p0+1:p0+1+rem][None]).to(dev)
            lg, _ = model(x, y)
            bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
            nb += int(el[y.reshape(-1).cpu()].sum())
    return bits / max(nb, 1), nb


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"recovering the DECIDING P62 code-val BPB from the saved stage-C checkpoints  (dev={dev})\n")
    rows = []
    for arm, V, pkg, kaggle_tail in ARMS:
        ckdir = os.path.join(RES, arm)
        if not os.path.isdir(ckdir):
            print(f"  {arm}: no checkpoint at {ckdir} -- skipping"); continue
        ck = load_extracted(ckdir)
        data = os.path.join(ROOT, "kaggle_rung1", pkg, "data")
        p62 = np.fromfile(os.path.join(data, f"p62_s0.u16"), dtype=np.uint16).astype(np.int64)
        ids = np.fromfile(os.path.join(data, f"ts_s0.u16"), dtype=np.uint16).astype(np.int64)
        el = torch.tensor(Bpe.load(os.path.join(data, f"bpe{V}_ts.bin")).exp_len, dtype=torch.long)
        meta = json.load(open(os.path.join(data, "meta_s0.json")))
        p62_bytes = int(meta["p62_bytes"]); ntr = int(meta["n_train_tok"])

        model = MVEStudent(V, **S0).to(dev)
        # The checkpoint's own embedding width names its vocabulary: assert it against V BEFORE loading, so
        # a swapped checkpoint is a named refusal, not a silent strict=False shape drop.
        ck_vocab = ck["model"]["emb.weight"].shape[0]
        assert ck_vocab == V, f"{arm}: checkpoint vocab {ck_vocab} != expected {V} -- wrong file"
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        real_missing = [m for m in missing if "recall" not in m]   # recall slot is absent at stage C by design
        if real_missing or unexpected:
            print(f"  {arm}: load missing={real_missing[:4]} unexpected={unexpected[:4]}")

        # PLANTED CONTROL first: reproduce the tail val Kaggle printed, same weights, same code, same cap.
        tail = ids[ntr:]
        tail_bpb, _ = bpb_on(model, tail, V, el, dev, cap=EVAL_TOK)
        d_tail = abs(tail_bpb - kaggle_tail)
        faithful = d_tail < 0.002
        print(f"  {arm}  V={V}")
        print(f"    tail-val cross-check : offline {tail_bpb:.4f}  vs Kaggle-printed {kaggle_tail:.4f}  "
              f"|delta| {d_tail:.4f}  -> {'REPRODUCES (pipeline faithful)' if faithful else 'DIVERGES -- recovery suspect'}")

        bpb, nb = bpb_on(model, p62, V, el, dev)
        total = nb + int(el[int(p62[0])])
        ok = (total == p62_bytes)
        rows.append((arm, V, bpb, nb, total, p62_bytes, ok and faithful))
        print(f"    P62 code-val BPB [DECIDING]  = {bpb:.4f}")
        print(f"    over {nb} bytes + {total-nb} B unscored first token = {total} "
              f"(declared {p62_bytes})  [byte invariant {'HOLDS' if ok else 'FAILED'}]\n")

    if len(rows) == 2:
        (a1, v1, b1, *_), (a2, v2, b2, *_) = rows
        lo, hi = (a1, b1), (a2, b2)
        if b2 < b1: lo, hi = (a2, b2), (a1, b1)
        print(f"  screening result (P62, the decider): {lo[0]} lower at BPB {lo[1]:.4f} vs {hi[1]:.4f} "
              f"-- delta {hi[1]-lo[1]:+.4f}")
        print(f"  sigma_seed = 0.005; delta is {abs(b1-b2)/0.005:.1f} sigma "
              f"({'decisive' if abs(b1-b2) > 0.010 else 'INSIDE noise -- not separable'})")
        if not all(r[6] for r in rows):
            print("  !! a byte invariant FAILED -- the recovered number is NOT comparable; investigate before quoting")
    print("\nSTOP. Recovered offline; no commit. Deciding metric is declared recovered-offline in the report.")


if __name__ == "__main__":
    main()
