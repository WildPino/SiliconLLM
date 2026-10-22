"""C50-S4: one arm's training, in sessions that checkpoint and resume (S4 protocol, training recipe).

- **Data:** block 1's shards 0-469 in stored order, documents packed to 8,192 tokens per step, one epoch. Validation:
  the 33-bucket KL on shards 470-471 at step 0, every 250 steps and at the end.
- **Recipe:** AdamW (0.9, 0.999), weight decay 0, lr 1e-4 with 30 warm-up steps then cosine to 1e-5 at the last step,
  global gradient-norm clip 1.0 after unscaling, fp16 forward, fp32 trainables, gradient scaling, seed 20261016.
- **Arm (a):** the copies' bias moves by 0.01 x sign(mean load - load) after each step; the copies' share is logged.
- **Arm (b):** the bank is read with the value leave-one-out at training positions, (n v - own) / (n - 1), own = that
  position's sum_k p_k W_out[k]; validation reads the full mean.
- **Start:** a fresh run compares the arm with T0 on 512 tokens and stops (VOID, exit 3) unless max abs dlogit is 0.
- **Sessions:** after `--cap-h` hours of this session, or every 100 steps, the trainables, optimizer, scaler, step and
  arm (a)'s biases go to `ckpt.pt`; `--resume` continues from one. A non-finite loss saves and stops (VOID, exit 4).

    python -I benchmarks/chatbot50/c50_s4_train.py --arm a|b|c48|c496 --gguf G --teacher DIR --corpus TSV
        [--bank DIR] --out DIR [--resume CKPT] [--cap-h 9.5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_s4_bank import BankReader  # noqa: E402
from c50_s4_student import Student, Teacher, bank_none, identity, pack  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402

SEED, TOKENS, LR, LR_END, WARMUP, CLIP, RATE_A = 20261016, 8192, 1e-4, 1e-5, 30, 1.0, 0.01
TRAIN_SHARDS, VAL_SHARDS, VAL_EVERY, CKPT_EVERY = range(470), range(470, 472), 250, 100
ARMS = {"a": ("a", 0), "b": ("b", 0), "c48": ("c", 48), "c496": ("c", 496)}


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    f = (step - WARMUP) / max(1, total - 1 - WARMUP)
    return LR_END + 0.5 * (LR - LR_END) * (1 + math.cos(math.pi * min(1.0, f)))


def bank_inputs(S: Student, bank: BankReader | None, ids, segs, tid=None, tlp=None):
    """Arm (b)'s bank rows for one packed batch; with tid/tlp (training), leave-one-out at the scored positions."""
    if S.arm != "b":
        return None, None
    if bank is None:
        return bank_none(S, torch.from_numpy(ids))
    v, hit, row, cnt = bank.read(ids, segs)
    if tid is not None:
        keep = np.ones(len(ids), dtype=bool)
        for a, b in segs:
            keep[b - 1] = False
        w = S.W.get("output.weight")                      # fp16, on the head's device
        own = F.embedding_bag(tid.to(w.device), w, per_sample_weights=tlp.exp().to(w.device, w.dtype),
                              mode="sum").float().cpu()
        k = np.flatnonzero(keep)
        sel = hit.numpy()[k]
        i = k[sel]
        n = torch.from_numpy(cnt[i]).float()[:, None]
        assert (n >= 2).all()
        v[i] = (n * v[i] - own[sel]) / (n - 1)
    return v, hit


@torch.no_grad()
def val_kl(S: Student, V: list, bank: BankReader | None) -> float:
    tot, npos = 0.0, 0
    for batch in V:
        ids, segs, tid, tlp = pack(batch)
        bv, bh = bank_inputs(S, bank, ids.numpy(), segs)
        loss, n = S.loss(ids, segs, tid, tlp, bv, bh)
        tot += float(loss)
        npos += n
    return tot / npos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--cap-h", type=float, default=9.5)
    ap.add_argument("--stop-at", type=int, default=0, help="end this session (PAUSED) once this step is reached")
    ap.add_argument("--cpu-smoke", type=int, default=0,
                    help="local: CPU, fp32, this many steps of 128 tokens on shard 0, 48-token identity, 1 val batch")
    a = ap.parse_args()
    t_session = time.time()
    smoke = a.cpu_smoke > 0
    devs = ("cpu", "cpu") if smoke else ("cuda:0", "cuda:1")
    kind, rank = ARMS[a.arm]
    cfg = {"cell": "C50-S4 train", "arm": a.arm, "kind": kind, "rank": rank, "torch": torch.__version__,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if not smoke else [],
           "tokens_per_step": TOKENS, "lr": LR, "lr_end": LR_END, "warmup": WARMUP, "clip": CLIP, "seed": SEED,
           "optimizer": "AdamW (0.9, 0.999) wd 0", "rate_a": RATE_A if kind == "a" else None,
           "train_shards": [TRAIN_SHARDS[0], TRAIN_SHARDS[-1]], "val_shards": [VAL_SHARDS[0], VAL_SHARDS[-1]],
           "bank": str(a.bank) if a.bank else None, "resume": str(a.resume) if a.resume else None,
           "cap_h": a.cap_h, "stop_at": a.stop_at, "smoke": a.cpu_smoke}
    print("CONFIG", json.dumps(cfg), flush=True)
    if not smoke:
        assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert (kind == "b") == (a.bank is not None)
    a.out.mkdir(parents=True, exist_ok=True)
    rec_path = a.out / "train.json"

    def finish(status: str, extra: dict, code: int) -> int:
        rec = {**cfg, "status": status, **extra, "session_s": round(time.time() - t_session, 1)}
        rec_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print("RESULT", json.dumps({k: v for k, v in rec.items() if k != "log"}), flush=True)
        return code

    tr_shards = [0] if smoke else list(TRAIN_SHARDS)
    va_shards = [1] if smoke else list(VAL_SHARDS)
    T = Teacher(a.teacher, a.corpus, tr_shards)
    batches = list(T.batches(TOKENS if not smoke else 128))
    first = next(T.docs())[1][:48 if smoke else 512]
    V = list(Teacher(a.teacher, a.corpus, va_shards).batches(TOKENS if not smoke else 128))[:1 if smoke else None]
    del T
    total = len(batches) if not smoke else min(len(batches), a.cpu_smoke)
    print("DATA", json.dumps({"steps": total, "val_batches": len(V)}), flush=True)
    W = T0Quant(a.gguf, devs, wdtype=torch.float32 if smoke else torch.float16)
    torch.manual_seed(SEED)
    S = Student(W, kind, rank, seed=SEED, bias_rate=RATE_A) if kind == "a" else Student(W, kind, rank, seed=SEED)
    bank = BankReader(a.bank) if kind == "b" else None
    params = [p for p in S.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=not smoke)
    log, vals, step = [], [], 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        S.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, log, vals = ck["step"], ck["log"], ck["vals"]
        assert ck["arm"] == a.arm and ck["total"] == total, (ck["arm"], ck["total"], total)
        print("RESUME", json.dumps({"step": step}), flush=True)
    else:
        ident = identity(S, torch.from_numpy(first))
        cfg["identity_step0"] = ident
        print("IDENTITY", json.dumps(ident), flush=True)
        if ident["max_abs_logit"] != 0.0:
            return finish("VOID: the arm is not T0 at step 0", {}, 3)
        vals.append({"step": 0, "val_kl": val_kl(S, V, bank)})
        print("VAL", json.dumps(vals[-1]), flush=True)

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"arm": a.arm, "total": total, "step": step, "model": S.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "log": log, "vals": vals}, tmp)
        os.replace(tmp, path)

    n_train = sum(p.numel() for p in params)
    cfg["trainable"] = n_train
    while step < total:
        if time.time() - t_session > a.cap_h * 3600 or (a.stop_at and step == a.stop_at):
            save(a.out / "ckpt.pt")
            return finish("PAUSED", {"step": step, "total": total, "log": log, "vals": vals}, 0)
        ids, segs, tid, tlp = pack(batches[step])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total)
        t1 = time.time()
        bv, bh = bank_inputs(S, bank, ids.numpy(), segs, tid, tlp)
        loss, npos = S.loss(ids, segs, tid, tlp, bv, bh)
        if not torch.isfinite(loss):
            save(a.out / "ckpt.pt")
            return finish("VOID: non-finite loss", {"step": step, "total": total, "log": log, "vals": vals}, 4)
        scaler.scale(loss / npos).backward()
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        e = {"step": step, "loss": round(float(loss) / npos, 5), "pos": npos, "lr": opt.param_groups[0]["lr"],
             "gnorm": round(gn, 4), "scale": scaler.get_scale() if not smoke else 1.0}
        if kind == "a":
            e["copy_share"] = round(S.load_step(), 5)
        e["s"] = round(time.time() - t1, 3)
        log.append(e)
        step += 1
        if step < 4 or step % 20 == 0:
            print("STEP", json.dumps(e), flush=True)
        if step % VAL_EVERY == 0 or step == total:
            vals.append({"step": step, "val_kl": val_kl(S, V, bank)})
            print("VAL", json.dumps(vals[-1]), flush=True)
        if step % CKPT_EVERY == 0 and step < total:
            save(a.out / "ckpt.pt")
    torch.save({k: v for k, v in S.state_dict().items()}, a.out / "final.pt")
    save(a.out / "ckpt.pt")
    trained = vals[-1]["val_kl"] < vals[0]["val_kl"]
    return finish("COMPLETE", {"step": step, "total": total, "val_first": vals[0]["val_kl"],
                               "val_last": vals[-1]["val_kl"], "trained": trained, "log": log, "vals": vals}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
