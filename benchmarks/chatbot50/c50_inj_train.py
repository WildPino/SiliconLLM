"""C50 injection cell: one arm's training on v0 (T4 x 2). Design: C50_INJECTION_REVIEW_20261027.md.

Arms (all on v0, frozen; the trainables start so that the student is v0 bit for bit):
  e       N's branch (`c50_t2_n.NStudent`: 4 SwiGLU experts of T0's size in layers 1-19, down zero) with ReMoE's
          adaptive L1 on the gate. At step 0 the router bias c of each layer is set so that the gate fires on
          FIRE0 of the first replay batch's (token, layer) pairs (D = 0, so the output is still v0's)
  edense  the same branch, no fire cost, c = 0 (N's start)
  s       arm e on the answer-swapped passages (`train_S.jsonl`)
  qa1     arm e on one chat line per fact, "original question -> gold" (`train_QA1.jsonl`)
  l       LoRA r48 on the always-on modules (S4's c48 configuration, `c50_t1_train.AnchoredStudent`)
Each step is two passes, gradients summed:
- **replay:** 4,096 tokens of block 1's shards 0-469 in order, with N's anchored KL33 target (v0's own distribution
  where v0 is sure of the right token and F0 is no better, F0's top-32 elsewhere);
- **text:** 4,096 tokens of the arm's training documents, cross-entropy on every next token (a one-hot target in the
  same KL33, which is then the cross-entropy).
Exposures: every fact is seen EXPOSURES times. A fact's documents are cycled up to the largest count any fact has
(so a fact with 3 kept passages repeats 2 of them), and epochs = EXPOSURES / that count; the order is shuffled, seeded,
each epoch. A dead layer (fire below 0.01 for DEAD_RUN consecutive steps) stops the arm as VOID. Checkpoints `ck25.pt`, `ck50.pt` and `final.pt` at a quarter, a half and all of the epochs.
LR, warm-up, cosine, clip and fp16 are S4's, with the peak lr from `--lr` (picked by the dev pilot).

    python -I c50_inj_train.py --arm e --gguf V0 --teacher DIR --corpus TSV --inputs DIR --out DIR [--lr X] [--seed N]
    python -I c50_inj_train.py --arm e ... --mve       lambda0 grid, MVE_STEPS steps each, then exit
    python -I c50_inj_train.py --arm e ... --dev       the dev pilot: the dev facts only (train_dev_*.jsonl)
    --smoke-steps N (engineering smoke only, never a registered run): N steps, marks at N/4, N/2, N; with --mve, N
    steps at lambda0 1e-4 only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_train as S4  # noqa: E402
from c50_s4_student import identity  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402
from c50_t1_train import TAU, AnchoredStudent, TeacherL, anchored, pack  # noqa: E402
from c50_t2_n import AUX_LAYERS, NStudent, val_n  # noqa: E402

HALF_TOKENS, EXPOSURES, FIRE0, FIRE_TARGET, FIRE_CEIL, DEAD, DEAD_RUN = 4096, 100, 0.10, 0.10, 0.15, 0.01, 20
LAMBDA_GRID, ALPHA, MVE_STEPS = (1e-6, 1e-4, 1e-2), 1.2, 30
TEXT = {"e": "E", "edense": "E", "l": "E", "s": "S", "qa1": "QA1"}
V_SIZE = 157184


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def equalize(docs: list[dict]) -> list[dict]:
    """Every fact gets as many documents as the fact with the most: its own, cycled in order."""
    by = {}
    for d in docs:
        by.setdefault(d["fact"], []).append(d)
    k = max(len(v) for v in by.values())
    return [v[j % len(v)] for v in by.values() for j in range(k)]


def onehot(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KL33 targets that make it the cross-entropy: the next token at log-prob 0, 31 other distinct ids at -1e4."""
    y = ids[1:].astype(np.int64)
    tid = (y[:, None] + np.arange(32)[None, :]) % V_SIZE
    tlp = np.full((len(y), 32), -1e4, dtype=np.float32)
    tlp[:, 0] = 0.0
    return tid, tlp


def text_batches(docs: list[dict], epochs: int, seed: int) -> tuple[list, list[int]]:
    """Each epoch: every document once, in a seeded order, packed up to HALF_TOKENS. Returns the batches and the
    batch index at which each epoch ends."""
    rng = np.random.default_rng(seed)
    out, ends = [], []
    for _ in range(epochs):
        cur, n = [], 0
        for j in rng.permutation(len(docs)):
            ids = np.asarray(docs[j]["ids"], dtype=np.int64)
            if cur and n + len(ids) > HALF_TOKENS:
                out.append(cur)
                cur, n = [], 0
            tid, tlp = onehot(ids)
            cur.append((docs[j]["fact"], ids, tid, tlp))
            n += len(ids)
        if cur:
            out.append(cur)
        ends.append(len(out))
    return out, ends


def pack_text(batch):
    ids = np.concatenate([b[1] for b in batch])
    segs, a = [], 0
    for b in batch:
        segs.append((a, a + len(b[1])))
        a += len(b[1])
    return (torch.from_numpy(ids), segs, torch.from_numpy(np.concatenate([b[2] for b in batch])),
            torch.from_numpy(np.concatenate([b[3] for b in batch])))


@torch.no_grad()
def set_fire_bias(S: NStudent, ids, segs, share: float) -> dict:
    """c of every layer = -(1 - share) quantile of max_k (R h) on this batch, so that the gate fires on `share` of
    its pairs at step 0 (sel = 0 and D = 0, so every layer's input is v0's whatever c is)."""
    rec, hooks = {}, []
    for i, ax in S.aux.items():
        hooks.append(ax.register_forward_hook(
            lambda m, inp, out, i=i: rec.setdefault(i, []).append((inp[0] @ m.R.T).max(-1).values.float().cpu())))
    S.hidden(ids, segs, ckpt=False)
    for h in hooks:
        h.remove()
    out = {}
    for i, ax in S.aux.items():
        z = torch.cat(rec[i])
        c = -float(torch.quantile(z, 1 - share))
        ax.c.fill_(c)
        out[i] = round(c, 4)
    return out


def build(arm: str, W: T0Quant, seed: int):
    torch.manual_seed(seed)
    if arm == "l":
        return AnchoredStudent(W, 48, seed, "all")
    return NStudent(W, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(TEXT))
    for k in ("gguf", "teacher", "corpus", "inputs", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=S4.LR)
    ap.add_argument("--seed", type=int, default=S4.SEED)
    ap.add_argument("--lambda0", type=float, default=0.0, help="arms e, s, qa1: the fire cost's starting lambda")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--cap-h", type=float, default=9.5)
    ap.add_argument("--mve", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=0)
    a = ap.parse_args()
    t_session = time.time()
    fired = a.arm in ("e", "s", "qa1")
    tag = "dev_" if a.dev else ""
    src = a.inputs / f"train_{tag}{TEXT[a.arm]}.jsonl"
    kept = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    docs = equalize(kept)
    per_fact = len(docs) / len({d["fact"] for d in docs})
    epochs = max(1, round(EXPOSURES / per_fact))
    cfg = {"cell": "C50 injection train" + (" MVE" if a.mve else " dev pilot" if a.dev else ""), "arm": a.arm,
           "text": str(src), "text_sha256": sha256(src), "text_docs_kept": len(kept), "text_docs": len(docs), "facts": len({d["fact"] for d in docs}),
           "docs_per_fact": round(per_fact, 3), "epochs": epochs, "exposures_target": EXPOSURES,
           "half_tokens": HALF_TOKENS, "lr": a.lr, "lr_end": S4.LR_END, "warmup": S4.WARMUP, "clip": S4.CLIP,
           "seed": a.seed, "tau": TAU, "gguf": str(a.gguf), "fire_cost": "adaptive L1 on g (ReMoE)" if fired else None,
           "fire0": FIRE0 if fired else None, "fire_target": FIRE_TARGET if fired else None,
           "lambda0": (list(LAMBDA_GRID) if a.mve else a.lambda0) if fired else None, "alpha": ALPHA if fired else None,
           "torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "resume": str(a.resume) if a.resume else None, "cap_h": a.cap_h, "mem_start": S4.host_mem()}
    print("CONFIG", json.dumps(cfg), flush=True)
    assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
    if fired and not a.mve:
        assert a.lambda0 > 0, "arms e, s, qa1 need --lambda0 (from the MVE)"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a.out.mkdir(parents=True, exist_ok=True)
    rec_path = a.out / ("mve.json" if a.mve else "train.json")

    def finish(status: str, extra: dict, code: int) -> int:
        r = {**cfg, "status": status, **extra, "session_s": round(time.time() - t_session, 1)}
        rec_path.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
        print("RESULT", json.dumps({k: v for k, v in r.items() if k not in ("log", "mve")}), flush=True)
        return code

    tb, ends = text_batches(docs, epochs, a.seed)
    mve_steps, grid = MVE_STEPS, LAMBDA_GRID
    if a.smoke_steps:
        cfg["cell"] += " SMOKE"
        tb = tb[:a.smoke_steps]
        ends = [max(1, a.smoke_steps * j // 4) for j in range(1, 5)]
        epochs, mve_steps, grid = 4, a.smoke_steps, (1e-4,)
        print("SMOKE", json.dumps({"steps": len(tb), "mve_grid": grid}), flush=True)
    total = len(tb)
    T = TeacherL(a.teacher, a.corpus, list(S4.TRAIN_SHARDS))
    rb = []
    for b in T.batches(HALF_TOKENS):
        rb.append(b)
        if len(rb) == total:
            break
    V = list(TeacherL(a.teacher, a.corpus, list(S4.VAL_SHARDS)).batches(S4.TOKENS))[:2]
    del T
    marks = {"ck25": ends[max(0, math.ceil(epochs / 4) - 1)], "ck50": ends[max(0, math.ceil(epochs / 2) - 1)],
             "final": total}
    cfg.update({"steps": total, "epoch_ends": ends, "checkpoints": marks, "val_batches": len(V)})
    print("DATA", json.dumps({"steps": total, "checkpoints": marks}), flush=True)
    W = T0Quant(a.gguf, ("cuda:0", "cuda:1"), wdtype=torch.float16)

    def fresh():
        S = build(a.arm, W, a.seed)
        params = [p for p in S.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
        return S, params, opt, torch.amp.GradScaler("cuda")

    def one_step(S, params, opt, scaler, step, lam):
        for g in opt.param_groups:
            g["lr"] = S4.lr_at(step, total) * a.lr / S4.LR
        t1 = time.time()
        ids, segs, tid, tlp, lpt = pack(rb[step])
        mid, mlp, _ = anchored(S, ids, segs, tid, tlp, lpt, TAU)
        losses = {}
        for name, (i_, s_, ti, tl) in (("replay", (ids, segs, mid, mlp)), ("text", pack_text(tb[step]))):
            if fired:
                S.set_pen(scaler.get_scale() * lam / (len(i_) * len(AUX_LAYERS)))
            loss, npos = S.loss(i_, s_, ti, tl)
            if not torch.isfinite(loss):
                return None
            scaler.scale(loss / npos).backward()
            if fired:
                S.set_pen(0.0)
            losses[name] = round(float(loss) / npos, 5)
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(params, S4.CLIP))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        e = {"step": step, **losses, "lr": opt.param_groups[0]["lr"], "gnorm": round(gn, 4), "s": round(time.time() - t1, 3)}
        if isinstance(S, NStudent):
            fl = S.fire_by_layer(reset=False)
            e.update(S.sel_step())
            e["fire_min_layer"] = round(min(v for v in fl.values() if v is not None), 5)
        if fired:
            e["lam"] = lam
        e["mem"] = S4.host_mem()
        e["mem"]["gpu"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(2)]
        return e

    def start(S):
        """Identity against v0 at step 0, then (fired arms) the bias that sets the step-0 fire share."""
        ident = identity(S, torch.from_numpy(rb[0][0][1][:512]))
        out = {"identity_step0": ident}
        if fired:
            ids, segs, *_ = pack(rb[0])
            out["fire_bias"] = set_fire_bias(S, ids, segs, FIRE0)
            S.fire_by_layer()
            with torch.no_grad():
                S.hidden(ids, segs, ckpt=False)
            out["fire_step0"] = round(float(np.mean([v for v in S.fire_by_layer().values() if v is not None])), 5)
            out["identity_after_bias"] = identity(S, torch.from_numpy(rb[0][0][1][:512]))
        return out

    if a.mve:
        runs = {}
        for lam0 in (grid if fired else (0.0,)):
            S, params, opt, scaler = fresh()
            st = start(S)
            lam, log = lam0, []
            for step in range(mve_steps):
                e = one_step(S, params, opt, scaler, step, lam)
                if e is None:
                    log.append({"step": step, "status": "non-finite"})
                    break
                if fired:
                    lam = lam * ALPHA if e["fire"] > FIRE_TARGET else lam / ALPHA if e["fire"] < FIRE_TARGET else lam
                log.append(e)
                print("MVE", lam0, json.dumps(e), flush=True)
            runs[str(lam0)] = {**st, "log": log}
            del S, params, opt, scaler
            torch.cuda.empty_cache()
        ok = {k: (len(r["log"]) == mve_steps and r["identity_step0"]["max_abs_logit"] == 0.0
                  and (not fired or (r["identity_after_bias"]["max_abs_logit"] == 0.0
                                     and np.mean([x["fire"] for x in r["log"][min(10, mve_steps // 3):]]) <= FIRE_CEIL
                                     and r["log"][-1]["fire_min_layer"] >= DEAD))) for k, r in runs.items()}
        ok = {k: bool(v) for k, v in ok.items()}
        pick = min((float(k) for k, v in ok.items() if v), default=None)
        s_all = [x["s"] for r in runs.values() for x in r["log"][3:] if "s" in x]
        return finish("MVE COMPLETE", {"ok_by_lambda0": ok, "lambda0_pick": pick, "s_median": float(np.median(s_all)),
                                       "mve": runs}, 0 if pick is not None or not fired else 6)

    S, params, opt, scaler = fresh()
    log, vals, step, lam = [], [], 0, a.lambda0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        S.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, log, vals, lam = ck["step"], ck["log"], ck["vals"], ck["lam"]
        assert ck["arm"] == a.arm and ck["total"] == total and ck["seed"] == a.seed and ck["lr"] == a.lr
        cfg["start"] = ck["start"]
        cfg["marks_sha256"] = ck.get("marks_sha256", {})
        print("RESUME", json.dumps({"step": step}), flush=True)
    else:
        cfg["start"] = start(S)
        print("START", json.dumps(cfg["start"]), flush=True)
        if cfg["start"]["identity_step0"]["max_abs_logit"] != 0.0 or (
                fired and cfg["start"]["identity_after_bias"]["max_abs_logit"] != 0.0):
            return finish("VOID: the arm is not v0 at step 0", {}, 3)
        vals.append({"step": 0, **(val_n(S, V) if isinstance(S, NStudent) else {})})

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"arm": a.arm, "total": total, "seed": a.seed, "lr": a.lr, "step": step, "model": S.state_dict(),
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(), "log": log, "vals": vals, "lam": lam,
                    "start": cfg["start"], "marks_sha256": cfg.get("marks_sha256", {})}, tmp)
        os.replace(tmp, path)

    while step < total:
        if time.time() - t_session > a.cap_h * 3600:
            save(a.out / "ckpt.pt")
            return finish("PAUSED", {"step": step, "log": log, "vals": vals}, 0)
        e = one_step(S, params, opt, scaler, step, lam)
        if e is None:
            save(a.out / "ckpt.pt")
            return finish("VOID: non-finite loss", {"step": step, "log": log, "vals": vals}, 4)
        if fired:
            lam = lam * ALPHA if e["fire"] > FIRE_TARGET else lam / ALPHA if e["fire"] < FIRE_TARGET else lam
            run = log[-(DEAD_RUN - 1):] + [e] if DEAD_RUN > 1 else [e]
            if len(run) == DEAD_RUN and all(x["fire_min_layer"] < DEAD for x in run):
                save(a.out / "ckpt.pt")
                return finish(f"VOID: a dead layer (fire below {DEAD} for {DEAD_RUN} steps)",
                              {"step": step, "log": log + [e], "vals": vals}, 5)
        log.append(e)
        step += 1
        if step < 4 or step % 20 == 0:
            print("STEP", json.dumps(e), flush=True)
        for name, at in marks.items():
            if step == at:
                vals.append({"step": step, "mark": name, **(val_n(S, V) if isinstance(S, NStudent) else {})})
                print("VAL", json.dumps(vals[-1]), flush=True)
                torch.save({k: v for k, v in S.state_dict().items()}, a.out / f"{name}.pt")
                cfg.setdefault("marks_sha256", {})[name] = sha256(a.out / f"{name}.pt")
        mm_ = e["mem"]
        if mm_.get("max") and mm_.get("anon") is not None and mm_["max"] - mm_["anon"] < S4.MEM_FLOOR_GIB:
            save(a.out / "ckpt.pt")
            return finish("PAUSED-LOWMEM", {"step": step, "log": log, "vals": vals}, 0)
        if step % S4.CKPT_EVERY == 0 and step < total:
            save(a.out / "ckpt.pt")
    save(a.out / "ckpt.pt")
    return finish("COMPLETE", {"step": step, "log": log, "vals": vals}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
