"""C50-T1 part 2b: c48's LoRA trained with its loss anchored to T0 where T0 is sure of the right token.

S4's c48 recipe unchanged (data, order, rank 48, modules, optimizer, schedule, seed, sessions, checks; the S4 trainer's
own functions), with one change in the target. At each scored position with target y:
- **anchored** iff p_T0(y) >= TAU and p_T0(y) >= p_F0(y): T0 is sure of the right token and the teacher is no better.
  The target there is T0's own distribution (its top-32 and the rest), read from the frozen base in the same step
  (LoRA off, no gradient);
- **free** otherwise: the teacher's top-32 and the rest, as in c48.
The loss is S4's 33-bucket KL, summed over both kinds of position. p_F0(y) is the shard's `lpt`.

Validation logs both S4's KL to F0 (comparable with c48's curve) and the anchored KL, with the anchored share.

    python -I benchmarks/chatbot50/c50_t1_train.py --gguf G --teacher DIR --corpus TSV --out DIR [--tau 0.5]
        [--only all|attn] [--resume CKPT] [--cap-h 9.5]
    python -I benchmarks/chatbot50/c50_t1_train.py --controls --gguf G --teacher DIR --corpus TSV --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_train as S4  # noqa: E402
from c50_s4_student import LORA_DENSE, LORA_MODULES, Student, Teacher, identity  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402
import c50_s4_t0_torch as t0  # noqa: E402
from c50_s4_t0_torch import mm  # noqa: E402

TAU = 0.5
ONLY = {"all": None, "attn": ("attn_qkv", "attn_output")}


class TeacherL(Teacher):
    """S4's teacher data, with each position's teacher log-prob of the target (`lpt`) as well."""

    def docs(self):
        for r in self.recs:
            bounds = np.flatnonzero(np.diff(r["doc"])) + 1
            for part in np.split(r, bounds):
                d = int(part["doc"][0])
                ids = self.ids[d]
                assert np.array_equal(part["pos"], np.arange(1, len(ids))), d
                assert np.array_equal(part["target"], ids[1:]), d
                yield d, ids, part["top"]["id"].astype(np.int64), part["top"]["lp"], part["lpt"]


def pack(batch):
    ids = np.concatenate([b[1] for b in batch])
    segs, a = [], 0
    for b in batch:
        segs.append((a, a + len(b[1])))
        a += len(b[1])
    return (torch.from_numpy(ids), segs, torch.from_numpy(np.concatenate([b[2] for b in batch])),
            torch.from_numpy(np.concatenate([b[3] for b in batch]).astype(np.float32)),
            torch.from_numpy(np.concatenate([b[4] for b in batch]).astype(np.float32)))


class AnchoredStudent(Student):
    """Arm "c" with a switch that turns the LoRA off (the frozen base is T0 exactly, as S4's identity check shows)."""

    def __init__(self, W: T0Quant, rank: int, seed: int, only: str):
        super().__init__(W, "c", rank, seed=seed)
        keep = ONLY[only]
        if keep is not None:                     # drop the modules outside the group, before the optimizer sees them
            for k in [k for k in self.lora if k.split("_", 1)[1] not in keep]:
                del self.lora[k]
        self.base_only = False

    def lin(self, x, i, m):
        y = mm(x, self.W.get(f"blk.{i}.{m}.weight"))
        if not self.base_only and f"{i}_{m}" in self.lora:
            y = y + self.lora[f"{i}_{m}"](x)
        return y


@torch.no_grad()
def t0_targets(S: AnchoredStudent, ids, segs, chunk=1024):
    """T0's top-32 (ids, log-probs) and log-prob of the target at every scored position, from the frozen base."""
    S.base_only = True
    try:
        h = S.hidden(ids, segs, ckpt=False)
    finally:
        S.base_only = False
    keep = torch.ones(len(ids), dtype=torch.bool)
    for a, b in segs:
        keep[b - 1] = False
    y = torch.roll(ids, -1)[keep]
    h = h[keep.to(h.device)]
    out_w = S.W.get("output.weight")
    tid, tlp, lpy = [], [], []
    for a in range(0, len(h), chunk):
        ls = torch.log_softmax(mm(h[a:a + chunk], out_w).float(), -1)
        v, i = ls.topk(32, dim=-1)
        tid.append(i.cpu())
        tlp.append(v.cpu())
        lpy.append(ls.gather(1, y[a:a + chunk, None].to(ls.device)).squeeze(1).cpu())
    return torch.cat(tid), torch.cat(tlp), torch.cat(lpy)


def anchored(S, ids, segs, tid, tlp, lpt, tau, force_all=False):
    """The mixed target: T0's top-32 at anchored positions, the teacher's elsewhere; and the anchor mask."""
    t_id, t_lp, lpy = t0_targets(S, ids, segs)
    m = torch.ones_like(lpy, dtype=torch.bool) if force_all else (lpy.exp() >= tau) & (lpy >= lpt)
    return torch.where(m[:, None], t_id, tid), torch.where(m[:, None], t_lp, tlp), m


@torch.no_grad()
def val(S, V, tau) -> dict:
    kl_f0 = kl_anc = 0.0
    npos = nanc = 0
    for batch in V:
        ids, segs, tid, tlp, lpt = pack(batch)
        l0, n = S.loss(ids, segs, tid, tlp)
        mid, mlp, m = anchored(S, ids, segs, tid, tlp, lpt, tau)
        l1, _ = S.loss(ids, segs, mid, mlp)
        kl_f0, kl_anc, npos, nanc = kl_f0 + float(l0), kl_anc + float(l1), npos + n, nanc + int(m.sum())
    return {"val_kl": kl_f0 / npos, "val_kl_anchored": kl_anc / npos, "val_anchored_share": nanc / npos}


def controls(a) -> int:
    """Planted controls on the CPU, at step 0 (the student is T0): with TAU above 1 nothing is anchored and the loss is
    S4's loss bit for bit; with every position anchored the loss is KL(T0 || T0) = 0 up to rounding."""
    W = T0Quant(a.gguf, ("cpu", "cpu"), wdtype=torch.float32)
    torch.manual_seed(S4.SEED)
    S = AnchoredStudent(W, 48, S4.SEED, "all")
    d, i0, top, lp, lpt0 = next(TeacherL(a.teacher, a.corpus, [0]).docs())
    ids, segs, tid, tlp, lpt = pack([(d, i0[:128], top[:127], lp[:127], lpt0[:127])])   # one document, 128 tokens
    ref, n = S.loss(ids, segs, tid, tlp)
    mid, mlp, m = anchored(S, ids, segs, tid, tlp, lpt, 2.0)
    none, _ = S.loss(ids, segs, mid, mlp)
    aid, alp, _ = anchored(S, ids, segs, tid, tlp, lpt, 0.0, force_all=True)
    allk, _ = S.loss(ids, segs, aid, alp)
    rec = {"positions": n, "s4_loss": float(ref), "tau_above_1_loss": float(none), "anchored_none": int(m.sum()) == 0,
           "K_none": float(none) == float(ref), "all_anchored_loss_per_pos": float(allk) / n,
           "K_all": float(allk) / n < 1e-5, "identity": identity(S, ids[:48])}
    # K-switch: with B != 0 the student differs from T0, and the anchor's targets must still be T0's (the switch turns
    # the LoRA off); at step 0, B = 0, so K-none and K-all cannot tell an off switch from a broken one.
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        for lo in S.lora.values():
            lo.B.copy_(torch.randn(lo.B.shape, generator=g) * 0.02)
        sid = ids[:48]
        # in fp32, as the anchor computes it: the first run compared against an fp64 log-softmax and missed by
        # 1.06e-4 on rounding alone (24 October)
        ref_lp = torch.log_softmax(t0.logits(sid, S.W).float(), -1)[:-1]
        stu_lp = torch.log_softmax(mm(S.hidden(sid, [(0, 48)], ckpt=False), S.W.get("output.weight")).double(), -1)[:-1]
        _, _, lpy = t0_targets(S, sid, [(0, 48)])
        y = sid[1:]
        rec["switch_lpy_vs_t0_max"] = float((lpy - ref_lp.gather(1, y[:, None]).squeeze(1)).abs().max())
        rec["student_vs_t0_max"] = float((stu_lp - ref_lp).abs().max())
    rec["K_switch"] = rec["switch_lpy_vs_t0_max"] < 1e-6 and rec["student_vs_t0_max"] > 1e-2
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "controls.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("CONTROLS", json.dumps(rec), flush=True)
    return 0 if rec["K_none"] and rec["K_all"] and rec["anchored_none"] and rec["K_switch"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--rank", type=int, default=48)
    ap.add_argument("--only", default="all", choices=list(ONLY))
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--cap-h", type=float, default=9.5)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--controls", action="store_true", help="local: the planted controls on the CPU, then exit")
    a = ap.parse_args()
    if a.controls:
        return controls(a)
    t_session = time.time()
    cfg = {"cell": "C50-T1 part 2b train", "arm": f"c{a.rank}-anchored-{a.only}", "tau": a.tau, "rank": a.rank,
           "only": a.only, "modules": {"layer0": LORA_DENSE, "layers1-19": LORA_MODULES}, "torch": torch.__version__,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "tokens_per_step": S4.TOKENS, "lr": S4.LR, "lr_end": S4.LR_END, "warmup": S4.WARMUP, "clip": S4.CLIP,
           "seed": S4.SEED, "optimizer": "AdamW (0.9, 0.999) wd 0",
           "train_shards": [S4.TRAIN_SHARDS[0], S4.TRAIN_SHARDS[-1]], "val_shards": [S4.VAL_SHARDS[0], S4.VAL_SHARDS[-1]],
           "resume": str(a.resume) if a.resume else None, "cap_h": a.cap_h, "stop_at": a.stop_at,
           "mem_floor_gib": S4.MEM_FLOOR_GIB, "mem_start": S4.host_mem()}
    print("CONFIG", json.dumps(cfg), flush=True)
    assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a.out.mkdir(parents=True, exist_ok=True)
    rec_path = a.out / "train.json"

    def finish(status: str, extra: dict, code: int) -> int:
        rec = {**cfg, "status": status, **extra, "session_s": round(time.time() - t_session, 1)}
        rec_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print("RESULT", json.dumps({k: v for k, v in rec.items() if k != "log"}), flush=True)
        return code

    T = TeacherL(a.teacher, a.corpus, list(S4.TRAIN_SHARDS))
    batches = list(T.batches(S4.TOKENS))
    first = next(T.docs())[1][:512]
    V = list(TeacherL(a.teacher, a.corpus, list(S4.VAL_SHARDS)).batches(S4.TOKENS))
    del T
    total = len(batches)
    print("DATA", json.dumps({"steps": total, "val_batches": len(V)}), flush=True)
    W = T0Quant(a.gguf, ("cuda:0", "cuda:1"), wdtype=torch.float16)
    torch.manual_seed(S4.SEED)
    S = AnchoredStudent(W, a.rank, S4.SEED, a.only)
    params = [p for p in S.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=S4.LR, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    log, vals, step = [], [], 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        S.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, log, vals = ck["step"], ck["log"], ck["vals"]
        assert ck["arm"] == cfg["arm"] and ck["total"] == total and ck["tau"] == a.tau, (ck["arm"], ck["total"])
        print("RESUME", json.dumps({"step": step}), flush=True)
    else:
        ident = identity(S, torch.from_numpy(first))
        cfg["identity_step0"] = ident
        print("IDENTITY", json.dumps(ident), flush=True)
        if ident["max_abs_logit"] != 0.0:
            return finish("VOID: the arm is not T0 at step 0", {}, 3)
        vals.append({"step": 0, **val(S, V, a.tau)})
        print("VAL", json.dumps(vals[-1]), flush=True)

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"arm": cfg["arm"], "tau": a.tau, "total": total, "step": step, "model": S.state_dict(),
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(), "log": log, "vals": vals}, tmp)
        os.replace(tmp, path)

    cfg["trainable"] = sum(p.numel() for p in params)
    while step < total:
        if time.time() - t_session > a.cap_h * 3600 or (a.stop_at and step == a.stop_at):
            save(a.out / "ckpt.pt")
            return finish("PAUSED", {"step": step, "total": total, "log": log, "vals": vals}, 0)
        ids, segs, tid, tlp, lpt = pack(batches[step])
        for g in opt.param_groups:
            g["lr"] = S4.lr_at(step, total)
        t1 = time.time()
        mid, mlp, m = anchored(S, ids, segs, tid, tlp, lpt, a.tau)
        t_anchor = time.time() - t1
        loss, npos = S.loss(ids, segs, mid, mlp)
        if not torch.isfinite(loss):
            save(a.out / "ckpt.pt")
            return finish("VOID: non-finite loss", {"step": step, "total": total, "log": log, "vals": vals}, 4)
        scaler.scale(loss / npos).backward()
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(params, S4.CLIP))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        e = {"step": step, "loss": round(float(loss) / npos, 5), "pos": npos, "anchored": round(float(m.float().mean()), 4),
             "lr": opt.param_groups[0]["lr"], "gnorm": round(gn, 4), "scale": scaler.get_scale(),
             "s_anchor": round(t_anchor, 3), "s": round(time.time() - t1, 3), "mem": S4.host_mem()}
        e["mem"]["gpu"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(2)]
        log.append(e)
        step += 1
        mm_ = e["mem"]
        if mm_.get("max") and mm_.get("anon") is not None and mm_["max"] - mm_["anon"] < S4.MEM_FLOOR_GIB:
            save(a.out / "ckpt.pt")
            return finish("PAUSED-LOWMEM", {"step": step, "total": total, "log": log, "vals": vals}, 0)
        if step < 4 or step % 20 == 0:
            print("STEP", json.dumps(e), flush=True)
        if step % S4.VAL_EVERY == 0 or step == total:
            vals.append({"step": step, **val(S, V, a.tau)})
            print("VAL", json.dumps(vals[-1]), flush=True)
        if step % S4.CKPT_EVERY == 0 and step < total:
            save(a.out / "ckpt.pt")
    torch.save({k: v for k, v in S.state_dict().items()}, a.out / "final.pt")
    save(a.out / "ckpt.pt")
    return finish("COMPLETE", {"step": step, "total": total, "val_first": vals[0], "val_last": vals[-1],
                               "trained": vals[-1]["val_kl_anchored"] < vals[0]["val_kl_anchored"], "log": log,
                               "vals": vals}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
