"""C50-T2 option N: new experts, trained with F0 as teacher, attached to v0 in a gated additive branch.

At each MoE layer i (1-19), with h the MoE input (v0's ffn_norm of the residual):

    y = MoE_v0(h) + relu(z_k*(h)) * E_k*(h),   z = R h + c,   k* = argmax(z + sel)

- **v0 is frozen and unchanged:** its router, top-8 and weights are its own; nothing competes for its slots.
- **E_k** are K new SwiGLU experts of T0's size (2048 -> 512 -> 2048), trained from scratch: gate and up random,
  down zero, so the branch outputs exactly 0 at step 0 and the student is v0 bit for bit.
- **The gate is the ReLU of the picked expert's router logit:** where it is 0 the expert is not read. `sel` is a
  selection-only bias that follows the base's aux-loss-free load rule (S4's arm (a) rate), so the K experts share
  the picks.
- **Target:** C50-T1 part 2b's anchored KL (`c50_t1_train.anchored`), with v0 as the anchor's base: where v0 is sure
  of the right token (p >= 0.5) and the teacher is no better, the target is v0's own distribution; elsewhere F0's
  top-32. The recipe is S4's and T1's: block 1's shards 0-469, one epoch, 8,192 tokens per step, AdamW, lr 1e-4 with
  30 warm-up steps and cosine to 1e-5, clip 1.0, fp16 forward with fp32 trainables, seed 20261016.

    python -I benchmarks/chatbot50/c50_t2_n.py --gguf V0 --teacher DIR --corpus TSV --out DIR [--resume CKPT]
    python -I benchmarks/chatbot50/c50_t2_n.py --controls --gguf V0 --teacher DIR --corpus TSV --out DIR   (CPU)
    python -I benchmarks/chatbot50/c50_t2_n.py --mve --gguf V0 --teacher DIR --corpus TSV --out DIR       (T4 x2)
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_train as S4  # noqa: E402
import c50_s4_t0_torch as t0  # noqa: E402
from c50_s4_student import Student, identity  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402
from c50_s4_t0_torch import H, N_LAYERS, mm, rms  # noqa: E402
from c50_t1_train import TAU, TeacherL, anchored, pack, val  # noqa: E402

K, FF, RATE_SEL = 4, 512, 0.01
AUX_LAYERS = range(1, N_LAYERS)
MVE_STEPS = 20


class Aux(nn.Module):
    """The K new experts of one layer, their router rows and the selection bias."""

    def __init__(self, dev, gen: torch.Generator):
        super().__init__()
        r = lambda *s: nn.Parameter((torch.randn(*s, generator=gen) / math.sqrt(s[-1])).to(dev))
        self.R, self.G, self.U = r(K, H), r(K, FF, H), r(K, FF, H)
        self.c = nn.Parameter(torch.zeros(K, device=dev))
        self.D = nn.Parameter(torch.zeros(K, H, FF, device=dev))
        self.register_buffer("sel", torch.zeros(K, device=dev))
        self.register_buffer("load", torch.zeros(K, device=dev), persistent=False)
        self.register_buffer("fired", torch.zeros((), device=dev), persistent=False)
        self.register_buffer("seen", torch.zeros((), device=dev), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h @ self.R.T + self.c
        k = (z.detach() + self.sel).argmax(-1)
        g = F.relu(z.gather(1, k[:, None]))[:, 0]
        on = g.detach() > 0
        with torch.no_grad():
            if torch.is_grad_enabled() or self.training:
                self.load += torch.bincount(k, minlength=K).float()
            self.fired += on.sum()
            self.seen += len(h)
        out = torch.zeros_like(h)
        for e in range(K):
            t = ((k == e) & on).nonzero(as_tuple=True)[0]
            if len(t) == 0:
                continue
            ht = h[t]
            a = F.silu(ht @ self.G[e].T) * (ht @ self.U[e].T)
            out = out.index_add(0, t, (a @ self.D[e].T) * g[t, None])
        return out


class NStudent(Student):
    """v0 frozen (arm "n" takes Student's plain path) plus the auxiliary branch; `base_only` turns the branch off, so
    the anchor reads v0 itself (`c50_t1_train.t0_targets`)."""

    def __init__(self, W: T0Quant, seed: int):
        super().__init__(W, "n")
        gen = torch.Generator().manual_seed(seed)
        self.aux = nn.ModuleDict({str(i): Aux(W.dev(i), gen) for i in AUX_LAYERS})
        self.base_only = False

    def ffn(self, x, i):
        y = super().ffn(x, i)
        if i == 0 or self.base_only:
            return y
        h = rms(x, self.W.get(f"blk.{i}.ffn_norm.weight"))
        return y + self.aux[str(i)](h)

    @torch.no_grad()
    def sel_step(self) -> dict:
        """After each step: sel += rate * sign(mean load - load), per layer. Returns the fire share and the busiest
        expert's share of the picks, over the step's forward (the checkpoint recompute counts again, equally)."""
        fire, top = [], []
        for ax in self.aux.values():
            if ax.load.sum() > 0:
                top.append(float(ax.load.max() / ax.load.sum()))
                ax.sel += RATE_SEL * torch.sign(ax.load.mean() - ax.load)
            if ax.seen > 0:
                fire.append(float(ax.fired / ax.seen))
            ax.load.zero_()
            ax.fired.zero_()
            ax.seen.zero_()
        return {"fire": round(float(np.mean(fire)), 5) if fire else None,
                "top_share": round(float(np.mean(top)), 5) if top else None}

    @torch.no_grad()
    def fire_by_layer(self, reset: bool = True) -> dict:
        out = {i: (float(ax.fired / ax.seen) if ax.seen > 0 else None) for i, ax in self.aux.items()}
        if reset:
            for ax in self.aux.values():
                ax.load.zero_()
                ax.fired.zero_()
                ax.seen.zero_()
        return out


@torch.no_grad()
def shuffle(S: NStudent) -> None:
    """The shuffle control: in each layer, expert e's weights move to slot (e + 1) mod K. The router rows, biases and
    gates stay, so every token reaches another expert of its own layer at the same gate value."""
    for ax in S.aux.values():
        p = torch.tensor([(e + 1) % K for e in range(K)], device=ax.G.device)
        for t in (ax.G, ax.U, ax.D):
            t.copy_(t[p])


def val_n(S: NStudent, V: list) -> dict:
    S.fire_by_layer()
    r = val(S, V, TAU)
    f = [v for v in S.fire_by_layer().values() if v is not None]
    return {**r, "val_fire": round(float(np.mean(f)), 5) if f else None}


# ---------------------------------------------------------------- planted controls (CPU)
def controls(a) -> int:
    """At step 0 the student is v0: max |dlogit| = 0 against v0's own forward. The anchor's switch must give v0's
    targets once the branch is non-zero, the branch must then move the student, and the shuffle must move it again.
    Gradients at step 0 reach only D (the others are multiplied by D = 0); with D non-zero they reach every part."""
    W = T0Quant(a.gguf, ("cpu", "cpu"), wdtype=torch.float32)
    S = NStudent(W, S4.SEED)
    d, i0, top, lp, lpt0 = next(TeacherL(a.teacher, a.corpus, [0]).docs())
    ids, segs, tid, tlp, lpt = pack([(d, i0[:128], top[:127], lp[:127], lpt0[:127])])
    rec = {"identity": identity(S, ids[:48])}
    rec["identity_ok"] = rec["identity"]["max_abs_logit"] == 0.0
    S.fire_by_layer()
    loss, n = S.loss(ids, segs, tid, tlp)
    rec["fire_step0"] = {k: round(v, 4) for k, v in S.fire_by_layer().items()}
    loss.backward()
    g0 = {nm: float(p.grad.abs().max()) if p.grad is not None else 0.0 for nm, p in S.named_parameters()}
    rec["grad_step0_D"] = max(v for k, v in g0.items() if k.endswith(".D"))
    rec["grad_step0_other"] = max(v for k, v in g0.items() if not k.endswith(".D"))
    rec["grads_step0_ok"] = rec["grad_step0_D"] > 0 and rec["grad_step0_other"] == 0.0
    S.zero_grad(set_to_none=True)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for ax in S.aux.values():
            ax.D.copy_(torch.randn(ax.D.shape, generator=gen) * 0.02)
    loss, _ = S.loss(ids, segs, tid, tlp)
    loss.backward()
    parts = {}
    for nm, p in S.named_parameters():
        key = nm.rsplit(".", 1)[1]
        parts[key] = max(parts.get(key, 0.0), float(p.grad.abs().max()) if p.grad is not None else 0.0)
    rec["grad_nonzero_D_by_part"] = parts
    rec["grads_ok"] = all(v > 0 for v in parts.values())
    S.zero_grad(set_to_none=True)
    sid = ids[:48]
    with torch.no_grad():
        ref = torch.log_softmax(t0.logits(sid, S.W).float(), -1)[:-1]
        stu = torch.log_softmax(mm(S.hidden(sid, [(0, 48)], ckpt=False), S.W.get("output.weight")).float(), -1)[:-1]
        from c50_t1_train import t0_targets
        _, _, lpy = t0_targets(S, sid, [(0, 48)])
        rec["switch_lpy_vs_v0_max"] = float((lpy - ref.gather(1, sid[1:, None]).squeeze(1)).abs().max())
        rec["student_vs_v0_max"] = float((stu - ref).abs().max())
        shuffle(S)
        shf = torch.log_softmax(mm(S.hidden(sid, [(0, 48)], ckpt=False), S.W.get("output.weight")).float(), -1)[:-1]
        rec["shuffle_vs_student_max"] = float((shf - stu).abs().max())
    rec["switch_ok"] = rec["switch_lpy_vs_v0_max"] < 1e-6 and rec["student_vs_v0_max"] > 1e-2
    rec["shuffle_ok"] = rec["shuffle_vs_student_max"] > 1e-2
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "controls.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("CONTROLS", json.dumps(rec), flush=True)
    return 0 if all(rec[k] for k in ("identity_ok", "grads_step0_ok", "grads_ok", "switch_ok", "shuffle_ok")) else 1


# ---------------------------------------------------------------- training
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--cap-h", type=float, default=9.5)
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--controls", action="store_true", help="local: the planted controls on the CPU, then exit")
    ap.add_argument("--mve", action="store_true", help="T4: identity, then MVE_STEPS steps for time and memory")
    a = ap.parse_args()
    if a.controls:
        return controls(a)
    t_session = time.time()
    cfg = {"cell": "C50-T2 option N train" + (" MVE" if a.mve else ""), "arm": "n", "experts_per_layer": K,
           "expert_ff": FF, "layers": [AUX_LAYERS[0], AUX_LAYERS[-1]], "gate": "relu(router logit of the pick)",
           "rate_sel": RATE_SEL, "tau": TAU, "anchor_base": "the --gguf model (v0)", "gguf": str(a.gguf),
           "torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "tokens_per_step": S4.TOKENS, "lr": S4.LR, "lr_end": S4.LR_END, "warmup": S4.WARMUP, "clip": S4.CLIP,
           "seed": S4.SEED, "optimizer": "AdamW (0.9, 0.999) wd 0",
           "train_shards": [S4.TRAIN_SHARDS[0], S4.TRAIN_SHARDS[-1]], "val_shards": [S4.VAL_SHARDS[0], S4.VAL_SHARDS[-1]],
           "resume": str(a.resume) if a.resume else None, "cap_h": a.cap_h, "stop_at": a.stop_at,
           "mem_floor_gib": S4.MEM_FLOOR_GIB, "mem_start": S4.host_mem()}
    print("CONFIG", json.dumps(cfg), flush=True)
    assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a.out.mkdir(parents=True, exist_ok=True)
    rec_path = a.out / ("mve.json" if a.mve else "train.json")

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
    S = NStudent(W, S4.SEED)
    params = [p for p in S.parameters() if p.requires_grad]
    cfg["trainable"] = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=S4.LR, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    log, vals, step = [], [], 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        S.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, log, vals = ck["step"], ck["log"], ck["vals"]
        assert ck["arm"] == "n" and ck["total"] == total and ck["tau"] == TAU and ck["k"] == K, (ck["arm"], ck["total"])
        print("RESUME", json.dumps({"step": step}), flush=True)
    else:
        ident = identity(S, torch.from_numpy(first))
        cfg["identity_step0"] = ident
        print("IDENTITY", json.dumps(ident), flush=True)
        if ident["max_abs_logit"] != 0.0:
            return finish("VOID: the arm is not v0 at step 0", {}, 3)
        if not a.mve:
            vals.append({"step": 0, **val_n(S, V)})
            print("VAL", json.dumps(vals[-1]), flush=True)

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"arm": "n", "k": K, "tau": TAU, "total": total, "step": step, "model": S.state_dict(),
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(), "log": log, "vals": vals}, tmp)
        os.replace(tmp, path)

    end = MVE_STEPS if a.mve else total
    while step < end:
        if not a.mve and (time.time() - t_session > a.cap_h * 3600 or (a.stop_at and step == a.stop_at)):
            save(a.out / "ckpt.pt")
            return finish("PAUSED", {"step": step, "total": total, "log": log, "vals": vals}, 0)
        ids, segs, tid, tlp, lpt = pack(batches[step])
        for g in opt.param_groups:
            g["lr"] = S4.lr_at(step, total)
        t1 = time.time()
        mid, mlp, m = anchored(S, ids, segs, tid, tlp, lpt, TAU)
        t_anchor = time.time() - t1
        S.fire_by_layer()                        # the anchor's pass runs with the branch off: nothing is counted
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
             "lr": opt.param_groups[0]["lr"], "gnorm": round(gn, 4), "scale": scaler.get_scale(), **S.sel_step(),
             "s_anchor": round(t_anchor, 3), "s": round(time.time() - t1, 3), "mem": S4.host_mem()}
        e["mem"]["gpu"] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(2)]
        log.append(e)
        step += 1
        mm_ = e["mem"]
        if mm_.get("max") and mm_.get("anon") is not None and mm_["max"] - mm_["anon"] < S4.MEM_FLOOR_GIB:
            if a.mve:
                return finish("MVE: PAUSED-LOWMEM", {"step": step, "log": log}, 5)
            save(a.out / "ckpt.pt")
            return finish("PAUSED-LOWMEM", {"step": step, "total": total, "log": log, "vals": vals}, 0)
        if step < 4 or step % 20 == 0:
            print("STEP", json.dumps(e), flush=True)
        if a.mve:
            continue
        if step % S4.VAL_EVERY == 0 or step == total:
            vals.append({"step": step, **val_n(S, V)})
            print("VAL", json.dumps(vals[-1]), flush=True)
        if step % S4.CKPT_EVERY == 0 and step < total:
            save(a.out / "ckpt.pt")
    if a.mve:
        s = [x["s"] for x in log[3:]]
        return finish("MVE COMPLETE", {"step": step, "s_median": float(np.median(s)),
                                       "gpu_max_gib": [max(x["mem"]["gpu"][i] for x in log) for i in range(2)],
                                       "log": log}, 0)
    torch.save({k: v for k, v in S.state_dict().items()}, a.out / "final.pt")
    save(a.out / "ckpt.pt")
    return finish("COMPLETE", {"step": step, "total": total, "val_first": vals[0], "val_last": vals[-1],
                               "trained": vals[-1]["val_kl_anchored"] < vals[0]["val_kl_anchored"], "log": log,
                               "vals": vals}, 0)


if __name__ == "__main__":
    raise SystemExit(main())
