"""Stage 2 part 1, test 2c: the organs without a router, ternary on every layer, trained block by block as stage 1 (b).

    teacher   the original model's residual stream on stage 1's corpus, written once: h_0 (embeddings) to h_20, flat bf16.
    train     --arm A: for each layer of the arm, in order, on the student stream H_s (the layers already trained):
              the organ's tensors of that layer become ternary in IQ2_BN's layout (test 2's quantizer: codes {-1, 0, 1},
              one scale per row, Lloyd start, no outlier rule, latent fp32 weights, STE, LSQ on the scale) and train on the
              block's output against the teacher's, in the space the later norms read (stage 1 (b), test 2):
              mean of (nrm(o_s) - nrm(o_t))^2 * gw.  The target is the teacher's stream, so each block also absorbs the
              error carried in.  Every other weight of the block is the original bf16 one, routing at k = 8.  Then H_s
              moves through the trained block.
    posthoc   the same organs quantized from the initial scales, no data, no training (the arm ph_both).

Arms:  st_attn     attention (fused QKV, output projection), layers 0-19;
       st_ffn      dense FFN (layer 0) and shared expert (layers 1-19);
       st_both     both; attention and FFN of a layer train together, on the same block output;
       st_attn_hi  attention, layers 10-19, on the original stream up to layer 10.  (st_attn_lo needs no training: it is
                   st_attn's layers 0-9, the trainer being sequential.)
lr per tensor: test 2's picks (attention 1e-3, shared expert 3e-3, dense FFN 1e-3) times --lr-scale, which the lr pick sets:
st_both's layers 0-1, then layer 2 from that stream at each scale (--pick-from DIR writes no student stream).  One epoch,
Adafactor, about 4,096 tokens of whole sequences per step, 1.5 M tokens train and 75 k are held out.
Logged per layer on held-out tokens: the block loss with the original weights on the student stream (what the carried
error alone costs), post-hoc, trained; after the move, the error of H_s against the teacher, plain and in the norms' space.
Resumable: after each layer it saves layer_<i>.pt and the student stream; a killed run restarts at the next layer.

Usage: python -I benchmarks/stage2/s2s_train.py teacher --dir D
       python -I benchmarks/stage2/s2s_train.py train --arm st_both --dir D --out DIR --json F
       python -I benchmarks/stage2/s2s_train.py posthoc --arm st_both --out DIR
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
import s1t_model as M  # noqa: E402
from s1t_train import EXP, Tern, downstream_gain2, load_corpus  # noqa: E402
from s2o_train import init_row_scale, nrm  # noqa: E402

ATTN = ["attention.query_key_value.weight", "attention.dense.weight"]
ARMS = {"st_attn": ("attn", 0, 19), "st_ffn": ("ffn", 0, 19), "st_both": ("both", 0, 19), "st_attn_hi": ("attn", 10, 19)}
LR_ATTN, LR_SHARED, LR_DENSE = 1e-3, 3e-3, 1e-3


class Asym(torch.autograd.Function):
    """S2L's grid: w [1, O, I], s [1, O, 1], sign g [1, O, 1] -> bf16 s * g * clamp(round(g * w / s), -1, 2).  The estimators are Tern's: STE for w, LSQ for s."""

    @staticmethod
    def forward(ctx, w, s, g):
        ctx.save_for_backward(w, s, g)
        x = w * g / torch.where(s > 0, s, 1.0)
        return (x.clamp_(-1, 2).round_() * (s.to(torch.bfloat16).float() * g)).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, go):
        w, s, g = ctx.saved_tensors
        go = go.float()
        x = w * g / torch.where(s > 0, s, 1.0)
        inside = ((x >= -1) & (x <= 2) & (s > 0)).float()
        return go * inside, (go * g * (x.clamp(-1, 2).round() - x * inside)).sum(-1, keepdim=True), None


def asym_init(w: torch.Tensor, iters: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """w [1, O, I] -> (s, g) [1, O, 1]: Lloyd on {-1, 0, 1, 2} for w and for -w, the better sign per row (S2I's q_asym)."""
    f = w.float()

    def fit(a):
        s = a.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-20)
        for _ in range(iters):
            q = (a / s).round().clamp(-1, 2)
            s = ((a * q).sum(-1, keepdim=True) / q.pow(2).sum(-1, keepdim=True).clamp_min(1e-20)).clamp_min(1e-20)
        return s, (a - (a / s).round().clamp(-1, 2) * s).pow(2).sum(-1, keepdim=True)
    sp, ep = fit(f)
    sn, en = fit(-f)
    pos = ep <= en
    return torch.where(pos, sp, sn), torch.where(pos, 1.0, -1.0)


def organ_names(kind: str, i: int, scale: float = 1.0) -> dict[str, float]:
    """tensor name in the layer -> lr."""
    out: dict[str, float] = {}
    if kind in ("attn", "both"):
        out.update({n: LR_ATTN for n in ATTN})
    if kind in ("ffn", "both"):
        if i == 0:
            out.update({f"mlp.{p}.weight": LR_DENSE for p in EXP})
        else:
            out.update({f"mlp.shared_experts.{p}.weight": LR_SHARED for p in EXP})
    return {k: v * scale for k, v in out.items()}


def write_flat(path: Path, h: torch.Tensor) -> None:
    tmp = path.with_suffix(".tmp")
    h.view(torch.int16).numpy().tofile(tmp)
    os.replace(tmp, path)


def read_flat(path: Path, n: int, start: int = 0, count: int | None = None) -> torch.Tensor:
    c = n - start if count is None else count
    a = np.fromfile(path, dtype=np.int16, count=c * M.D, offset=start * M.D * 2)
    return torch.from_numpy(a).view(torch.bfloat16).view(c, M.D)


def setup(tokens: int, heldout: int):
    tr, he = load_corpus(tokens, heldout)
    seqs = tr + he
    lens = [len(s) for s in seqs]
    offs = np.concatenate([[0], np.cumsum(lens)]).tolist()
    return seqs, len(tr), lens, offs


def teacher(a: argparse.Namespace) -> int:
    seqs, n_tr_seq, lens, offs = setup(a.tokens, a.heldout)
    n = offs[-1]
    a.dir.mkdir(parents=True, exist_ok=True)
    print("CONFIG", json.dumps({"cmd": "teacher", "tokens": a.tokens, "heldout": a.heldout, "positions": n,
                                "sequences": [n_tr_seq, len(seqs) - n_tr_seq], "dir": str(a.dir),
                                "torch": torch.__version__}), flush=True)
    dev = torch.device("cuda")
    W = M.Weights()
    emb = W.get(["model.word_embeddings.weight"])["model.word_embeddings.weight"]
    h = torch.empty(n, M.D, dtype=torch.bfloat16)
    for j, s in enumerate(seqs):
        h[offs[j]:offs[j + 1]] = emb[torch.tensor(s)]
    del emb
    write_flat(a.dir / "h_0.bin", h)
    css: dict = {}
    t0 = time.time()
    for i in range(M.N_LAYER):
        if (a.dir / f"h_{i + 1}.bin").exists():
            h = read_flat(a.dir / f"h_{i + 1}.bin", n)
            continue
        w = M.to(W.layer(i), dev)
        with torch.no_grad():
            for j in range(len(seqs)):
                m = lens[j]
                if m not in css:
                    css[m] = M.rope(m, dev)
                h[offs[j]:offs[j + 1]] = M.layer_forward(i, w, h[offs[j]:offs[j + 1]].to(dev), css[m], M.Form()).cpu()
        del w
        torch.cuda.empty_cache()
        write_flat(a.dir / f"h_{i + 1}.bin", h)
        print("LAYER", i, round(time.time() - t0, 1), flush=True)
    (a.dir / "meta.json").write_text(json.dumps({"lens": lens, "n_train_seq": n_tr_seq, "positions": n, "tokens": a.tokens,
                                                 "heldout": a.heldout}) + "\n", encoding="utf-8")
    return 0


def posthoc(a: argparse.Namespace) -> int:
    kind, lo, hi = ARMS[a.arm]
    a.out.mkdir(parents=True, exist_ok=True)
    W = M.Weights()
    for i in range(lo, hi + 1):
        names = list(organ_names(kind, i))
        w = W.get([f"model.layers.{i}.{n}" for n in names])
        save = {}
        for n in names:
            o = w[f"model.layers.{i}.{n}"].unsqueeze(0)
            s, _ = init_row_scale(o, "lloyd", 0.0, 0)
            save[n] = Tern.apply(o.float(), s)[0]
            save["ternary_scale:" + n] = s.to(torch.bfloat16)
        torch.save(save, a.out / f"layer_{i}.pt")
        print("LAYER", i, flush=True)
    return 0


def train(a: argparse.Namespace) -> int:
    kind, lo, hi = ARMS[a.arm]
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
    meta = json.loads((a.dir / "meta.json").read_text(encoding="utf-8"))
    lens, n_tr_seq, n = meta["lens"], meta["n_train_seq"], meta["positions"]
    offs = np.concatenate([[0], np.cumsum(lens)]).tolist()
    n_tr = offs[n_tr_seq]
    a.out.mkdir(parents=True, exist_ok=True)
    state, flat = a.out / "state.json", a.out / "student.bin"
    log = json.loads(a.json.read_text(encoding="utf-8")) if a.json.exists() else {"arm": a.arm, "layers": {}}
    done = json.loads(state.read_text(encoding="utf-8"))["layer"] if state.exists() else lo - 1
    print("CONFIG", json.dumps({"cmd": "train", "arm": a.arm, "kind": kind, "layers": [lo, hi], "resume_after": done,
                                "positions": n, "train_positions": n_tr, "batch": a.batch, "seed": a.seed, "group": "row",
                                "init": "lloyd", "outlier": 0, "lr": {"attn": LR_ATTN, "shared": LR_SHARED, "dense": LR_DENSE},
                                "lr_scale": a.lr_scale, "form": a.form, "pick_from": str(a.pick_from) if a.pick_from else None,
                                "dir": str(a.dir), "torch": torch.__version__}), flush=True)
    dev = torch.device("cuda")
    W = M.Weights()
    hs = read_flat(a.pick_from / "student.bin", n) if a.pick_from else read_flat(flat, n) if done >= lo else read_flat(a.dir / f"h_{lo}.bin", n)
    css: dict = {}

    def cs(m):
        if m not in css:
            css[m] = M.rope(m, dev)
        return css[m]

    for i in range(done + 1, hi + 1):
        t0 = time.time()
        names = organ_names(kind, i, a.lr_scale)
        torch.manual_seed(a.seed + i)
        gen = torch.Generator().manual_seed(a.seed + i)
        wl = M.to(W.layer(i), dev)
        gw = downstream_gain2(W, i).to(dev)
        tt = read_flat(a.dir / f"h_{i + 1}.bin", n)
        orig = {k: wl[k].unsqueeze(0).cpu() for k in names}
        sign: dict = {}

        def start():
            p = {}
            for k, w in orig.items():
                if a.form == "asym":
                    s, sign[k] = asym_init(w.to(dev))
                else:
                    s, _ = init_row_scale(w.to(dev), "lloyd", 0.0, 0)
                p[k] = torch.nn.Parameter(w.float().to(dev))
                p["s:" + k] = torch.nn.Parameter(s)
            return p

        def weights(p, tern=True):
            if not tern:
                return {k: orig[k][0].to(dev) for k in names}
            if a.form == "asym":
                return {k: Asym.apply(p[k], p["s:" + k], sign[k])[0] for k in names}
            return {k: Tern.apply(p[k], p["s:" + k])[0] for k in names}

        def seq_loss(wt, j):
            x = hs[offs[j]:offs[j + 1]].to(dev)
            o = M.layer_forward(i, {**wl, **wt}, x, cs(lens[j]), M.Form()).float()
            t = tt[offs[j]:offs[j + 1]].to(dev).float()
            e = ((nrm(o) - nrm(t)) ** 2 * gw).mean(-1)
            return e.sum(), e.numel()

        def evaluate(wt, seqs) -> float:
            tot = cnt = 0
            with torch.no_grad():
                for j in seqs:
                    s, c = seq_loss(wt, j)
                    tot, cnt = tot + float(s), cnt + c
            return tot / cnt

        held = range(n_tr_seq, len(lens))
        p = start()
        rec = {"carried_only": evaluate(weights(p, False), held), "posthoc": evaluate(weights(p), held)}
        groups = [{"params": [p[k], p["s:" + k]], "lr": lr} for k, lr in names.items()]
        opt = torch.optim.Adafactor(groups, lr=LR_ATTN)
        order, batches, cur, m = torch.randperm(n_tr_seq, generator=gen).tolist(), [], [], 0
        for j in order:
            cur.append(j)
            m += lens[j]
            if m >= a.batch:
                batches.append(cur)
                cur, m = [], 0
        losses = []
        for st, b in enumerate(batches):
            wt = weights(p)
            tot = cnt = 0
            for j in b:
                s, c = seq_loss(wt, j)
                tot, cnt = tot + s, cnt + c
            loss = tot / cnt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if st % 20 == 0:
                losses.append(round(float(loss.detach()), 6))
        wt = {k: v.detach() for k, v in weights(p).items()}
        rec["trained"] = evaluate(wt, held)
        rec["trained_trainseqs"] = evaluate(wt, range(0, min(n_tr_seq, len(held))))
        rec["loss_every_20"] = losses
        save = {}
        for k in names:
            save[k] = wt[k].cpu()
            save["ternary_scale:" + k] = p["s:" + k].detach().to(torch.bfloat16).cpu()
            if a.form == "asym":
                save["sign:" + k] = sign[k].cpu()
        torch.save(save, a.out / f"layer_{i}.pt")
        with torch.no_grad():
            w2 = {**wl, **wt}
            for j in range(len(lens)):
                hs[offs[j]:offs[j + 1]] = M.layer_forward(i, w2, hs[offs[j]:offs[j + 1]].to(dev), cs(lens[j]), M.Form()).cpu()
            d = hs[offs[n_tr_seq]:].float()
            t = tt[offs[n_tr_seq]:].float()
            rec["stream_ms"] = float((d - t).pow(2).mean())
            rec["stream_norm_err"] = float((((nrm(d) - nrm(t)) ** 2) * gw.cpu()).mean())
            rec["stream_cos"] = float(torch.nn.functional.cosine_similarity(d, t, dim=-1).mean())
        if not a.pick_from:
            write_flat(flat, hs)
            state.write_text(json.dumps({"layer": i}) + "\n", encoding="utf-8")
        rec["seconds"] = round(time.time() - t0, 1)
        log["layers"][str(i)] = rec
        a.json.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
        print("LAYER", i, json.dumps({k: v for k, v in rec.items() if k != "loss_every_20"}), flush=True)
        del p, wl, wt, tt, opt, groups, orig
        torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("teacher")
    t.add_argument("--dir", type=Path, required=True)
    t.add_argument("--tokens", type=int, default=1_500_000)
    t.add_argument("--heldout", type=int, default=75_000)
    t.set_defaults(fn=teacher)
    r = sp.add_parser("train")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--json", type=Path, required=True)
    r.add_argument("--layers", default="", help="dry runs only: a part of the arm's layers")
    r.add_argument("--lr-scale", type=float, default=1.0)
    r.add_argument("--pick-from", type=Path, help="lr pick only: the student stream to start from")
    r.add_argument("--batch", type=int, default=4096)
    r.add_argument("--seed", type=int, default=20261104)
    r.add_argument("--form", choices=("tern", "asym"), default="tern", help="the organs' grid (S2L)")
    r.set_defaults(fn=train)
    q = sp.add_parser("posthoc")
    q.add_argument("--arm", required=True, choices=sorted(ARMS))
    q.add_argument("--out", type=Path, required=True)
    q.set_defaults(fn=posthoc)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
