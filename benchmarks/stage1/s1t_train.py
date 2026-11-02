"""Stage 1 (a) part 2: block-wise distillation of Ling-mini's routers and routed experts to a k = 4 routing form.

Sequential, one layer at a time (the model never sits whole on the GPU):
- teacher stream H_t: the model as trained (k = 8); student stream H_s: the layers already trained, at the form;
- for layer i >= 1 (layer 0 is dense and identical in both streams):
  - teacher: H_t <- layer_i(H_t) at k = 8;
  - student, frozen parts: a = H_s + attn_i(H_s), u = norm(a), sh = shared_i(u);
  - target of the routed part: r = H_t(new) - a - sh, so the routed experts also absorb the error carried in H_s;
  - train layer i's router weight and its 256 routed experts (fp32 masters, bf16 compute, Adafactor) to minimize
    the error of routed_form(u) against r in the space the later layers read: each dim weighted by g^2 averaged over
    every RMSNorm after layer i, each token divided by the mean square of its teacher residual (the norm's
    denominator).  A plain MSE is dominated by the massive dims that layer 1 writes (dim 1149), which the norms
    scale down; the selection bias stays as it is;
  - with --keep8 w, the loss adds w times the same error at k = 8 with the trained weights: a decode-only form runs the
    prompt at k = 8 on the same weights, and that path must stay close to the teacher;
  - H_s <- a + sh + routed_form(u) with the trained weights, on every sequence (training and held-out).
- Logged per layer, on held-out tokens: rel = weighted error / weighted mean(r^2) before and after training, and the same at k = 8 with
  the original weights (the part of the error that only comes from the student's input), and at k = 8 with the
  trained weights; also mean(r^2) and the
  mean square of H_s - H_t at the layer's input (the error carried in).
Saves layer_<i>.pt (router fp32, expert stacks bf16) for s1t_export.py, and a JSON log.

Stage 1 (b), --ternary G: the routed experts become ternary, w = s * q with q in {-1, 0, +1} and one scale s per G
consecutive weights along the input dim (G = 256 is llama.cpp's TQ2_0 block, so the export is exact).  Latent fp32
weights start at the original ones, s starts at the least-squares ternary scale of its group (TWN), and both train
through a straight-through estimator (round passes the gradient inside [-1, 1]; s gets the LSQ gradient).  The router
is not trained and the form is k = 8.  rel_*_orig is then the post-hoc ternary model (initial scales, no training).
The saved expert stacks are s * q in bf16 (s rounded to bf16 first); the scales are saved beside them.
--layers L-L with --alone trains layer L only, on the original model's inputs (the layers before it are not changed).

Usage: python -I s1t_train.py --k 2 --mass 16 --tokens 1500000 --heldout 75000 --out DIR --json F [--lr 3e-3]
       [--epochs 1] [--batch 4096] [--keep8 1] [--layers 1-19] (--layers and --lr-grid are for the registered lr pick on layer 1)
       python -I s1t_train.py --k 8 --ternary 256 --out DIR --json F [--layers 10-10 --alone]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1t_model as M  # noqa: E402

CORPUS = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s3/corpus_v1/teacher.tsv")
EXP = ("gate_proj", "up_proj", "down_proj")


def load_corpus(n_train: int, n_held: int) -> tuple[list[list[int]], list[list[int]]]:
    tr, he, nt, nh = [], [], 0, 0
    with CORPUS.open(encoding="utf-8") as f:
        for ln in f:
            ids = [int(t) for t in ln.rstrip("\n").split("\t")[1].split(",")]
            if nt < n_train:
                tr.append(ids)
                nt += len(ids)
            elif nh < n_held:
                he.append(ids)
                nh += len(ids)
            else:
                break
    return tr, he


def student_pre(w: dict, x: torch.Tensor, cs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen part of a MoE layer on the student stream: (attention residual a, MoE input u, shared expert out, fp32)."""
    aa = x + M.attention(w, M.rms(x, w["input_layernorm.weight"]), cs)
    u = M.rms(aa, w["post_attention_layernorm.weight"])
    return aa, u, M.mlp(u, *(w[f"mlp.shared_experts.{p}.weight"] for p in EXP)).float()


class Tern(torch.autograd.Function):
    """w [E, O, I] fp32 and s [E, O, I/G] fp32 -> bf16 s * clamp(round(w / s), -1, 1); STE for w, LSQ for s."""

    # Some groups are all zero in the checkpoint (layer 1: 45,928 of gate's and up's): their scale is 0, and they
    # stay 0 (the division uses 1 there, and w gets no gradient while s is 0).

    @staticmethod
    def forward(ctx, w, s):
        ctx.save_for_backward(w, s)
        e, o, i = w.shape
        x = w.view(e, o, s.shape[-1], -1) / torch.where(s > 0, s, 1.0).unsqueeze(-1)
        return (x.clamp_(-1, 1).round_() * s.to(torch.bfloat16).float().unsqueeze(-1)).view(e, o, i).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, g):
        w, s = ctx.saved_tensors
        e, o, i = w.shape
        g = g.float().view(e, o, s.shape[-1], -1)
        x = w.view(e, o, s.shape[-1], -1) / torch.where(s > 0, s, 1.0).unsqueeze(-1)
        inside = (x.abs() <= 1).float() * (s > 0).float().unsqueeze(-1)
        gs = (g * (x.clamp(-1, 1).round() - x * inside)).sum(-1)
        return (g * inside).view(e, o, i), gs


def twn_scale(w: torch.Tensor, grp: int) -> torch.Tensor:
    """Least-squares ternary scale per group (TWN): threshold 0.7 mean|w|, scale = mean |w| above it."""
    a = w.float().abs().view(*w.shape[:-1], w.shape[-1] // grp, grp)
    keep = (a > 0.7 * a.mean(-1, keepdim=True)).float()
    return (a * keep).sum(-1) / keep.sum(-1).clamp_min(1)


OUTLIER = 8.0


def init_scale(w: torch.Tensor, grp: int) -> tuple[torch.Tensor, torch.Tensor]:
    """TWN scale, except in outlier groups (max |w| > OUTLIER x the TWN scale), whose scale is that max, so the
    outlier is kept exactly (layer 1's expert 13 down_proj[1149, 256] = 2.52 writes the massive activation; TWN
    would make it 0.21).  Returns (scale, outlier mask); outlier scales are not trained."""
    s = twn_scale(w, grp)
    mx = w.float().abs().view(*w.shape[:-1], w.shape[-1] // grp, grp).amax(-1)
    big = mx > OUTLIER * s
    return torch.where(big, mx, s), big


def downstream_gain2(W: M.Weights, i: int) -> torch.Tensor:
    """Mean of g^2 over every RMSNorm that reads the residual stream after layer i (the error's weight per dim)."""
    names = [f"model.layers.{j}.{n}.weight" for j in range(i + 1, M.N_LAYER)
             for n in ("input_layernorm", "post_attention_layernorm")] + ["model.norm.weight"]
    g = W.get(names)
    return torch.stack([g[n].float() ** 2 for n in names]).mean(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--mass", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=1_500_000)
    ap.add_argument("--heldout", type=int, default=75_000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-grid", default="")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--layers", default="1-19")
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--keep8", type=float, default=0.0,
                    help="weight of a second term: the same loss at k = 8 with the trained weights (decode-only forms)")
    ap.add_argument("--ternary", type=int, default=0, help="group size of ternary experts (0 = not ternary)")
    ap.add_argument("--alone", action="store_true", help="the layers before --layers stay as the original model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    if a.ternary and (a.k != 8 or a.mass or a.keep8):
        raise SystemExit("--ternary runs at k = 8, without --mass or --keep8")
    lo, hi = (int(x) for x in a.layers.split("-"))
    form = M.Form(a.k, a.mass)
    lrs = [float(x) for x in a.lr_grid.split(",")] if a.lr_grid else [a.lr]
    a.out.mkdir(parents=True, exist_ok=True)
    print("CONFIG", json.dumps({**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                                "form": [form.k, form.mass], "torch": torch.__version__}), flush=True)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    tr, he = load_corpus(a.tokens, a.heldout)
    seqs = tr + he
    n_tr = sum(map(len, tr))
    W = M.Weights()
    emb = W.get(["model.word_embeddings.weight"])["model.word_embeddings.weight"]
    ht = [emb[torch.tensor(s)] for s in seqs]               # CPU bf16, one tensor per sequence
    hs = [h.clone() for h in ht]
    del emb
    css = {}
    log = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "train_tokens": n_tr, "heldout_tokens": sum(map(len, he)), "sequences": [len(tr), len(he)], "layers": {}}

    def cs(n):
        if n not in css:
            css[n] = M.rope(n, dev)
        return css[n]

    for i in range(0, hi + 1):
        t0 = time.time()
        w = M.to(W.layer(i), dev)
        with torch.no_grad():
            if i == 0 or (a.alone and i < lo):
                ht = [M.layer_forward(i, w, h.to(dev), cs(len(h)), M.Form()).cpu() for h in ht]
                hs = [h.clone() for h in ht]
                del w
                continue
            # one buffer each, filled in place (RAM: the layer's inputs and targets are ~6 GB each at 1.5 M tokens)
            n_all = sum(map(len, seqs))
            u_all = torch.empty(n_all, M.D, dtype=torch.bfloat16)
            r_all = torch.empty(n_all, M.D, dtype=torch.bfloat16)
            inv_ms = torch.empty(n_all)
            t_all = torch.empty(n_all, M.D, dtype=torch.bfloat16) if a.ternary else None
            carried, r_sq, off = 0.0, 0.0, 0
            for j in range(len(seqs)):
                n = len(seqs[j])
                carried += float((hs[j].float() - ht[j].float()).pow(2).sum())
                t_new = M.layer_forward(i, w, ht[j].to(dev), cs(n), M.Form())
                aa, u, sh = student_pre(w, hs[j].to(dev), cs(n))
                r = (t_new.float() - aa.float() - sh).to(torch.bfloat16)
                r_sq += float(r.float().pow(2).sum())
                r_all[off:off + n] = r.cpu()
                inv_ms[off:off + n] = 1.0 / t_new.float().pow(2).mean(-1).cpu()
                u_all[off:off + n] = u.cpu()
                if a.ternary:
                    t_all[off:off + n] = t_new.to(torch.bfloat16).cpu()
                ht[j] = t_new.cpu()
                off += n
        gw = downstream_gain2(W, i).to(dev)
        held = slice(n_tr, None)
        bias = w["mlp.gate.expert_bias"]
        orig = {"gate": w["mlp.gate.weight"].float().cpu(), **{p: w[f"mlp.experts.{p}"].cpu() for p in EXP}}
        del w
        torch.cuda.empty_cache()
        wl = M.to(W.layer(i), dev)                           # attention and shared expert, for the propagation
        for p in EXP:
            del wl[f"mlp.experts.{p}"]
        torch.cuda.empty_cache()

        def experts(params):
            if a.ternary:
                return (Tern.apply(params[p], params["s_" + p]).unbind(0) for p in EXP)
            return (params[p].to(torch.bfloat16).unbind(0) for p in EXP)

        def nrm(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + M.EPS)

        def err(y, rows):
            """Per-element weighted error of routed output y on these rows.  Ternary: exactly in the space the later
            norms read, h / rms(h) against the teacher's, so the denominator moves with the student (layer 1 writes
            the massive dim 1149, and the linearized form below gives its tokens almost no weight)."""
            r = r_all[rows].to(dev, non_blocking=True).float()
            if a.ternary:
                t = t_all[rows].to(dev, non_blocking=True).float()
                return (nrm(t - r + y) - nrm(t)) ** 2 * gw
            return (y - r) ** 2 * inv_ms[rows].to(dev, non_blocking=True).unsqueeze(-1) * gw

        def rel(params, f: M.Form, rows: slice) -> float:
            num = den = 0.0
            eg, eu, ed = experts(params)
            with torch.no_grad():
                for s in range(rows.start or 0, rows.stop or len(u_all), a.batch):
                    e = min(s + a.batch, rows.stop or len(u_all))
                    u = u_all[s:e].to(dev)
                    ids, wts = M.route(u, params["gate"], bias, f)
                    y = M.routed(u, ids, wts, eg, eu, ed)
                    num += float(err(y, slice(s, e)).sum())
                    den += float(err(torch.zeros_like(y), slice(s, e)).sum())
            return num / den

        outliers = {}

        def fresh():
            if a.ternary:                    # the router stays as trained; latent experts and their group scales
                d = {"gate": orig["gate"].to(dev), **{p: torch.nn.Parameter(orig[p].float().to(dev)) for p in EXP}}
                for p in EXP:
                    s, big = init_scale(orig[p].to(dev), a.ternary)
                    d["s_" + p] = torch.nn.Parameter(s)
                    d["s_" + p].register_hook(lambda g, big=big: g.masked_fill(big, 0.0))
                    outliers[p] = int(big.sum())
                return d
            return {"gate": torch.nn.Parameter(orig["gate"].to(dev)),
                    **{p: torch.nn.Parameter(orig[p].float().to(dev)) for p in EXP}}

        def train(params, lr: float) -> list[float]:
            opt = torch.optim.Adafactor([v for v in params.values() if v.requires_grad], lr=lr)
            steps = int(a.epochs * n_tr / a.batch)
            losses = []
            for st in range(steps):
                idx = torch.randint(0, n_tr, (a.batch,), generator=gen)
                u = u_all[idx].to(dev, non_blocking=True)
                ids, wts = M.route(u, params["gate"], bias, form)
                eg, eu, ed = experts(params)
                loss = err(M.routed(u, ids, wts, eg, eu, ed), idx).mean()
                if a.keep8:
                    ids8, wts8 = M.route(u, params["gate"], bias, M.Form())
                    loss = loss + a.keep8 * err(M.routed(u, ids8, wts8, eg, eu, ed), idx).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if st % 20 == 0:
                    losses.append(round(float(loss.detach()), 6))
            return losses

        hrows = slice(n_tr, len(u_all))
        p0 = {k: v.to(dev) for k, v in fresh().items()}
        rec = {"target_ms": r_sq / (n_all * M.D), "carried_ms": carried / (n_all * M.D),
               "rel_k8_orig": rel(p0, M.Form(), hrows), "rel_form_orig": rel(p0, form, hrows)}
        del p0
        if len(lrs) > 1:
            rec["lr_grid"] = {}
            for lr in lrs:
                p = fresh()
                train(p, lr)
                rec["lr_grid"][str(lr)] = rel(p, form, hrows) + (a.keep8 * rel(p, M.Form(), hrows) if a.keep8 else 0.0)
                del p
                torch.cuda.empty_cache()
            lr = min(lrs, key=lambda x: rec["lr_grid"][str(x)])
        else:
            lr = lrs[0]
        p = fresh()
        rec["lr"] = lr
        if a.ternary:
            rec["outlier_groups"] = dict(outliers)
        rec["loss_every_20"] = train(p, lr)
        rec["rel_form_trained"] = rel(p, form, hrows)
        rec["rel_k8_trained"] = rel(p, M.Form(), hrows)
        rec["rel_form_trained_trainrows"] = rel(p, form, slice(0, min(n_tr, 200_000)))
        with torch.no_grad():
            if a.ternary:
                torch.save({"mlp.gate.weight": p["gate"].detach().cpu(),
                            **{f"mlp.experts.{q}": Tern.apply(p[q], p["s_" + q]).cpu() for q in EXP},
                            **{f"ternary_scale.{q}": p["s_" + q].detach().to(torch.bfloat16).cpu() for q in EXP}},
                           a.out / f"layer_{i}.pt")
            else:
                torch.save({"mlp.gate.weight": p["gate"].detach().cpu(),
                            **{f"mlp.experts.{q}": p[q].detach().to(torch.bfloat16).cpu() for q in EXP}},
                           a.out / f"layer_{i}.pt")
            eg, eu, ed = experts(p)
            for j in range(len(seqs)):          # attention and shared expert recomputed: no stored bases
                aa, u, sh = student_pre(wl, hs[j].to(dev), cs(len(hs[j])))
                ids, wts = M.route(u, p["gate"], bias, form)
                hs[j] = (aa.float() + sh + M.routed(u, ids, wts, eg, eu, ed)).to(torch.bfloat16).cpu()
        del p, wl, u_all, r_all, inv_ms, t_all, orig
        torch.cuda.empty_cache()
        rec["seconds"] = round(time.time() - t0, 1)
        log["layers"][i] = rec
        print("LAYER", i, json.dumps({k: v for k, v in rec.items() if k != "loss_every_20"}), flush=True)
        a.json.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
        if len(lrs) > 1:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
