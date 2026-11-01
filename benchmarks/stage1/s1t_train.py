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

Usage: python -I s1t_train.py --k 2 --mass 16 --tokens 1500000 --heldout 75000 --out DIR --json F [--lr 3e-3]
       [--epochs 1] [--batch 4096] [--keep8 1] [--layers 1-19] (--layers and --lr-grid are for the registered lr pick on layer 1)
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
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
            if i == 0:
                ht = [M.layer_forward(0, w, h.to(dev), cs(len(h)), M.Form()).cpu() for h in ht]
                hs = [h.clone() for h in ht]
                del w
                continue
            us, rs, base, tok_ms = [], [], [], []
            carried = sum(float((hs[j].float() - ht[j].float()).pow(2).sum()) for j in range(len(seqs)))
            for j in range(len(seqs)):
                t_new = M.layer_forward(i, w, ht[j].to(dev), cs(len(ht[j])), M.Form())
                x = hs[j].to(dev)
                aa = x + M.attention(w, M.rms(x, w["input_layernorm.weight"]), cs(len(x)))
                u = M.rms(aa, w["post_attention_layernorm.weight"])
                sh = M.mlp(u, *(w[f"mlp.shared_experts.{p}.weight"] for p in EXP)).float()
                rs.append((t_new.float() - aa.float() - sh).to(torch.bfloat16).cpu())
                tok_ms.append(t_new.float().pow(2).mean(-1).cpu())
                us.append(u.cpu())
                base.append((aa.float() + sh).to(torch.bfloat16).cpu())
                ht[j] = t_new.cpu()
        u_all, r_all, inv_ms = torch.cat(us), torch.cat(rs), 1.0 / torch.cat(tok_ms)
        gw = downstream_gain2(W, i).to(dev)
        del us, rs
        held = slice(n_tr, None)
        bias = w["mlp.gate.expert_bias"]
        orig = {"gate": w["mlp.gate.weight"].float().cpu(), **{p: w[f"mlp.experts.{p}"].cpu() for p in EXP}}
        del w
        torch.cuda.empty_cache()
        wl = M.to(W.layer(i), dev)                           # attention and shared expert, for the propagation
        for p in EXP:
            del wl[f"mlp.experts.{p}"]
        torch.cuda.empty_cache()

        def rel(params, f: M.Form, rows: slice) -> float:
            num = den = 0.0
            eg, eu, ed = (params[p].to(torch.bfloat16).unbind(0) for p in EXP)
            with torch.no_grad():
                for s in range(rows.start or 0, rows.stop or len(u_all), a.batch):
                    e = min(s + a.batch, rows.stop or len(u_all))
                    u = u_all[s:e].to(dev)
                    r = r_all[s:e].to(dev).float()
                    m = inv_ms[s:e].to(dev).unsqueeze(-1) * gw
                    ids, wts = M.route(u, params["gate"], bias, f)
                    y = M.routed(u, ids, wts, eg, eu, ed)
                    num += float(((y - r) ** 2 * m).sum())
                    den += float((r ** 2 * m).sum())
            return num / den

        def fresh():
            return {"gate": torch.nn.Parameter(orig["gate"].to(dev)),
                    **{p: torch.nn.Parameter(orig[p].float().to(dev)) for p in EXP}}

        def train(params, lr: float) -> list[float]:
            opt = torch.optim.Adafactor(params.values(), lr=lr)
            steps = int(a.epochs * n_tr / a.batch)
            losses = []
            for st in range(steps):
                idx = torch.randint(0, n_tr, (a.batch,), generator=gen)
                u = u_all[idx].to(dev, non_blocking=True)
                r = r_all[idx].to(dev, non_blocking=True).float()
                m = inv_ms[idx].to(dev, non_blocking=True).unsqueeze(-1) * gw
                ids, wts = M.route(u, params["gate"], bias, form)
                eg, eu, ed = (params[p].to(torch.bfloat16).unbind(0) for p in EXP)
                loss = ((M.routed(u, ids, wts, eg, eu, ed) - r) ** 2 * m).mean()
                if a.keep8:
                    ids8, wts8 = M.route(u, params["gate"], bias, M.Form())
                    loss = loss + a.keep8 * ((M.routed(u, ids8, wts8, eg, eu, ed) - r) ** 2 * m).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if st % 20 == 0:
                    losses.append(round(float(loss.detach()), 6))
            return losses

        hrows = slice(n_tr, len(u_all))
        p0 = {k: v.to(dev) for k, v in fresh().items()}
        n_all = len(u_all)
        rec = {"target_ms": float(r_all.float().pow(2).mean()), "carried_ms": carried / (n_all * M.D),
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
        rec["loss_every_20"] = train(p, lr)
        rec["rel_form_trained"] = rel(p, form, hrows)
        rec["rel_k8_trained"] = rel(p, M.Form(), hrows)
        rec["rel_form_trained_trainrows"] = rel(p, form, slice(0, min(n_tr, 200_000)))
        torch.save({"mlp.gate.weight": p["gate"].detach().cpu(),
                    **{f"mlp.experts.{q}": p[q].detach().to(torch.bfloat16).cpu() for q in EXP}}, a.out / f"layer_{i}.pt")
        with torch.no_grad():
            eg, eu, ed = (p[q].to(torch.bfloat16).unbind(0) for q in EXP)
            off = 0
            for j in range(len(seqs)):
                n = len(hs[j])
                u = u_all[off:off + n].to(dev)
                ids, wts = M.route(u, p["gate"], bias, form)
                hs[j] = (base[j].to(dev).float() + M.routed(u, ids, wts, eg, eu, ed)).to(torch.bfloat16).cpu()
                off += n
        del p, wl, u_all, r_all, inv_ms, base, orig
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
