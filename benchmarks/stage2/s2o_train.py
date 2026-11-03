"""Stage 2 part 1, test 2: one organ of Ling-mini ternary in IQ2_BN's layout (codes {-1, 0, 1}, one scale per row),
trained alone on the original model's inputs, as stage 1 (b)'s layer 10 (s1t_train.py --layers 10-10 --alone).

Organs (--organ), every other weight the original bf16 one, routing at k = 8:
    attn   layer 10: attention.query_key_value and attention.dense;
    shexp  layer 10: the shared expert's gate, up and down;
    exps   layer 10: the 256 routed experts' gate, up and down (stage 1 (b)'s tern_l10 with one scale per row instead
           of one per 256 weights);
    dense  layer 0: the dense FFN's gate, up and down (the only dense FFN of the model);
    head   lm_head on the final norm's output; with --alpha a, the column scale c = rms^a of s2h_head.py (fixed, not
           trained), so the matrix is W diag(c) and the final norm's gain becomes g / c.
Quantizer: stage 1 (b)'s Tern with the group = the whole row (--group 0, registered).  Latent fp32 weights start at the
original ones and pass a straight-through estimator; the scale per row gets LSQ's gradient and starts (--init lloyd,
registered) at the fixed point of s = mean(|w| where |w| > s / 2) from TWN's, the least-squares scale for the rounding
the quantizer applies.  Stage 1 (b)'s outlier rule (a row whose max |w| exceeds f times its scale keeps the max,
untrained) is off (--outlier 0, registered).  --group 256, --init twn and --outlier 8 give (b)'s quantizer back, for
the dry runs that chose these settings.
Loss:
    layer organs: the layer's output against the teacher's, in the space the later norms read (stage 1 (b)):
                  mean of (nrm(o_s) - nrm(o_t))^2 * gw, gw = g^2 averaged over every RMSNorm after the layer;
    head:         KL(teacher || student) over the full vocabulary, per token, teacher = the original head in bf16.
Data: stage 1's corpus (the first 1.5 M tokens train, the next 75 k are held out), one epoch, Adafactor, batch 4,096
tokens (whole sequences for attn; 1,024 for the head, whose logits fill the GPU).  lr: each of --lr-grid trains from the same start, the pick is the least held-out
loss; the trained weights are the pick's.
Logged per lr and for the pick, on held-out tokens: the loss with the original weights (the floor: bf16 rounding),
post-hoc (initial scales, no training), trained, and with the organ's output zeroed (the scale of the loss); for the
head also top-1 agreement and KL in nats.
Saves organ.pt: the trained tensors as s * q in bf16 under their HF names, the scales beside them, and for the head
the gain g / c.  s2o_export.py writes them into the GGUF as TQ2_0, which holds them exactly.

Usage: python -I benchmarks/stage2/s2o_train.py --organ attn --out DIR --json F
       python -I benchmarks/stage2/s2o_train.py --organ head --alpha -0.35 --imatrix F --out DIR --json F
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
import s1t_model as M  # noqa: E402
from s1t_train import EXP, Tern, downstream_gain2, load_corpus  # noqa: E402

ORGANS = {   # organ: (layer, trained tensors under s1t_model's layer names, or the head's)
    "attn": (10, ["attention.query_key_value.weight", "attention.dense.weight"]),
    "shexp": (10, [f"mlp.shared_experts.{p}.weight" for p in EXP]),
    "exps": (10, [f"mlp.experts.{p}" for p in EXP]),
    "dense": (0, [f"mlp.{p}.weight" for p in EXP]),
    "head": (M.N_LAYER, ["lm_head.weight"]),
}


def init_row_scale(w: torch.Tensor, init: str, outlier: float, group: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """w [E, O, I] -> (scale [E, O, 1], outlier rows [E, O, 1]).  TWN per row; with init 'lloyd', then the fixed point
    of s = mean(|w| where |w| > s / 2), the least-squares scale for the rounding the quantizer applies; with outlier
    f > 0, the max where it exceeds f times the scale."""
    e, o, i = w.shape
    g = group or i
    a = w.float().abs().view(e, o, i // g, g)
    keep = (a > 0.7 * a.mean(-1, keepdim=True)).float()
    s = (a * keep).sum(-1, keepdim=True) / keep.sum(-1, keepdim=True).clamp_min(1)
    if init == "lloyd":
        for _ in range(30):
            keep = (a > s / 2).float()
            s = (a * keep).sum(-1, keepdim=True) / keep.sum(-1, keepdim=True).clamp_min(1)
    mx = a.amax(-1, keepdim=True)
    big = (mx > outlier * s) if outlier > 0 else torch.zeros_like(s, dtype=torch.bool)
    return torch.where(big, mx, s).squeeze(-1), big.squeeze(-1)


def nrm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + M.EPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", required=True, choices=sorted(ORGANS))
    ap.add_argument("--tokens", type=int, default=1_500_000)
    ap.add_argument("--heldout", type=int, default=75_000)
    ap.add_argument("--lr-grid", default="1e-3,3e-3,1e-2")
    ap.add_argument("--batch", type=int, default=0, help="tokens per step: 4,096, or 1,024 for the head")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--group", type=int, default=0, help="weights per scale; 0 = the row (IQ2_BN), the registered one")
    ap.add_argument("--init", default="lloyd", choices=["twn", "lloyd"])
    ap.add_argument("--outlier", type=float, default=0.0, help="stage 1 (b)'s outlier rule factor (8), 0 = off")
    ap.add_argument("--alpha", type=float, default=0.0, help="head only: the column scale's exponent")
    ap.add_argument("--imatrix", type=Path, help="head only, with --alpha: s2h_head.py's imatrix of the head's input")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    if a.alpha and (a.organ != "head" or not a.imatrix):
        raise SystemExit("--alpha is for --organ head, with --imatrix")
    L, names = ORGANS[a.organ]
    head = a.organ == "head"
    tok = a.organ in ("shexp", "exps", "dense")
    a.batch = a.batch or (1024 if head else 4096)
    lrs = [float(x) for x in a.lr_grid.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    print("CONFIG", json.dumps({**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                                "layer": L, "tensors": names, "group": a.group or "row", "torch": torch.__version__}), flush=True)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    tr, he = load_corpus(a.tokens, a.heldout)
    seqs = tr + he
    n_tr, n_all = sum(map(len, tr)), sum(map(len, seqs))
    W = M.Weights()
    emb = W.get(["model.word_embeddings.weight"])["model.word_embeddings.weight"]
    hs = [emb[torch.tensor(s)] for s in seqs]
    del emb
    css: dict = {}

    def cs(n):
        if n not in css:
            css[n] = M.rope(n, dev)
        return css[n]

    t0 = time.time()
    with torch.no_grad():                       # the original model up to the organ's layer (exclusive)
        for i in range(L):
            w = M.to(W.layer(i), dev)
            hs = [M.layer_forward(i, w, h.to(dev), cs(len(h)), M.Form()).cpu() for h in hs]
            del w
            torch.cuda.empty_cache()
    rec = {"organ": a.organ, "layer": L, "train_tokens": n_tr, "heldout_tokens": n_all - n_tr,
           "prefix_seconds": round(time.time() - t0, 1)}

    # ---- the organ's original weights, the frozen rest, and the data each step reads
    if head:
        g = W.get(["model.norm.weight", "lm_head.weight"])
        gain, w_head = g["model.norm.weight"].to(dev), g["lm_head.weight"]
        c = torch.ones(M.D)
        if a.alpha:
            from s2h_head import input_ms, scale_of
            c = scale_of(torch.from_numpy(input_ms(a.imatrix)).float(), a.alpha)
        c = c.to(dev)
        z_all = torch.empty(n_all, M.D, dtype=torch.bfloat16)          # the final norm's output (teacher = student)
        off = 0
        with torch.no_grad():
            for h in hs:
                z_all[off:off + len(h)] = M.rms(h.to(dev), gain).cpu()
                off += len(h)
        del hs
        orig = {"lm_head.weight": (w_head.float() * c.cpu()).unsqueeze(0)}          # [1, V, D], W diag(c)
        w_teacher = w_head.to(dev)
    else:
        wl = M.to(W.layer(L), dev)
        orig = {n: (wl[n] if n.startswith("mlp.experts.") else wl[n].unsqueeze(0)).cpu() for n in names}
        gw = downstream_gain2(W, L).to(dev)
        if tok:      # per token: the FFN input u, the frozen rest of the layer's output, the teacher's output
            u_all, base_all, t_all = (torch.empty(n_all, M.D, dtype=torch.bfloat16) for _ in range(3))
            off = 0
            with torch.no_grad():
                for h in hs:
                    n = len(h)
                    x = h.to(dev)
                    aa = x + M.attention(wl, M.rms(x, wl["input_layernorm.weight"]), cs(n))
                    u = M.rms(aa, wl["post_attention_layernorm.weight"])
                    if L == 0:
                        rest = torch.zeros_like(aa, dtype=torch.float32)
                    else:
                        ids, wts = M.route(u, wl["mlp.gate.weight"], wl["mlp.gate.expert_bias"], M.Form())
                        ro = M.routed(u, ids, wts, *(wl[f"mlp.experts.{p}"].unbind(0) for p in EXP))
                        sh = M.mlp(u, *(wl[f"mlp.shared_experts.{p}.weight"] for p in EXP)).float()
                        rest = sh if a.organ == "exps" else ro
                    u_all[off:off + n], base_all[off:off + n] = u.cpu(), (aa.float() + rest).to(torch.bfloat16).cpu()
                    t_all[off:off + n] = M.layer_forward(L, wl, x, cs(n), M.Form()).cpu()
                    off += n
            del hs
        else:        # attention: whole sequences, the input and the teacher's output
            with torch.no_grad():
                ts = [M.layer_forward(L, wl, h.to(dev), cs(len(h)), M.Form()).cpu() for h in hs]
    rec["data_seconds"] = round(time.time() - t0, 1)

    # ---- quantizer and the organ's forward with given weights
    def start():
        p, outl = {}, {}
        for n, w in orig.items():
            s, big = init_row_scale(w.to(dev), a.init, a.outlier, a.group)
            p[n] = torch.nn.Parameter(w.float().to(dev))
            p["s:" + n] = torch.nn.Parameter(s)
            p["s:" + n].register_hook(lambda gr, big=big: gr.masked_fill(big, 0.0))
            outl[n] = int(big.sum())
        return p, outl

    def weights(p, mode):
        """mode: 'tern' (s * q), 'orig' (the original bf16), 'zero' (the organ removed)."""
        out = {}
        for n in orig:
            if mode == "tern":
                v = Tern.apply(p[n], p["s:" + n])
            else:
                v = orig[n].to(dev).to(torch.bfloat16) * (0 if mode == "zero" else 1)
            out[n] = v if n.startswith("mlp.experts.") else v[0]
        return out

    def organ_out(wt, u):
        if a.organ == "exps":
            ids, wts = M.route(u, wl["mlp.gate.weight"], wl["mlp.gate.expert_bias"], M.Form())
            return M.routed(u, ids, wts, *(wt[f"mlp.experts.{p}"].unbind(0) for p in EXP))
        return M.mlp(u, *(wt[n] for n in names)).float()

    def batch_loss(wt, rows=None, seq_ids=None):
        """Sum of the per-token loss over the rows (token organs, head) or the sequences (attn); and the token count."""
        if head:
            z = z_all[rows].to(dev, non_blocking=True)
            with torch.no_grad():
                lt = (z @ w_teacher.t()).float().log_softmax(-1)
            ls = ((z.float() / c).to(torch.bfloat16) @ wt["lm_head.weight"].t()).float().log_softmax(-1)
            kl = (lt.exp() * (lt - ls)).sum(-1)
            return kl.sum(), kl.numel(), int((lt.argmax(-1) == ls.argmax(-1)).sum())
        if tok:
            u = u_all[rows].to(dev, non_blocking=True)
            o = base_all[rows].to(dev, non_blocking=True).float() + organ_out(wt, u)
            t = t_all[rows].to(dev, non_blocking=True).float()
            e = ((nrm(o) - nrm(t)) ** 2 * gw).mean(-1)
            return e.sum(), e.numel(), 0
        tot, cnt = 0.0, 0
        w2 = {**wl, **wt}
        for j in seq_ids:
            x = hs[j].to(dev)
            o = M.layer_forward(L, w2, x, cs(len(x)), M.Form()).float()
            e = ((nrm(o) - nrm(ts[j].to(dev).float())) ** 2 * gw).mean(-1)
            tot, cnt = tot + e.sum(), cnt + e.numel()
        return tot, cnt, 0

    held_rows = range(n_tr, n_all)
    held_seqs = list(range(len(tr), len(seqs)))

    def evaluate(wt) -> dict:
        tot = cnt = top = 0
        with torch.no_grad():
            if head or tok:
                for s in range(n_tr, n_all, a.batch):
                    l, n, t1 = batch_loss(wt, rows=slice(s, min(s + a.batch, n_all)))
                    tot, cnt, top = tot + float(l), cnt + n, top + t1
            else:
                for j in held_seqs:
                    l, n, _ = batch_loss(wt, seq_ids=[j])
                    tot, cnt = tot + float(l), cnt + n
        r = {"loss": tot / cnt}
        if head:
            r["top1_agreement"] = top / cnt
        return r

    gen = torch.Generator().manual_seed(a.seed)

    def train(p, lr) -> list[float]:
        opt = torch.optim.Adafactor(list(p.values()), lr=lr)
        losses = []
        if head or tok:
            steps = n_tr // a.batch
            batches = (torch.randint(0, n_tr, (a.batch,), generator=gen) for _ in range(steps))
        else:                                   # whole sequences, about a.batch tokens per step, one epoch
            order, batches, cur, n = torch.randperm(len(tr), generator=gen).tolist(), [], [], 0
            for j in order:
                cur.append(j)
                n += len(seqs[j])
                if n >= a.batch:
                    batches.append(cur)
                    cur, n = [], 0
        for st, b in enumerate(batches):
            l, n, _ = batch_loss(weights(p, "tern"), rows=b) if (head or tok) else batch_loss(weights(p, "tern"),
                                                                                                 seq_ids=b)
            loss = l / n
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if st % 20 == 0:
                losses.append(round(float(loss.detach()), 6))
        return losses

    p, outl = start()
    rec["outlier_rows"] = outl
    rec["heldout"] = {"orig": evaluate(weights(p, "orig")), "posthoc": evaluate(weights(p, "tern"))}
    if not head:
        rec["heldout"]["zero"] = evaluate(weights(p, "zero"))
    del p
    rec["lr_grid"], best = {}, None
    for lr in lrs:
        t1 = time.time()
        p, _ = start()
        losses = train(p, lr)
        ev = evaluate(weights(p, "tern"))
        rec["lr_grid"][str(lr)] = {**ev, "loss_every_20": losses, "seconds": round(time.time() - t1, 1)}
        print("LR", lr, json.dumps(ev), flush=True)
        if best is None or ev["loss"] < best[1]["loss"]:
            best = (lr, ev, {k: v.detach().cpu() for k, v in p.items()})
        del p
        torch.cuda.empty_cache()
    lr, ev, p = best
    rec["lr"], rec["heldout"]["trained"] = lr, ev
    with torch.no_grad():
        save = {}
        for n in orig:
            v = Tern.apply(p[n], p["s:" + n]).cpu()
            save[n] = v if n.startswith("mlp.experts.") else v[0]
            save["ternary_scale:" + n] = p["s:" + n].to(torch.bfloat16).cpu()
        if head:
            save["output_norm.weight"] = (gain.float() / c).cpu()
            save["column_scale"] = c.cpu()
        torch.save(save, a.out / "organ.pt")
    rec["seconds"] = round(time.time() - t0, 1)
    print("ORGAN", json.dumps({k: v for k, v in rec.items() if k != "lr_grid"}), flush=True)
    a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
