"""C50-T2 trained stitch: can a map TRAINED on F0's experts' own output make them read v0's state?

The linear cell (c50_t2_stitch.py) fitted ridge maps on the states and failed; this trains the map on what F0's
routed experts output.  For each v0 MoE layer l, at its depth-matched F0 layer j = round(l * 31 / 19), from the same
two dumps and the same fit / select / test documents (positions SKIP..LEN-1):
  arms, each starting from the ridge map fitted on the fit rows (lambda on the select R^2, as the linear cell):
    ridge   the ridge map itself, untrained (reproduces the linear post-hoc);
    linear  the affine map trained;
    mlp     the affine map plus a residual MLP (hidden HIDDEN, output layer zero-initialised), trained;
  training: F0's frozen routed-expert block (bf16 weights on the GPU) on the mapped state, against the same block on
  F0's own state: loss = (1 - cosine of the routed output) + ALPHA * (router-logit MSE / logit variance); the top-k
  selection is the block's own (discrete), gradients flow through the selected weights and the experts.  AdamW,
  EPOCHS epochs; the epoch (0 = the ridge start) is kept on the select cosine.  The measured map of l is the arm
  (linear or mlp) of higher select cosine; its test numbers are the measure;
  controls, through the same code:
    tool      the bf16 block on F0's own test states vs llama.cpp's ffn_moe_topk / ffn_moe_out;
    ridge     the ridge arm's test overlap and cosine vs the linear post-hoc record (same rows, same lambda);
    positive  input = F0's own state on its top PCA_DIM principal components (fit rows), map started at the
              reconstruction from the first PCA_START of them (the rest weighted 0) and trained as the linear arm:
              a map that stitches exists in the family, and the start reads about as badly as the ridge maps do;
    shuffled  linear and mlp arms on fit pairs permuted across positions (no map exists), selected the same way;
    noise     bars: F0's own test state plus Gaussian noise of known R^2 (0.3, 0.5), read by the same block;
  diagnostic: the linear arm on the first half of the fit documents (is the cell data-limited?).
Writes the numbers only; c50_t2_tstitch_verdict.py applies the rule.

    python -I benchmarks/chatbot50/c50_t2_tstitch.py --inputs DIR --v0 DUMP --f0 DUMP --f0-gguf G --posthoc R --json OUT
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_t2_moe as MOE  # noqa: E402
import c50_t2_stitch as S  # noqa: E402

SEED = 20261026
EPOCHS, BATCH = 16, 512
LR_LIN, LR_MLP, WD_MLP = 3e-4, 3e-4, 1e-2
ALPHA, HIDDEN = 0.1, 2048
PCA_DIM, PCA_START = 3584, 1024
EVAL_CHUNK = 2048
NOISE_R2 = (0.3, 0.5)
FIT_EVAL_ROWS = 4096


def pairs_of(lv: list[int]) -> dict[int, int]:
    return {l: round(l * 31 / 19) for l in lv}


class Block:
    """F0's routed-expert block on the GPU: c50_t2_moe.Layer's routing in torch, experts in bf16."""

    def __init__(self, L: MOE.Layer, dev: torch.device):
        m = L.m
        self.k, self.ng, self.ngu, self.norm_w, self.scale, self.E = m.k, m.n_group, m.n_group_used, m.norm_w, m.scale, m.n_expert
        self.router = torch.from_numpy(L.router.copy()).to(dev)
        self.bias = torch.from_numpy(L.bias).to(dev)
        g, u, dn = L.expert(0)
        bf = torch.bfloat16
        self.g = torch.empty((self.E, *g.shape), dtype=bf, device=dev)
        self.u = torch.empty((self.E, *u.shape), dtype=bf, device=dev)
        self.dn = torch.empty((self.E, *dn.shape), dtype=bf, device=dev)
        for e in range(self.E):
            g, u, dn = L.expert(e)
            self.g[e], self.u[e], self.dn[e] = (torch.from_numpy(a).to(dev, bf) for a in (g, u, dn))

    def route(self, x: torch.Tensor):
        logits = x @ self.router.T
        probs = torch.sigmoid(logits)
        sel = (probs + self.bias).detach()
        n = len(x)
        if self.ng > 1:
            gs = sel.view(n, self.ng, -1)
            keep = gs.topk(2, dim=2).values.sum(2).topk(self.ngu, dim=1).indices
            mask = torch.full((n, self.ng), float("-inf"), device=x.device).scatter(1, keep, 0.0)
            sel = (gs + mask[:, :, None]).view(n, -1)
        top = sel.topk(self.k, dim=1).indices
        w = probs.gather(1, top)
        if self.norm_w:
            w = w / w.sum(1, keepdim=True).clamp_min(6.103515625e-5)
        if self.scale not in (0.0, 1.0):
            w = w * self.scale
        return logits, top, w

    def experts(self, x: torch.Tensor, top: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        xb = x.to(torch.bfloat16)
        flat = top.reshape(-1)
        order = torch.argsort(flat)
        rows_all, slot_all = order // self.k, order % self.k
        off = 0
        for e, c in enumerate(torch.bincount(flat, minlength=self.E).tolist()):
            if c == 0:
                continue
            r, s = rows_all[off:off + c], slot_all[off:off + c]
            off += c
            xe = xb[r]
            h = (torch.nn.functional.silu((xe @ self.g[e].T).float()) * (xe @ self.u[e].T).float()).to(torch.bfloat16)
            out.index_add_(0, r, (h @ self.dn[e].T).float() * w[r, s][:, None])
        return out

    def __call__(self, x: torch.Tensor):
        logits, top, w = self.route(x)
        return logits, top, self.experts(x, top, w)


class Map(torch.nn.Module):
    def __init__(self, w: torch.Tensor, b: torch.Tensor, mx: torch.Tensor, sx: torch.Tensor, mlp: bool):
        super().__init__()
        self.lin = torch.nn.Linear(w.shape[0], w.shape[1])
        with torch.no_grad():
            self.lin.weight.copy_(w.T)
            self.lin.bias.copy_(b)
        self.register_buffer("mx", mx)
        self.register_buffer("sx", sx)
        self.mlp = None
        if mlp:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(w.shape[0], HIDDEN), torch.nn.GELU(), torch.nn.Linear(HIDDEN, w.shape[1]))
            torch.nn.init.zeros_(self.mlp[2].weight)
            torch.nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        return y if self.mlp is None else y + self.mlp((x - self.mx) / self.sx)


def overlap_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, :, None] == b[:, None, :]).any(2).float().mean(1)


def cos_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row cosine in float64: in float32 a diverged map's huge outputs gave cosines far above 1."""
    a, b = a.double(), b.double()
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-300)).float()


class Own:
    """F0's block on F0's own states of one split: targets of the training and references of the measures."""

    def __init__(self, blk: Block, y: torch.Tensor):
        self.y = y
        lg, tp, out = [], [], []
        with torch.no_grad():
            for i in range(0, len(y), EVAL_CHUNK):
                a, b, c = blk(y[i:i + EVAL_CHUNK])
                lg.append(a), tp.append(b), out.append(c)
        self.logits, self.top, self.out = torch.cat(lg), torch.cat(tp), torch.cat(out)


def measure(blk: Block, mapped_fn, x: torch.Tensor, own: Own, idx: torch.Tensor | None = None) -> dict:
    ov, cs, se = [], [], []
    rows = torch.arange(len(x), device=x.device) if idx is None else idx
    with torch.no_grad():
        for i in range(0, len(rows), EVAL_CHUNK):
            r = rows[i:i + EVAL_CHUNK]
            m = mapped_fn(x[r])
            _, top, out = blk(m)
            ov.append(overlap_t(top, own.top[r]))
            cs.append(cos_t(out, own.out[r]))
            se.append(((m - own.y[r]) ** 2).sum())
    yv = own.y[rows]
    return {"topk_overlap": float(torch.cat(ov).mean()), "moe_cos_mean": float(torch.cat(cs).mean()),
            "moe_cos_median": float(torch.cat(cs).median()), "r2": float(1 - sum(se) / ((yv - yv.mean(0)) ** 2).sum())}


def ridge_init(xf: torch.Tensor, yf: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor):
    R = S.Ridge(xf)
    uty = R.u.T @ (yf - yf.mean(0))
    sel = [S.r2(ys, R.apply(xs, *R.weights(yf, lam, uty))) for lam in S.LAMBDAS]
    li = int(np.argmax(sel))
    w, my = R.weights(yf, S.LAMBDAS[li], uty)
    return w, my - R.mx @ w, R.mx, S.LAMBDAS[li], sel


def train(blk: Block, model: Map, xf: torch.Tensor, own_f: Own, xs: torch.Tensor, own_s: Own, gen: torch.Generator,
          lr_lin: float, log: str) -> tuple[Map, dict]:
    t0 = time.time()
    lg_var = float(own_f.logits.var())
    groups = [{"params": model.lin.parameters(), "lr": lr_lin, "weight_decay": 0.0}]
    if model.mlp is not None:
        groups.append({"params": model.mlp.parameters(), "lr": LR_MLP, "weight_decay": WD_MLP})
    opt = torch.optim.AdamW(groups)
    hist = [measure(blk, model, xs, own_s)["moe_cos_mean"]]
    best, best_ep = copy.deepcopy(model.state_dict()), 0
    losses, diverged = [], None
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(len(xf), generator=gen, device="cpu").to(xf.device)
        tot = 0.0
        for i in range(0, len(xf) - BATCH + 1, BATCH):
            r = perm[i:i + BATCH]
            m = model(xf[r])
            logits, _, out = blk(m)
            loss = (1 - cos_t(out, own_f.out[r])).mean() + ALPHA * ((logits - own_f.logits[r]) ** 2).mean() / lg_var
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(r)
        losses.append(tot / (len(xf) // BATCH * BATCH))
        hist.append(measure(blk, model, xs, own_s)["moe_cos_mean"])
        if not (np.isfinite(losses[-1]) and -1.0 <= hist[-1] <= 1.0 + 1e-4):
            diverged = ep
            break
        if hist[-1] > hist[best_ep]:
            best, best_ep = copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best)
    print(f"    {log}: {time.time() - t0:.0f}s, select cos {hist[0]:.3f} -> best {hist[best_ep]:.3f} at epoch {best_ep}; train loss {losses[0]:.3f} -> {losses[-1]:.3f}"
          + (f"; DIVERGED at epoch {diverged}" if diverged else ""), flush=True)
    return model, {"select_cos_by_epoch": hist, "train_loss_by_epoch": losses, "best_epoch": best_ep, "diverged_at_epoch": diverged}


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--inputs", "--v0", "--f0", "--f0-gguf", "--json"):
        ap.add_argument(k, required=True)
    ap.add_argument("--posthoc", help="the linear cell's post-hoc record (depth-matched ridge numbers); required unless --smoke")
    ap.add_argument("--layers", help="comma list of v0 layers (default all MoE layers)")
    ap.add_argument("--positive-only", action="store_true", help="the planted positive alone (calibration of the control)")
    ap.add_argument("--smoke", action="store_true", help="the 4 dev documents split 2/1/1: checks the code, numbers mean nothing")
    a = ap.parse_args()
    global BATCH
    if a.smoke:
        BATCH = 64
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    man = json.loads((Path(a.inputs) / "manifest.json").read_text(encoding="utf-8"))
    v0d, f0d = Path(a.v0), Path(a.f0)
    rv, rf = (json.loads((d / "run.json").read_text(encoding="utf-8")) for d in (v0d, f0d))
    which = "dev" if a.smoke else "main"
    ids_sha = S.sha256(Path(a.inputs) / f"{which}.ids")
    assert ids_sha == man[f"{which}_ids_sha256"], f"{which}.ids differs from its manifest"
    n_docs = len(man["dev"]) if a.smoke else len(man["main_order"])
    assert rv["ids_fnv1a64"] == rf["ids_fnv1a64"] and rv["positions"] == rf["positions"] == n_docs * S.LEN
    nf, ns, nt = (2, 1, 1) if a.smoke else (len(man["split"][k]) for k in ("fit", "select", "test"))
    fit, sel, test = S.rows_of(range(nf)), S.rows_of(range(nf, nf + ns)), S.rows_of(range(nf + ns, nf + ns + nt))
    half = S.rows_of(range(nf // 2))
    mf = MOE.Model(a.f0_gguf)
    lv = [int(x) for x in a.layers.split(",")] if a.layers else list(range(1, 20))
    pairs = pairs_of(lv)
    post = None
    if not a.smoke and not a.positive_only:
        post = json.loads(Path(a.posthoc).read_text(encoding="utf-8"))["layers"]
    config = {"f0_gguf": a.f0_gguf, "v0_dump": str(v0d), "f0_dump": str(f0d), "ids_sha256": ids_sha,
              "rows": {"fit": len(fit), "select": len(sel), "test": len(test), "half_fit": len(half)}, "pairs": pairs,
              "epochs": EPOCHS, "batch": BATCH, "lr_lin": LR_LIN, "lr_mlp": LR_MLP, "wd_mlp": WD_MLP, "alpha": ALPHA,
              "hidden": HIDDEN, "pca_dim": PCA_DIM, "pca_start": PCA_START, "noise_r2": NOISE_R2, "lambdas": S.LAMBDAS, "seed": SEED,
              "expert_dtype": "bfloat16", "smoke": a.smoke, "positive_only": a.positive_only}
    print("CONFIG", json.dumps(config), flush=True)
    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)
    out_layers, per_j = {}, {}
    for l in lv:
        j = pairs[l]
        t0 = time.time()
        blk = Block(mf.layer(j), dev)
        ld = lambda d, name, run, idx: torch.from_numpy(S.load(d, name, run, idx)).to(dev)
        yf, ys, yt = (ld(f0d, f"ffn_norm-{j}", rf, r) for r in (fit, sel, test))
        own_f, own_s, own_t = Own(blk, yf), Own(blk, ys), Own(blk, yt)
        t_load = time.time() - t0
        # tool: the bf16 block on F0's own test states vs llama.cpp
        ref_top = torch.from_numpy(S.load(f0d, f"ffn_moe_topk-{j}", rf, test)).to(dev).long()
        ref_out = torch.from_numpy(S.load(f0d, f"ffn_moe_out-{j}", rf, test)).to(dev)
        c = cos_t(own_t.out, ref_out)
        tool = {"topk_overlap": float(overlap_t(own_t.top, ref_top).mean()), "cos_median": float(c.median()),
                "cos_p01": float(torch.quantile(c, 0.01))}
        del ref_top, ref_out
        # noise bars on the test rows
        bars = {}
        var = yt.var(0)
        for r2v in NOISE_R2:
            noise = torch.from_numpy(rng.standard_normal(tuple(yt.shape)).astype(np.float32)).to(dev) * torch.sqrt((1 - r2v) * var)
            m = measure(blk, lambda x: x, yt + noise, own_t)
            bars[f"noise_r2_{r2v}"] = {k: m[k] for k in ("topk_overlap", "moe_cos_mean")}
            del noise
        print(f"v0 {l} -> F0 {j}: load {t_load:.0f}s  tool {tool}  bars {bars}", flush=True)
        # planted positive: F0's own state on PCA_DIM components, map started at the PCA_START reconstruction
        mu = yf.mean(0)
        _, _, vh = torch.linalg.svd(yf - mu, full_matrices=False)
        V = vh[:min(PCA_DIM, vh.shape[0])].T.contiguous()
        zf, zs, zt = ((y - mu) @ V for y in (yf, ys, yt))
        W0 = V.T.clone()
        W0[min(PCA_START, V.shape[1] // 2):] = 0
        pmodel = Map(W0, mu.clone(), zf.mean(0), zf.std(0) + 1e-6, False).to(dev)
        p_start = measure(blk, pmodel, zt, own_t)
        pmodel, ptr = train(blk, pmodel, zf, own_f, zs, own_s, gen, LR_LIN, f"positive (PCA-{V.shape[1]} from PCA-{PCA_START})")
        positive = {**measure(blk, pmodel, zt, own_t), **ptr, "start": p_start,
                    "pca_r2_test": float(1 - ((zt @ V.T + mu - yt) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())}
        del pmodel, zf, zs, zt, V, vh, W0
        rec_l = {"f0_layer": j, "tool": tool, "bars": bars, "positive": positive}
        if a.positive_only:
            out_layers[l] = rec_l
            print(f"  positive: {positive['topk_overlap']:.3f} / cos {positive['moe_cos_mean']:.3f}  bar 0.5 {bars['noise_r2_0.5']}  {time.time() - t0:.0f}s", flush=True)
            del blk, yf, ys, yt, own_f, own_s, own_t
            torch.cuda.empty_cache()
            continue
        xf, xs, xt = (ld(v0d, f"ffn_norm-{l}", rv, r) for r in (fit, sel, test))
        sx = xf.std(0) + 1e-6
        w, b, mx, lam, sel_r2 = ridge_init(xf, yf, xs, ys)
        ridge = Map(w, b, mx, sx, False).to(dev)
        arms = {"ridge": {"lambda": lam, "select_r2_grid": sel_r2, "select": measure(blk, ridge, xs, own_s),
                          "test": measure(blk, ridge, xt, own_t)}}
        fit_idx = torch.from_numpy(rng.choice(len(fit), min(FIT_EVAL_ROWS, len(fit)), replace=False)).to(dev)
        for name, mlp in (("linear", False), ("mlp", True)):
            mdl, tr = train(blk, Map(w, b, mx, sx, mlp).to(dev), xf, own_f, xs, own_s, gen, LR_LIN, name)
            arms[name] = {**tr, "select": measure(blk, mdl, xs, own_s), "test": measure(blk, mdl, xt, own_t),
                          "fit": measure(blk, mdl, xf, own_f, fit_idx)}
            del mdl
        # shuffled pairs: the fit targets permuted across positions; ridge and training on the permuted pairs
        perm = torch.from_numpy(rng.permutation(len(fit))).to(dev)
        own_fp = copy.copy(own_f)
        own_fp.y, own_fp.logits, own_fp.top, own_fp.out = own_f.y[perm], own_f.logits[perm], own_f.top[perm], own_f.out[perm]
        ws, bs, _, lams, _ = ridge_init(xf, yf[perm], xs, ys)
        for name, mlp in (("shuffled_linear", False), ("shuffled_mlp", True)):
            mdl, tr = train(blk, Map(ws, bs, mx, sx, mlp).to(dev), xf, own_fp, xs, own_s, gen, LR_LIN, name)
            arms[name] = {**tr, "lambda": lams, "select": measure(blk, mdl, xs, own_s), "test": measure(blk, mdl, xt, own_t)}
            del mdl
        del own_fp
        # diagnostic: the linear arm on half the fit documents
        hn = len(half)
        own_h = copy.copy(own_f)
        own_h.y, own_h.logits, own_h.top, own_h.out = own_f.y[:hn], own_f.logits[:hn], own_f.top[:hn], own_f.out[:hn]
        wh, bh, mxh, lamh, _ = ridge_init(xf[:hn], yf[:hn], xs, ys)
        mdl, tr = train(blk, Map(wh, bh, mxh, sx, False).to(dev), xf[:hn], own_h, xs, own_s, gen, LR_LIN, "half_linear")
        arms["half_linear"] = {**tr, "lambda": lamh, "select": measure(blk, mdl, xs, own_s), "test": measure(blk, mdl, xt, own_t)}
        del mdl, own_h
        chosen = max(("linear", "mlp"), key=lambda k: arms[k]["select"]["moe_cos_mean"])
        chosen_shuf = max(("shuffled_linear", "shuffled_mlp"), key=lambda k: arms[k]["select"]["moe_cos_mean"])
        rec_l.update({"arms": arms, "chosen": chosen, "chosen_shuffled": chosen_shuf, "wall_s": time.time() - t0})
        if post is not None:
            rec_l["posthoc"] = {k: post[str(l)][k] for k in ("f0_layer", "lambda", "test_r2", "topk_overlap", "moe_cos_mean")}
        out_layers[l] = rec_l
        A = arms
        print(f"  v0 {l} -> F0 {j}: ridge {A['ridge']['test']['topk_overlap']:.3f}/{A['ridge']['test']['moe_cos_mean']:.3f}"
              f"  linear {A['linear']['test']['topk_overlap']:.3f}/{A['linear']['test']['moe_cos_mean']:.3f}"
              f"  mlp {A['mlp']['test']['topk_overlap']:.3f}/{A['mlp']['test']['moe_cos_mean']:.3f}  chosen {chosen}"
              f"  shuffled {A[chosen_shuf]['test']['topk_overlap']:.3f}/{A[chosen_shuf]['test']['moe_cos_mean']:.3f}"
              f"  positive {positive['topk_overlap']:.3f}/{positive['moe_cos_mean']:.3f}"
              f"  bar {bars['noise_r2_0.5']['topk_overlap']:.3f}/{bars['noise_r2_0.5']['moe_cos_mean']:.3f}  {time.time() - t0:.0f}s", flush=True)
        del blk, yf, ys, yt, xf, xs, xt, own_f, own_s, own_t
        torch.cuda.empty_cache()
    rec = {"cell": "C50-T2 trained stitch: maps trained on F0's routed-expert output" + (" (SMOKE: not evidence)" if a.smoke else "")
           + (" (planted positive only: calibration)" if a.positive_only else ""),
           "config": {**config, "pairs": {str(k): v for k, v in pairs.items()},
                      "inputs_manifest_sha256": S.sha256(Path(a.inputs) / "manifest.json"),
                      "f0_gguf_sha256": S.sha256(Path(a.f0_gguf)), "v0_run": {k: v for k, v in rv.items() if k != "tensors"},
                      "f0_run": {k: v for k, v in rf.items() if k != "tensors"}, "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0)},
           "layers": {str(l): out_layers[l] for l in lv}, "wall_s": time.time() - t_start}
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DONE", a.json, f"{time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
