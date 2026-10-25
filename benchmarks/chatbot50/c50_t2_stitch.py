"""C50-T2 stitching cell: can v0's MoE input be mapped linearly onto F0's, well enough for F0's experts to read it?

From the two dumps (c50_llama_dump, same documents and tokens), positions SKIP..LEN-1 of each document:
  1. pairing: for every v0 MoE layer l and F0 MoE layer j, a ridge map (centred, lambda on a grid) fitted on the
     fit documents and scored by R^2 on the select documents; each l takes the (j, lambda) of highest select R^2;
  2. on the test documents, for each l at its (j, lambda): R^2, and F0's own routed-expert block (c50_t2_moe) read
     on the mapped state against F0's own state: top-k overlap of the selected experts, and the cosine of the
     routed output;
  3. controls, through the same code:
     - numpy block vs llama.cpp's own ffn_moe_out and ffn_moe_topk on F0's states, every chosen j (the tool);
     - F0 -> F0: the same ridge pipeline from F0 layer j onto itself (the map exists: must read ~1);
     - shuffled pairs: the map fitted with the fit rows of F0 permuted across positions (no map exists);
     - noise calibration (reported): F0's own state plus Gaussian noise of known R^2.
Writes the numbers only; c50_t2_stitch_verdict.py applies the rule.

    python -I benchmarks/chatbot50/c50_t2_stitch.py --inputs DIR --v0 DUMP --f0 DUMP --v0-gguf G --f0-gguf G --json OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_t2_moe as MOE  # noqa: E402

LEN, SKIP = 288, 32
SEED = 20261026
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # times the mean squared singular value of the fit design
NOISE_R2 = (0.3, 0.5, 0.7, 0.9)
CAL_ROWS = 2048


def rows_of(docs: range) -> np.ndarray:
    return np.concatenate([np.arange(k * LEN + SKIP, (k + 1) * LEN) for k in docs])


def load(dump: Path, name: str, run: dict, idx: np.ndarray | None = None) -> np.ndarray:
    meta = run["tensors"][name]
    dt = np.float32 if meta["dtype"] == "f32" else np.int32
    a = np.memmap(dump / f"{name}.bin", dtype=dt, mode="r", shape=(meta["rows"], meta["width"]))
    return np.array(a if idx is None else a[idx])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def r2(y: torch.Tensor, p: torch.Tensor) -> float:
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


class Ridge:
    """Centred ridge from X (fit rows) to any Y, through the SVD of X; lambda relative to mean(s^2)."""

    def __init__(self, x: torch.Tensor):
        self.mx = x.mean(0)
        u, s, vh = torch.linalg.svd(x - self.mx, full_matrices=False)
        self.u, self.s, self.v = u, s, vh.T
        self.scale = float((s ** 2).mean())

    def weights(self, y: torch.Tensor, lam: float, uty: torch.Tensor | None = None):
        my = y.mean(0)
        if uty is None:
            uty = self.u.T @ (y - my)
        f = self.s / (self.s ** 2 + lam * self.scale)
        return self.v @ (f[:, None] * uty), my

    def apply(self, x: torch.Tensor, w: torch.Tensor, my: torch.Tensor) -> torch.Tensor:
        return (x - self.mx) @ w + my


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)]) / a.shape[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--inputs", "--v0", "--f0", "--v0-gguf", "--f0-gguf", "--json"):
        ap.add_argument(k, required=True)
    ap.add_argument("--smoke", action="store_true", help="the 4 dev documents split 2/1/1: checks the code, numbers mean nothing")
    a = ap.parse_args()
    global CAL_ROWS
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = json.loads((Path(a.inputs) / "manifest.json").read_text(encoding="utf-8"))
    v0d, f0d = Path(a.v0), Path(a.f0)
    rv, rf = (json.loads((d / "run.json").read_text(encoding="utf-8")) for d in (v0d, f0d))
    which = "dev" if a.smoke else "main"
    ids_sha = sha256(Path(a.inputs) / f"{which}.ids")
    assert ids_sha == man[f"{which}_ids_sha256"], f"{which}.ids differs from its manifest"
    n_docs = len(man["dev"]) if a.smoke else len(man["main_order"])
    assert rv["ids_fnv1a64"] == rf["ids_fnv1a64"] and rv["positions"] == rf["positions"] == n_docs * LEN
    nf, ns, nt = (2, 1, 1) if a.smoke else (len(man["split"][k]) for k in ("fit", "select", "test"))
    if a.smoke:
        CAL_ROWS = 128
    fit, sel, test = rows_of(range(nf)), rows_of(range(nf, nf + ns)), rows_of(range(nf + ns, nf + ns + nt))
    mv, mf = MOE.Model(a.v0_gguf), MOE.Model(a.f0_gguf)
    Lv = list(range(mv.dense_lead, mv.n_layer))
    Lf = list(range(mf.dense_lead, mf.n_layer))
    print("CONFIG", json.dumps({"v0_gguf": a.v0_gguf, "f0_gguf": a.f0_gguf, "v0_dump": str(v0d), "f0_dump": str(f0d),
                                "main_ids_sha256": ids_sha, "rows": {"fit": len(fit), "select": len(sel), "test": len(test)},
                                "len": LEN, "skip": SKIP, "lambdas": LAMBDAS, "v0_layers": [Lv[0], Lv[-1]],
                                "f0_layers": [Lf[0], Lf[-1]], "device": str(dev), "seed": SEED, "smoke": a.smoke}), flush=True)

    # 1. pairing on the select documents
    ridge, xs = {}, {}
    for l in Lv:
        x = torch.from_numpy(load(v0d, f"ffn_norm-{l}", rv, np.concatenate([fit, sel]))).to(dev)
        ridge[l] = Ridge(x[:len(fit)])
        xs[l] = x[len(fit):]
        del x
    grid = {}
    for j in Lf:
        y = torch.from_numpy(load(f0d, f"ffn_norm-{j}", rf, np.concatenate([fit, sel]))).to(dev)
        yf, ys = y[:len(fit)], y[len(fit):]
        for l in Lv:
            R = ridge[l]
            uty = R.u.T @ (yf - yf.mean(0))
            grid[(l, j)] = [r2(ys, R.apply(xs[l], *R.weights(yf, lam, uty))) for lam in LAMBDAS]
        del y, yf, ys
        print(f"PAIR f0 layer {j}: best v0 layer {max(Lv, key=lambda l: max(grid[(l, j)]))}", flush=True)
    choice = {}
    for l in Lv:
        j, li = max(((j, i) for j in Lf for i in range(len(LAMBDAS))), key=lambda t: grid[(l, t[0])][t[1]])
        choice[l] = {"f0_layer": j, "lambda": LAMBDAS[li], "select_r2": grid[(l, j)][li]}

    # 2-3. test documents, grouped by the chosen F0 layer so that each layer's experts are read once
    rng = np.random.default_rng(SEED)
    out_layers, tool, cal = {}, {}, {}
    for j in sorted({c["f0_layer"] for c in choice.values()}):
        t0 = time.time()
        yall = torch.from_numpy(load(f0d, f"ffn_norm-{j}", rf, np.concatenate([fit, test]))).to(dev)
        yf, yt = yall[:len(fit)], yall[len(fit):]
        perm = torch.from_numpy(rng.permutation(len(fit))).to(dev)
        own = yt.cpu().numpy()
        blocks, tags = [own], [("own", None)]
        for l in [l for l in Lv if choice[l]["f0_layer"] == j]:
            R, lam = ridge[l], choice[l]["lambda"]
            xt = torch.from_numpy(load(v0d, f"ffn_norm-{l}", rv, test)).to(dev)
            mapped = R.apply(xt, *R.weights(yf, lam))
            shuf = R.apply(xt, *R.weights(yf[perm], lam))
            out_layers[l] = {**choice[l], "test_r2": r2(yt, mapped), "shuffled_test_r2": r2(yt, shuf)}
            blocks += [mapped.cpu().numpy(), shuf.cpu().numpy()]
            tags += [("mapped", l), ("shuffled", l)]
            del xt, mapped, shuf
        # F0 -> F0 through the same pipeline, and the noise calibration, on a subset of rows
        Rs = Ridge(yf)
        self_map = Rs.apply(yt, *Rs.weights(yf, LAMBDAS[0]))
        self_r2 = r2(yt, self_map)
        sub = rng.choice(len(test), CAL_ROWS, replace=False)
        blocks.append(self_map.cpu().numpy()[sub])
        tags.append(("self", None))
        var = own.var(0)
        for r in NOISE_R2:
            blocks.append(own[sub] + rng.standard_normal((CAL_ROWS, own.shape[1])).astype(np.float32) * np.sqrt((1 - r) * var))
            tags.append(("noise", r))
        del yall, yf, yt, Rs, self_map
        torch.cuda.empty_cache()

        L = mf.layer(j)
        xcat = np.concatenate(blocks)
        top, w = L.route(xcat)
        y = L.forward(xcat, top, w, device=dev if dev.type == "cuda" else None)
        off, parts = 0, {}
        for blk, tag in zip(blocks, tags):
            parts[tag] = (top[off:off + len(blk)], y[off:off + len(blk)])
            off += len(blk)
        own_top, own_y = parts[("own", None)]
        ref_y = load(f0d, f"ffn_moe_out-{j}", rf, test)
        ref_top = load(f0d, f"ffn_moe_topk-{j}", rf, test)
        cos_tool = cosine(own_y, ref_y)
        tool[j] = {"topk_overlap": float(overlap(own_top, ref_top).mean()), "cos_median": float(np.median(cos_tool)),
                   "cos_p01": float(np.percentile(cos_tool, 1))}
        for (kind, l), (tp, yy) in parts.items():
            if kind in ("mapped", "shuffled"):
                out_layers[l][f"{kind}_topk_overlap"] = float(overlap(tp, own_top).mean())
                out_layers[l][f"{kind}_moe_cos_mean"] = float(cosine(yy, own_y).mean())
                out_layers[l][f"{kind}_moe_cos_median"] = float(np.median(cosine(yy, own_y)))
        cal[j] = {"self_r2": self_r2}
        for (kind, r), (tp, yy) in parts.items():
            if kind in ("self", "noise"):
                cal[j][kind if r is None else f"noise_r2_{r}"] = {
                    "topk_overlap": float(overlap(tp, own_top[sub]).mean()),
                    "moe_cos_mean": float(cosine(yy, own_y[sub]).mean())}
        print(f"F0 LAYER {j}: tool {tool[j]}  {time.time() - t0:.0f}s", flush=True)
        for l in sorted(l for l in out_layers if out_layers[l]["f0_layer"] == j):
            o = out_layers[l]
            print(f"  v0 {l} -> f0 {j}: R2 select {o['select_r2']:.3f} test {o['test_r2']:.3f}  overlap {o['mapped_topk_overlap']:.3f}"
                  f" (shuffled {o['shuffled_topk_overlap']:.3f})  cos {o['mapped_moe_cos_mean']:.3f} (shuffled {o['shuffled_moe_cos_mean']:.3f})", flush=True)

    rec = {"cell": "C50-T2 stitching: v0's MoE input mapped onto F0's, read by F0's experts" + (" (SMOKE: not evidence)" if a.smoke else ""),
           "config": {"len": LEN, "skip": SKIP, "lambdas": LAMBDAS, "noise_r2": NOISE_R2, "cal_rows": CAL_ROWS,
                      "seed": SEED, "rows": {"fit": len(fit), "select": len(sel), "test": len(test)},
                      "main_ids_sha256": ids_sha, "inputs_manifest_sha256": sha256(Path(a.inputs) / "manifest.json"),
                      "v0_gguf_sha256": sha256(Path(a.v0_gguf)), "f0_gguf_sha256": sha256(Path(a.f0_gguf)),
                      "v0_run": rv, "f0_run": {k: v for k, v in rf.items() if k != "tensors"}},
           "pairing_select_r2": {f"{l},{j}": grid[(l, j)] for (l, j) in grid},
           "layers": {str(l): out_layers[l] for l in Lv}, "tool": {str(j): v for j, v in tool.items()},
           "calibration": {str(j): v for j, v in cal.items()}, "wall_s": time.time() - t_start}
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DONE", a.json, f"{time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
