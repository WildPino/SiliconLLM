"""C50-T2 stitching, post-hoc (not registered): the same measures at matched depth, j = round(l * 31 / 19).

The registered pairing took, for each v0 layer l, the F0 layer of highest select R^2; it put 17 of 19 layers on
F0 layer 2. This reads each l at its depth-matched F0 layer instead, at that pair's best select lambda, with the
same ridge, the same test rows, the same F0 block and the same noise-0.5 bar. It decides nothing.

    python -I benchmarks/chatbot50/c50_t2_stitch_posthoc.py --inputs DIR --v0 DUMP --f0 DUMP --f0-gguf G --record R --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_t2_moe as MOE  # noqa: E402
import c50_t2_stitch as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--inputs", "--v0", "--f0", "--f0-gguf", "--record", "--json"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    rec = json.loads(Path(a.record).read_text(encoding="utf-8"))
    man = json.loads((Path(a.inputs) / "manifest.json").read_text(encoding="utf-8"))
    v0d, f0d = Path(a.v0), Path(a.f0)
    rv, rf = (json.loads((d / "run.json").read_text(encoding="utf-8")) for d in (v0d, f0d))
    nf, ns, nt = (len(man["split"][k]) for k in ("fit", "select", "test"))
    fit, test = S.rows_of(range(nf)), S.rows_of(range(nf + ns, nf + ns + nt))
    mf = MOE.Model(a.f0_gguf)
    grid = {tuple(map(int, k.split(","))): v for k, v in rec["pairing_select_r2"].items()}
    pairs = {l: round(l * 31 / 19) for l in range(1, 20)}
    rng = np.random.default_rng(S.SEED + 1)
    out = {}
    for j in sorted(set(pairs.values())):
        t0 = time.time()
        yall = torch.from_numpy(S.load(f0d, f"ffn_norm-{j}", rf, np.concatenate([fit, test]))).to(dev)
        yf, yt = yall[:len(fit)], yall[len(fit):]
        own = yt.cpu().numpy()
        blocks, ls = [own], []
        for l in [l for l in pairs if pairs[l] == j]:
            li = int(np.argmax(grid[(l, j)]))
            R = S.Ridge(torch.from_numpy(S.load(v0d, f"ffn_norm-{l}", rv, fit)).to(dev))
            xt = torch.from_numpy(S.load(v0d, f"ffn_norm-{l}", rv, test)).to(dev)
            mapped = R.apply(xt, *R.weights(yf, S.LAMBDAS[li]))
            out[l] = {"f0_layer": j, "lambda": S.LAMBDAS[li], "select_r2": grid[(l, j)][li], "test_r2": S.r2(yt, mapped)}
            blocks.append(mapped.cpu().numpy())
            ls.append(l)
            del R, xt, mapped
        sub = rng.choice(len(test), S.CAL_ROWS, replace=False)
        var = own.var(0)
        blocks.append(own[sub] + rng.standard_normal((S.CAL_ROWS, own.shape[1])).astype(np.float32) * np.sqrt(0.5 * var))
        del yall, yf, yt
        torch.cuda.empty_cache()
        L = mf.layer(j)
        xcat = np.concatenate(blocks)
        top, w = L.route(xcat)
        y = L.forward(xcat, top, w, device=dev)
        n = len(own)
        own_top, own_y = top[:n], y[:n]
        for k, l in enumerate(ls):
            sl = slice(n * (k + 1), n * (k + 2))
            out[l]["topk_overlap"] = float(S.overlap(top[sl], own_top).mean())
            out[l]["moe_cos_mean"] = float(S.cosine(y[sl], own_y).mean())
        nz = slice(n * (len(ls) + 1), None)
        bar = {"topk_overlap": float(S.overlap(top[nz], own_top[sub]).mean()),
               "moe_cos_mean": float(S.cosine(y[nz], own_y[sub]).mean())}
        for l in ls:
            out[l]["noise_0.5"] = bar
            out[l]["stitches"] = out[l]["topk_overlap"] >= bar["topk_overlap"] and out[l]["moe_cos_mean"] >= bar["moe_cos_mean"]
            o = out[l]
            print(f"v0 {l} -> f0 {j}: R2 test {o['test_r2']:.3f}  overlap {o['topk_overlap']:.3f} (bar {bar['topk_overlap']:.3f})"
                  f"  cos {o['moe_cos_mean']:.3f} (bar {bar['moe_cos_mean']:.3f})  {time.time() - t0:.0f}s", flush=True)
    res = {"cell": "C50-T2 stitching post-hoc (not registered): depth-matched pairs j = round(l*31/19)",
           "record": a.record, "layers": {str(l): out[l] for l in sorted(out)},
           "stitching_layers": sum(o["stitches"] for o in out.values()), "wall_s": time.time() - t_start}
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("DONE", res["stitching_layers"], "stitch;", f"{res['wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
