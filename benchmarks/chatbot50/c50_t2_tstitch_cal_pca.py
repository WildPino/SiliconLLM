"""Calibration of the trained-stitch controls on F0's own states only (no v0): PCA-k ceilings and a rotated positive."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "D:/_THINGS/Progetti/SiliconLLM/benchmarks/chatbot50")
import c50_t2_moe as MOE
import c50_t2_stitch as S
import c50_t2_tstitch as T

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
f0d = Path("C:/SiliconLLM_stage/t2/main/f0r")
rf = json.loads((f0d / "run.json").read_text(encoding="utf-8"))
mf = MOE.Model("C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf")
fit, sel, test = S.rows_of(range(64)), S.rows_of(range(64, 96)), S.rows_of(range(96, 128))
js = [int(x) for x in sys.argv[1].split(",")]
lrs = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else []
out = {}
for j in js:
    blk = T.Block(mf.layer(j), dev)
    yf, ys, yt = (torch.from_numpy(S.load(f0d, f"ffn_norm-{j}", rf, r)).to(dev) for r in (fit, sel, test))
    own_f, own_s, own_t = T.Own(blk, yf), T.Own(blk, ys), T.Own(blk, yt)
    mu = yf.mean(0)
    _, sv, vh = torch.linalg.svd(yf - mu, full_matrices=False)
    res = {"pca": {}}
    for k in ((1024, 2048, 3072, 3584) if "--pca" in sys.argv else ()):
        V = vh[:k].T
        m = T.measure(blk, lambda x: (x - mu) @ V @ V.T + mu, yt, own_t)
        res["pca"][k] = m
        print(f"F0 {j}: PCA-{k} reconstruction: R2 {m['r2']:.3f} overlap {m['topk_overlap']:.3f} cos {m['moe_cos_mean']:.3f}", flush=True)
    g = torch.Generator(device="cpu").manual_seed(1)
    Q, _ = torch.linalg.qr(torch.randn(4096, 4096, generator=g))
    Q = Q.to(dev)
    zf, zs, zt = yf @ Q, ys @ Q, yt @ Q
    for lr in lrs:
        gen = torch.Generator().manual_seed(T.SEED)
        mdl = T.Map(torch.zeros(4096, 4096, device=dev), mu.clone(), zf.mean(0), zf.std(0) + 1e-6, False).to(dev)
        mdl, tr = T.train(blk, mdl, zf, own_f, zs, own_s, gen, lr, f"rotated positive lr {lr}")
        m = T.measure(blk, mdl, zt, own_t)
        res[f"rot_lr_{lr}"] = {**m, **tr}
        print(f"F0 {j}: rotated positive lr {lr}: R2 {m['r2']:.3f} overlap {m['topk_overlap']:.3f} cos {m['moe_cos_mean']:.3f}", flush=True)
        del mdl
    out[j] = res
    del blk, yf, ys, yt, own_f, own_s, own_t, Q, zf, zs, zt
    torch.cuda.empty_cache()
Path(sys.argv[3] if len(sys.argv) > 3 else "C:/SiliconLLM_stage/t2/dev/tstitch_cal2.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
