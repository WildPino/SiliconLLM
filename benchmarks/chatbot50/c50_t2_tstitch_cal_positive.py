"""Calibration (F0 only): positive P4 = PCA-K input, map started at the PCA-K0 reconstruction (rows past K0 zero)."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, "D:/_THINGS/Progetti/SiliconLLM/benchmarks/chatbot50")
import c50_t2_moe as MOE, c50_t2_stitch as S, c50_t2_tstitch as T
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
f0d = Path("C:/SiliconLLM_stage/t2/main/f0r")
rf = json.loads((f0d / "run.json").read_text(encoding="utf-8"))
mf = MOE.Model("C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf")
fit, sel, test = S.rows_of(range(64)), S.rows_of(range(64, 96)), S.rows_of(range(96, 128))
js = [int(x) for x in sys.argv[1].split(",")]
lrs = [float(x) for x in sys.argv[2].split(",")]
K, K0 = int(sys.argv[3]), int(sys.argv[4])
T.EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else T.EPOCHS
out = {}
for j in js:
    blk = T.Block(mf.layer(j), dev)
    yf, ys, yt = (torch.from_numpy(S.load(f0d, f"ffn_norm-{j}", rf, r)).to(dev) for r in (fit, sel, test))
    own_f, own_s, own_t = T.Own(blk, yf), T.Own(blk, ys), T.Own(blk, yt)
    mu = yf.mean(0)
    _, _, vh = torch.linalg.svd(yf - mu, full_matrices=False)
    V = vh[:K].T.contiguous()
    zf, zs, zt = ((y - mu) @ V for y in (yf, ys, yt))
    W0 = V.T.clone()
    W0[K0:] = 0
    for lr in lrs:
        gen = torch.Generator().manual_seed(T.SEED)
        mdl = T.Map(W0, mu.clone(), zf.mean(0), zf.std(0) + 1e-6, False).to(dev)
        st = T.measure(blk, mdl, zt, own_t)
        mdl, tr = T.train(blk, mdl, zf, own_f, zs, own_s, gen, lr, f"P4 PCA-{K} from PCA-{K0}, lr {lr}")
        m = T.measure(blk, mdl, zt, own_t)
        print(f"F0 {j}: start {st['topk_overlap']:.3f}/{st['moe_cos_mean']:.3f} -> {m['topk_overlap']:.3f}/{m['moe_cos_mean']:.3f} (R2 {m['r2']:.3f})", flush=True)
        out[f"{j}_{lr}"] = {"start": st, "end": m, **tr}
        del mdl
    del blk, yf, ys, yt, own_f, own_s, own_t
    torch.cuda.empty_cache()
Path(f"C:/SiliconLLM_stage/t2/dev/tstitch_cal4_{K}_{K0}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
