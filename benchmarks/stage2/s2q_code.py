"""Stage 2, the fourth code, step 1: at equal bytes, how much does the output error fall if IQ2_BN's two bits use all four codes?

Forms, each with one f32 scale per row, 2 bits per weight, no training:
    tern   {-1, 0, 1}, the threshold that minimizes the row's squared error, by exact search over the sorted row  (the reference)
    asym   {-1, 0, 1, 2} with a sign per row, Lloyd, the better sign kept
    bal    {-1.5, -0.5, 0.5, 1.5}, Lloyd; in the kernel this is code - 1 with the correction -0.5 * s * sum(x) per row
Extra arm, attention only: rot_*, the same forms after a rotation per KV head between V and the output projection, absorbed in the
weights (a Hadamard matrix with random signs, 128 x 128): nothing is added at run time.

Where: layers 5, 10, 15 of Ling-mini, on attention, shared expert and routed experts, in float, on the first 8,192 held-out positions of
stage 1's corpus (the teacher stream of test 2c), which no panel uses.  Primary: the block loss of test 2c, one organ quantized at a time,
every other weight original: mean of (nrm(o) - nrm(t))^2 * gw.  Reported: the block output's plain relative error, the weights' relative
error.  Ling-flash, reported: weights only, the same layers, from the BF16 GGUF (no activations of flash are on disk).

Controls: `twin`, Gaussian weights of the same variance per row, on a tensor of each kind, through the same quantizers and inputs (the
trivial one); `plant`, weights built on each form's grid plus noise of 5% of the step, each quantizer on each; `identity`, the rotated
weights unquantized against the original block.  The labels are s2q_verdict.py's.

Usage: python -I benchmarks/stage2/s2q_code.py --json benchmarks/stage2/results/s2q_run.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
import s1t_model as M  # noqa: E402
from s1t_train import downstream_gain2  # noqa: E402
from s2o_train import nrm  # noqa: E402
from s2h_head import gguf  # noqa: E402

TEACHER = Path("E:/SiliconLLM_Artifacts/stage2/s2s/teacher")
FLASH = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/ling-flash-2.0-96f9b7a-bf16.gguf")
SEED = 20261105
LAYERS, POSITIONS, THREADS, ITERS, ROWS = (5, 10, 15), 8192, 4, 16, 8192
FLASH_EXPERTS = 16
FORMS = ("tern", "asym", "bal")
ORGANS = {"attention": ["attention.query_key_value.weight", "attention.dense.weight"],
          "shared": [f"mlp.shared_experts.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")],
          "experts": [f"mlp.experts.{p}" for p in ("gate_proj", "up_proj", "down_proj")]}


def q_tern(w: torch.Tensor) -> torch.Tensor:
    a = w.abs()
    v = a.sort(1, descending=True).values
    c = v.double().cumsum(1)
    t = torch.arange(1, w.shape[1] + 1, dtype=torch.float64)
    k = (c * c / t).argmax(1, keepdim=True)
    s = (c.gather(1, k) / (k + 1)).float()
    return torch.where(a >= v.gather(1, k), w.sign(), torch.zeros_like(w)) * s


def lloyd(w: torch.Tensor, grid, s0: float) -> tuple[torch.Tensor, torch.Tensor]:
    s = w.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-20) * s0
    for _ in range(ITERS):
        q = grid(w / s)
        s = ((w * q).sum(1, keepdim=True) / (q * q).sum(1, keepdim=True).clamp_min(1e-20)).clamp_min(1e-20)
    d = grid(w / s) * s
    return d, (w - d).pow(2).sum(1, keepdim=True)


def q_bal(w: torch.Tensor) -> torch.Tensor:
    return lloyd(w, lambda z: (z.floor() + 0.5).clamp(-1.5, 1.5), 1.0)[0]


def q_asym(w: torch.Tensor) -> torch.Tensor:
    g = lambda z: z.round().clamp(-1, 2)  # noqa: E731
    dp, ep = lloyd(w, g, 1.0)
    dn, en = lloyd(-w, g, 1.0)
    return torch.where(ep <= en, dp, -dn)


QUANT = {"tern": q_tern, "asym": q_asym, "bal": q_bal}


def quantize(w: torch.Tensor, form: str) -> torch.Tensor:
    f = w.reshape(-1, w.shape[-1])
    out = torch.empty_like(f)
    for i in range(0, f.shape[0], ROWS):
        out[i:i + ROWS] = QUANT[form](f[i:i + ROWS])
    return out.view_as(w)


def rel(w: torch.Tensor, d: torch.Tensor) -> float:
    return float((w - d).double().pow(2).sum() / w.double().pow(2).sum())


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / n ** 0.5


def rotate(w: dict, gen: torch.Generator) -> dict:
    """V's rows and the output projection's columns, per KV head; q-head j reads KV head j // (H / HKV)."""
    qkv, dense = w[ORGANS["attention"][0]].clone(), w[ORGANS["attention"][1]].clone()
    v0, group = (M.H + M.HKV) * M.HD, M.H // M.HKV
    for g in range(M.HKV):
        signs = (torch.randint(0, 2, (M.HD,), generator=gen) * 2 - 1).float()
        r = hadamard(M.HD) * signs
        rows = slice(v0 + g * M.HD, v0 + (g + 1) * M.HD)
        qkv[rows] = r @ qkv[rows]
        for j in range(g * group, (g + 1) * group):
            cols = slice(j * M.HD, (j + 1) * M.HD)
            dense[:, cols] = dense[:, cols] @ r.t()
    return {ORGANS["attention"][0]: qkv, ORGANS["attention"][1]: dense}


def plant() -> dict:
    gen = torch.Generator().manual_seed(SEED)
    rows, n = 512, 2048
    s = 0.01 + 0.02 * torch.rand(rows, 1, generator=gen)
    u = torch.rand(rows, n, generator=gen)
    grids = {"tern": torch.where(u < 0.3, -1.0, torch.where(u < 0.7, 0.0, 1.0)),
             "asym": (u * 4).floor() - 1, "bal": (u * 4).floor() - 1.5}
    out = {}
    for f, q in grids.items():
        sign = torch.where(torch.arange(rows)[:, None] % 2 == 0, 1.0, -1.0) if f == "asym" else 1.0
        clean = q * s * sign
        w = clean + 0.05 * s * torch.randn(rows, n, generator=gen)
        out[f] = {"noise_rel": rel(w, clean)}
        for g in FORMS:
            d = quantize(w, g)
            out[f][g] = rel(w, d)
            if g == f:
                out[f]["recovered"] = float(((d - clean).abs() < 0.25 * s).float().mean())
    return out


def out_err(x: torch.Tensor, w: torch.Tensor, d: torch.Tensor) -> float:
    return float(((x @ (w - d).t()).double().pow(2).sum()) / ((x @ w.t()).double().pow(2).sum()))


def twin(name: str, w: torch.Tensor, x: torch.Tensor, gen: torch.Generator) -> dict:
    g = torch.randn(w.shape, generator=gen) * w.std(1, keepdim=True)
    rec: dict = {"tensor": name}
    for tag, t in (("real", w), ("gauss", g)):
        for f in FORMS:
            d = quantize(t, f)
            rec[f"{tag}_{f}"] = {"weight": rel(t, d), "output": out_err(x, t, d)}
    return rec


def stream(i: int, n_want: int) -> tuple[torch.Tensor, list[int], dict]:
    meta = json.loads((TEACHER / "meta.json").read_text(encoding="utf-8"))
    lens, n_tr = meta["lens"], meta["n_train_seq"]
    start = sum(lens[:n_tr])
    take, m = [], 0
    for ln in lens[n_tr:]:
        if m >= n_want:
            break
        take.append(ln)
        m += ln
    a = np.fromfile(TEACHER / f"h_{i}.bin", dtype=np.int16, count=m * M.D, offset=start * M.D * 2)
    return torch.from_numpy(a).view(torch.bfloat16).view(m, M.D).float(), take, {"positions": m, "sequences": len(take), "first_position": start}


def mini_layer(i: int, positions: int, W, rec: dict, save) -> None:
    t0 = time.time()
    wf = {k: v.float() for k, v in W.layer(i).items()}
    gw = downstream_gain2(W, i).float().cpu()
    h, lens, info = stream(i, positions)
    offs = np.concatenate([[0], np.cumsum(lens)]).tolist()
    css = {m: M.rope(m, "cpu") for m in set(lens)}
    form = M.Form()

    def forward(w):
        return [M.layer_forward(i, w, h[offs[j]:offs[j + 1]], css[lens[j]], form).float() for j in range(len(lens))]

    with torch.no_grad():
        ref = forward(wf)
        delta = float(sum((r - h[offs[j]:offs[j + 1]]).double().pow(2).sum() for j, r in enumerate(ref)))

        def cell(wt: dict) -> dict:
            o = forward({**wf, **wt})
            loss = sum(float(((nrm(a) - nrm(b)) ** 2 * gw).mean(-1).sum()) for a, b in zip(o, ref)) / offs[-1]
            return {"block": loss, "plain": float(sum((a - b).double().pow(2).sum() for a, b in zip(o, ref))) / delta}

        layer: dict = {"stream": info, "organs": {}}
        for organ, names in ORGANS.items():
            layer["organs"][organ] = {}
            for f in FORMS:
                wt = {n: quantize(wf[n], f) for n in names}
                c = cell(wt)
                c["weight"] = sum(rel(wf[n], wt[n]) * wf[n].numel() for n in names) / sum(wf[n].numel() for n in names)
                layer["organs"][organ][f] = c
                del wt
                print(f"L{i} {organ} {f} {json.dumps(c)} {time.time() - t0:.0f}s", flush=True)
                rec["mini"][str(i)] = layer
                save()
        rw = rotate(wf, torch.Generator().manual_seed(SEED + i))
        layer["identity"] = cell(rw)
        print(f"L{i} identity {json.dumps(layer['identity'])}", flush=True)
        for f in FORMS:
            wt = {n: quantize(rw[n], f) for n in rw}
            c = cell(wt)
            c["weight"] = sum(rel(rw[n], wt[n]) * rw[n].numel() for n in rw) / sum(rw[n].numel() for n in rw)
            layer["organs"]["attention"]["rot_" + f] = c
            print(f"L{i} attention rot_{f} {json.dumps(c)} {time.time() - t0:.0f}s", flush=True)
        gen = torch.Generator().manual_seed(SEED + 100 + i)
        x = M.rms(h, wf["input_layernorm.weight"])
        a = torch.cat([h[offs[j]:offs[j + 1]] + M.attention(wf, x[offs[j]:offs[j + 1]], css[lens[j]]) for j in range(len(lens))])
        u = M.rms(a, wf["post_attention_layernorm.weight"])
        layer["twin"] = [twin("attn_qkv", wf[ORGANS["attention"][0]], x, gen), twin("shared_gate", wf[ORGANS["shared"][0]], u, gen),
                         twin("expert0_gate", wf[ORGANS["experts"][0]][0], u, gen)]
        print(f"L{i} twin {json.dumps(layer['twin'][0])} {time.time() - t0:.0f}s", flush=True)
        layer["seconds"] = time.time() - t0
        rec["mini"][str(i)] = layer
        save()


def bf16(t) -> torch.Tensor:
    a = np.ascontiguousarray(t).view(np.uint16)
    return torch.from_numpy(a.astype(np.int32) << 16).view(torch.float32)


def flash(rec: dict, save) -> None:
    r = gguf.GGUFReader(str(FLASH))
    by = {t.name: t for t in r.tensors}
    rng = np.random.default_rng(SEED)
    for i in LAYERS:
        ids = sorted(rng.choice(256, FLASH_EXPERTS, replace=False).tolist())
        layer: dict = {"experts_read": ids}
        for organ, names, ex in (("attention", ("attn_qkv", "attn_output"), False), ("shared", ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"), False),
                                 ("experts", ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"), True)):
            acc = {f: [0.0, 0] for f in FORMS}
            for n in names:
                t = by[f"blk.{i}.{n}.weight"]
                for e in (ids if ex else [None]):
                    w = bf16(t.data[e] if ex else t.data)
                    for f in FORMS:
                        acc[f][0] += rel(w, quantize(w, f)) * w.numel()
                        acc[f][1] += w.numel()
            layer[organ] = {f: v[0] / v[1] for f, v in acc.items()}
            print(f"flash L{i} {organ} {json.dumps(layer[organ])}", flush=True)
        rec["flash"][str(i)] = layer
        save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--positions", type=int, default=POSITIONS)
    ap.add_argument("--no-flash", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    rec: dict = {"cell": "S2Q-fourth-code", "layers": a.layers, "positions": a.positions, "iters": ITERS, "seed": SEED, "torch": torch.__version__,
                 "plant": plant(), "mini": {}, "flash": {}}
    print("CONFIG", json.dumps({k: rec[k] for k in ("layers", "positions", "iters", "seed", "torch")}), "threads", THREADS, flush=True)
    print("PLANT", json.dumps(rec["plant"]), flush=True)

    def save():
        a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    W = M.Weights()
    for i in (int(x) for x in a.layers.split(",")):
        mini_layer(i, a.positions, W, rec, save)
    if not a.no_flash:
        flash(rec, save)
    rec["status"] = "COMPLETE"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
