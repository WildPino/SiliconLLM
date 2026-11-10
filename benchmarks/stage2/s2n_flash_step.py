"""S2N cell F: one M2 step at Ling-flash's shapes, streamed from host RAM, with s2n_core's code.  Random codes in this stack's form (experts as 2-bit
codes with fp16 scales per 256; organs as bf16 values, as the training holds them; the head bf16 on the GPU), random teacher top-32.  The training's
step: 2 sequences of 2,048 positions with documents packed (S3's corpus's length, mean 350) under a block-diagonal mask; expert load sampled from
mini's measured shares (s2n_route_hist.py).  A timing: idle machine only.

Per kind (row: what M2 holds on flash; grp: the experts' per-256 scales trained): 2 warm-up steps, then 5 timed, k = 8.  Records peak GPU memory
(allocated and reserved), the process's peak host working set, seconds per step (median, min, max), the busiest expert's load over the mean, and
hours for 16.8 M positions (4,096 steps, median), and the dequantization's share of the median step (elements unpacked in one step, counted,
times the unpack's cost per element on one expert chunk).  Fits: peak reserved <= 11.5 GB, host <= 70 GB and hours <= 240.

Usage: python -I benchmarks/stage2/s2n_flash_step.py --selftest
       python -I benchmarks/stage2/s2n_flash_step.py --json benchmarks/stage2/results/s2n_flash_step.json [--kinds row grp] [--layers 32]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
import s2n_core as C  # noqa: E402

STEPS, WARM, B, T = 5, 2, 2, 2048
GPU_LIMIT, HOST_LIMIT = 11.5, 70.0
HOURS_LIMIT = 240.0       # what the ~360 h recipe leaves after the teacher, bal's M1, the lr pick and the export
SEED = 20261109


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def host_gb() -> float:
    """The process's peak working set, GB (Windows)."""
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = k32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    return c.PeakWorkingSetSize / 1e9 if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) else float("nan")


def rnd_codes(r: int, i: int, e: int | None = None):
    shp = (r, i // 4) if e is None else (e, r, i // 4)
    dsh = (r, i // 256) if e is None else (e, r, i // 256)
    return torch.randint(0, 256, shp, dtype=torch.uint8).pin_memory(), (torch.rand(dsh) * 0.004 + 0.003).to(torch.float16).pin_memory()


def build(sh: C.Shape, kind_exp: str, dev, layers: int, hist: list | None = None) -> list[C.Layer]:
    out = []
    qkv = (sh.heads + 2 * sh.heads_kv) * sh.head_dim
    for i in range(layers):
        L = C.Layer(i)
        h = L.host
        for n, size in (("attn_norm", sh.d), ("ffn_norm", sh.d), ("attn_q_norm", sh.head_dim), ("attn_k_norm", sh.head_dim)):
            h[n] = torch.ones(size, dtype=torch.bfloat16).pin_memory()
        mats = {"attn_qkv": (qkv, sh.d), "attn_output": (sh.d, sh.heads * sh.head_dim)}
        if i == 0:
            mats.update({"ffn_gate": (sh.dense_f, sh.d), "ffn_up": (sh.dense_f, sh.d), "ffn_down": (sh.d, sh.dense_f)})
        else:
            mats.update({"ffn_gate_shexp": (sh.shared_f, sh.d), "ffn_up_shexp": (sh.shared_f, sh.d), "ffn_down_shexp": (sh.d, sh.shared_f)})
            h["ffn_gate_inp"] = (torch.randn(sh.experts, sh.d) * 0.02).to(torch.bfloat16).pin_memory()
            h["exp_probs_b"] = torch.zeros(sh.experts).pin_memory()
            for n, (r, c) in (("ffn_gate_exps", (sh.expert_f, sh.d)), ("ffn_up_exps", (sh.expert_f, sh.d)), ("ffn_down_exps", (sh.d, sh.expert_f))):
                h[n] = rnd_codes(r, c, sh.experts)
                L.mats[n] = C.Mat(kind_exp, (sh.experts, r, c), dev)
        for n, (r, c) in mats.items():
            h[n] = (torch.randn(r, c) * 0.02).to(torch.bfloat16).pin_memory()      # organs as bf16 values, as the training holds them
            L.mats[n] = C.Mat("row", (r, c), dev)
        if i >= 1 and hist is not None:
            s = torch.tensor(hist[round((i - 1) * (len(hist) - 1) / (sh.layers - 2))])     # flash's MoE layer i -> mini's, by depth
            h["exp_probs_b"] = s[torch.randperm(sh.experts)].clamp_min(1e-9).log().float().pin_memory()     # read by skewed_route
        out.append(L)
    return out


LOAD: list[float] = []


def skewed_route(x, gate, bias, sh, form):
    """F's routing: k distinct experts per token sampled from the layer's measured shares (carried in bias as log shares); random weights."""
    n = x.shape[0]
    ids = torch.multinomial(bias.exp().expand(n, -1), form.k, replacement=False)
    wts = torch.rand(n, form.k, device=x.device)
    c = torch.bincount(ids.reshape(-1), minlength=sh.experts).float()
    LOAD.append(float(c.max() / c.mean()))
    return ids, wts / wts.sum(-1, keepdim=True) * sh.route_scale


def packed_seg(b: int, t: int, rng) -> C.Seg:
    """Documents of S3's corpus's length (mean 350, at most 2,048) packed into b sequences of t; the tail of a sequence is its last document."""
    pos, doc = torch.zeros(b, t, dtype=torch.long), torch.zeros(b, t, dtype=torch.long)
    for r in range(b):
        at, j = 0, 0
        while at < t:
            n = int(min(max(rng.exponential(350), 16), t - at))
            pos[r, at:at + n] = torch.arange(n)
            doc[r, at:at + n] = j
            at, j = at + n, j + 1
    return C.Seg(pos, doc)


def selftest() -> int:
    torch.manual_seed(0)
    ok = {}

    def rel(u, v):
        return float((u.detach().float() - v.detach().float()).abs().max() / v.detach().float().abs().max())
    codes = torch.randint(0, 4, (3, 8, 512), dtype=torch.uint8)
    d = (torch.rand(3, 8, 2) + 0.5).to(torch.float16)
    p = C.pack(codes)
    ref = ((codes.float() * 2 - 3).view(3, 8, 2, 256) * d.float().unsqueeze(-1)).view(3, 8, 512)
    ok["pack/unpack"] = rel(C.unpack(p, d), ref) <= 8e-3
    x = torch.randn(5, 512, dtype=torch.bfloat16)
    for kind in ("row", "grp", "glob", "none"):
        s = {"row": torch.rand(8) + 0.5, "grp": torch.rand(8, 2) + 0.5, "glob": torch.tensor(1.3), "none": torch.ones(())}[kind].requires_grad_(kind != "none")
        xa = x.float().requires_grad_(True)
        w0 = C.unpack(p[0], d[0]).float()
        y_ref = xa @ C._grp(w0, s).t() if kind == "grp" else (xa @ w0.t()) * s
        gy = torch.randn_like(y_ref)
        y_ref.backward(gy)
        gx_ref, gs_ref = xa.grad.clone(), (s.grad.clone() if s.grad is not None else None)
        s.grad = None
        xb = x.float().requires_grad_(True)
        y = C.Packed.apply(xb.to(torch.bfloat16), p[0], d[0], s, kind, None, None).float()
        y.backward(gy)
        good = rel(y, y_ref) <= 1e-2 and rel(xb.grad, gx_ref) <= 1e-2
        if gs_ref is not None:
            good = good and rel(s.grad, gs_ref) <= 1e-2
        ok[f"Packed {kind}"] = bool(good)
    for kind in ("row", "grp"):
        s = (torch.rand(3, 8) + 0.5 if kind == "row" else torch.rand(3, 8, 2) + 0.5).requires_grad_(True)
        xe = torch.randn(3, 6, 512, dtype=torch.bfloat16)
        y = C.Packed.apply(xe, p, d, s, kind, None, None)
        y.float().sum().backward()
        gb = s.grad.clone()
        s.grad = None
        ys = torch.stack([C.Packed.apply(xe[e], p[e], d[e], s[e], kind, None, None) for e in range(3)])
        ys.float().sum().backward()
        ok[f"batched {kind}"] = rel(y, ys) <= 1e-3 and rel(gb, s.grad) <= 1e-3
    import s1t_model as M
    xr = torch.randn(64, 2048)
    gate, bias = torch.randn(256, 2048) * 0.02, torch.randn(256) * 0.01
    for f in (C.Form(8, 0), C.Form(4, 0), C.Form(4, 8), C.Form(4, 16)):
        a = C.route(xr, gate, bias, C.MINI, f)
        b = M.route(xr, gate, bias, M.Form(f.k, f.mass))
        ok[f"route k={f.k} mass={f.mass}"] = torch.equal(a[0], b[0]) and torch.allclose(a[1], b[1])
    # packing, prefill and the validity mask on a tiny stack (CPU, fp32 activations)
    tiny = C.Shape(3, 256, 4, 2, 64, 32, 512, 16, 256, 256, 1000, groups=4, groups_kept=2, k_native=4, expert_chunk=8)
    dev = torch.device("cpu")
    lay = build(tiny, "row", dev, tiny.layers)
    for L in lay:
        for n, v in list(L.host.items()):
            L.host[n] = tuple(t.clone() for t in v) if isinstance(v, tuple) else v.float() if v.dtype == torch.bfloat16 else v
    emb = torch.randn(tiny.vocab, tiny.d) * 0.5
    a_tok, b_tok = torch.randint(0, tiny.vocab, (1, 7)), torch.randint(0, tiny.vocab, (1, 5))
    with torch.no_grad():
        ha = C.forward_hidden(lay, emb, a_tok, tiny, C.Form(4, 0), dev, False)[0]
        hb = C.forward_hidden(lay, emb, b_tok, tiny, C.Form(4, 0), dev, False)[0]
        seg = C.Seg(torch.cat([torch.arange(7), torch.arange(5), torch.zeros(4, dtype=torch.long)])[None],
                    torch.tensor([0] * 7 + [1] * 5 + [-1] * 4)[None])
        hp = C.forward_hidden(lay, emb, torch.cat([a_tok, b_tok, torch.zeros(1, 4, dtype=torch.long)], 1), tiny, C.Form(4, 0), dev, False, seg)[0]
    ok["packed = separate"] = rel(hp[0, :7], ha[0]) <= 1e-4 and rel(hp[0, 7:12], hb[0]) <= 1e-4
    L1 = lay[1]
    w = C.to_gpu(L1.host, dev)
    xf = torch.randn(10, tiny.d)
    pos = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    with torch.no_grad():
        y = C.moe(xf, w, L1.mats, tiny, C.Form(2, 0, 3), pos)
        y8 = C.moe(xf, w, L1.mats, tiny, C.Form(4, 0), pos)
        y2 = C.moe(xf, w, L1.mats, tiny, C.Form(2, 0), pos)
    pre = pos < 3
    ok["prefill routes at k native"] = rel(y[pre], y8[pre]) <= 1e-5 and rel(y[~pre], y2[~pre]) <= 1e-5 and rel(y2[pre], y8[pre]) > 1e-3
    hid = torch.randn(6, tiny.d)
    head = torch.randn(tiny.vocab, tiny.d) * 0.1
    nrm = torch.ones(tiny.d)
    tid = torch.randint(0, tiny.vocab, (6, 32))
    tlp = torch.log_softmax(torch.randn(6, 32), -1) - 0.1
    v = torch.tensor([True, True, False, True, False, True])
    t_all, _, _, _ = C.logits_kl(hid[v], head, nrm, tid[v], tlp[v], 1e-6, False)
    t_m, _, _, nv = C.logits_kl(hid, head, nrm, tid, tlp, 1e-6, False, v)
    ok["validity mask"] = abs(float(t_all) - float(t_m)) <= 1e-4 * abs(float(t_all)) and nv == 4
    for k, v in ok.items():
        print("selftest", "ok  " if v else "FAIL", k)
    print("selftest", "PASS" if all(ok.values()) else "FAIL", len(ok))
    return 0 if all(ok.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--kinds", nargs="+", default=["row", "grp"])
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--micro", type=int, default=1, choices=(1, 2), help="sequences of 2,048 per micro-step; the step is 4,096 tokens either way")
    ap.add_argument("--hist", type=Path, default=HERE / "results" / "s2n_route_hist.json", help="mini's measured expert shares (s2n_route_hist.py)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    dev = torch.device("cuda")
    sh = C.FLASH
    hist = json.loads(a.hist.read_text(encoding="utf-8"))
    C.route = skewed_route
    cfg = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version(), "shape": sh.__dict__,
           "tokens_per_step": B * T, "micro_sequences": a.micro, "steps": STEPS, "warm": WARM, "layers": a.layers, "form": "k = 8", "kinds": a.kinds,
           "organs": "bf16", "packing": "documents of mean 350 positions, block-diagonal mask", "routing": f"sampled from {a.hist.name}",
           "hist_busiest_over_mean_median": hist["busiest_over_mean_median"], "limits": {"gpu_gb": GPU_LIMIT, "host_gb": HOST_LIMIT, "hours": HOURS_LIMIT}}
    print("CONFIG", json.dumps(cfg), flush=True)
    out = {"config": cfg, "kinds": {}}
    embed = (torch.randn(sh.vocab, sh.d) * 0.02).to(torch.bfloat16).pin_memory()
    rng = np.random.default_rng(SEED)
    for kind in a.kinds:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        LOAD.clear()
        layers = build(sh, kind, dev, a.layers, hist["shares_sorted"])
        head = (torch.randn(sh.vocab, sh.d) * 0.02).to(torch.bfloat16).to(dev)
        norm = torch.ones(sh.d, device=dev, dtype=torch.bfloat16)
        params = [q for L in layers for q in L.params()]
        opt = torch.optim.Adafactor(params, lr=1e-4)
        times = []
        for s in range(WARM + STEPS):
            tokens = torch.randint(0, sh.vocab, (B, T))
            seg = packed_seg(B, T, rng)
            t_lp = torch.log_softmax(torch.randn(B * T, 32, device=dev) * 3, -1).sort(-1, descending=True)[0] - 0.05
            t_ids = torch.randint(0, sh.vocab, (B * T, 32), device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for m in range(0, B, a.micro):
                sl = slice(m * T, (m + a.micro) * T)
                sg = C.Seg(seg.pos[m:m + a.micro], seg.doc[m:m + a.micro])
                r = C.step(layers, embed, head, norm, tokens[m:m + a.micro], t_ids[sl], t_lp[sl], sh, C.Form(8, 0), dev, train=True,
                           weight=a.micro / B, seg=sg)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(kind, "step", s, f"{dt:.1f} s", f"kl {r['kl'] / r['n']:.3f}", flush=True)
            if s >= WARM:
                times.append(dt)
        # the dequantization's share: elements unpacked in one step (counted), times the unpack's measured cost per element on an expert chunk
        counted, orig = [0], C.unpack

        def counting(p, d, exi=None, exv=None):
            counted[0] += p.numel() * 4
            return orig(p, d, exi, exv)
        C.unpack = counting
        r = C.step(layers, embed, head, norm, tokens[:a.micro], t_ids[:a.micro * T], t_lp[:a.micro * T], sh, C.Form(8, 0), dev, train=True,
                   weight=a.micro / B, seg=C.Seg(seg.pos[:a.micro], seg.doc[:a.micro]))
        opt.zero_grad(set_to_none=True)
        C.unpack = orig
        per_step = counted[0] * (B // a.micro)
        hp, hd = layers[1].host["ffn_gate_exps"]
        p_c, d_c = hp[:sh.expert_chunk].to(dev), hd[:sh.expert_chunk].to(dev)
        C.unpack(p_c, d_c)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            C.unpack(p_c, d_c)
        torch.cuda.synchronize()
        per_elem = (time.perf_counter() - t0) / 20 / (p_c.numel() * 4)
        med = float(np.median(times))
        rec = {"seconds": times, "min": min(times), "median": med, "max": max(times), "hours_16_8M": med * 4096 / 3600,
               "hours_16_8M_min_max": [min(times) * 4096 / 3600, max(times) * 4096 / 3600],
               "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
               "host_peak_gb": host_gb(), "trainable": sum(q.numel() for q in params),
               "busiest_over_mean": {"median": float(np.median(LOAD)), "max": float(max(LOAD))},
               "unpacked_elements_per_step": per_step, "unpack_seconds_per_step": per_step * per_elem,
               "unpack_share_of_median": per_step * per_elem / med}
        rec["fits"] = rec["gpu_peak_reserved_gb"] <= GPU_LIMIT and rec["host_peak_gb"] <= HOST_LIMIT and rec["hours_16_8M"] <= HOURS_LIMIT
        rec["hours_without_unpack"] = (1 - rec["unpack_share_of_median"]) * rec["hours_16_8M"]      # an estimate: what a fused kernel could at most save
        rec["fits_without_unpack"] = rec["gpu_peak_reserved_gb"] <= GPU_LIMIT and rec["host_peak_gb"] <= HOST_LIMIT and rec["hours_without_unpack"] <= HOURS_LIMIT
        out["kinds"][kind] = rec
        print(kind, json.dumps(rec), flush=True)
        del layers, head, params, opt
        if a.json:
            a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
