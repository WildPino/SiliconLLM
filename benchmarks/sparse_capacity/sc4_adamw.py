#!/usr/bin/env python3
# Sparse Capacity, phase 4 (D-M): Python side of the fused host AdamW + ternary packing kernel (csrc/adamw_tern.c),
# and its exactness selftest. The kernel replaces, for the expert bank, torch's fused CPU AdamW followed by w_q and the
# 2-bit packing: the GPU then receives 2 bits per weight plus one fp32 scale per row instead of fp32 weights.
#
#   python benchmarks/sparse_capacity/sc4_adamw.py --selftest     (exactness; runs on a busy machine)
#   python benchmarks/sparse_capacity/sc4_adamw.py --time         (timings: only on a free machine)
import argparse, ctypes, json, os, shutil, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc2_model as SM      # noqa: E402

SRC = os.path.join(HERE, "csrc", "adamw_tern.c")
BUILD = os.path.join(HERE, "csrc", "build")
FLAGS = ["-O2", "-mavx2", "-mfma", "-ffp-contract=off", "-fopenmp", "-shared"]
BETAS, WD, EPS = (0.9, 0.95), 0.1, 1e-8              # phase 2's AdamW (torch default eps)


def build():
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, "adamw_tern.dll")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(SRC):
        subprocess.run(["gcc"] + FLAGS + ["-o", out, SRC], check=True)
    gcc = shutil.which("gcc")
    if gcc and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.dirname(gcc))                       # libgomp, libwinpthread
    lib = ctypes.CDLL(out)
    f, i64, d, P = ctypes.c_int, ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
    lib.adamw_tern.argtypes = [P, P, P, P, i64, i64, d, d, d, d, d, d, d, ctypes.c_float, P, P, f]
    lib.tern_pack.argtypes = [P, i64, i64, P, P, f]
    return lib


LIB = None


def _lib():
    global LIB
    if LIB is None:
        LIB = build()
    return LIB


def ptr(t):
    assert t.is_contiguous() and t.device.type == "cpu"
    return ctypes.c_void_p(t.data_ptr())


def step(p, g, m, v, stepno, lr, packed=None, scale=None, nthreads=6, betas=BETAS, wd=WD, eps=EPS, gscale=1.0):
    """One AdamW step on (p, g * gscale, m, v), all fp32 CPU contiguous, rows = last dim; then (packed, scale) of the
    new p. gscale is the clip coefficient (an fp32 value); g itself is not modified."""
    rowlen = p.shape[-1]; rows = p.numel() // rowlen
    assert rowlen % 8 == 0 and all(t.dtype == torch.float32 for t in (p, g, m, v))
    bc1, bc2 = 1 - betas[0] ** float(stepno), 1 - betas[1] ** float(stepno)
    _lib().adamw_tern(ptr(p), ptr(g), ptr(m), ptr(v), rows, rowlen, lr, betas[0], betas[1], wd, eps, bc1, bc2, float(gscale),
                      ptr(packed) if packed is not None else None, ptr(scale) if scale is not None else None, nthreads)


def pack(p, packed, scale, nthreads=6):
    rowlen = p.shape[-1]
    _lib().tern_pack(ptr(p), p.numel() // rowlen, rowlen, ptr(packed), ptr(scale), nthreads)


def buffers(p):
    return torch.empty(*p.shape[:-1], p.shape[-1] // 4, dtype=torch.uint8), torch.empty(*p.shape[:-1], 1)


def unpack(b, n):
    out = torch.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(*b.shape[:-1], n)
    return out.float() - 1.0


def selftest():
    res = []
    def check(name, val, tol):
        ok = bool(val <= tol); res.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<76} {val:.3e}  (tol {tol:.0e})")
    torch.set_num_threads(6)
    g0 = torch.Generator().manual_seed(0)
    for shape in ((512, 8, 256), (512, 256, 8)):                          # Wg/Wu rows of D, Wd rows of h
        p0 = torch.randn(*shape, generator=g0) * 0.02
        ref = torch.nn.Parameter(p0.clone()); opt = torch.optim.AdamW([ref], lr=3e-3, betas=BETAS, weight_decay=WD, fused=True)
        p, m, v = p0.clone(), torch.zeros_like(p0), torch.zeros_like(p0)
        pk, sc = buffers(p)
        for t in range(1, 6):                                            # 5 steps, varying lr and gradient scale
            gr = torch.randn(*shape, generator=g0) * (10.0 ** -t)
            lr = 3e-3 * (0.5 + 0.1 * t)
            for gp in opt.param_groups: gp["lr"] = lr
            ref.grad = gr.clone(); opt.step()
            step(p, gr, m, v, t, lr, pk, sc)
        st = opt.state[ref]
        tag = "x".join(map(str, shape))
        check(f"{tag}: weights after 5 steps vs torch fused AdamW (max abs, bitwise = 0)", float((p - ref.data).abs().max()), 0)
        check(f"{tag}: first moment (max abs)", float((m - st["exp_avg"]).abs().max()), 0)
        check(f"{tag}: second moment (max abs)", float((v - st["exp_avg_sq"]).abs().max()), 0)
        for dev in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            qr, sr = SM.w_q(ref.data.to(dev))
            qk, sk = unpack(pk, shape[-1]).to(dev), sc.to(dev)
            rel = float(((sk - sr).abs() / sr).max())
            flips = (qk != qr)
            x = (ref.data.to(dev) / sr).abs()
            hard = int((flips & ((x - 0.5).abs() > 1e-5)).sum())
            check(f"{tag}: scale vs w_q on {dev} (max relative)", rel, 1e-6)
            check(f"{tag}: ternary flips vs w_q on {dev} away from the 0.5 boundary (count)", hard, 0)
            print(f"       ({int(flips.sum())} flips in all, of {p.numel()})")
    # the clip coefficient inside the kernel = torch's clip (grad *= coef) then fused AdamW, bitwise
    p0 = torch.randn(64, 256, generator=g0) * 0.02; gr = torch.randn(64, 256, generator=g0) * 1e-2
    coef = torch.tensor(0.3712345, dtype=torch.float32)
    ref = torch.nn.Parameter(p0.clone()); opt = torch.optim.AdamW([ref], lr=3e-3, betas=BETAS, weight_decay=WD, fused=True)
    ref.grad = gr.clone(); torch._foreach_mul_([ref.grad], coef); opt.step()
    a = p0.clone(); za, zb = torch.zeros_like(p0), torch.zeros_like(p0); g_keep = gr.clone()
    step(a, gr, za, zb, 1, 3e-3, gscale=float(coef))
    check("clip coefficient in the kernel vs clip then torch fused AdamW (max abs, bitwise = 0)", float((a - ref.data).abs().max()), 0)
    check("the kernel leaves g unchanged (max abs)", float((gr - g_keep).abs().max()), 0)
    # planted negatives: a wrong beta2, and a skipped weight decay, must be seen
    p0 = torch.randn(64, 256, generator=g0) * 0.02; gr = torch.randn(64, 256, generator=g0) * 1e-2
    a = p0.clone(); b = p0.clone(); c = p0.clone()
    z = [torch.zeros_like(p0) for _ in range(6)]
    step(a, gr, z[0], z[1], 1, 3e-3); step(b, gr, z[2], z[3], 1, 3e-3, betas=(0.9, 0.999)); step(c, gr, z[4], z[5], 1, 3e-3, wd=0.0)
    check("planted negative: beta2 0.999 instead of 0.95 is seen (1e-9 / diff)", 1e-9 / max(float((a - b).abs().max()), 1e-30), 1)
    check("planted negative: no weight decay is seen (1e-12 / diff)", 1e-12 / max(float((a - c).abs().max()), 1e-30), 1)
    # packing round trip and the zero row
    w = torch.tensor([[0.0] * 8, [1.0, -1.0, 0.2, -0.2, 2.0, -2.0, 0.49, 0.51]])
    pk, sc = buffers(w); pack(w, pk, sc)
    qr, sr = SM.w_q(w)
    check("zero row and a hand row: packed trits equal w_q (count of differences)", float((unpack(pk, 8) != qr).sum()), 0)
    check("zero row: scale clamped to 1e-5 as w_q does (abs diff from w_q's scale)", abs(float(sc[0, 0]) - float(sr[0, 0])), 0)
    print("SELFTEST (fused AdamW + ternary)", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


def timing(nthreads=6, reps=7):
    """Per step, at the m16 bank shapes (8 layers x Wg, Wu, Wd): torch fused AdamW, torch AdamW + the torch
    quantize/pack of the phase-4 dry, and the fused kernel. Only meaningful on a free machine."""
    torch.set_num_threads(nthreads)
    L, E, H, D = 8, 2048, 8, 256
    shapes = [(E, H, D), (E, H, D), (E, D, H)] * L
    g = torch.Generator().manual_seed(0)
    ps = [torch.randn(*s, generator=g) * 0.02 for s in shapes]
    gs = [torch.randn(*s, generator=g) * 1e-3 for s in shapes]
    ms = [torch.zeros_like(p) for p in ps]; vs = [torch.zeros_like(p) for p in ps]
    bufs = [buffers(p) for p in ps]
    P = sum(p.numel() for p in ps)
    params = [torch.nn.Parameter(p.clone()) for p in ps]
    for q, gr in zip(params, gs): q.grad = gr
    opt = torch.optim.AdamW(params, lr=3e-3, betas=BETAS, weight_decay=WD, fused=True)
    def med(f):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts) // 2]
    def torch_quant():
        for q, (b, s) in zip(params, bufs):
            sv = q.data.abs().mean(-1, keepdim=True).clamp_min(1e-5)
            qq = torch.round(q.data / sv).clamp_(-1, 1)
            u = (qq + 1).to(torch.uint8).reshape(*qq.shape[:-1], qq.shape[-1] // 4, 4)
            b.copy_(u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)); s.copy_(sv)
    cnt = [0]
    def fused():
        cnt[0] += 1
        for p, gr, m, v, (b, s) in zip(ps, gs, ms, vs, bufs):
            step(p, gr, m, v, cnt[0], 3e-3, b, s, nthreads)
    out = dict(bank_params=P, threads=nthreads)
    for k, f in (("torch_adamw", opt.step), ("torch_quant_pack", torch_quant), ("fused_kernel", fused)):
        t = med(f); out[k] = dict(s=t, ns_per_param=t / P * 1e9)
        print(f"{k:18s} {t:.4f} s = {t / P * 1e9:.3f} ns per bank parameter", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true"); g.add_argument("--time", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    r = timing()
    path = os.path.join(HERE, "..", "..", "results", "sparse_capacity", "f4_kernel_time_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    json.dump(r, open(path, "w"), indent=1); print("->", os.path.normpath(path))
