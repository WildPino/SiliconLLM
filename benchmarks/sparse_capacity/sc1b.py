#!/usr/bin/env python3
# Sparse Capacity, phase 1b (D-M): "magazzino contro officina" -- the runner.
# Protocol: docs/research/SPARSE_CAPACITY/F1B_CROSSOVER_PROTOCOL.md. This script prints and records NUMBERS only:
# the desk prediction is sc1b_predict.py, the verdict is sc1b_verdict.py, neither chooses anything at run time.
#
#   python benchmarks/sparse_capacity/sc1b.py --selftest          kernels and both paths vs an fp64 reference
#   python benchmarks/sparse_capacity/sc1b.py --dry               small sizes, prints CONFIG; NOT a measurement
#   python benchmarks/sparse_capacity/sc1b.py --run               the measurement (unloaded machine, user's go)
#   python benchmarks/sparse_capacity/sc1b.py --run --resume DIR  continue a run after a crash (completed cells kept)
import argparse, ctypes, hashlib, json, math, os, platform, shutil, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import sc1b_model as M  # noqa: E402

SRC = os.path.join(HERE, "sc1b_cpu.c")
BIN = os.path.join(HERE, "bin")
DLL = os.path.join(BIN, "sc1b_cpu.dll")
CFLAGS = ["-O3", "-mavx2", "-mfma", "-march=znver2", "-fopenmp", "-shared"]   # NO -ffast-math

D = M.D
NTHR, STRIDE = 6, 2                       # one OpenMP thread per physical core of the 3600X
H_LIST = (128, 8, 1)
R_LIST = (1, 2, 4, 8, 16, 32, 64, 128)
P_BANK = 3 * 128 * D * 2048               # 201,326,592 parameters: one layer's bank, 805 MB in fp32
CHUNK_BYTES = 32 << 20                    # streaming chunk: 32 MiB of bf16 weights or of gathered activations
NSLOT, LOOK = 4, 2                        # device ring slots; chunks in flight ahead of the CPU update
WARMUP, REPS, ROUNDS = 1, 4, 3
T_POOL = 16384                            # tokens per step the bank slice draws from (PLAN: steps of 16k tokens or more)
LR, EPS, B1, B2 = 1e-4, 1e-8, 0.9, 0.999
SIGMA_W, DY_SCALE = 0.02, 0.01
ARM_LIST = tuple(M.ARMS)
MU = {k: (3.0 if v >= 1.0 else statistics.NormalDist().inv_cdf(math.sqrt(v))) for k, v in M.RHO.items()}
MU["zero"] = -6.0                         # the planted CPU arm: no hidden unit is ever active
GROUPS = tuple(M.PAIRS.values()) + (("C1",), ("C2",))   # a pair is timed cell by cell, C and S alternating
H1_SKIP = ("C3d", "S2d")
# Ternary arms: the absmean quantizer caps the column-0 offset at 1 x scale, so the density cannot be set from the
# weights. It is set from the token instead: x0 = |MU| x the std of the other 255 ternary terms (about 69% of the
# codes are nonzero), so g/s has mean +-x0 against std sqrt(0.69 x 255) on both paths (LUT at 63 levels and bf16).
X0_TERN = {k: max(1.0, abs(v) * math.sqrt(0.69 * (D - 1))) for k, v in MU.items()}                  # at h1 the primary density is already 1: these would repeat C3/S2

DRY = dict(P_BANK=3 * 128 * D * 32, R_LIST=(1, 4, 16), WARMUP=1, REPS=2, ROUNDS=1, RAM_MB=256, PCIE_MB=64)


# ----------------------------------------------------------------------------------------------- build and bind
def gcc_path():
    g = shutil.which("gcc")
    if not g:
        sys.exit("gcc not found on PATH")
    return g


def build():
    os.makedirs(BIN, exist_ok=True)
    cmd = [gcc_path()] + CFLAGS + ["-o", DLL, SRC, "-lm"]
    subprocess.run(cmd, check=True)
    return cmd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Ctx(ctypes.Structure):
    _fields_ = [("E", ctypes.c_int32), ("h", ctypes.c_int32), ("D", ctypes.c_int32), ("r", ctypes.c_int32),
                ("T", ctypes.c_int32), ("nthr", ctypes.c_int32),
                ("tok", ctypes.c_void_p), ("X", ctypes.c_void_p), ("DY", ctypes.c_void_p), ("Y", ctypes.c_void_p),
                ("DX", ctypes.c_void_p), ("acc", ctypes.c_void_p), ("GU", ctypes.c_void_p), ("XQ", ctypes.c_void_p),
                ("XSA", ctypes.c_void_p), ("LUT", ctypes.c_void_p),
                ("lr", ctypes.c_float), ("eps", ctypes.c_float), ("b1", ctypes.c_float), ("b2", ctypes.c_float),
                ("step", ctypes.c_int32), ("seed", ctypes.c_uint32),
                ("n_active", ctypes.c_int64), ("n_rows_upd", ctypes.c_int64)]


class Bank(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("W", "Wb", "codes", "sc", "S1", "S2", "G")]


_DLL_DIRS = []


def load_lib():
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")                     # no spinning workers while CUDA is driven
    _DLL_DIRS.append(os.add_dll_directory(os.path.dirname(gcc_path())))   # libgomp; the handle must stay alive
    lib = ctypes.CDLL(DLL)
    P, I, I64, F, DB, U32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_double, ctypes.c_uint32
    PC, PB = ctypes.POINTER(Ctx), ctypes.POINTER(Bank)
    sig = {"xalloc": ([I64], P), "xfree": ([P], None), "cpu_fwd": ([PC, I, PB], None),
           "cpu_bwd": ([PC, I, I, I, PB], None), "cpu_upd_unfused": ([PC, PB], None),
           "str_update": ([I, I, I64, I64, I, PC, PB, P, P, P], None), "tern_init": ([PC, PB, I], None),
           "tern_init_rowmajor": ([I64, I, PB, I], None), "f32_to_bf16_all": ([P, P, I64, I], None),
           "k_read": ([P, I64, I], DB), "k_write_nt": ([P, I64, F, I], None), "k_write": ([P, I64, F, I], None),
           "k_rmw": ([P, I64, F, I], None), "fma_plant": ([I64, I], DB), "count_nonfinite_f32": ([P, I64], I64),
           "count_nonfinite_bf16": ([P, I64], I64), "pin_threads": ([I, I], I), "omp_max": ([], I),
           "lut_selftest": ([I, I, U32], I64), "acc_zero": ([PC], None), "acc_reduce": ([PC, I], None)}
    for name, (a, r) in sig.items():
        fn = getattr(lib, name); fn.argtypes = a; fn.restype = r
    return lib


def ptr(a):
    return None if a is None else a.ctypes.data


FMT = {"f32": 0, "bf16": 1, "tern": 2}
OPT = {"sgd": 0, "rowada": 1, "adam": 2}


# ----------------------------------------------------------------------------------------------- problem and bank
class Problem:
    """Balanced routing: every expert gets exactly r tokens, every token goes to k = A/h experts (active hidden A
    held fixed), so T = E*r/k. Slots are shuffled, so an expert's tokens are scattered over the token array."""
    def __init__(s, h, r, E, seed, k=None, T=None, x0=1.0):
        s.h, s.r, s.E, s.x0 = h, r, E, x0
        rng = np.random.default_rng(seed)
        s.f_fixed = 1.0
        if T is not None:
            # Review obj. 1: the bank is a SLICE of a layer's bank and its E*r routed slots draw from a step of
            # T >= 16k tokens, so the token rows are as scattered and as cold as in the registered regime. T is a power
            # of two and each expert's r tokens are a + b*j mod T (b odd): distinct, pseudo-randomly scattered.
            assert T & (T - 1) == 0 and r <= T
            s.T = T
            a = rng.integers(0, T, size=(E, 1), dtype=np.int64)
            b = 2 * rng.integers(0, T // 2, size=(E, 1), dtype=np.int64) + 1
            s.tok = np.ascontiguousarray(((a + b * np.arange(r, dtype=np.int64)) % T).astype(np.int32))
            # per-layer fixed costs (activation transfers, accumulator zero/reduce) are paid once per T tokens by the
            # whole layer bank; this slice is charged its share of the layer's routed slots, E*r / (T * A/h)
            s.f_fixed = min(1.0, E * r * h / (T * M.A))
        else:
            k = k or max(1, M.A // h)
            s.T = -(-E * r // k)
            exp_at_pos = np.repeat(np.arange(E, dtype=np.int32), r)[rng.permutation(E * r)]
            tok_of_pos = (np.arange(E * r, dtype=np.int64) // k).astype(np.int32)
            order = np.argsort(exp_at_pos, kind="stable")
            s.tok = np.ascontiguousarray(tok_of_pos[order].reshape(E, r))
        s.X = rng.standard_normal((s.T, D), dtype=np.float32)
        s.X[:, 0] = x0                                                # carries the density offset of gate/up rows
        s.DY = (rng.standard_normal((s.T, D), dtype=np.float32) * DY_SCALE).astype(np.float32)


def make_W0(E, h, seed, sigma=SIGMA_W):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((E * 3 * h, D), dtype=np.float32) * sigma).astype(np.float32)


def with_offset(W0, E, h, rho_key, sigma=SIGMA_W):
    W = W0.copy()
    W.reshape(E, 3, h, D)[:, :2, :, 0] = MU[rho_key] * sigma * math.sqrt(D - 1)
    return W


# ----------------------------------------------------------------------------------------------- CPU path
class CpuArm:
    def __init__(s, lib, torch, name, W, E, h, lr=LR, nthr=NTHR):
        s.lib, s.torch, s.name, s.E, s.h, s.nthr = lib, torch, name, E, h, nthr
        _, s.fmt, s.opt, s.fused, _ = M.ARMS[name]
        nrows = E * 3 * h
        s.W = s.Wb = s.codes = s.sc = s.S1 = s.S2 = s.G = None
        if s.fmt == "bf16":
            s.Wb = np.empty((nrows, D), np.uint16); lib.f32_to_bf16_all(ptr(W), ptr(s.Wb), W.size, nthr)
        else:
            s.W = W.copy()
        if s.opt == "rowada":
            s.S1 = np.zeros(nrows, np.float32)
        elif s.opt == "adam":
            s.S1 = np.zeros((nrows, D), np.float32); s.S2 = np.zeros((nrows, D), np.float32)
        if not s.fused:
            s.G = np.zeros((nrows, D), np.float32)
        if s.fmt == "tern":
            s.codes = np.zeros((nrows, D // 2), np.uint8); s.sc = np.zeros(nrows, np.float32)
        s.bank = Bank(*[ptr(a) for a in (s.W, s.Wb, s.codes, s.sc, s.S1, s.S2, s.G)])
        s.ctx = Ctx(E=E, h=h, D=D, r=1, T=1, nthr=nthr, lr=lr, eps=EPS, b1=B1, b2=B2, step=0, seed=1)
        if s.fmt == "tern":
            lib.tern_init(ctypes.byref(s.ctx), ctypes.byref(s.bank), 1)

    def set_problem(s, prob):
        c, T, t = s.ctx, prob.T, s.torch
        s.prob = prob
        s.Y = np.zeros((T, D), np.float32); s.DX = np.zeros((T, D), np.float32)
        s.acc = np.zeros(s.nthr * T * D, np.float32); s.GU = np.zeros(s.E * prob.r * s.h * 2, np.float32)
        s.XQ = s.XSA = s.LUT = None
        if s.fmt == "tern":
            s.XQ = np.zeros((T, D), np.int8); s.XSA = np.zeros(T, np.float32); s.LUT = np.zeros(T * (D // 2) * 16, np.int8)
        c.r, c.T = prob.r, T
        c.tok, c.X, c.DY, c.Y, c.DX, c.acc, c.GU = (ptr(prob.tok), ptr(prob.X), ptr(prob.DY), ptr(s.Y), ptr(s.DX),
                                                    ptr(s.acc), ptr(s.GU))
        c.XQ, c.XSA, c.LUT = ptr(s.XQ), ptr(s.XSA), ptr(s.LUT)
        # the activations cross PCIe on this path: x and dy to the host, y and dx back, bf16, T x D each
        bf = t.bfloat16
        s.ag = [t.empty((T, D), dtype=bf, device="cuda") for _ in range(4)]
        s.ah = [t.empty((T, D), dtype=bf, pin_memory=True) for _ in range(4)]

    def step(s):
        """t = the forward and backward kernels over this bank slice + f_fixed x the per-layer fixed costs (activation
        transfers over PCIe, zero and reduce of the per-thread accumulators), which a layer pays once per step whatever
        the size of its bank: the slice is charged its share of the layer's routed slots (Problem.f_fixed)."""
        lib, c, t = s.lib, s.ctx, s.torch
        pc, ff = ctypes.byref(c), time.perf_counter
        c.step += 1; c.seed = (c.seed * 1103515245 + 12345) & 0xFFFFFFFF; c.n_rows_upd = 0
        t0 = ff(); s.ah[0].copy_(s.ag[0])                                       # x: GPU -> host
        t1 = ff(); lib.acc_zero(pc)
        t2 = ff(); lib.cpu_fwd(pc, FMT[s.fmt], ctypes.byref(s.bank))
        t3 = ff(); lib.acc_reduce(pc, 0)
        t4 = ff(); s.ag[1].copy_(s.ah[1]); s.ah[2].copy_(s.ag[2]); t.cuda.synchronize()   # y back, dy over
        t5 = ff(); lib.acc_zero(pc)
        t6 = ff(); lib.cpu_bwd(pc, FMT[s.fmt], OPT[s.opt], int(s.fused), ctypes.byref(s.bank))
        if not s.fused:
            lib.cpu_upd_unfused(pc, ctypes.byref(s.bank))
        t7 = ff(); lib.acc_reduce(pc, 1)
        t8 = ff(); s.ag[3].copy_(s.ah[3]); t.cuda.synchronize()                  # dx back
        t9 = ff()
        fixed = (t2 - t0) + (t4 - t3) + (t6 - t4) + (t9 - t7)
        f = s.prob.f_fixed
        n_ph = s.E * s.prob.r * s.h
        return dict(t=(t3 - t2) + (t7 - t6) + f * fixed, t_wall=t9 - t0, t_fwd=t3 - t2, t_bwd=t7 - t6,
                    t_fixed=fixed, f_fixed=f, density=c.n_active / n_ph, rows_upd=c.n_rows_upd / (s.E * 3 * s.h))

    def nonfinite(s):
        if s.fmt == "bf16":
            return s.lib.count_nonfinite_bf16(ptr(s.Wb), s.Wb.size)
        return s.lib.count_nonfinite_f32(ptr(s.W), s.W.size)

    def master(s):
        """fp32 view of the master weights (latent for TERN), for the selftest."""
        if s.fmt == "bf16":
            return (s.Wb.astype(np.uint32) << 16).view(np.float32)
        return s.W

    def release(s):
        """Drop the large buffers now (two arms of a pair are alive at once)."""
        for k in ("W", "Wb", "codes", "sc", "S1", "S2", "G", "Y", "DX", "acc", "GU", "XQ", "XSA", "LUT", "ag", "ah"):
            setattr(s, k, None)


# ----------------------------------------------------------------------------------------------- streaming path
class StreamArm:
    """Bank and optimizer state in host RAM; weights streamed to the GPU chunk by chunk in the forward sweep and
    AGAIN in the backward sweep (the bank does not fit the VRAM, so nothing is kept between the sweeps), forward and
    backward on the GPU in bf16, gradient (or, for bf16 master, the updated master) streamed back. Copy, compute and
    copy-back run on three CUDA streams over a ring of NSLOT device buffers; the CPU update of chunk c overlaps the
    GPU work on chunks c+1..c+LOOK."""
    def __init__(s, lib, torch, name, W, E, h, lr=LR, nthr=NTHR, plant=False):
        t = torch
        s.lib, s.t, s.name, s.E, s.h, s.nthr, s.plant, s.lr = lib, torch, name, E, h, nthr, plant, lr
        _, s.fmt, s.opt, _, _ = M.ARMS[name]
        s.nrows = nrows = E * 3 * h
        s.W = s.codes_np = s.sc_np = s.S1 = s.S2 = None
        bf = t.bfloat16
        if s.fmt == "f32":
            s.W = W.copy()
            s.stg = t.empty((nrows, D), dtype=bf, pin_memory=True)
            lib.f32_to_bf16_all(ptr(s.W), s.stg.data_ptr(), s.W.size, nthr)
            s.S1 = np.zeros(nrows, np.float32) if s.opt == "rowada" else np.zeros((nrows, D), np.float32)
            if s.opt == "adam":
                s.S2 = np.zeros((nrows, D), np.float32)
        elif s.fmt == "bf16":
            s.stg = t.empty((nrows, D), dtype=bf, pin_memory=True)    # the master itself
            lib.f32_to_bf16_all(ptr(W), s.stg.data_ptr(), W.size, nthr)
            s.st_h = t.zeros(nrows, dtype=t.float32, pin_memory=True)
        else:
            s.W = W.copy()
            s.codes_h = t.empty((nrows, D // 2), dtype=t.uint8, pin_memory=True)
            s.sc_h = t.empty(nrows, dtype=t.float32, pin_memory=True)
            s.S1 = np.zeros(nrows, np.float32)
        s.grad_h = t.empty((nrows, D), dtype=bf, pin_memory=True)
        s.flag_h = t.empty(nrows, dtype=t.uint8, pin_memory=True)
        cptr = s.codes_h.data_ptr() if s.fmt == "tern" else None
        scptr = s.sc_h.data_ptr() if s.fmt == "tern" else None
        s.bank = Bank(ptr(s.W), None, cptr, scptr, ptr(s.S1), ptr(s.S2), None)
        if s.fmt == "tern":
            lib.tern_init_rowmajor(nrows, D, ctypes.byref(s.bank), nthr)
        s.ctx = Ctx(E=E, h=h, D=D, r=1, T=1, nthr=nthr, lr=lr, eps=EPS, b1=B1, b2=B2, step=0, seed=7)
        s.h2d, s.cmp, s.d2h = t.cuda.Stream(), t.cuda.Stream(), t.cuda.Stream()
        s.tv = t.tensor([[a, b] for a in (-1, 0, 1) for b in (-1, 0, 1)], dtype=bf, device="cuda")

    def set_problem(s, prob, chunk_experts=None):
        t, h, E = s.t, s.h, s.E
        s.prob, s.r = prob, prob.r
        s.Ec = chunk_experts or max(1, CHUNK_BYTES // (D * 2 * max(3 * h, prob.r)))
        s.Ec = min(s.Ec, E)
        s.nch = -(-E // s.Ec)
        rows = s.Ec * 3 * h
        bf = t.bfloat16
        s.X_g = t.from_numpy(prob.X).to("cuda").to(bf)
        s.DY_g = t.from_numpy(prob.DY).to("cuda").to(bf)
        s.Y_g = t.zeros((prob.T, D), dtype=t.float32, device="cuda")
        s.DX_g = t.zeros((prob.T, D), dtype=t.float32, device="cuda")
        s.tok_g = t.from_numpy(prob.tok).to("cuda").long()
        s.GU_g = t.zeros((E, prob.r, h, 2), dtype=bf, device="cuda")
        if s.fmt == "tern":
            s.wbuf = [t.empty((rows, D // 2), dtype=t.uint8, device="cuda") for _ in range(NSLOT)]
            s.scbuf = [t.empty(rows, dtype=t.float32, device="cuda") for _ in range(NSLOT)]
        else:
            s.wbuf = [t.empty((rows, D), dtype=bf, device="cuda") for _ in range(NSLOT)]
        if s.fmt == "bf16":
            s.stbuf = [t.empty(rows, dtype=t.float32, device="cuda") for _ in range(NSLOT)]
        s.gbuf = [t.zeros((rows, D), dtype=bf, device="cuda") for _ in range(NSLOT)]
        s.fbuf = [t.zeros(rows, dtype=t.uint8, device="cuda") for _ in range(NSLOT)]
        s.free = [t.cuda.Event() for _ in range(NSLOT)]
        s.ev_in = [t.cuda.Event() for _ in range(NSLOT)]
        s.ev_c = [t.cuda.Event() for _ in range(NSLOT)]
        for e in s.free:
            e.record(t.cuda.current_stream())
        t.cuda.synchronize()

    # -- device-side expert math on one chunk (experts e0..e1)
    def _weights(s, k, n, ne):
        t = s.t
        if s.fmt == "tern":
            w = s.tv[s.wbuf[k][:n].long()].reshape(n, D) * s.scbuf[k][:n, None]
            w = w.to(t.bfloat16)
        else:
            w = s.wbuf[k][:n]
        return w.reshape(ne, 3 * s.h, D)

    def _fwd(s, Wc, e0, e1):
        t, h = s.t, s.h
        tk = s.tok_g[e0:e1]
        xs = s.X_g[tk]                                                    # (ne, r, D)
        g = t.bmm(xs, Wc[:, :h].transpose(1, 2)); u = t.bmm(xs, Wc[:, h:2 * h].transpose(1, 2))
        s.GU_g[e0:e1] = t.stack([g, u], -1)
        hv = t.relu(g) * t.relu(u)
        y = t.bmm(hv, Wc[:, 2 * h:])                                      # (ne, r, D)
        s.Y_g.index_add_(0, tk.reshape(-1), y.reshape(-1, D).float())

    def _bwd(s, Wc, e0, e1, k, n):
        t, h = s.t, s.h
        tk = s.tok_g[e0:e1]
        xs, dys = s.X_g[tk], s.DY_g[tk]
        g, u = s.GU_g[e0:e1, ..., 0], s.GU_g[e0:e1, ..., 1]
        m = ((g > 0) & (u > 0)).to(g.dtype)
        dh = t.bmm(dys, Wc[:, 2 * h:].transpose(1, 2))                    # (ne, r, h)
        dg, du, hv = dh * u * m, dh * g * m, g * u * m
        dx = t.bmm(dg, Wc[:, :h]) + t.bmm(du, Wc[:, h:2 * h])
        s.DX_g.index_add_(0, tk.reshape(-1), dx.reshape(-1, D).float())
        gw = t.cat([t.bmm(dg.transpose(1, 2), xs), t.bmm(du.transpose(1, 2), xs), t.bmm(hv.transpose(1, 2), dys)], 1)
        gw = gw.reshape(n, D)
        s.fbuf[k][:n].copy_((gw != 0).any(-1))
        if s.fmt != "bf16":
            s.gbuf[k][:n].copy_(gw)
            return
        # bf16 master: row-wise Adagrad on the GPU, stochastic rounding back to bf16
        gf = gw.float()
        st = s.stbuf[k][:n]
        st += (gf * gf).mean(-1)
        w = s.wbuf[k][:n].float() - (s.lr / (st.sqrt() + EPS))[:, None] * gf
        bits = w.view(t.int32) + t.randint(0, 65536, w.shape, dtype=t.int32, device="cuda")
        s.wbuf[k][:n].copy_((bits & -65536).view(t.float32).to(t.bfloat16))

    def _h2d(s, k, R0, n):
        if s.fmt == "tern":
            s.wbuf[k][:n].copy_(s.codes_h[R0:R0 + n], non_blocking=True)
            s.scbuf[k][:n].copy_(s.sc_h[R0:R0 + n], non_blocking=True)
        else:
            s.wbuf[k][:n].copy_(s.stg[R0:R0 + n], non_blocking=True)

    def _cpu_update(s, c):
        e0 = c * s.Ec; e1 = min(s.E, e0 + s.Ec); R0 = e0 * 3 * s.h; n = (e1 - e0) * 3 * s.h
        stg = s.stg.data_ptr() if s.fmt == "f32" else None
        s.lib.str_update(FMT[s.fmt], OPT[s.opt], R0, n, D, ctypes.byref(s.ctx), ctypes.byref(s.bank),
                         s.grad_h.data_ptr() + R0 * D * 2, s.flag_h.data_ptr() + R0, stg)

    def step(s):
        t, h = s.t, s.h
        c_ = s.ctx; c_.step += 1; c_.n_rows_upd = 0
        cpu_upd = (s.fmt != "bf16") and not s.plant
        t.cuda.synchronize(); t0 = time.perf_counter()
        s.Y_g.zero_(); s.DX_g.zero_()
        busy_f, busy_b = [], []                                             # GPU-busy time per sweep (review obj. 2)
        for c in range(s.nch):                                              # ---- forward sweep
            k = c % NSLOT; e0 = c * s.Ec; e1 = min(s.E, e0 + s.Ec); R0 = e0 * 3 * h; n = (e1 - e0) * 3 * h
            s.h2d.wait_event(s.free[k])
            with t.cuda.stream(s.h2d):
                s._h2d(k, R0, n); s.ev_in[k].record(s.h2d)
            if s.plant:
                s.free[k].record(s.h2d); continue
            with t.cuda.stream(s.cmp):
                s.cmp.wait_event(s.ev_in[k])
                ea, eb = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
                ea.record(s.cmp); s._fwd(s._weights(k, n, e1 - e0), e0, e1); eb.record(s.cmp)
                busy_f.append((ea, eb))
                s.free[k].record(s.cmp)
        t.cuda.synchronize(); t1 = time.perf_counter()
        ev_d = [None] * s.nch; cursor = 0
        for c in range(s.nch):                                              # ---- backward sweep
            k = c % NSLOT; e0 = c * s.Ec; e1 = min(s.E, e0 + s.Ec); R0 = e0 * 3 * h; n = (e1 - e0) * 3 * h
            s.h2d.wait_event(s.free[k])
            with t.cuda.stream(s.h2d):
                s._h2d(k, R0, n)
                if s.fmt == "bf16":
                    s.stbuf[k][:n].copy_(s.st_h[R0:R0 + n], non_blocking=True)
                s.ev_in[k].record(s.h2d)
            with t.cuda.stream(s.cmp):
                s.cmp.wait_event(s.ev_in[k])
                if not s.plant:
                    ea, eb = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
                    ea.record(s.cmp); s._bwd(s._weights(k, n, e1 - e0), e0, e1, k, n); eb.record(s.cmp)
                    busy_b.append((ea, eb))
                s.ev_c[k].record(s.cmp)
            with t.cuda.stream(s.d2h):
                s.d2h.wait_event(s.ev_c[k])
                if s.fmt == "bf16" and not s.plant:
                    s.stg[R0:R0 + n].copy_(s.wbuf[k][:n], non_blocking=True)
                    s.st_h[R0:R0 + n].copy_(s.stbuf[k][:n], non_blocking=True)
                else:
                    s.grad_h[R0:R0 + n].copy_(s.gbuf[k][:n], non_blocking=True)
                    s.flag_h[R0:R0 + n].copy_(s.fbuf[k][:n], non_blocking=True)
                ev = t.cuda.Event(); ev.record(s.d2h); ev_d[c] = ev
                s.free[k].record(s.d2h)
            if cpu_upd:
                while cursor <= c - LOOK:
                    ev_d[cursor].synchronize(); s._cpu_update(cursor); cursor += 1
        if cpu_upd:
            while cursor < s.nch:
                ev_d[cursor].synchronize(); s._cpu_update(cursor); cursor += 1
        t.cuda.synchronize(); t2 = time.perf_counter()
        gf = sum(a.elapsed_time(b) for a, b in busy_f) / 1e3
        gb = sum(a.elapsed_time(b) for a, b in busy_b) / 1e3
        return dict(t=t2 - t0, t_wall=t2 - t0, t_fwd=t1 - t0, t_bwd=t2 - t1, t_fixed=0.0, gpu_busy_fwd=gf,
                    gpu_busy_bwd=gb, rows_upd=c_.n_rows_upd / s.nrows, nch=s.nch)

    def density(s):
        g, u = s.GU_g[..., 0], s.GU_g[..., 1]
        return float(((g > 0) & (u > 0)).float().mean())

    def nonfinite(s):
        if s.fmt == "bf16":
            return int((~s.t.isfinite(s.stg.float())).sum())
        return s.lib.count_nonfinite_f32(ptr(s.W), s.W.size)

    def master(s):
        if s.fmt == "bf16":
            return s.stg.float().numpy()
        return s.W

    def release(s):
        for k in ("W", "S1", "S2", "stg", "st_h", "codes_h", "sc_h", "grad_h", "flag_h", "X_g", "DY_g", "Y_g", "DX_g",
                  "tok_g", "GU_g", "wbuf", "scbuf", "stbuf", "gbuf", "fbuf"):
            if hasattr(s, k):
                setattr(s, k, None)


def make_arm(lib, torch, name, W, E, h, **kw):
    cls = CpuArm if M.ARMS[name][0] == "cpu" else StreamArm
    return cls(lib, torch, name, W, E, h, **kw)


# ----------------------------------------------------------------------------------------------- controls
def ctl_ram(lib, mb, reps, nthr=NTHR):
    nbytes = mb << 20; n = nbytes // 4
    p = lib.xalloc(nbytes); lib.k_write_nt(p, n, 1.0, nthr)
    out = {}
    tests = (("read", lambda: lib.k_read(p, n, nthr), 1), ("write_nt", lambda: lib.k_write_nt(p, n, 1.0, nthr), 1),
             ("write", lambda: lib.k_write(p, n, 1.0, nthr), 1), ("rmw", lambda: lib.k_rmw(p, n, 1.0, nthr), 2))
    for name, fn, mult in tests:
        ts = []
        for i in range(reps + 1):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        g = [mult * nbytes / x / 1e9 for x in ts[1:]]
        out[name] = summ(g)
    lib.xfree(p)
    return out


def ctl_pcie(torch, mb, reps):
    t = torch; nb = mb << 20
    h1 = t.empty(nb, dtype=t.uint8, pin_memory=True); h2 = t.empty(nb, dtype=t.uint8, pin_memory=True)
    d1 = t.empty(nb, dtype=t.uint8, device="cuda"); d2 = t.empty(nb, dtype=t.uint8, device="cuda")
    hp = t.empty(nb, dtype=t.uint8)
    s1, s2 = t.cuda.Stream(), t.cuda.Stream()

    def tm(fn):
        ts = []
        for i in range(reps + 1):
            t.cuda.synchronize(); t0 = time.perf_counter(); fn(); t.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return [nb / x / 1e9 for x in ts[1:]]

    def bidir():
        with t.cuda.stream(s1):
            d1.copy_(h1, non_blocking=True)
        with t.cuda.stream(s2):
            h2.copy_(d2, non_blocking=True)
    out = dict(h2d=summ(tm(lambda: d1.copy_(h1, non_blocking=True))),
               d2h=summ(tm(lambda: h1.copy_(d1, non_blocking=True))),
               bidir_per_dir=summ(tm(bidir)),
               h2d_pageable=summ(tm(lambda: d1.copy_(hp))))
    out["link"] = pcie_link_under_load(torch, h1, d1)
    return out


def pcie_link_under_load(torch, h, d):
    """The link trains down at idle: read the generation and width while a copy loop is running."""
    import threading
    stop = [False]

    def loop():
        while not stop[0]:
            d.copy_(h, non_blocking=True); torch.cuda.synchronize()
    th = threading.Thread(target=loop); th.start()
    time.sleep(0.5)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current,"
                            "pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        q = f"nvidia-smi failed: {ex}"
    stop[0] = True; th.join()
    return q


def ctl_fma(lib, nthr):
    lib.fma_plant(1000000, nthr)
    t0 = time.perf_counter(); lib.fma_plant(20000000, nthr); dt = time.perf_counter() - t0
    iters = max(1000000, int(20000000 * 1.0 / max(dt, 1e-3)))
    t0 = time.perf_counter(); lib.fma_plant(iters, nthr); dt = time.perf_counter() - t0
    gflops = iters * 10 * 8 * 2 * nthr / dt / 1e9
    return dict(gflops=gflops, ghz_implied=gflops / (nthr * 32.0), threads=nthr, seconds=dt)


def ctl_gpu_mm(torch, n=4096, reps=20):
    t = torch
    a = t.randn(n, n, dtype=t.bfloat16, device="cuda"); b = t.randn(n, n, dtype=t.bfloat16, device="cuda")
    for _ in range(3):
        a @ b
    t.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    t.cuda.synchronize(); dt = time.perf_counter() - t0
    return dict(tflops_bf16=2 * n ** 3 * reps / dt / 1e12)


def cpu_busy(sec=3.0):
    if not hasattr(ctypes, "windll"):
        return float("nan")
    k32 = ctypes.windll.kernel32
    FT = ctypes.c_ulonglong

    def snap():
        i, k, u = FT(), FT(), FT(); k32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
        return i.value, k.value, u.value
    a = snap(); time.sleep(sec); b = snap()
    idle = b[0] - a[0]; tot = (b[1] - a[1]) + (b[2] - a[2])
    return 1.0 - idle / tot if tot else float("nan")


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        return f"nvidia-smi failed: {ex}"


def summ(xs):
    xs = sorted(xs)
    return dict(med=statistics.median(xs), min=xs[0], max=xs[-1], n=len(xs))


# ----------------------------------------------------------------------------------------------- plants
def plant_cpu(lib, torch, W0, E, h, prob, ram, reps):
    """Review obj. 7: a REAL arm kernel at a known roofline. C2 (fp32, fused SGD) with no unit ever active: the forward
    must read the gate and up rows once (8/3 B/param) and nothing else of the bank, the backward touches no weight.
    Through the same step() and timing as the arms; expected from the RAM read control."""
    arm = CpuArm(lib, torch, "C2", with_offset(W0, E, h, "zero"), E, h)
    arm.set_problem(prob)
    res = [arm.step() for _ in range(reps + 1)][1:]
    arm.release()
    meas = statistics.median(x["t"] for x in res)
    expect = E * 3 * h * D * (8.0 / 3.0) / (ram["read"]["med"] * 1e9)
    return dict(t_med=meas, t_expected=expect, ratio=meas / expect, density=statistics.median(x["density"] for x in res))


def plant_str(lib, torch, W, E, h, prob, pcie, reps):
    """The streaming harness with compute and update removed: only the forward H2D, the backward H2D and the D2H
    of a bf16 bank. Expected from the PCIe control (backward: both directions at the bidirectional rate)."""
    arm = StreamArm(lib, torch, "S2", W, E, h, plant=True)
    arm.set_problem(prob)
    ts = [arm.step()["t"] for _ in range(reps + 1)][1:]
    arm.release()
    wb = E * 3 * h * D * 2
    expect = wb / (pcie["h2d"]["med"] * 1e9) + wb / (pcie["bidir_per_dir"]["med"] * 1e9)
    meas = statistics.median(ts)
    return dict(t_med=meas, t_expected=expect, ratio=meas / expect, nch=arm.nch)


# ----------------------------------------------------------------------------------------------- selftest
def ref_step(W, tok, X, DY, h, fwd_x=None, wgt_x=None, gu=None):
    """fp64 reference of one expert-bank forward+backward. W: (E,3h,D) effective forward weights. Returns
    Y, DX and the weight gradient (E,3h,D). fwd_x / wgt_x: the x used in the gate/up product and in their weight
    gradient (default X). gu: (g, u) computed elsewhere (the ternary path, where the integer dot is often exactly 0
    and only an exact integer product reproduces the engine's g > 0 decision)."""
    E, r = tok.shape; T = X.shape[0]
    Xf = (X if fwd_x is None else fwd_x).astype(np.float64); Xw = (X if wgt_x is None else wgt_x).astype(np.float64)
    W = W.astype(np.float64); dyv = DY.astype(np.float64)
    Wg, Wu, Wd = W[:, :h], W[:, h:2 * h], W[:, 2 * h:]
    xs, xw, dys = Xf[tok], Xw[tok], dyv[tok]
    if gu is None:
        g = np.einsum("erd,ehd->erh", xs, Wg); u = np.einsum("erd,ehd->erh", xs, Wu)
    else:
        g, u = gu
    m = (g > 0) & (u > 0)
    hv = np.where(m, g * u, 0.0)
    Y = np.zeros((T, D)); np.add.at(Y, tok.ravel(), np.einsum("erh,ehd->erd", hv, Wd).reshape(-1, D))
    dh = np.einsum("erd,ehd->erh", dys, Wd)
    dg, du = np.where(m, dh * u, 0.0), np.where(m, dh * g, 0.0)
    DXr = np.zeros((T, D)); np.add.at(DXr, tok.ravel(), (np.einsum("erh,ehd->erd", dg, Wg) + np.einsum("erh,ehd->erd", du, Wu)).reshape(-1, D))
    G = np.concatenate([np.einsum("erh,erd->ehd", dg, xw), np.einsum("erh,erd->ehd", du, xw), np.einsum("erh,erd->ehd", hv, dys)], 1)
    return Y, DXr, G


def ref_opt(W, G, opt, lr, st=None):
    """fp64 optimizer step; st (a dict) carries the state across steps (row-wise Adagrad sum, Adam moments, step)."""
    st = {} if st is None else st
    if opt == "sgd":
        return W - lr * G
    if opt == "rowada":
        st["s"] = st.get("s", 0.0) + (G * G).mean(-1, keepdims=True)
        return W - lr * G / (np.sqrt(st["s"]) + EPS)
    t = st["t"] = st.get("t", 0) + 1
    st["m"] = B1 * st.get("m", 0.0) + (1 - B1) * G; st["v"] = B2 * st.get("v", 0.0) + (1 - B2) * G * G
    return W - lr * (st["m"] / (1 - B1 ** t)) / (np.sqrt(st["v"] / (1 - B2 ** t)) + EPS)


def bf16_round(a):
    b = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return (b.astype(np.uint32) << 16).view(np.float32)


def bf16_bits(a):
    b = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def decode_codes(codes, E, h, tile):
    """engine packing c = (w0+1)*3+(w1+1); gate/up blocks tile-major [t][i] when tile, everything else row-major."""
    Dp = D // 2
    blk = codes.reshape(E, 3 * h * Dp)
    q = np.empty((E, 3, h, D), np.float32)
    for qq in range(3):
        part = blk[:, qq * h * Dp:(qq + 1) * h * Dp]
        part = part.reshape(E, Dp, h).transpose(0, 2, 1) if (qq < 2 and tile) else part.reshape(E, h, Dp)
        c = part.astype(np.int64)
        q[:, qq, :, 0::2] = c // 3 - 1
        q[:, qq, :, 1::2] = c % 3 - 1
    return q.reshape(E * 3 * h, D)


def quant_ref(X):
    amax = np.abs(X).max(1)
    sa = (amax / np.float32(63)).astype(np.float32)
    inv = (np.float32(1) / sa).astype(np.float32)
    xq = np.clip(np.rint((X * inv[:, None]).astype(np.float32)), -63, 63)
    return xq.astype(np.float32), sa


def tern_ref(W):
    s = np.maximum(np.abs(W).mean(-1, keepdims=True, dtype=np.float32), np.float32(1e-5)).astype(np.float32)
    q = np.clip(np.rint((W / s).astype(np.float32)), -1, 1)
    return q, s


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def selftest(lib, torch):
    ok = True

    def check(name, val, tol):
        nonlocal ok
        good = val <= tol
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:<46} {val:.3e}  (tol {tol:.0e})")

    print("SELFTEST")
    for hh in (32, 128):
        bad = lib.lut_selftest(hh, D, 12345)
        check(f"LUT kernel vs scalar int, h={hh} (mismatches)", float(bad), 0)
    lr = 1e-2
    for h in (128, 32, 8, 1):
        E = 7                                             # with 2 experts per streaming chunk the last chunk is partial
        for r in (1, 2, 3, 4, 9):
            # r >= 4: tokens drawn from a pool, as in the run (Problem with T); r < 4: the balanced k = 2 routing
            prob = Problem(h, r, E, seed=100 + h + r, T=64) if r >= 4 else Problem(h, r, E, seed=100 + h + r, k=2)
            W0 = make_W0(E, h, seed=7 + h, sigma=0.05)
            W = with_offset(W0, E, h, "mid", sigma=0.05)
            W3 = W.reshape(E, 3 * h, D)
            # ---- CPU arms
            for name in ("C1", "C2", "C3", "C4", "C5", "C6"):
                _, fmt, opt, fused, _ = M.ARMS[name]
                arm = CpuArm(lib, torch, name, W, E, h, lr=lr)
                arm.set_problem(prob)
                if fmt == "tern":
                    q, s = tern_ref(W)
                    weff = (q * s).reshape(E, 3 * h, D)
                    xq, sa = quant_ref(prob.X)
                    xd = xq * sa[:, None]
                    q3, s3 = q.reshape(E, 3 * h, D), s.reshape(E, 3 * h).astype(np.float64)
                    xqt, sat = xq[prob.tok].astype(np.float64), sa[prob.tok].astype(np.float64)
                    Sg = np.einsum("erd,ehd->erh", xqt, q3[:, :h]); Su = np.einsum("erd,ehd->erh", xqt, q3[:, h:2 * h])
                    gu = (Sg * sat[..., None] * s3[:, None, :h], Su * sat[..., None] * s3[:, None, h:2 * h])
                    Y, DXr, G = ref_step(weff, prob.tok, prob.X, prob.DY, h, fwd_x=xd, wgt_x=xd, gu=gu)
                elif fmt == "bf16":
                    Y, DXr, G = ref_step(bf16_round(W).reshape(E, 3 * h, D), prob.tok, prob.X, prob.DY, h)
                else:
                    Y, DXr, G = ref_step(W3, prob.tok, prob.X, prob.DY, h)
                base = bf16_round(W) if fmt == "bf16" else W
                Wn = ref_opt(base.reshape(E, 3 * h, D).astype(np.float64), G, opt, lr)
                arm.step()
                check(f"{name} h={h} r={r} Y", rel(arm.Y, Y), 1e-5)
                check(f"{name} h={h} r={r} DX", rel(arm.DX, DXr), 1e-5)
                d_ref = Wn.reshape(-1, D) - base.astype(np.float64)
                d_got = arm.master().astype(np.float64) - base.astype(np.float64)
                # Adam's first step is ~G/|G|: it amplifies the fp32 rounding of gradients near eps
                tol = 1e-3 if opt == "adam" else {"f32": 1e-4, "bf16": 3e-2, "tern": 1e-4}[fmt]
                check(f"{name} h={h} r={r} update", rel(d_got, d_ref), tol)
                if fmt == "tern":
                    q2, _ = tern_ref(arm.W)
                    got = decode_codes(arm.codes, E, h, tile=(h % 32 == 0))
                    check(f"{name} h={h} r={r} repacked codes (mismatch frac)", float((got != q2).mean()), 1e-3)
                if fmt == "f32" and r in (2, 9):
                    st = {}
                    W1 = ref_opt(W3.astype(np.float64), G, opt, lr, st)
                    _, _, G1 = ref_step(W1, prob.tok, prob.X, prob.DY, h)
                    W2 = ref_opt(W1, G1, opt, lr, st)
                    arm.step()
                    check(f"{name} h={h} r={r} second update", rel(arm.W.astype(np.float64) - W1.reshape(-1, D),
                                                                   (W2 - W1).reshape(-1, D)), 1e-3)
                if arm.nonfinite():
                    check(f"{name} non-finite", 1.0, 0)
                arm.release()
            # ---- streaming arms
            for name in ("S1", "S2", "S3", "S4"):
                _, fmt, opt, _, _ = M.ARMS[name]
                arm = StreamArm(lib, torch, name, W, E, h, lr=lr)
                arm.set_problem(prob, chunk_experts=2)
                xb = bf16_round(prob.X); dyb = bf16_round(prob.DY)
                if fmt == "tern":
                    q, s = tern_ref(W)
                    weff = bf16_round((q * s).astype(np.float32)).reshape(E, 3 * h, D)
                    base = W
                elif fmt == "bf16":
                    weff = bf16_round(W).reshape(E, 3 * h, D); base = bf16_round(W)
                else:
                    weff = bf16_round(W).reshape(E, 3 * h, D); base = W
                Y, DXr, G = ref_step(weff, prob.tok, xb, dyb, h)
                Wn = ref_opt(base.reshape(E, 3 * h, D).astype(np.float64), G, opt, lr)
                arm.step()
                check(f"{name} h={h} r={r} Y", rel(arm.Y_g.cpu().numpy(), Y), 2e-2)
                check(f"{name} h={h} r={r} DX", rel(arm.DX_g.cpu().numpy(), DXr), 2e-2)
                d_ref = Wn.reshape(-1, D) - base.astype(np.float64)
                d_got = arm.master().astype(np.float64) - base.astype(np.float64)
                check(f"{name} h={h} r={r} update", rel(d_got, d_ref), 0.1 if opt == "adam" else 5e-2)
                if fmt == "f32":
                    got = arm.stg.view(torch.int16).numpy().view(np.uint16)
                    check(f"{name} h={h} r={r} staging != bf16(master) (frac)", float((got != bf16_bits(arm.W)).mean()), 0)
                if fmt == "f32" and r in (2, 9):
                    st = {}
                    W1 = ref_opt(W.reshape(E, 3 * h, D).astype(np.float64), G, opt, lr, st)
                    _, _, G1 = ref_step(bf16_round(arm.W).reshape(E, 3 * h, D), prob.tok, xb, dyb, h)
                    W2 = ref_opt(W1, G1, opt, lr, st)
                    w1 = arm.W.astype(np.float64)
                    arm.step()
                    check(f"{name} h={h} r={r} second update", rel(arm.W.astype(np.float64) - w1,
                                                                   (W2 - W1).reshape(-1, D)), 0.1 if opt == "adam" else 5e-2)
                if arm.nonfinite():
                    check(f"{name} non-finite", 1.0, 0)
                arm.release()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------------------------------- run
def t_of(E, h, r):
    """Tokens of the step this bank slice belongs to: at least T_POOL, and at least the tokens the slice alone
    implies at k = A/h; rounded up to a power of two (distinct tokens per expert, see Problem)."""
    t = max(T_POOL, -(-E * r * h // M.A))
    return 1 << (t - 1).bit_length()


def git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True).stdout.strip()


def emit(fh, rec):
    fh.write(json.dumps(rec) + "\n"); fh.flush()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true"); g.add_argument("--dry", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()

    cmd = None if a.no_build else build()
    lib = load_lib()
    import torch
    torch.set_num_threads(1)                    # torch's own CPU pool stays out of the way of the OpenMP team
    pinned = lib.pin_threads(NTHR, STRIDE)
    if a.selftest:
        sys.exit(0 if selftest(lib, torch) else 1)
    if a.run and not selftest(lib, torch):
        sys.exit("REFUSED: selftest failed")

    dry = a.dry
    p_bank = DRY["P_BANK"] if dry else P_BANK
    r_list = DRY["R_LIST"] if dry else R_LIST
    warm, reps, rounds = (DRY["WARMUP"], DRY["REPS"], DRY["ROUNDS"]) if dry else (WARMUP, REPS, ROUNDS)
    ram_mb, pcie_mb = (DRY["RAM_MB"], DRY["PCIE_MB"]) if dry else (2048, 512)

    head, upstream = git("rev-parse", "HEAD"), git("rev-parse", "@{u}")
    dirty = git("status", "--porcelain", "--", "benchmarks/sparse_capacity", "docs/research/SPARSE_CAPACITY")
    busy = cpu_busy(3.0)
    cfg = dict(mode="DRY (not a measurement)" if dry else "RUN", git_head=head, pushed=(head == upstream),
               dirty_files=dirty.splitlines(), src_sha256=sha256(SRC), dll_sha256=sha256(DLL), build=cmd,
               python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), gpu_state=gpu_state(), cpu=platform.processor(),
               nthr=NTHR, pin_stride=STRIDE, pinned_threads=pinned, omp_max=lib.omp_max(), cpu_busy_frac=busy,
               D=D, A=M.A, P_bank=p_bank, h_list=H_LIST, r_list=r_list,
               E={h: p_bank // (3 * h * D) for h in H_LIST}, arms=ARM_LIST, pairs=M.PAIRS, gate_pairs=M.GATE_PAIRS,
               gate_h=M.GATE_H, rho=M.RHO, mu_offset=MU, chunk_bytes=CHUNK_BYTES, nslot=NSLOT, look=LOOK,
               warmup=warm, reps=reps, rounds=rounds, lr=LR, eps=EPS, b1=B1, b2=B2, sigma_w=SIGMA_W, dy_scale=DY_SCALE,
               ram_ctl_mb=ram_mb, pcie_ctl_mb=pcie_mb, t_pool=T_POOL, groups=GROUPS, h1_skip=H1_SKIP,
               h1_primary_density=M.H1_PRIMARY, rho_key_by_h={h: {a: M.rho_key(a, h) for a in ARM_LIST} for h in H_LIST},
               torch_threads=torch.get_num_threads(), omp_wait_policy=os.environ.get("OMP_WAIT_POLICY"),
               x0_tern=X0_TERN)
    for k, v in cfg.items():
        print(f"CONFIG {k} = {v}")
    refuse = []
    if pinned != NTHR:
        refuse.append(f"pinned {pinned} threads, not {NTHR}")
    if not dry:
        if dirty:
            refuse.append("uncommitted changes in the apparatus")
        if head != upstream:
            refuse.append("HEAD is not pushed (protocol must be pushed before the run)")
        if "3060" not in cfg["gpu"]:
            refuse.append("not the RTX 3060 of the protocol")
        if not (busy <= 0.15):
            refuse.append(f"machine not unloaded: CPU busy {busy:.0%} > 15%")
    if refuse:
        sys.exit("REFUSED: " + "; ".join(refuse))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = a.resume or os.path.join(ROOT, "results", "sparse_capacity", ("f1b_dry_" if dry else "f1b_") + stamp)
    os.makedirs(out, exist_ok=True)
    done = set()
    cells_path = os.path.join(out, "cells.jsonl")
    if a.resume and os.path.exists(cells_path):
        for line in open(cells_path):
            rec = json.loads(line)
            if rec.get("kind") == "cell_done":
                done.add((rec["arm"], rec["h"], rec["r"], rec["round"]))
    with open(os.path.join(out, "config.json" if not a.resume else f"config_resume_{stamp}.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    fc = open(os.path.join(out, "controls.jsonl"), "a"); fh = open(cells_path, "a")

    # ---- controls (start)
    print("CONTROLS (start)")
    ram = ctl_ram(lib, ram_mb, 5); pcie = ctl_pcie(torch, pcie_mb, 5)
    fma6 = ctl_fma(lib, NTHR); fma1 = ctl_fma(lib, 1); mm = ctl_gpu_mm(torch)
    if lib.pin_threads(NTHR, STRIDE) != NTHR:                         # re-pin after the 1-thread team
        sys.exit("REFUSED: re-pinning failed")
    ctl = dict(kind="controls", when="start", ram=ram, pcie=pcie, fma6=fma6, fma1=fma1, gpu_mm=mm, cpu_busy=busy)
    emit(fc, ctl); print(json.dumps(ctl, indent=1))
    # ---- desk prediction with the measured bandwidths, BEFORE any arm is timed
    import sc1b_predict
    pred = sc1b_predict.predict(ctl)
    emit(fc, dict(kind="desk2", table=pred)); sc1b_predict.print_table(pred)
    # ---- planted arms
    h0 = H_LIST[0]; E0 = p_bank // (3 * h0 * D)
    W0 = make_W0(E0, h0, seed=1000 + h0)
    pr0 = Problem(h0, 1, E0, seed=1, T=t_of(E0, h0, 1))
    pc = plant_cpu(lib, torch, W0, E0, h0, pr0, ram, 5)
    ps = plant_str(lib, torch, W0, E0, h0, pr0, pcie, 3)
    emit(fc, dict(kind="plants", cpu=pc, str=ps)); print("PLANT cpu", pc); print("PLANT str", ps)
    del W0
    if not dry and not (0.8 <= pc["ratio"] <= 1.5 and 0.8 <= ps["ratio"] <= 1.3):
        sys.exit("REFUSED: a planted arm is off its roofline -- the harness does not measure what it claims")

    # ---- cells: per (round, h), the groups in rotated order; inside a pair, C and S alternate cell by cell
    for rnd in range(rounds):
        for h in H_LIST:
            E = p_bank // (3 * h * D)
            W0 = make_W0(E, h, seed=1000 + h)
            probs = {}
            fm = ctl_fma(lib, NTHR)                                          # throttling probe between groups
            emit(fc, dict(kind="fma_mid", round=rnd, h=h, fma6=fm))
            order = GROUPS[rnd % len(GROUPS):] + GROUPS[:rnd % len(GROUPS)]
            for grp in order:
                names = [n for n in grp if not (h == 1 and n in H1_SKIP)]
                todo = {n: [r for r in r_list if (n, h, r, rnd) not in done] for n in names}
                names = [n for n in names if todo[n]]
                if not names:
                    continue
                arms = {n: make_arm(lib, torch, n, with_offset(W0, E, h, M.rho_key(n, h)), E, h) for n in names}
                tern = M.ARMS[names[0]][1] == "tern"
                for ir, r in enumerate(r_list):
                    key = (r, tern)
                    if key not in probs:
                        x0 = X0_TERN[M.rho_key(names[0], h)] if tern else 1.0
                        probs.clear()                                    # at most one problem alive (h1, r = 128)
                        probs[key] = Problem(h, r, E, seed=10 * h + r, T=t_of(E, h, r), x0=x0)
                    seq = names if (rnd + ir) % 2 == 0 else names[::-1]
                    for name in seq:
                        if r not in todo[name]:
                            continue
                        arm = arms[name]
                        arm.set_problem(probs[key])
                        for i in range(warm + reps):
                            res = arm.step()
                            if i >= warm:
                                rec = dict(kind="sample", arm=name, h=h, r=r, round=rnd, rep=i - warm, E=E,
                                           T=probs[key].T, x0=probs[key].x0, P=p_bank, rho_key=M.rho_key(name, h),
                                           ns_per_param=res["t"] / p_bank * 1e9, **res)
                                if isinstance(arm, StreamArm):
                                    rec["density"] = arm.density()
                                emit(fh, rec)
                        nf = arm.nonfinite()
                        emit(fh, dict(kind="cell_done", arm=name, h=h, r=r, round=rnd, nonfinite=nf))
                        print(f"cell {name:4s} h={h:3d} r={r:3d} round={rnd} last={res['t'] * 1e3:9.1f} ms "
                              f"({res['t'] / p_bank * 1e9:6.3f} ns/param) nonfinite={nf}", flush=True)
                        if nf:
                            sys.exit(f"ABORT: non-finite master weights in {name} h={h} r={r} (PLAN section 3)")
                for arm in arms.values():
                    arm.release()
                del arms, arm
                torch.cuda.empty_cache()
            del W0, probs

    # ---- controls (end): drift and throttling
    print("CONTROLS (end)")
    ctl_end = dict(kind="controls", when="end", ram=ctl_ram(lib, ram_mb, 5), fma6=ctl_fma(lib, NTHR), cpu_busy=cpu_busy(3.0))
    emit(fc, ctl_end); print(json.dumps(ctl_end, indent=1))
    print(f"DONE -> {out}\nverdict: python benchmarks/sparse_capacity/sc1b_verdict.py {out}")


if __name__ == "__main__":
    main()
