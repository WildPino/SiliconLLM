#!/usr/bin/env python3
# Sparse Capacity, the new R probe (before phase 4, D-R): variable tracing in short Python programs, native to the code
# corpus of phase 2, at zero training cost. Protocol: docs/research/SPARSE_CAPACITY/RT_TRACE_PROTOCOL.md.
#
# A program is m chains of assignments, each of depth d, interleaved at random (the order inside a chain is kept):
#     q = 7          <- root of a chain: a literal
#     k = q          <- one hop
#     ...
# and a question on the last variable of one chain ("assert k == ", or the doctest ">>> k"). Format after Wu, Geiger,
# Millière (arXiv 2505.20896, var = const | var = var, query on one chain, distractor chains) and RULER's variable
# tracking (arXiv 2404.06654). All chains have the same depth, so by symmetry the answer is the first, the last, or any
# given literal with probability 1/m: the line-position heuristics that Wu et al. found in training (line-1, line-2)
# score 1/m by construction. The controls below measure that on the items themselves.
#
# Answer: one token (a literal that the BPE writes as one token in that place), scored as the greedy top-1 over the
# whole vocabulary at the answer position. Reading: the same, restricted to the program's literals (multiple choice).
#
#   python benchmarks/sparse_capacity/sc_trace.py --selftest
#   python benchmarks/sparse_capacity/sc_trace.py --tune --data-dir <m0>     (m1 s0 and s1 ONLY: the calibration)
#   python benchmarks/sparse_capacity/sc_trace.py --run --data-dir <m0>      (all ten phase-2 models, after the push)
import argparse, json, os, random, string, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc2_model as SM      # noqa: E402
import sc2_train as T       # noqa: E402

ROOT = T.ROOT
F2_CKPT = "E:/SiliconLLM_Artifacts/sparse_capacity/f2"
F2_RES = os.path.join(ROOT, "results", "sparse_capacity")
TUNE_MODELS = (("m1", 0), ("m1", 1))                     # the calibration looks at these two and nothing else
RUN_MODELS = tuple((a, s) for a in ("m1", "m4", "m16", "m64") for s in (0, 1)) + (("dense", 0), ("dense-fp", 0))
NAMES = [c for c in string.ascii_lowercase if c not in "lo"]      # one token each; no l/o (read as 1/0)
TUNE_SEED, RUN_SEED = 31, 32                             # different items for the calibration and for the run
FORMATS = ("assert", "doctest")
CONTEXTS = ("code", "demos")


# ------------------------------------------------------------------------------------------------------------ programs
def make_program(rng, m, d, pool, rebind=False):
    """m chains of depth d (d lines each: a root `v = literal`, then d-1 hops `w = v`), interleaved uniformly at random
    keeping each chain's order. rebind: after the query chain's second line, its root variable is assigned a new
    literal (the old binding is the answer: resolving names by their final value gives the trap literal)."""
    names = rng.sample(NAMES, m * d + (1 if rebind else 0))
    lits = rng.sample(pool, m + (1 if rebind else 0))
    chains = []
    for c in range(m):
        vs = names[c * d:(c + 1) * d]
        chains.append([(vs[0], lits[c])] + [(vs[i], vs[i - 1]) for i in range(1, d)])
    qc = rng.randrange(m)
    q = chains[qc][-1][0]                                                    # the end of the query chain
    if rebind:
        assert d >= 2
        ch = chains[qc]
        chains[qc] = ch[:2] + [(ch[0][0], lits[m])] + ch[2:]          # root re-assigned after its first use
    slots = [c for c in range(m) for _ in chains[c]]
    rng.shuffle(slots)
    pos = [0] * m; lines = []
    for c in slots:
        lines.append(chains[c][pos[c]]); pos[c] += 1
    return dict(lines=lines, q=q, ans=lits[qc], cands=lits[:m] + ([lits[m]] if rebind else []), m=m, d=d,
                rebind=rebind, trap=lits[m] if rebind else None)


def evaluate(lines, q):
    env = {}
    for v, rhs in lines:
        env[v] = rhs if isinstance(rhs, int) else env[rhs]
    return env[q]


def program_text(p, fmt):
    if fmt == "assert":
        return "".join(f"{v} = {r}\n" for v, r in p["lines"]) + f"assert {p['q']} == "
    return "".join(f">>> {v} = {r}\n" for v, r in p["lines"]) + f">>> {p['q']}\n"


def answer_text(p, fmt):
    return f"{p['ans']}\n"


# ------------------------------------------------------------------------------------------------- no-model controls
def ctl_first(p, rng):
    return next(r for _, r in p["lines"] if isinstance(r, int))


def ctl_last(p, rng):
    return [r for _, r in p["lines"] if isinstance(r, int)][-1]


def ctl_nearest(p, rng):
    """The last literal written before the line that defines the queried variable (a local look-back)."""
    i = max(j for j, (v, _) in enumerate(p["lines"]) if v == p["q"])
    prev = [r for _, r in p["lines"][:i + 1] if isinstance(r, int)]
    return prev[-1] if prev else ctl_first(p, rng)


def ctl_onehop(p, rng):
    """Resolve one hop by name, then guess among the literals (solves d <= 2 only)."""
    defs = {}
    for v, r in p["lines"]:
        defs[v] = r
    r = defs[p["q"]]
    if isinstance(r, int):
        return r
    r2 = defs[r]
    return r2 if isinstance(r2, int) else rng.choice(p["cands"])


def ctl_final_binding(p, rng):
    """Resolve every name by its FINAL assignment (ignores line order): the trap of the rebind cell."""
    defs = {}
    for v, r in p["lines"]:
        defs[v] = r
    x = p["q"]
    for _ in range(20):
        if isinstance(defs[x], int):
            return defs[x]
        x = defs[x]
    return rng.choice(p["cands"])


def ctl_random(p, rng):
    return rng.choice(p["cands"])


CONTROLS = dict(first=ctl_first, last=ctl_last, nearest=ctl_nearest, onehop=ctl_onehop, final_binding=ctl_final_binding,
                random=ctl_random)


# ------------------------------------------------------------------------------------------------------------- items
def single_token_pool(enc):
    """Literals the BPE writes as exactly one token after '== ' and after '\\n' (both formats), excluding 0 and 1."""
    out = []
    for n in range(2, 1000):
        a = enc(f"assert x == {n}\n"); b = enc(f"assert x == 2\n")
        c = enc(f">>> x\n{n}\n"); e = enc(f">>> x\n2\n")
        if len(a) == len(b) and len(c) == len(e):
            out.append(n)
    return out


def build_items(data, cells, n_per_cell, fmt, ctx, seed, need):
    """Items: (context tokens, answer token, program, controls). Context = code from the validation split (ctx 'code')
    or solved programs of the same cell (ctx 'demos'), always filling the window (no padding, as in phase 2)."""
    rng = random.Random(seed); crng = random.Random(seed + 1)
    enc = data.enc
    pool = single_token_pool(enc)
    val0 = data.ntr; nval = len(data.ids) - data.ntr
    items = []
    for cell in cells:
        m, d, rb = cell
        for _ in range(n_per_cell):
            p = make_program(rng, m, d, pool, rb)
            assert evaluate(p["lines"], p["q"]) == p["ans"]
            pre = enc(program_text(p, fmt)); ans = enc(answer_text(p, fmt))
            assert len(ans) == 2, ans                                           # the literal, then the newline
            if ctx == "code":
                off = val0 + rng.randrange(nval - need - 1)
                filler = [int(t) for t in data.ids[off:off + need]] + enc("\n\n")
            else:
                filler = []
                while len(filler) < need:
                    dp = make_program(rng, m, d, pool, rb)
                    filler += enc(program_text(dp, fmt) + answer_text(dp, fmt) + "\n")
            ctxt = (filler + pre)[-need:]
            items.append(dict(cell=f"m{m}_d{d}" + ("_rb" if rb else ""), ctx=ctxt, ans=ans[0],
                              cand=[enc(answer_text(dict(ans=c), fmt))[0] for c in p["cands"]],
                              ctl={k: f(p, crng) == p["ans"] for k, f in CONTROLS.items()}))
    return items, pool


@torch.no_grad()
def score(model, items, dev, mb=32):
    """Greedy top-1 over the vocabulary at the answer position, and the multiple choice among the program's literals."""
    model.eval(); top1, mc = [], []
    for i in range(0, len(items), mb):
        ch = items[i:i + mb]
        x = torch.tensor([it["ctx"] for it in ch], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            lg, _ = model(x)
        last = lg[:, -1].float()
        pred = last.argmax(-1).tolist()
        for j, it in enumerate(ch):
            top1.append(pred[j] == it["ans"])
            c = it["cand"]; mc.append(c[int(last[j, c].argmax())] == it["ans"])
    return top1, mc


def load_model(arm, seed, V, dev):
    d = os.path.join(F2_CKPT, f"f2_{arm}_s{seed}")
    path = os.path.join(d, "final_model.pt")
    want = [json.loads(l) for l in open(os.path.join(F2_RES, f"f2_{arm}_s{seed}", "train.jsonl"))]
    want = [r for r in want if r.get("kind") == "done"][-1]["final_weights_sha256"]
    if T.sha256(path) != want:
        sys.exit(f"REFUSED: {path} is not the phase-2 final model (SHA-256)")
    a = T.ARMS[arm]
    m = SM.Student(V, M=a["M"], quant=a["quant"]).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True), strict=True)
    m.eval()
    return m


def summarize(items, top1, mc):
    out = {}
    for cell in sorted({it["cell"] for it in items}):
        ix = [i for i, it in enumerate(items) if it["cell"] == cell]
        r = dict(n=len(ix), top1=float(np.mean([top1[i] for i in ix])), mc=float(np.mean([mc[i] for i in ix])))
        for k in CONTROLS:
            r["ctl_" + k] = float(np.mean([items[i]["ctl"][k] for i in ix]))
        # final_binding resolves names fully: it is the answer key on cells without rebind, a trap control only on rebind
        # cells; nearest solves d = 1 and onehop d <= 2 (local rules, like phase 2's lookahead)
        triv = [k for k in CONTROLS if k != "final_binding" or cell.endswith("_rb")]
        r["ctl_max"] = max(r["ctl_" + k] for k in triv)
        out[cell] = r
    return out


# ------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    res = []
    def check(name, ok, val=""):
        res.append(bool(ok)); print(f"  {'ok  ' if ok else 'FAIL'} {name:<72} {val}")
    rng = random.Random(0); crng = random.Random(1)
    pool = list(range(2, 32))
    for m, d, rb in ((3, 1, False), (3, 3, False), (4, 4, False), (3, 2, True)):
        ps = [make_program(rng, m, d, pool, rb) for _ in range(6000)]
        check(f"m{m} d{d}{' rb' if rb else ''}: every answer is the interpreter's", all(evaluate(p["lines"], p["q"]) == p["ans"] for p in ps))
        acc = {k: np.mean([f(p, crng) == p["ans"] for p in ps]) for k, f in CONTROLS.items()}
        if not rb:
            for k in ("first", "last"):
                check(f"m{m} d{d}: control '{k}' at 1/m by symmetry (|acc - 1/m| < 0.02)", abs(acc[k] - 1 / m) < 0.02, f"{acc[k]:.3f}")
            if d >= 3:
                check(f"m{m} d{d}: one-hop control near 1/m (< 1/m + 0.02)", acc["onehop"] < 1 / m + 0.02, f"{acc['onehop']:.3f}")
            if d == 1:
                check(f"m{m} d1: one-hop solves it (a lookup, not a chain)", acc["onehop"] == 1.0, f"{acc['onehop']:.3f}")
        else:
            check("rebind: resolving by final binding is never right (the trap)", acc["final_binding"] == 0.0, f"{acc['final_binding']:.3f}")
    ex = make_program(random.Random(5), 2, 3, pool)
    print("  example:\n" + "".join("    " + l + "\n" for l in program_text(ex, "assert").splitlines()) + f"    -> {ex['ans']}")
    print("SELFTEST (trace probe)", "PASS" if all(res) else "FAIL", f"({sum(res)}/{len(res)})")
    return all(res)


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true"); g.add_argument("--tune", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--ctx-len", type=int, default=T.SEQ)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    data = T.Data(a.data_dir, 0)
    if a.tune:
        cells = [(m, d, False) for m in (2, 3, 4) for d in (1, 2, 3, 4)] + [(3, 2, True), (3, 3, True)]
        out = dict(mode="TUNE", models=TUNE_MODELS, n_per_cell=a.n, ctx_len=a.ctx_len, seed=TUNE_SEED, res={})
        models = {k: load_model(*k, data.V, dev) for k in TUNE_MODELS}
        for fmt in FORMATS:
            for ctx in CONTEXTS:
                items, pool = build_items(data, cells, a.n, fmt, ctx, TUNE_SEED, a.ctx_len)
                out["pool"] = pool
                for (arm, s), mdl in models.items():
                    t1, mc = score(mdl, items, dev)
                    out["res"][f"{fmt}|{ctx}|{arm}_s{s}"] = summarize(items, t1, mc)
                    print(f"{fmt:8s} {ctx:6s} {arm} s{s}:", {c: (round(v['top1'], 3), round(v['ctl_max'], 3))
                                                              for c, v in out["res"][f"{fmt}|{ctx}|{arm}_s{s}"].items()},
                          flush=True)
        path = os.path.join(F2_RES, "rt_tune_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
        json.dump(out, open(path, "w"), indent=1)
        print("TUNE ->", path)
        return
    sys.exit("--run is written after the calibration, with the registered cells (protocol)")


if __name__ == "__main__":
    main()
