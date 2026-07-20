#!/usr/bin/env python3
"""WS6 decisive test: are the 4 replicas of each seed slice functionally distinct, or decoration?

The telemetry says the pool spans 8 fp32 directions (same-slice cosine 0.96, cross-slice 0.01) while the
ROUTER has learned to separate replicas (same-slice router rows anti-correlated at -0.13). Two readings fit
that evidence and they imply opposite things for the rung-1 upcycle:
  (a) the pool is 8 experts wearing 32 hats -- stage E bought parameters and routing cost, not capacity;
  (b) the replicas carry small but load-bearing deltas that the router exploits.

Ablation decides it. Replace every expert by the MEAN of its replica group (32 experts -> 8 distinct
functions, replicated exactly), keep the router and everything else untouched, and re-measure val BPB.
  * BPB barely moves  -> (a): the differentiation never happened, and the upcycle recipe must change.
  * BPB jumps          -> (b): the deltas matter and the pool is doing real work.

Control: collapse groups of the SAME SIZE but built from experts that do NOT share a seed slice. That
destroys a comparable amount of parameter detail without touching replica structure, so it separates
"averaging 4 experts hurts" from "averaging REPLICAS hurts".

Run: python benchmarks/phase64/mve/ws6_collapse.py
"""
import os, sys, json, math
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
from mve_model import MVEStudent, S0, MOE       # noqa: E402
from ws6_probe import load_final, NSLICE        # noqa: E402

DATA = os.path.join(ROOT, "kaggle_mve", "account_1", "mve_data", "data")
dev = "cuda" if torch.cuda.is_available() else "cpu"
TAG = "full"; EVAL_TOK = 200_000

meta = json.load(open(os.path.join(DATA, f"meta_{TAG}.json")))
V = meta["V_student"]
ids = np.fromfile(os.path.join(DATA, f"ts_{TAG}.u16"), dtype=np.uint16).astype(np.int64)
val = ids[meta["n_train_tok"]:]
from cartography import Bpe                      # noqa: E402
el = torch.tensor(Bpe.load(os.path.join(DATA, f"bpe{V}_ts.bin")).exp_len, dtype=torch.long)


def build(ck):
    cfg = ck["cfg"]
    m = MVEStudent(V, **S0)
    m.qat_ternary()
    m.upcycle_moe(dev_type=("cuda" if dev == "cuda" else "cpu"), seed=cfg.get("seed", 0))
    if cfg.get("recall") == "on":
        m.add_recall(lam_nce=cfg.get("lam_nce", 0.0))
    m.load_state_dict(ck["model"])
    return m.to(dev).eval()


@torch.no_grad()
def bpb(m, cap=EVAL_TOK, W=512):
    bits = 0.0; nb = 0; pos = 0; lim = min(cap, len(val) - 1)
    while pos + W + 1 <= lim:
        x = torch.from_numpy(val[pos:pos+W][None]).to(dev)
        y = torch.from_numpy(val[pos+1:pos+1+W][None]).to(dev)
        lg, _ = m(x, y)
        bits += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="sum").item() / math.log(2)
        nb += int(el[y.reshape(-1).cpu()].sum()); pos += W
    return bits / max(nb, 1)


@torch.no_grad()
def collapse(m, groups):
    """groups: list of expert-index lists. Each group is replaced by its elementwise mean."""
    hid_e = MOE["hid_e"]
    for blk in m.blocks:
        if not hasattr(blk.mlp, "Wd"): continue
        m_ = blk.mlp
        for grp in groups:
            for name in ("gate", "up"):
                W = getattr(m_, name).weight
                sl = [slice(e * hid_e, (e + 1) * hid_e) for e in grp]
                mean = torch.stack([W[s] for s in sl]).mean(0)
                for s in sl: W[s] = mean
            mean = m_.Wd[list(grp)].mean(0)
            for e in grp: m_.Wd[e] = mean
    return m


E = MOE["E"]
replica_groups = [[e for e in range(E) if e % NSLICE == g] for g in range(NSLICE)]
# control: same group sizes, but members deliberately drawn from DIFFERENT seed slices
rng = np.random.default_rng(0)
perm = rng.permutation(E)
control_groups = [list(perm[i::NSLICE]) for i in range(NSLICE)]
assert all(len(g) == E // NSLICE for g in replica_groups + control_groups)
assert all(len({e % NSLICE for e in g}) > 1 for g in control_groups), "control must mix slices"

print(f"device={dev}  E={E} groups of {E//NSLICE}  eval on {EVAL_TOK} val tokens")
print(f"  replica groups (share a seed slice): {replica_groups[0]} ...")
print(f"  control  groups (mixed slices)     : {control_groups[0]} ...\n")

for arm in ("A", "B", "C"):
    ck = load_final(arm)
    base = bpb(build(ck))
    rep = bpb(collapse(build(ck), replica_groups))
    ctl = bpb(collapse(build(ck), control_groups))
    print(f"ARM {arm} (arm={ck['cfg'].get('arm')} recall={ck['cfg'].get('recall')})")
    print(f"   baseline (32 experts)          {base:.4f}")
    print(f"   collapse REPLICAS  -> 8 fn     {rep:.4f}   ({rep-base:+.4f})")
    print(f"   collapse CONTROL   -> 8 fn     {ctl:.4f}   ({ctl-base:+.4f})")

print("\nSTOP. WS6 ablation above. No commit.")
