#!/usr/bin/env python3
"""WS6 -- CPU probes on the run-3 MVE checkpoints, aimed at the open D->E discontinuity.

SCOPE, stated before the numbers:
  * The stage-E pool IS available: the run-3 finals carry the trained MoE. That is the product-relevant
    read S3's own scope note asked for -- moe_gran.pt is the from-scratch WORST case, whereas the ladder
    upcycles with 4x replica seeding, which may create redundancy that the worst case cannot show.
  * A stage-D checkpoint does NOT exist. The trainer deletes the resume file on completion, so the only
    surviving states are end-of-F. S2 is therefore run on the end-of-F ternary weights -- still QAT-trained
    and still the pool the engine would stream, but NOT the stage-D state the brief named. Anything the
    ternarization does between D and F is invisible here. Fixing this costs one flag in the next run.

  Replica telemetry (the new measurement): upcycle_moe seeds expert e from dense hidden slice (e % 8), so
  {e, e+8, e+16, e+24} start as the SAME weights plus 2% noise. How far they have travelled by end-of-F
  says whether stage E produced 32 experts or 8 experts wearing 32 hats -- which bears directly on why the
  MoE rung costs +0.07 BPB and never repays it inside its step budget.

Run: python benchmarks/phase64/mve/ws6_probe.py
"""
import os, sys, glob, zipfile, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(ROOT, "kaggle_mve", "results_run3")
NSLICE = 8                      # upcycle: hid(1024) / hid_e(128) -> experts e and e+8k share a seed slice
RNG = np.random.default_rng(0)


def load_final(arm):
    """The finals came back unpacked into directories; rebuild the zip container torch.load expects."""
    d = os.path.join(RES, f"final_{arm}3")
    if os.path.isfile(d + ".pt"): return torch.load(d + ".pt", map_location="cpu", weights_only=False)
    tmp = os.path.join(tempfile.gettempdir(), f"ws6_final_{arm}3.pt")
    if os.path.isfile(tmp):                      # cache: the ablation loads each arm three times
        return torch.load(tmp, map_location="cpu", weights_only=False)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, os.path.dirname(d)).replace("\\", "/"))
    return torch.load(tmp, map_location="cpu", weights_only=False)


def ternarize(w, dim):
    """EXACT BitLinear158 / e4_export semantics: per-row absmean over `dim`, round, clamp."""
    scale = w.abs().mean(dim=dim, keepdim=True).clamp_min(1e-5)
    return (w / scale).round().clamp(-1, 1).to(torch.int8).numpy()


def H(counts):
    p = np.asarray(counts, dtype=np.float64); p = p / max(p.sum(), 1); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_bounds(t):
    """t: (rows, len) int8 trits. Returns the coding bounds in bits/weight."""
    flat = t.reshape(-1)
    h0 = H(np.bincount(flat + 1, minlength=3))
    rows = [H(np.bincount(r + 1, minlength=3)) for r in t]
    h0r = float(np.mean(rows))
    pairs = (t[:, :-1].astype(np.int32) + 1) * 3 + (t[:, 1:].astype(np.int32) + 1)
    hp = H(np.bincount(pairs.reshape(-1), minlength=9)) / 2.0
    # order-1 along the row: H(next | current)
    cur, nxt = t[:, :-1].reshape(-1) + 1, t[:, 1:].reshape(-1) + 1
    joint = np.bincount(cur * 3 + nxt, minlength=9).reshape(3, 3).astype(np.float64)
    h1 = 0.0
    for i in range(3):
        n = joint[i].sum()
        if n > 0: h1 += (n / joint.sum()) * H(joint[i])
    return dict(H0=h0, H0row=h0r, Hpair2=hp, H1=h1, zeros=float((flat == 0).mean()))


def expert_blocks(sd, li):
    """Per-expert weight blocks of layer li, as fp32 (E, ...) and engine-exact trits."""
    Wd = sd[f"blocks.{li}.mlp.Wd"]                       # (E,D,hid_e)
    g = sd[f"blocks.{li}.mlp.gate.weight"]               # (E*hid_e, D)
    u = sd[f"blocks.{li}.mlp.up.weight"]
    E = Wd.shape[0]; hid_e = Wd.shape[2]
    fp = [torch.cat([g[e*hid_e:(e+1)*hid_e].reshape(-1), u[e*hid_e:(e+1)*hid_e].reshape(-1),
                     Wd[e].reshape(-1)]) for e in range(E)]
    tr = [np.concatenate([ternarize(g[e*hid_e:(e+1)*hid_e], 1).reshape(-1),
                          ternarize(u[e*hid_e:(e+1)*hid_e], 1).reshape(-1),
                          ternarize(Wd[e], -1).reshape(-1)]) for e in range(E)]
    return torch.stack(fp), np.stack(tr), E


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    for arm in ("A", "B", "C"):
        try:
            ck = load_final(arm)
        except Exception as e:
            print(f"arm {arm}: could not load ({str(e)[:60]})"); continue
        sd = ck["model"]
        nl = sum(1 for k in sd if k.endswith("mlp.Wd"))
        print("=" * 92)
        print(f"ARM {arm}  (end of stage F; {nl} MoE layers)   arm={ck['cfg'].get('arm')} recall={ck['cfg'].get('recall')}")

        # ---- S2: how many bits/weight does the streamed pool actually need? ------------------------
        allt = []
        for li in range(nl):
            g = sd[f"blocks.{li}.mlp.gate.weight"]
            allt.append(ternarize(g, 1))
        t = np.concatenate(allt, 0)
        b = entropy_bounds(t)
        print(f"  S2 ternary entropy over the pool gate rows ({t.shape[0]} rows x {t.shape[1]}):")
        print(f"     zeros={100*b['zeros']:.1f}%  |  flat(log2 3)=1.585   H0={b['H0']:.3f}   H0/row={b['H0row']:.3f}"
              f"   Hpair/2={b['Hpair2']:.3f}   H1={b['H1']:.3f}  bits/weight")
        print(f"     vs packs: int8-pair = 4.000 b/w   |   5-trits/byte = 1.600 b/w")

        # ---- replica-divergence telemetry (the D->E-relevant read) ---------------------------------
        # Experts sharing a seed slice started identical + 2% noise. If they are still near-identical the
        # pool has 8 distinct functions, not 32, and the router is choosing among duplicates.
        win, acr, wint, acrt = [], [], [], []
        for li in range(nl):
            fp, tr, E = expert_blocks(sd, li)
            for i in range(E):
                for j in range(i + 1, E):
                    c = cos(fp[i], fp[j]); a = float((tr[i] == tr[j]).mean())
                    (win if (i % NSLICE) == (j % NSLICE) else acr).append(c)
                    (wint if (i % NSLICE) == (j % NSLICE) else acrt).append(a)
        print(f"  replica divergence (experts sharing a seed slice vs not):")
        print(f"     fp32 cosine   same-slice {np.mean(win):+.4f} +/- {np.std(win):.4f}   "
              f"cross-slice {np.mean(acr):+.4f} +/- {np.std(acr):.4f}   gap {np.mean(win)-np.mean(acr):+.4f}")
        print(f"     trit agreement same-slice {100*np.mean(wint):.2f}%   cross-slice {100*np.mean(acrt):.2f}%"
              f"   (chance for this marginal ~= {100*((1-b['zeros'])**2/2 + b['zeros']**2):.2f}%)")

        # ---- S3: pool-level nearest-neighbour redundancy, with a shuffled control -------------------
        fp0, tr0, E = expert_blocks(sd, 0)
        rows = np.concatenate([ternarize(sd[f"blocks.{li}.mlp.gate.weight"], 1) for li in range(nl)], 0)
        # The queries are a subsample, but each is searched against the WHOLE pool. Searching a subsample against
        # ITSELF was the first version and it was blind by construction: a row and its replica both landing in a
        # 2000-of-32768 draw has probability ~6%, so the search could not see the very structure the replica
        # telemetry proves is there. Query-vs-full-pool costs one 2000x32768 matmul and actually asks the question.
        qi = RNG.choice(len(rows), size=min(2000, len(rows)), replace=False)
        ctl_pool = np.stack([RNG.permutation(r) for r in rows])      # same per-row marginals, structure destroyed
        def nn_agree(Q, pool, qidx):
            sim = Q.astype(np.float32) @ pool.astype(np.float32).T
            sim[np.arange(len(Q)), qidx] = -1e9                      # never match a row with itself
            best = sim.argmax(1)
            return float(np.mean([(Q[i] == pool[best[i]]).mean() for i in range(len(Q))]))
        print(f"  S3 nearest-other-row trit agreement (query vs FULL pool, {len(rows)} rows):")
        print(f"     real {100*nn_agree(rows[qi], rows, qi):.2f}%   "
              f"shuffled control {100*nn_agree(ctl_pool[qi], ctl_pool, qi):.2f}%")

    print("\nSTOP. WS6 probes above. No commit.")


if __name__ == "__main__":
    main()
