#!/usr/bin/env python3
"""WS4 -- MQAR diagnostic: the pre-registered detector for the recall demotion clause (D4, clause 2).

WHAT DECIDES: the difference between the recall arm and the no-recall arm on the SAME probe set.
The paired control inside each probe (below) is a variance reducer, not the verdict.

Two probe families, two different jobs -- this is not "both because we could not choose":

  code   (GATED)      Associative recall in the form the product actually uses: an identifier bound
                      earlier in the file and referenced later. Native to the domain, so no separate
                      synthetic corpus to decontaminate.
  synth  (CALIBRATION, NOT GATED)   The positive control for the DETECTOR itself. Code is redundant and
                      an identifier is often guessable from local context, so a blind probe would return
                      a null -- and a pre-registered demotion would fire on that null, killing an
                      architectural component over a measurement artefact. An instrument never seen to
                      trip is an assumption, not a measurement.

Pre-registered reading of all four cells (written before any number, so none is interpreted after):

    synth  code   reading
      +     +     recall stays in v1
      +     -     mechanism works, code does not need it -> demotion, clean and well diagnosed
      -     -     tier neither learned nor used -> demotion, FLAGGED as a training failure
                  (InfoNCE / lam_nce); the architectural question stays on the books as untested
      -     +     incoherent -> apparatus fault; investigate before reading the gate

This specifies what the gate reads. It does not soften it: a null from a calibrated instrument demotes
recall exactly as written.

DISTANCE. Every probe distance must EXCEED the attention window (win=128): inside it the single SWA
layer carries the reference and the probe measures something other than the recall tier. The gate reads
the CURVE over 128/512/2048 -- the tier should help more as distance grows, and the shape is worth more
than any single point. d=8 is also run, outside the gate, as an in-band detector sanity check: if the
harness cannot see a copy at distance 8, it is broken and no other row means anything.

Run (apparatus smoke on whatever tag is present):
    python benchmarks/phase64/mve/ws4_mqar.py --ckpt A.pt --ckpt B.pt --smoke
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from mve_model import MVEStudent, S0, MOE   # noqa: E402

DATA_DEF = os.path.join(ROOT, "results", "phase64", "mve")
GATED_D = (128, 512, 2048)      # must all exceed win=128 ... 128 is the boundary case, kept as the floor
SANITY_D = 8                    # NOT gated: detector sanity. Inside the attention window on purpose.


# ------------------------------------------------------------------ model reconstruction -------------
def build_from_ckpt(path, V, dev):
    """Replay the stage surgery the trainer applied, then load STRICT.

    Strict is the point: it is the assertion that the replay matched the run. A permissive load would
    silently give us a half-initialised model and a plausible-looking BPB, which is the exact failure
    class that produced the alpha bug."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    stages = cfg.get("stages", "CDEF")
    m = MVEStudent(V, **S0)
    if cfg.get("xproj_rank", 0) > 0:
        m.lowrank_xproj(cfg["xproj_rank"])
    if "D" in stages:
        m.qat_ternary(alpha_sched=cfg.get("qat_alpha", 0) > 0)
    if "E" in stages:
        if not cfg.get("dense_paired", False):
            m.upcycle_moe(dev_type="cpu", load_w=cfg.get("load_w", 0.01),
                          seed=cfg.get("seed", 0), sparse=cfg.get("sparse_moe", False))
        if cfg.get("recall", "off") == "on":
            m.add_recall(lam_nce=cfg.get("lam_nce", 0.0))
    m.load_state_dict(ck["model"], strict=True)
    if "D" in stages and cfg.get("qat_alpha", 0) > 0:
        m.set_qat_alpha(1.0)          # stage D is complete in any finished checkpoint; see restore_qat_alpha
    return m.to(dev).eval(), cfg


# ------------------------------------------------------------------ probe construction ---------------
def freq_table(ids, V):
    return np.bincount(ids, minlength=V).astype(np.float64)


def substitute_map(freq, rng):
    """For each token, a replacement of the closest global frequency. The paired control must differ from
    the probe ONLY in the identity of the antecedent -- if the substitute were drawn uniformly it would
    also differ in how surprising a token it is, and the delta would partly measure that instead."""
    order = np.argsort(-freq)
    pos = np.empty_like(order); pos[order] = np.arange(len(order))
    sub = np.empty(len(freq), dtype=np.int64)
    for t in range(len(freq)):
        p = pos[t]
        cands = [order[q] for q in (p - 2, p - 1, p + 1, p + 2) if 0 <= q < len(order)]
        sub[t] = int(rng.choice(cands)) if cands else t
    return sub


def code_probes(ids, freq, sub, d, n, rng, ctx_slack=256, freq_pct=60.0):
    """Probes mined from the corpus itself: a token bound earlier in the window and referenced later.

    Constraints, each of which removes a way of measuring the wrong thing:
      * the answer must be RARE (below the freq_pct percentile of the frequency mass) -- a common token
        is guessable from local syntax and its delta measures local context, not recall;
      * the antecedent is the token's LAST earlier occurrence, at distance ~d, and there must be no other
        occurrence between: one antecedent, one attribution;
      * the substitute must not itself appear in the window, or the control is not a control.
    """
    thr = np.percentile(freq[freq > 0], freq_pct)
    L = d + ctx_slack
    out = []
    tries = 0
    while len(out) < n and tries < n * 400:
        tries += 1
        q = int(rng.integers(L, len(ids) - 1))
        t = int(ids[q])
        if freq[t] > thr:
            continue
        w0 = q - L
        prev = np.nonzero(ids[w0:q] == t)[0]
        if len(prev) == 0:
            continue
        p = w0 + int(prev[-1])
        if not (d <= q - p < d + ctx_slack):
            continue
        if len(prev) > 1:
            continue                               # more than one antecedent in the window -> ambiguous attribution
        s = int(sub[t])
        if s == t or np.any(ids[w0:q] == s):
            continue
        ctx = ids[w0:q].copy()
        ctl = ctx.copy(); ctl[p - w0] = s
        out.append((ctx, ctl, t, q - p))
    return out


def synth_probes(V, freq, d, n, rng, n_pairs=16, ctx_slack=64):
    """Synthetic MQAR: key->value bindings, filler, then a query key. The answer is recoverable ONLY from
    the binding, so a null here is a statement about the model or the harness, never about the domain."""
    live = np.nonzero(freq > 0)[0]
    out = []
    for _ in range(n):
        toks = rng.choice(live, size=2 * n_pairs, replace=False)
        keys, vals = toks[:n_pairs], toks[n_pairs:]
        qi = int(rng.integers(0, n_pairs))
        body = np.empty(2 * n_pairs, dtype=np.int64)
        body[0::2], body[1::2] = keys, vals
        fill = rng.choice(live, size=max(0, d - ctx_slack))
        ctx = np.concatenate([body, fill, [keys[qi]]]).astype(np.int64)
        ctl = ctx.copy()
        ctl[2 * qi + 1] = int(rng.choice([v for v in vals if v != vals[qi]]))   # rebind, same shape
        out.append((ctx, ctl, int(vals[qi]), len(ctx) - 1 - (2 * qi + 1)))
    return out


# ------------------------------------------------------------------ scoring ---------------------------
@torch.no_grad()
def score(model, probes, dev, bs=8):
    """Delta = logP(answer | context) - logP(answer | context with the antecedent replaced).

    Both members of the pair are the same length and the answer sits at the same position, so the
    difference cancels position, length, and the local continuation -- what is left is the antecedent."""
    dl, hit = [], []
    for i in range(0, len(probes), bs):
        chunk = probes[i:i + bs]
        L = min(len(p[0]) for p in chunk)
        x = torch.tensor(np.stack([p[0][-L:] for p in chunk]), device=dev)
        c = torch.tensor(np.stack([p[1][-L:] for p in chunk]), device=dev)
        ans = torch.tensor([p[2] for p in chunk], device=dev)
        lx = F.log_softmax(model(x)[0][:, -1].float(), -1)
        lc = F.log_softmax(model(c)[0][:, -1].float(), -1)
        a = lx.gather(1, ans[:, None]).squeeze(1)
        b = lc.gather(1, ans[:, None]).squeeze(1)
        dl += (a - b).tolist()
        hit += (lx.argmax(-1) == ans).float().tolist()
    d = np.asarray(dl)
    return dict(n=len(d), delta=float(d.mean()), sem=float(d.std(ddof=1) / max(np.sqrt(len(d)), 1)),
                acc=float(np.mean(hit)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="one per arm; the gate reads the DIFFERENCE")
    ap.add_argument("--data-dir", default=DATA_DEF)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260719, help="the seed sealed in the rung-1 pre-registration")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="", help="write the table as JSON for the prereg record")
    a = ap.parse_args()
    if a.smoke: a.n = 32

    meta = json.load(open(os.path.join(a.data_dir, f"meta_{a.tag}.json")))
    V = meta["V_student"]
    ids = np.fromfile(os.path.join(a.data_dir, f"ts_{a.tag}.u16"), dtype=np.uint16).astype(np.int64)
    val = ids[meta["n_train_tok"]:]
    freq = freq_table(ids[:meta["n_train_tok"]], V)
    rng = np.random.default_rng(a.seed)
    sub = substitute_map(freq, rng)

    print(f"MQAR diagnostic   corpus={meta['corpus']}  V={V}  val={len(val)} tok  seed={a.seed}  n={a.n}/cell")
    print(f"  gated distances {GATED_D} (all > win=128) | d={SANITY_D} = detector sanity, NOT gated\n")

    dists = [SANITY_D] + list(GATED_D)
    bank = {}
    for d in dists:
        bank[d] = dict(code=code_probes(val, freq, sub, d, a.n, rng),
                       synth=synth_probes(V, freq, d, a.n, rng))

    rec = {}
    for path in a.ckpt:
        model, cfg = build_from_ckpt(path, V, a.device)
        name = os.path.basename(path)
        tag = f"arm={cfg.get('arm','?')} recall={cfg.get('recall','?')}"
        print(f"== {name}   {tag}")
        print(f"   {'d':>6}  {'family':6}  {'n':>4}  {'delta (nats)':>16}  {'top1':>6}")
        rec[name] = dict(cfg=tag, rows={})
        for d in dists:
            for fam in ("synth", "code"):
                p = bank[d][fam]
                if not p:
                    print(f"   {d:6d}  {fam:6}  {0:4d}  {'(no probes mined)':>16}")
                    continue
                r = score(model, p, a.device)
                mark = "" if d in GATED_D else "   [sanity]"
                print(f"   {d:6d}  {fam:6}  {r['n']:4d}  {r['delta']:+9.4f} +/-{r['sem']:.4f}  "
                      f"{100*r['acc']:5.1f}%{mark}")
                rec[name]["rows"][f"{fam}@{d}"] = r
        print()

    if len(rec) >= 2:
        ks = list(rec)
        print(f"== CROSS-ARM (what the gate reads): {ks[0]}  MINUS  {ks[1]}")
        print(f"   {'d':>6}  {'family':6}  {'d(delta)':>12}")
        for d in dists:
            for fam in ("synth", "code"):
                k = f"{fam}@{d}"
                if k in rec[ks[0]]["rows"] and k in rec[ks[1]]["rows"]:
                    dd = rec[ks[0]]["rows"][k]["delta"] - rec[ks[1]]["rows"][k]["delta"]
                    print(f"   {d:6d}  {fam:6}  {dd:+12.4f}{'' if d in GATED_D else '   [sanity]'}")
    elif len(rec) == 1:
        print("Only one arm given: this is apparatus exercise, NOT the gate. The gate reads the difference\n"
              "between the recall and the no-recall arm on the same probe bank.")

    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    print("\nSTOP. MQAR diagnostic above. No commit.")


if __name__ == "__main__":
    main()
