"""Stage 2, routing by conversation: analysis and verdict, by the rule registered in S2H_ROUTING_BY_CONVERSATION_PROTOCOL_20261104.md.

Per document and MoE layer, at k = 8 and k = 4 (slots of a token: its ids with a non-zero weight):
    spread   how many of the 256 experts cover 90, 95 and 99% of the slots and of the router's mass;
    hot set  the experts most used in the first 256 tokens, at 10, 30 and 50% of the experts; the share of the later slots and
             mass it covers, and the share of tokens with every slot inside it;
    freq     the same size, by frequency over the other documents (trivial control);
    shuffle  the hot set of another document: of another category (the registered one, document d + 24) and of the same (reported);
    lru      a cache of the same size, least recently used, warmed on the first 256 tokens (reported).
Means over documents of the means over layers; CI95 by 10,000 draws of documents.
Plant: on the concatenated pairs the hot set comes from the first half's first 256 tokens; fall = coverage of tokens 256-1,023
minus coverage of tokens 1,024-2,047.  The same fall on the single documents is the baseline.
Line, at k = 4 and 30%:
    "A STABLE SUBSET"   hot-set slot coverage >= 0.97, above freq and shuffle by the CI95;
    "A LEANING SUBSET"  coverage >= freq + 0.10 and >= shuffle + 0.05, each by the CI95's lower end;
    "NO SUBSET"         otherwise.
VOID: the hot set beats shuffle by 0.03 or more, yet the plant's fall does not exceed the baseline's (CI95 above 0).

Usage: python -I benchmarks/stage2/s2r_verdict.py --selftest
       python -I benchmarks/stage2/s2r_verdict.py --work E:/SiliconLLM_Artifacts/stage2/s2r --json benchmarks/stage2/results/s2r_verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
E, WARM, DRAWS, SEED = 256, 256, 10000, 20261104
FRACS = (0.10, 0.30, 0.50)
QS = (0.90, 0.95, 0.99)
OFFSET = 24
HALF = 1024
STABLE, FREQ_GAP, SHUF_GAP, VOID_GAP = 0.97, 0.10, 0.05, 0.03
CHAT_BASE = 400000


def load(path: Path) -> dict:
    """layer -> (ids [T, n] int16, weights [T, n] float32), tokens in order."""
    b = path.read_bytes()
    off, parts = 0, {}
    while off < len(b):
        lay, t, n = np.frombuffer(b, "<i4", 3, off)
        off += 12
        ids = np.frombuffer(b, "<i2", t * n, off).reshape(t, n)
        off += 2 * t * n
        w = np.frombuffer(b, "<f4", t * n, off).reshape(t, n)
        off += 4 * t * n
        parts.setdefault(int(lay), []).append((ids, w))
    return {k: (np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])) for k, v in parts.items()}


def lengths(tsv: Path) -> tuple[list[int], list[int]]:
    """The tokens the scorer feeds: all but the last of each sequence, which is only a target."""
    ids, n = [], []
    for ln in tsv.read_text(encoding="ascii").splitlines():
        a, b = ln.split("\t")
        ids.append(int(a))
        n.append(b.count(","))
    return ids, n


def slot_counts(ids: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = w > 0
    return np.bincount(ids[m].astype(np.int64), minlength=E).astype(float), np.bincount(ids[m].astype(np.int64), weights=w[m], minlength=E)


def needed(c: np.ndarray, q: float) -> int:
    s = np.sort(c)[::-1].cumsum()
    return int(np.searchsorted(s, q * s[-1]) + 1)


def top_set(cnt: np.ndarray, mass: np.ndarray, size: int) -> np.ndarray:
    order = np.lexsort((-mass, -cnt))
    s = np.zeros(E, bool)
    s[order[:size]] = True
    return s


def coverage(s: np.ndarray, ids: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    m = w > 0
    h = s[ids.astype(np.int64)]
    slots = float((h & m).sum() / max(m.sum(), 1))
    mass = float(w[h & m].sum() / max(w[m].sum(), 1e-30))
    tokens = float(((h | ~m).all(1)).mean()) if len(ids) else float("nan")
    return slots, mass, tokens


def lru(size: int, warm: np.ndarray, ids: np.ndarray, w: np.ndarray) -> float:
    c: OrderedDict = OrderedDict((int(e), None) for e in warm)
    while len(c) > size:
        c.popitem(last=False)
    hit = tot = 0
    for row, wr in zip(ids.tolist(), (w > 0).tolist()):
        for e, on in zip(row, wr):
            if not on:
                continue
            tot += 1
            if e in c:
                hit += 1
                c.move_to_end(e)
            else:
                c[e] = None
                if len(c) > size:
                    c.popitem(last=False)
    return hit / max(tot, 1)


def boot(x: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    v = x[rng.integers(0, len(x), (DRAWS, len(x)))].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def docs_table(R: dict, starts: list[int], n: list[int], with_lru: bool = True) -> dict:
    """Arrays [doc, layer] for each measure."""
    L = sorted(R)
    D = len(n)
    out: dict = {f"spread_slots_{q}": np.zeros((D, len(L))) for q in QS} | {f"spread_mass_{q}": np.zeros((D, len(L))) for q in QS}
    arms = ("hot", "freq", "shuffle", "shuffle_same")
    for f in FRACS:
        for a in arms:
            for m in ("slots", "mass", "tokens"):
                out[f"{a}_{f}_{m}"] = np.full((D, len(L)), np.nan)
        out[f"lru_{f}_slots"] = np.full((D, len(L)), np.nan)
    for li, l in enumerate(L):
        ids, w = R[l]
        cnt = np.zeros((D, E))
        ms = np.zeros((D, E))
        wc = np.zeros((D, E))
        wm = np.zeros((D, E))
        for d in range(D):
            a, b = starts[d], starts[d] + n[d]
            cnt[d], ms[d] = slot_counts(ids[a:b], w[a:b])
            wc[d], wm[d] = slot_counts(ids[a:a + WARM], w[a:a + WARM])
            for q in QS:
                out[f"spread_slots_{q}"][d, li] = needed(cnt[d], q)
                out[f"spread_mass_{q}"][d, li] = needed(ms[d], q)
        tc, tm = cnt.sum(0), ms.sum(0)
        for d in range(D):
            a, b = starts[d] + WARM, starts[d] + n[d]
            if b <= a:
                continue
            other, same = (d + OFFSET) % D, (d // OFFSET) * OFFSET + (d + 1) % OFFSET
            for f in FRACS:
                size = int(round(f * E))
                sets = {"hot": top_set(wc[d], wm[d], size), "freq": top_set(tc - cnt[d], tm - ms[d], size),
                        "shuffle": top_set(wc[other], wm[other], size), "shuffle_same": top_set(wc[same], wm[same], size)}
                for k, s in sets.items():
                    c = coverage(s, ids[a:b], w[a:b])
                    for m, v in zip(("slots", "mass", "tokens"), c):
                        out[f"{k}_{f}_{m}"][d, li] = v
                if with_lru:
                    out[f"lru_{f}_slots"][d, li] = lru(size, np.flatnonzero(sets["hot"]), ids[a:b], w[a:b])
    return out


def plant_falls(R: dict, starts: list[int], n: list[int], f: float = 0.30) -> np.ndarray:
    """Per sequence, coverage of tokens WARM..HALF minus coverage of tokens HALF..2 * HALF, mean over layers."""
    size = int(round(f * E))
    out = np.full(len(n), np.nan)
    for d in range(len(n)):
        if n[d] < 2 * HALF - 1:
            continue
        v = []
        for ids, w in R.values():
            a = starts[d]
            s = top_set(*slot_counts(ids[a:a + WARM], w[a:a + WARM]), size)
            v.append(coverage(s, ids[a + WARM:a + HALF], w[a + WARM:a + HALF])[0] - coverage(s, ids[a + HALF:a + 2 * HALF], w[a + HALF:a + 2 * HALF])[0])
        out[d] = float(np.mean(v))
    return out


def decide(hot: np.ndarray, freq: np.ndarray, shuffle: np.ndarray, fall_pairs: np.ndarray, fall_single: np.ndarray) -> dict:
    fp, fs = fall_pairs[~np.isnan(fall_pairs)], fall_single[~np.isnan(fall_single)]
    rng = np.random.default_rng(SEED)
    diff = fp[rng.integers(0, len(fp), (DRAWS, len(fp)))].mean(1) - fs[rng.integers(0, len(fs), (DRAWS, len(fs)))].mean(1)
    plant_ci = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    gap_f, gap_s = boot(hot - freq), boot(hot - shuffle)
    out = {"hot": float(hot.mean()), "hot_ci95": boot(hot), "freq": float(freq.mean()), "shuffle": float(shuffle.mean()),
           "hot_minus_freq_ci95": gap_f, "hot_minus_shuffle_ci95": gap_s, "plant_fall": float(fp.mean()),
           "single_fall": float(fs.mean()), "plant_minus_single_ci95": plant_ci, "plant_fires": plant_ci[0] > 0}
    if float((hot - shuffle).mean()) >= VOID_GAP and not out["plant_fires"]:
        out["outcome"] = "VOID"
    elif hot.mean() >= STABLE and gap_f[0] > 0 and gap_s[0] > 0:
        out["outcome"] = "A STABLE SUBSET"
    elif gap_f[0] >= FREQ_GAP and gap_s[0] >= SHUF_GAP:
        out["outcome"] = "A LEANING SUBSET"
    else:
        out["outcome"] = "NO SUBSET"
    return out


def selftest() -> int:
    r = np.random.default_rng(1)

    def v(m):
        return np.clip(m + 0.01 * r.standard_normal(72), 0, 1)
    fall, flat = v(0.15)[:24], v(0.01)
    cases = [("stable", v(0.985), v(0.5), v(0.6), fall, flat, "A STABLE SUBSET"),
             ("leaning", v(0.75), v(0.5), v(0.6), fall, flat, "A LEANING SUBSET"),
             ("near freq", v(0.55), v(0.5), v(0.4), fall, flat, "NO SUBSET"),
             ("no gap, plant silent", v(0.55), v(0.5), v(0.55), flat[:24], flat, "NO SUBSET"),
             ("gap, plant silent", v(0.75), v(0.5), v(0.6), flat[:24], flat, "VOID")]
    for name, h, f, s, fp, fs, want in cases:
        got = decide(h, f, s, fp, fs)["outcome"]
        if got != want:
            print("FAIL", name, got)
            return 1
    # the measures on planted routing: tokens of one document keep to 64 experts
    ids = np.stack([r.choice(64, 8, replace=False) for _ in range(1200)]).astype(np.int16)
    w = np.full(ids.shape, 0.125, np.float32)
    s = top_set(*slot_counts(ids[:WARM], w[:WARM]), 77)
    c = coverage(s, ids[WARM:], w[WARM:])
    w4 = w.copy()
    w4[:, 4:] = 0
    if c[0] < 0.999 or needed(slot_counts(ids, w)[0], 0.99) > 64 or slot_counts(ids, w4)[0].sum() != 1200 * 4:
        print("FAIL measures", c)
        return 1
    if lru(77, np.flatnonzero(s), ids[WARM:], w[WARM:]) < 0.999:
        print("FAIL lru")
        return 1
    print(f"selftest PASS, {len(cases) + 1} cases")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--work", type=Path)
    ap.add_argument("--docs-tsv", type=Path, default=Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs/docs.tsv"))
    ap.add_argument("--gen-tsv", type=Path, default=Path("C:/SiliconLLM_fast/s2t/sets/gen.tsv"))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    _, n = lengths(a.docs_tsv)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(int).tolist()
    _, npairs = lengths(a.work / "pairs.tsv")
    pstarts = np.concatenate([[0], np.cumsum(npairs)[:-1]]).astype(int).tolist()
    gid, ng = lengths(a.gen_tsv)
    gstarts = np.concatenate([[0], np.cumsum(ng)[:-1]]).astype(int).tolist()
    out: dict = {"cell": "S2R-routing-by-conversation", "documents": len(n), "runs": {}}
    for k in (8, 4):
        R = load(a.work / f"docs_k{k}.route")
        if any(len(v[0]) != sum(n) for v in R.values()):
            raise SystemExit(f"docs k = {k}: {len(next(iter(R.values()))[0])} tokens, the documents hold {sum(n)}")
        T = docs_table(R, starts, n)
        rec = {name: {"mean": float(np.nanmean(v)), "ci95": boot(np.nanmean(v, 1)), "by_layer": np.nanmean(v, 0).round(4).tolist()} for name, v in T.items()}
        fs = plant_falls(R, starts, n)
        P = load(a.work / f"pairs_k{k}.route")
        fp = plant_falls(P, pstarts, npairs)
        G = load(a.work / f"gen_k{k}.route")
        keep = [i for i, d in enumerate(gid) if d // 100000 == CHAT_BASE // 100000]
        G2 = {l: (np.concatenate([v[0][gstarts[i]:gstarts[i] + ng[i]] for i in keep]), np.concatenate([v[1][gstarts[i]:gstarts[i] + ng[i]] for i in keep])) for l, v in G.items()}
        cn = [ng[i] for i in keep]
        cs_ = np.concatenate([[0], np.cumsum(cn)[:-1]]).astype(int).tolist()
        spread = {f"spread_slots_{q}": float(np.mean([needed(slot_counts(v[0][s:s + m], v[1][s:s + m])[0], q) for v in G2.values() for s, m in zip(cs_, cn)])) for q in QS}
        out["runs"][f"k{k}"] = {"measures": rec, "chat": {"items": len(cn), "tokens_mean": float(np.mean(cn)), **spread},
                                "verdict": decide(np.nanmean(T["hot_0.3_slots"], 1), np.nanmean(T["freq_0.3_slots"], 1), np.nanmean(T["shuffle_0.3_slots"], 1), fp, fs)}
        print(k, json.dumps(out["runs"][f"k{k}"]["verdict"]), flush=True)
    out["outcome"] = out["runs"]["k4"]["verdict"]["outcome"]
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("outcome", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
