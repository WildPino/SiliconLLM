"""C50 cell M (the metric): the reading, from `c50_m_eval.py`'s records and the recorded evaluations only.

On S4's 36 documents, per arm:
- R as recorded (temperature 1);
- the trivial controls: R(T0 at a temperature fit on the validation shards, by NLL or by KL to F0), and R(T0 at its
  oracle temperature on the documents themselves);
- R_cal: each arm at its NLL temperature against T0 at its own. F0 keeps temperature 1, so R_cal is an upper bound;
- R_acc on top-1 accuracy, also without the decile of documents where F0 gains most over T0; top-5 accuracy and the
  target's median rank;
- the readings that S4, C50-T1 part 2b and option N rested on, under R, R_cal and R_acc, with a paired document
  bootstrap, and the original decision rules re-applied with their margins rescaled by metric(v0) / R(v0).

    python -I benchmarks/chatbot50/c50_m_verdict.py --selftest
    python -I benchmarks/chatbot50/c50_m_verdict.py --m DIR --json OUT     (DIR holds one folder per arm)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_verdict as V  # noqa: E402

C = V.C
LN2 = math.log(2)
RECORDED = {**{k: C / "s4/verdict/evals" / k for k in ("t0", "c48", "c496", "a", "b")},
            **{k: C / "t2/n/evals" / k for k in ("v0", "e2", "n", "n_shuffle")}}
READINGS = [("S4: a - c496", "a", "c496"), ("S4: b - c48", "b", "c48"), ("T1 part 2b: v0 - T0", "v0", "t0"),
            ("N: N - v0", "n", "v0"), ("N: N - E2", "n", "e2"), ("N: E2 - v0 (the data-bound reading)", "e2", "v0"),
            ("N: N-shuffle - v0", "n_shuffle", "v0"), ("null: N-shuffle - N", "n_shuffle", "n")]
# the original primaries, re-applied with the margin rescaled by metric(v0) / R(v0): (reading names, margin)
RULES = {"S4 (a, b against their LoRAs)": (["S4: a - c496", "S4: b - c48"], 0.20, "any"),
         "N (over v0 and over E2)": (["N: N - v0", "N: N - E2"], 0.05, "all")}
TRIVIAL_SHARE = 0.25          # R(T0 at one temperature) / R(v0) at or above this: R does not decide alone
TOL_LOGP, V0_SHA, T0_SHA = 1e-3, "e490f803", "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"


def boot_acc(c_t0, c_x, c_ref, c_f0, seed=V.SEED, draws=V.DRAWS) -> dict:
    """Difference of R_acc between x and ref, with a paired document bootstrap; c_* are correct counts per document."""
    f = lambda i, c: (c[i].sum() - c_t0[i].sum()) / (c_f0[i].sum() - c_t0[i].sum())  # noqa: E731
    i0 = np.arange(len(c_t0))
    d = float(f(i0, c_x) - f(i0, c_ref))
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(draws):
        i = rng.integers(0, len(c_t0), len(c_t0))
        b.append(f(i, c_x) - f(i, c_ref))
    return {"delta": d, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def compare(r: dict, new: dict) -> str:
    """'flips': the new CI95 excludes 0 on the other side of R's estimate; 'loses CI': R's CI excluded 0 and the new
    one does not (reported as underpowered, not as a contradiction); 'gains CI'; else 'holds'."""
    excl = lambda x: x["ci95"][0] > 0 or x["ci95"][1] < 0  # noqa: E731
    if excl(new) and np.sign(new["delta"]) != np.sign(r["delta"]) and r["delta"] != 0:
        return "flips"
    if excl(r) and not excl(new):
        return "loses CI"
    if excl(new) and not excl(r):
        return "gains CI"
    return "holds"


def decide(s_fit: float, s_oracle: float) -> dict:
    """s_fit: the larger of the two validation-fit controls' shares of R(v0); s_oracle: the oracle temperature's."""
    if s_fit >= TRIVIAL_SHARE:
        return {"outcome": "R DEMOTED", "why": f"one temperature fit off the documents takes {s_fit:.2f} of v0's R"}
    if s_oracle < TRIVIAL_SHARE:
        return {"outcome": "R KEPT WITH R_acc", "why": f"even the documents' own best temperature takes only "
                                                       f"{s_oracle:.2f} of v0's R"}
    return {"outcome": "INCONCLUSIVE", "why": f"the fitted temperature takes {s_fit:.2f}, the oracle {s_oracle:.2f}: "
                                              "the control cannot decide; R stays suspect"}


def per_doc(r: np.ndarray, temp: np.ndarray, rank: np.ndarray, docs: list[int], g: int):
    bits, right, top5 = [], [], []
    for d in docs:
        s = r["doc"] == d
        bits.append(float(-temp[s, g].astype(np.float64).sum() / LN2))
        right.append(int((r["top1"][s] == r["target"][s]).sum()))
        top5.append(int((rank[s] < 5).sum()))
    return np.array(bits), np.array(right), np.array(top5)


def run(a) -> int:
    split = json.loads(V.SPLIT.read_text(encoding="utf-8"))
    docs = split["s4"]["docs"]
    blen = V.byte_lengths(Path(V.A4.TOKENIZER))
    checks, arms = {}, {}
    for k, rec_dir in RECORDED.items():
        e = json.loads((a.m / k / "eval.json").read_text(encoding="utf-8"))
        r = np.fromfile(a.m / k / "docs.pos.bin", dtype=V.A4.REC)
        temp = np.fromfile(a.m / k / "docs.temp.f32", dtype=np.float32).reshape(len(r), len(e["taus"]))
        rank = np.fromfile(a.m / k / "docs.rank.i32", dtype=np.int32)
        old = np.fromfile(rec_dir / "docs.pos.bin", dtype=V.A4.REC)
        same = len(old) == len(r) and np.array_equal(old["target"], r["target"]) and np.array_equal(old["doc"], r["doc"])
        dl = float(np.abs(old["logp_target"] - r["logp_target"]).max()) if same else None
        mode = {"n_shuffle": ("n_mode", "shuffle"), "a": ("a_mode", "full"), "b": ("bank_mode", "full")}
        f, want = mode.get(k, ("n_mode", "full"))
        checks[f"{k}_arm"] = e["arm"] == k.split("_")[0] and e[f] == want
        checks[f"{k}_gguf"] = e["gguf_sha256"].startswith(V0_SHA) if e["family"] == "t2" else e["gguf_sha256"] == T0_SHA
        checks[f"{k}_full"] = e["limit"] == 0 and e["docs"] == len(docs) and e["docs_tsv"] is None and len(rank) == len(r)
        checks[f"{k}_instrument"] = same and dl <= TOL_LOGP and e["col_1_vs_logp_target_max"] <= 1e-5
        checks[f"{k}_tau_inside_grid"] = not e["tau_at_edge"]
        taus = np.array(e["taus"])
        arms[k] = {"e": e, "r": r, "temp": temp, "rank": rank, "dl": dl, "taus": taus,
                   "g_nll": int(np.flatnonzero(taus == e["tau_nll"])[0]),
                   "g_kl": int(np.flatnonzero(taus == e["tau_kl"])[0]), "i1": int(np.flatnonzero(taus == 1.0)[0])}
    out = {"cell": "C50 cell M (the metric), part 1", "checks": checks,
           "logp_vs_recorded_max": {k: x["dl"] for k, x in arms.items()}}
    if not all(checks.values()):
        out["verdict"] = "STOP: " + ", ".join(k for k, v in checks.items() if not v)
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=1))
        return 0
    gap = V.doc_bits(V.LLAMA_T0["a4"], blen, docs)[0] - V.doc_bits(V.LLAMA_F0, blen, docs)[0]
    f0 = np.fromfile(V.LLAMA_F0 / "docs.pos.bin", dtype=V.A4.REC)
    c_f0 = np.array([int(((f0["top1"] == f0["target"]) & (f0["doc"] == d)).sum()) for d in docs])
    npos = np.array([int((arms["t0"]["r"]["doc"] == d).sum()) for d in docs])
    b1, bn, bk, cr, c5 = {}, {}, {}, {}, {}
    for k, x in arms.items():
        b1[k], cr[k], c5[k] = per_doc(x["r"], x["temp"], x["rank"], docs, x["i1"])
        bn[k] = per_doc(x["r"], x["temp"], x["rank"], docs, x["g_nll"])[0]
        bk[k] = per_doc(x["r"], x["temp"], x["rank"], docs, x["g_kl"])[0]
    # T0's oracle temperature: the grid point that minimises T0's own bits on the 36 documents
    t0 = arms["t0"]
    g_or = int(np.argmin([-t0["temp"][:, g].astype(np.float64).sum() for g in range(len(t0["taus"]))]))
    bo = per_doc(t0["r"], t0["temp"], t0["rank"], docs, g_or)[0]
    R = {k: V.recovered(b1["t0"], b1[k], gap) for k in arms}
    triv = {"nll": V.recovered(b1["t0"], bn["t0"], gap), "kl": V.recovered(b1["t0"], bk["t0"], gap),
            "oracle": V.recovered(b1["t0"], bo, gap)}
    shares = {f"{c}_over_{x}": triv[c] / R[x] for c in triv for x in ("v0", "a")}
    gap_cal = gap - (b1["t0"] - bn["t0"])
    # the decile of documents where F0 gains most over T0 in top-1 accuracy (defined on F0 - T0 only)
    gain = (c_f0 - cr["t0"]) / npos
    top = np.argsort(-gain)[:math.ceil(0.1 * len(docs))]
    rest = np.setdiff1d(np.arange(len(docs)), top)
    racc = lambda c, i: float((c[i].sum() - cr["t0"][i].sum()) / (c_f0[i].sum() - cr["t0"][i].sum()))  # noqa: E731
    table = {}
    for k, x in arms.items():
        table[k] = {"tau_nll": x["e"]["tau_nll"], "tau_kl": x["e"]["tau_kl"], "R": R[k],
                    "R_cal": V.recovered(bn["t0"], bn[k], gap_cal), "R_acc": racc(cr[k], np.arange(len(docs))),
                    "R_acc_without_top_decile": racc(cr[k], rest), "top1": float(cr[k].sum() / npos.sum()),
                    "top5": float(c5[k].sum() / npos.sum()), "median_rank": float(np.median(x["rank"])),
                    "rank0_vs_top1_mismatch": x["e"]["rank0_vs_top1_mismatch"]}
    readings = {}
    for name, x, c in READINGS:
        rr = V.delta(b1["t0"], b1[x], b1[c], gap)
        rc = V.delta(bn["t0"], bn[x], bn[c], gap_cal)
        ra = boot_acc(cr["t0"], cr[x], cr[c], c_f0)
        readings[name] = {"R": rr, "R_cal": rc, "R_acc": ra, "R_cal_vs_R": compare(rr, rc),
                          "R_acc_vs_R": compare(rr, ra)}
    rules = {}
    for rname, (names, margin, how) in RULES.items():
        rules[rname] = {}
        for m in ("R", "R_cal", "R_acc"):
            mg = margin * table["v0"][m] / table["v0"]["R"]
            ok = [readings[n][m]["delta"] >= mg and readings[n][m]["ci95"][0] > 0 for n in names]
            rules[rname][m] = {"margin": mg, "passes": any(ok) if how == "any" else all(ok)}
    out.update({"trivial_control": {"R_T0": triv, "tau_T0": {"nll": t0["e"]["tau_nll"], "kl": t0["e"]["tau_kl"],
                                                             "oracle": float(t0["taus"][g_or])}, "shares": shares},
                "top_decile_docs": [docs[i] for i in top], "arms": table, "readings": readings, "rules": rules,
                "t1_part2b_g1": "not re-applied: an absolute threshold on R (0.559); rescaled by v0 it is tautological",
                **decide(max(shares["nll_over_v0"], shares["kl_over_v0"]), shares["oracle_over_v0"])})
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("trivial_control", "outcome", "why", "top_decile_docs")}, indent=1))
    print(f"{'arm':10s} {'tau_nll':>7s} {'tau_kl':>6s} {'R':>7s} {'R_cal':>7s} {'R_acc':>7s} {'Racc-10':>7s} "
          f"{'top1':>6s} {'top5':>6s} {'rank':>5s}")
    for k, t in table.items():
        print(f"{k:10s} {t['tau_nll']:7.3f} {t['tau_kl']:6.3f} {t['R']:7.3f} {t['R_cal']:7.3f} {t['R_acc']:7.3f} "
              f"{t['R_acc_without_top_decile']:7.3f} {t['top1']:6.3f} {t['top5']:6.3f} {t['median_rank']:5.1f}")
    for name, v in readings.items():
        print(f"{name:38s} " + "  ".join(f"{m} {v[m]['delta']:+.4f} [{v[m]['ci95'][0]:+.4f}, {v[m]['ci95'][1]:+.4f}]"
                                         for m in ("R", "R_cal", "R_acc")) +
              f"  cal: {v['R_cal_vs_R']}, acc: {v['R_acc_vs_R']}")
    print(json.dumps(rules, indent=1))
    return 0


def selftest() -> int:
    ok = True
    for s_fit, s_or, want in ((0.30, 0.40, "R DEMOTED"), (0.10, 0.20, "R KEPT WITH R_acc"),
                              (0.10, 0.30, "INCONCLUSIVE")):
        got = decide(s_fit, s_or)["outcome"]
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'} shares fit {s_fit} oracle {s_or}: {got} (want {want})")
    cases = [({"delta": 0.010, "ci95": [0.008, 0.013]}, {"delta": 0.002, "ci95": [-0.001, 0.005]}, "loses CI"),
             ({"delta": 0.010, "ci95": [0.008, 0.013]}, {"delta": 0.004, "ci95": [0.001, 0.007]}, "holds"),
             ({"delta": -0.19, "ci95": [-0.25, -0.13]}, {"delta": 0.02, "ci95": [0.01, 0.03]}, "flips"),
             ({"delta": -0.002, "ci95": [-0.004, 0.0001]}, {"delta": 0.001, "ci95": [-0.002, 0.004]}, "holds"),
             ({"delta": 0.001, "ci95": [-0.002, 0.004]}, {"delta": 0.010, "ci95": [0.004, 0.016]}, "gains CI")]
    for r, new, want in cases:
        got = compare(r, new)
        ok &= got == want
        print(f"{'ok ' if got == want else 'BAD'} compare({r['delta']:+.3f} -> {new['delta']:+.3f}): {got} (want {want})")
    rng = np.random.default_rng(3)
    n = 36
    c_t0 = rng.integers(1500, 2500, n)
    c_f0 = c_t0 + rng.integers(80, 160, n)
    c_x = c_t0 + rng.integers(20, 40, n)
    d = boot_acc(c_t0, c_x, c_t0, c_f0)
    want = (c_x.sum() - c_t0.sum()) / (c_f0.sum() - c_t0.sum())
    good = abs(d["delta"] - want) < 1e-12 and d["ci95"][0] > 0
    ok &= good
    print(f"{'ok ' if good else 'BAD'} boot_acc on a planted gain: {d['delta']:+.4f} (want {want:+.4f}), ci95 {d['ci95']}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--m", type=Path)
    ap.add_argument("--json")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else run(a))
