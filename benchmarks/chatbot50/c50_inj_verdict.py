"""C50 injection cell: the verdict, from the records only (it reads every arm's outputs, chooses nothing itself).

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_INJECTION_PROTOCOL_20261027.md. Per fact: acc on the rewordings
(the mean over its valid rewordings), the original question and the cloze. Δ(arm) = mean acc(injected) - mean
acc(twin); every contrast is paired across arms on the same facts, with a bootstrap that redraws facts within each half
(10,000 draws). Arms E and L are the two seeds' per-fact mean. Arms are read at their final checkpoint (100
exposures); ck25 and ck50 are reported (the dose curve). Before any number is read, the provenance chain is checked:
inputs manifest -> each training's text and lr/lambda0 (params.json) -> each mark's weights -> each evaluation.

    python -I c50_inj_verdict.py --inputs E:/.../inj --evals E:/.../inj/evals --train E:/.../inj/train --json OUT
    python -I c50_inj_verdict.py --pick lambda --inputs E:/.../inj --mve MVE.json --params E:/.../inj/params.json
    python -I c50_inj_verdict.py --pick lr --inputs E:/.../inj --dev E:/.../inj/dev --params E:/.../inj/params.json
    python -I c50_inj_verdict.py --selftest          (decide() on planted numbers, then measure() on a planted folder)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED, DRAWS = 20261028, 10000
PRIMARY_MIN, OPEN_MIN, FIRE_CEIL, SELECTIVE, DOCS_TOL, CALIB_MAX, BAG_MAX = 0.10, 0.50, 0.15, 1.5, 0.005, 0.08, 0.05
TRAINED = ("e1", "e2", "edense", "l1", "l2", "s", "qa1")
TEXT = {"e1": "E", "e2": "E", "edense": "E", "l1": "E", "l2": "E", "s": "S", "qa1": "QA1"}
FIRED = ("e1", "e2", "s", "qa1")
MARKS = ("ck25", "ck50", "final")
SETS = ("rew", "orig", "cloze")
LRS = ("1e-4", "3e-4", "1e-3")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- statistics
def paired_ok(ref: np.ndarray, x: np.ndarray) -> dict:
    """A4's paired rule: b = ref right and x wrong, c = the reverse; not worse iff b - c <= max(2 sqrt(b + c), 1)."""
    b, c = int((ref & ~x).sum()), int((x & ~ref).sum())
    return {"b": b, "c": c, "ok": b - c <= max(2 * math.sqrt(b + c), 1)}


def delta(acc: np.ndarray, inj: np.ndarray) -> float:
    return float(np.nanmean(acc[inj]) - np.nanmean(acc[~inj]))


def _draws(ii: np.ndarray, tt: np.ndarray, draws: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    return rng.choice(ii, (draws, len(ii))), rng.choice(tt, (draws, len(tt)))


def boot_d(d: np.ndarray, inj: np.ndarray, draws: int = DRAWS) -> list[float]:
    """CI95 of mean d(injected) - mean d(twin), resampling facts within each half (the same draws for every call)."""
    Ii, It = _draws(np.flatnonzero(inj), np.flatnonzero(~inj), draws)
    with np.errstate(invalid="ignore"):
        v = np.nanmean(d[Ii], 1) - np.nanmean(d[It], 1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def contrast(a: np.ndarray, b: np.ndarray, inj: np.ndarray, draws: int = DRAWS) -> dict:
    """Δ(a) - Δ(b), paired on the same facts (a and b share their missing facts)."""
    d = a - b
    return {"value": delta(d, inj), "ci95": boot_d(d, inj, draws)}


def half_vs(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    """acc(a) - acc(b) on one half, with a fact bootstrap."""
    ix = np.flatnonzero(mask & ~np.isnan(a) & ~np.isnan(b))
    d = a[ix] - b[ix]
    bs = d[np.random.default_rng(SEED).integers(0, len(d), (DRAWS, len(d)))].mean(1)
    return {"value": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(ix))}


# ---------------------------------------------------------------- the rule
def decide(m: dict) -> dict:
    """m: the measured numbers (see `measure`). Returns the checks, each arm's pass and the outcome row."""
    chk = {
        "provenance": m["provenance_ok"],
        "trainings_complete": all(m["train_ok"].values()),
        "open_book": m["open_rew_v0"] >= OPEN_MIN,                  # the planted positive, on the primary's prompts
        "calibration": abs(m["delta"]["v0"]["value"]) <= CALIB_MAX,
        "balance": m["balance"]["ok"],                               # v0's open book, injected against twin
    }
    if not all(chk.values()):
        return {"checks": chk, "outcome": "STOP: " + ", ".join(k for k, v in chk.items() if not v)}
    passes = {arm: m["vs0"][arm]["value"] >= PRIMARY_MIN and m["vs0"][arm]["ci95"][0] > 0
              for arm in ("e", "edense", "l", "qa1", "s")}
    e_t0 = m["vs0_t0rew"]["e"]["ci95"][0] > 0                       # the co-gate: T0's rewordings alone agree
    e_pass = passes["e"] and e_t0
    shuffle_ok = m["e_vs_s"]["ci95"][0] > 0 and m["delta"]["s"]["value"] - m["delta"]["v0"]["value"] <= m["vs0"]["e"]["value"] / 2
    bag_ok = m["answer_bag_e1"] <= BAG_MAX
    fire_ok = m["fire"]["e"]["replay"] <= FIRE_CEIL and m["fire"]["e"]["min_layer"] >= 0.01
    selective = m["fire"]["e"]["inj_prompts"] >= SELECTIVE * m["fire"]["e"]["twin_prompts"]
    guards_ok = all(g["ok"] for g in m["guards"]["e"].values())
    stored = {arm: m["cloze_vs0"][arm]["value"] >= PRIMARY_MIN and m["cloze_vs0"][arm]["ci95"][0] > 0
              for arm in ("e", "edense", "l")}
    if e_pass and not (shuffle_ok and bag_ok):
        row = "the gain is not the fact (S ~ E, or the answer bag is high): redesign the text"
    elif e_pass and fire_ok and guards_ok and selective:
        row = "PASS: E stores and extracts, sparse and selective"
    elif e_pass and fire_ok and guards_ok:
        row = "PASS, NOT SELECTIVE: a sparse adapter, not memory"
    elif e_pass:
        row = "E passes the primary but fails " + ", ".join(k for k, v in (("fire", fire_ok), ("guards", guards_ok)) if not v)
    elif passes["edense"]:
        row = "E fails, E-dense passes: the fire cost stops learning"
    elif passes["l"]:
        row = "E and E-dense fail, L passes: the expert branch does not store"
    elif any(stored.values()):
        row = "all fail, cloze passes: stored, not extractable"
    else:
        row = "all fail, cloze fails: not stored at 100 exposures"
    return {"checks": chk, "passes": passes, "e_t0_rewordings": e_t0, "shuffle_ok": shuffle_ok, "bag_ok": bag_ok,
            "fire_ok": fire_ok, "selective": selective, "guards_ok": guards_ok, "stored": stored, "outcome": row}


# ---------------------------------------------------------------- reading the records
def provenance(inputs: Path, evals: Path, train: Path, params: dict) -> dict:
    """Every link from the inputs manifest to each evaluation must hold; returns the failures (empty = ok)."""
    man = read_json(inputs / "manifest.json")["files"]
    bad = []

    def ev(d: Path, want_sha: str | None, sets: list[str], rot: bool = False):
        e = read_json(d / "eval.json")
        if e.get("weights_sha256") != want_sha:
            bad.append(f"{d}: weights")
        if bool(e.get("rot")) != rot:
            bad.append(f"{d}: rot flag")
        for s in sets:
            f = "known.tsv" if s == "known" else f"eval_{s}.tsv"
            if e.get("inputs_sha256", {}).get(s) != man.get(f):
                bad.append(f"{d}: input {s}")

    ev(evals / "v0", None, list(SETS) + ["known", "open", "open_rew"])
    for k in TRAINED:
        t = read_json(train / k / "train.json")
        if t.get("text_sha256") != man.get(f"train_{TEXT[k]}.jsonl"):
            bad.append(f"{k}: text")
        if t.get("lr") != params["lr_l" if k.startswith("l") else "lr_e"]:
            bad.append(f"{k}: lr")
        if (t.get("lambda0") if k in FIRED else None) != (params["lambda0"] if k in FIRED else None):
            bad.append(f"{k}: lambda0")
        for mk in MARKS:
            ev(evals / k / mk, t.get("marks_sha256", {}).get(mk, "missing"), list(SETS) + ["known"])
        if k in ("e1", "l1"):
            ev(evals / k / "general", t.get("marks_sha256", {}).get("final", "missing"), [])
    ev(evals / "e_rot" / "final", read_json(train / "e1" / "train.json").get("marks_sha256", {}).get("final"),
       list(SETS), rot=True)
    return {"ok": not bad, "failures": bad}


def general_guards(evals: Path, k: str, tok) -> dict:
    """gsm8k and chat (A4's paired rule) and docs top-1 accuracy (CI95 low > -DOCS_TOL), arm k against v0."""
    import c50_a4_checks as A4
    import c50_s4_verdict as V
    g = {}
    gen_v0 = V.correctness({"gsm8k": evals / "v0" / "general", "chat": evals / "v0" / "general"}, tok, ("gsm8k", "chat"))
    gen_x = V.correctness({"gsm8k": evals / k / "general", "chat": evals / k / "general"}, tok, ("gsm8k", "chat"))
    for p in ("gsm8k", "chat"):
        it = sorted(gen_v0[p])
        g[p] = paired_ok(np.array([gen_v0[p][i] for i in it]), np.array([gen_x[p][i] for i in it]))

    def top1(d: Path) -> np.ndarray:
        r = np.fromfile(d / "docs.pos.bin", dtype=A4.REC)
        return np.array([float((r["top1"][r["doc"] == x] == r["target"][r["doc"] == x]).mean()) for x in np.unique(r["doc"])])
    a0, ax = top1(evals / "v0" / "general"), top1(evals / k / "general")
    bs = (ax - a0)[np.random.default_rng(SEED).integers(0, len(a0), (DRAWS, len(a0)))].mean(1)
    lo = float(np.percentile(bs, 2.5))
    g["docs_top1"] = {"value": float((ax - a0).mean()), "ci95_low": lo, "ok": lo > -DOCS_TOL}
    return g


def measure(inputs: Path, evals: Path, train: Path, params: dict, general=general_guards) -> dict:
    import c50_a4_checks as A4
    from c50_s2_checks import qa_check
    tok = A4.Tokenizer.from_file(A4.TOKENIZER)
    sp = read_json(inputs / "split_typed.json")
    facts = [f for f in sp["facts"] if f["half"] in ("injected", "twin")]
    fid = {f["fact"]: j for j, f in enumerate(facts)}
    inj = np.array([f["half"] == "injected" for f in facts])
    gold = {f["fact"]: f["aliases"] for f in facts}
    meta = {s: read_jsonl(inputs / f"eval_{s}_meta.jsonl") for s in SETS + ("open", "open_rew")}
    m = {"provenance": provenance(inputs, evals, train, params)}
    m["provenance_ok"] = m["provenance"]["ok"]

    def acc(d: Path, s: str, variant: str | None = None) -> np.ndarray:
        rows = {r["item"]: r for r in read_jsonl(d / f"{s}.jsonl")}
        tot, n = np.zeros(len(facts)), np.zeros(len(facts))
        for mm in meta[s]:
            if mm["fact"] not in fid or (variant and mm["variant"] != variant):
                continue
            j = fid[mm["fact"]]
            tot[j] += qa_check(A4.decode(tok, rows[mm["row"]]["gen"]), gold[mm["fact"]])["match"]
            n[j] += 1
        with np.errstate(invalid="ignore"):
            return np.where(n > 0, tot / np.maximum(n, 1), np.nan)

    def fire(d: Path, s: str) -> np.ndarray:
        f = np.fromfile(d / f"{s}.fire.f32", dtype=np.float32)
        out, n = np.zeros(len(facts)), np.zeros(len(facts))
        for mm in meta[s]:
            if mm["fact"] in fid and not np.isnan(f[mm["row"]]):
                out[fid[mm["fact"]]] += f[mm["row"]]
                n[fid[mm["fact"]]] += 1
        with np.errstate(invalid="ignore"):
            return np.where(n > 0, out / np.maximum(n, 1), np.nan)

    A = {("v0", "final"): {s: acc(evals / "v0", s) for s in SETS}}
    A[("v0", "final")]["rew_t0"] = acc(evals / "v0", "rew", "t0")
    for arm in TRAINED:
        for mk in MARKS:
            A[(arm, mk)] = {s: acc(evals / arm / mk, s) for s in SETS}
            A[(arm, mk)]["rew_t0"] = acc(evals / arm / mk, "rew", "t0")
    A[("e_rot", "final")] = {s: acc(evals / "e_rot" / "final", s) for s in SETS}
    for mk in MARKS:                                        # arms E and L = the two seeds, per fact
        for k, (x, y) in (("e", ("e1", "e2")), ("l", ("l1", "l2"))):
            with np.errstate(invalid="ignore"):
                A[(k, mk)] = {s: np.nanmean([A[(x, mk)][s], A[(y, mk)][s]], 0) for s in SETS + ("rew_t0",)}
    ok = ~np.isnan(A[("v0", "final")]["rew"])
    ok_t0 = ~np.isnan(A[("v0", "final")]["rew_t0"])
    okc = ~np.isnan(A[("v0", "final")]["cloze"])
    m["facts"] = {"injected": int(inj.sum()), "twin": int((~inj).sum()),
                  "with_rewording": {"injected": int((ok & inj).sum()), "twin": int((ok & ~inj).sum())}}
    # the planted positive and the balance check: v0's open book on the rewordings, per half
    orw = acc(evals / "v0", "open_rew")
    pi, pt = float(np.nanmean(orw[inj])), float(np.nanmean(orw[~inj]))
    ni, nt = int((~np.isnan(orw) & inj).sum()), int((~np.isnan(orw) & ~inj).sum())
    se = math.sqrt(max(pi * (1 - pi), 0.01) / max(ni, 1) + max(pt * (1 - pt), 0.01) / max(nt, 1))
    m["open_rew_v0"] = float(np.nanmean(orw))
    m["open_rew_v0_by_source"] = {v: float(np.nanmean(acc(evals / "v0", "open_rew", v))) for v in ("f0", "t0")}
    m["open_orig_v0"] = float(np.nanmean(acc(evals / "v0", "open")))
    m["balance"] = {"injected": pi, "twin": pt, "se": se, "ok": abs(pi - pt) <= 3 * se}
    keys = ("v0", "e", "e1", "e2", "edense", "l", "l1", "l2", "qa1", "s", "e_rot")
    m["delta"] = {k: {"value": delta(A[(k, "final")]["rew"][ok], inj[ok])} for k in keys}
    m["delta"]["v0"]["ci95"] = boot_d(A[("v0", "final")]["rew"][ok], inj[ok])
    v0 = A[("v0", "final")]
    m["vs0"] = {k: contrast(A[(k, "final")]["rew"][ok], v0["rew"][ok], inj[ok]) for k in keys if k != "v0"}
    m["vs0_t0rew"] = {k: contrast(A[(k, "final")]["rew_t0"][ok_t0], v0["rew_t0"][ok_t0], inj[ok_t0]) for k in ("e", "l")}
    m["e_vs_s"] = contrast(A[("e", "final")]["rew"][ok], A[("s", "final")]["rew"][ok], inj[ok])
    m["qa1_vs_e"] = contrast(A[("qa1", "final")]["rew"][ok], A[("e", "final")]["rew"][ok], inj[ok])
    m["orig_vs0"] = {k: contrast(A[(k, "final")]["orig"], v0["orig"], inj) for k in ("e", "edense", "l", "qa1", "s")}
    m["cloze_vs0"] = {k: contrast(A[(k, "final")]["cloze"][okc], v0["cloze"][okc], inj[okc])
                      for k in ("e", "edense", "l", "qa1", "s")}
    m["halves_vs0"] = {k: {h: half_vs(A[(k, "final")]["rew"], v0["rew"], msk & ok)
                           for h, msk in (("injected", inj), ("twin", ~inj))} for k in ("e", "edense", "l", "qa1", "s")}
    m["dose"] = {k: {mk: delta(A[(k, mk)]["rew"][ok], inj[ok]) - m["delta"]["v0"]["value"] for mk in MARKS}
                 for k in ("e", "edense", "l", "qa1", "s")}
    # F0's tokens per extracted fact, along the dose curve: the passages' tokens per injected fact over (arm - 0)
    te = read_jsonl(inputs / "train_E.jsonl")
    n_inj = int(inj.sum())
    tok_all = sum(len(r["ids"]) for r in te) / n_inj
    tok_p1 = sum(len(r["ids"]) for r in te if r["kind"] == "p1") / n_inj
    m["price"] = {"f0_tokens_per_fact_E": tok_all, "f0_tokens_per_fact_p1": tok_p1,
                  "per_extracted_E": {mk: (tok_all / v if v > 0 else None) for mk, v in m["dose"]["e"].items()},
                  "per_extracted_QA1": {mk: (tok_p1 / v if v > 0 else None) for mk, v in m["dose"]["qa1"].items()}}
    # the answer bag: twin rewordings that arm e1 answers with an injected fact's gold
    inj_gold = [a for f in facts if f["half"] == "injected" for a in f["aliases"]]
    rows_e1 = {r["item"]: r for r in read_jsonl(evals / "e1" / "final" / "rew.jsonl")}
    tw = [mm for mm in meta["rew"] if mm["fact"] in fid and not inj[fid[mm["fact"]]]]
    m["answer_bag_e1"] = float(np.mean([qa_check(A4.decode(tok, rows_e1[mm["row"]]["gen"]), inj_gold)["match"] for mm in tw]))
    # trainings, fire (arm e1, the first seed): replay from the training's last validation, prompts from the evaluation
    tr = {k: read_json(train / k / "train.json") for k in TRAINED}
    m["train_ok"] = {k: v["status"] == "COMPLETE" and v["start"]["identity_step0"]["max_abs_logit"] == 0.0 for k, v in tr.items()}
    fe = fire(evals / "e1" / "final", "rew")
    m["fire"] = {"e": {"replay": tr["e1"]["vals"][-1]["val_fire"],
                       "min_layer": min(x.get("fire_min_layer", 1.0) for x in tr["e1"]["log"][-20:]),
                       "inj_prompts": float(np.nanmean(fe[inj & ok])), "twin_prompts": float(np.nanmean(fe[~inj & ok]))},
                 "edense": {"replay": tr["edense"]["vals"][-1]["val_fire"]}}
    # guards (arms e1 and l1 against v0): known items (A4's paired rule) at every mark, general panels at the final
    km = read_jsonl(inputs / "known.jsonl")

    def known(d: Path) -> np.ndarray:
        rows = {r["item"]: r for r in read_jsonl(d / "known.jsonl")}
        return np.array([qa_check(A4.decode(tok, rows[r["k"]]["gen"]), r["aliases"])["match"] for r in km])
    kv0 = known(evals / "v0")
    m["guards"] = {"known_by_mark": {}}
    for k in ("e1", "l1"):
        g = {"known": paired_ok(kv0, known(evals / k / "final")), **general(evals, k, tok)}
        m["guards"]["e" if k == "e1" else "l"] = g
        m["guards"]["known_by_mark"][k] = {mk: paired_ok(kv0, known(evals / k / mk))["ok"] for mk in MARKS}
    return m


# ---------------------------------------------------------------- the picks before the trainings (protocol step 7)
def pick(stage: str, inputs: Path, mve: Path | None, dev: Path | None, out: Path) -> int:
    """lambda: params.json gets the MVE's own pick (the smallest qualifying lambda0; None = STOP). lr: for arms e and
    l, the lr with the highest dev accuracy (the mean over rew, orig and cloze of each set's accuracy on the dev facts)
    minus v0's on the same facts; ties to the lower lr. `dev` holds v0/ and <arm>-<lr>/ evaluation folders."""
    rec = read_json(out) if out.exists() else {}
    if stage == "lambda":
        mv = read_json(mve)
        rec.update({"lambda0": mv["lambda0_pick"], "mve_ok": mv["ok_by_lambda0"], "mve_sha_status": mv["status"]})
        out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print("PICK", json.dumps(rec))
        return 0 if rec["lambda0"] is not None else 6
    import c50_a4_checks as A4
    from c50_s2_checks import qa_check
    tok = A4.Tokenizer.from_file(A4.TOKENIZER)
    sp = read_json(inputs / "split_typed.json")
    devf = {f["fact"]: f["aliases"] for f in sp["facts"] if f["half"] == "dev"}

    def score(d: Path) -> float:
        per_set = []
        for s in SETS:
            rows = {r["item"]: r for r in read_jsonl(d / f"{s}.jsonl")}
            per_set.append(float(np.mean([qa_check(A4.decode(tok, rows[mm["row"]]["gen"]), devf[mm["fact"]])["match"]
                                          for mm in read_jsonl(inputs / f"eval_{s}_meta.jsonl") if mm["fact"] in devf])))
        return float(np.mean(per_set))
    base = score(dev / "v0")
    rec.update({"dev_v0": base, "dev": {}})
    for arm in ("e", "l"):
        sc = {lr: score(dev / f"{arm}-{lr}") - base for lr in LRS}
        rec["dev"][arm] = sc
        best = max(sc.values())
        rec[f"lr_{arm}"] = float(min((lr for lr in LRS if sc[lr] == best), key=float))
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("PICK", json.dumps(rec))
    return 0


# ---------------------------------------------------------------- planted self-tests
def selftest_decide() -> list[tuple[str, str, str]]:
    rng = np.random.default_rng(7)
    n = 150
    inj = np.r_[np.ones(n, bool), np.zeros(n, bool)]

    def arm(p_inj, p_tw):
        return np.r_[rng.random(n) < p_inj, rng.random(n) < p_tw].astype(float)

    def build(e_inj, s_inj, ed_inj, l_inj, cl_inj, fire_rep=0.10, fire_inj=0.20, fire_tw=0.10, open_v0=0.8,
              guard=True, v0_shift=0.0, bag=0.0, prov=True, balance=True):
        v0 = arm(0.03 + v0_shift, 0.03)
        arms = {"v0": v0, "e": arm(e_inj, 0.03), "edense": arm(ed_inj, 0.03), "l": arm(l_inj, 0.03),
                "qa1": arm(0.10, 0.03), "s": arm(s_inj, 0.03)}
        cl = {k: arm(cl_inj if k != "v0" else 0.05, 0.05) for k in arms}
        m = {"provenance_ok": prov, "open_rew_v0": open_v0, "train_ok": {"e1": True}, "balance": {"ok": balance},
             "delta": {k: {"value": delta(v, inj)} for k, v in arms.items()}, "answer_bag_e1": bag}
        m["delta"]["v0"]["ci95"] = boot_d(v0, inj)
        m["vs0"] = {k: contrast(v, v0, inj) for k, v in arms.items() if k != "v0"}
        m["vs0_t0rew"] = {"e": m["vs0"]["e"]}
        m["e_vs_s"] = contrast(arms["e"], arms["s"], inj)
        m["cloze_vs0"] = {k: contrast(cl[k], cl["v0"], inj) for k in ("e", "edense", "l", "qa1", "s")}
        m["fire"] = {"e": {"replay": fire_rep, "min_layer": 0.05, "inj_prompts": fire_inj, "twin_prompts": fire_tw}}
        m["guards"] = {"e": {"known": {"ok": guard}}}
        return decide(m)["outcome"]

    good = (0.45, 0.03, 0.45, 0.30, 0.6)
    return [
        ("planted positive", build(*good), "PASS: E stores"),
        ("not selective", build(*good, fire_inj=0.10), "PASS, NOT SELECTIVE"),
        ("null", build(0.03, 0.03, 0.03, 0.03, 0.05), "all fail, cloze fails"),
        ("stored only", build(0.03, 0.03, 0.03, 0.03, 0.6), "all fail, cloze passes"),
        ("fire cost kills", build(0.03, 0.03, 0.45, 0.30, 0.6), "E fails, E-dense passes"),
        ("branch cannot store", build(0.03, 0.03, 0.03, 0.40, 0.6), "E and E-dense fail, L passes"),
        ("shuffle leaks", build(0.45, 0.45, 0.45, 0.30, 0.6), "the gain is not the fact"),
        ("answer bag", build(*good, bag=0.2), "the gain is not the fact"),
        ("open book broken", build(*good, open_v0=0.2), "STOP"),
        ("dense fire", build(*good, fire_rep=0.5), "E passes the primary but fails fire"),
        ("split broken", build(*good, v0_shift=0.25), "STOP"),
        ("provenance broken", build(*good, prov=False), "STOP"),
        ("balance broken", build(*good, balance=False), "STOP"),
    ]


def selftest_measure(break_chain: bool) -> str:
    """measure() + decide() on a planted folder: v0 knows no fact closed-book and every fact with the passage in the
    prompt; e1, e2, edense, l1, l2 know the injected facts; s, qa1 and e_rot know none; e1's gate fires 0.3 on the
    injected rewordings and 0.1 on the twin ones. With break_chain, one evaluation's weights hash is wrong."""
    import c50_a4_checks as A4
    tok = A4.Tokenizer.from_file(A4.TOKENIZER)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids + [156895]
    d = Path(tempfile.mkdtemp(prefix="inj_selftest_"))
    inp, ev, tr = d / "inj", d / "evals", d / "train"
    inp.mkdir()
    n = 120
    facts = [{"fact": i, "half": "injected" if i % 2 == 0 else "twin", "aliases": [f"Zorblax {i}"], "gold": f"Zorblax {i}"}
             for i in range(n)] + [{"fact": n + i, "half": "dev", "aliases": ["Q"], "gold": "Q"} for i in range(4)]
    (inp / "split_typed.json").write_text(json.dumps({"facts": facts}), encoding="utf-8")
    meta = {s: [] for s in SETS + ("open", "open_rew")}
    for f in facts:
        for s in SETS + ("open", "open_rew"):
            for v in (("f0", "t0") if s in ("rew", "open_rew") else ("x",)):
                meta[s].append({"fact": f["fact"], "variant": v, "row": len(meta[s])})
    for s, rows in meta.items():
        (inp / f"eval_{s}_meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (inp / "known.jsonl").write_text("".join(json.dumps({"k": k, "aliases": ["Paris"]}) + "\n" for k in range(50)), encoding="utf-8")
    (inp / "train_E.jsonl").write_text("".join(json.dumps({"fact": f["fact"], "kind": k, "ids": [1] * 70}) + "\n"
                                               for f in facts if f["half"] == "injected" for k in ("p1", "p2")), encoding="utf-8")
    files = {f"eval_{s}.tsv": f"sha-{s}" for s in meta} | {"known.tsv": "sha-known"} | \
            {f"train_{t}.jsonl": f"sha-train-{t}" for t in ("E", "S", "QA1")}
    (inp / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    params = {"lr_e": 3e-4, "lr_l": 1e-4, "lambda0": 1e-4}
    knows = {"e1": True, "e2": True, "edense": True, "l1": True, "l2": True, "s": False, "qa1": False, "e_rot": False}
    fid = {f["fact"]: f for f in facts}

    def write_eval(folder: Path, arm: str, weights: str | None, sets, rot=False, fire=False):
        folder.mkdir(parents=True, exist_ok=True)
        for s in sets:
            out = []
            if s == "known":
                out = [{"item": k, "gen": enc("Paris")} for k in range(50)]
            else:
                for mm in meta[s]:
                    f = fid[mm["fact"]]
                    right = (arm == "v0" and s in ("open", "open_rew")) or (knows.get(arm) and f["half"] == "injected")
                    out.append({"item": mm["row"], "gen": enc(f["gold"] if right else "I do not know")})
            (folder / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
            if fire and s == "rew":
                np.array([0.3 if fid[mm["fact"]]["half"] == "injected" else 0.1 for mm in meta[s]], dtype=np.float32).tofile(folder / "rew.fire.f32")
        (folder / "eval.json").write_text(json.dumps({"weights_sha256": weights, "rot": rot, "inputs_sha256":
                                                      {s: files["known.tsv" if s == "known" else f"eval_{s}.tsv"] for s in sets}}),
                                          encoding="utf-8")

    write_eval(ev / "v0", "v0", None, list(SETS) + ["known", "open", "open_rew"])
    for k in TRAINED:
        marks = {mk: f"w-{k}-{mk}" for mk in MARKS}
        (tr / k).mkdir(parents=True)
        (tr / k / "train.json").write_text(json.dumps({
            "status": "COMPLETE", "start": {"identity_step0": {"max_abs_logit": 0.0}}, "vals": [{"val_fire": 0.1}],
            "log": [{"fire_min_layer": 0.05}], "lr": params["lr_l" if k.startswith("l") else "lr_e"],
            "lambda0": params["lambda0"] if k in FIRED else None, "text_sha256": files[f"train_{TEXT[k]}.jsonl"],
            "marks_sha256": marks}), encoding="utf-8")
        for mk in MARKS:
            w = "wrong" if (break_chain and k == "s" and mk == "final") else marks[mk]
            write_eval(ev / k / mk, k, w, list(SETS) + ["known"], fire=(k == "e1"))
        if k in ("e1", "l1"):
            (ev / k / "general").mkdir()
            (ev / k / "general" / "eval.json").write_text(json.dumps({"weights_sha256": marks["final"], "rot": False,
                                                                      "inputs_sha256": {}}), encoding="utf-8")
    write_eval(ev / "e_rot" / "final", "e_rot", "w-e1-final", list(SETS), rot=True)
    stub = lambda evals, k, tok_: {"gsm8k": {"ok": True}, "chat": {"ok": True}, "docs_top1": {"ok": True}}
    m = measure(inp, ev, tr, params, general=stub)
    return decide(m)["outcome"]


def selftest() -> int:
    cases = selftest_decide()
    cases.append(("measure on a planted folder", selftest_measure(False), "PASS: E stores"))
    cases.append(("measure, broken provenance", selftest_measure(True), "STOP: provenance"))
    res = [(k, got, got.startswith(want)) for k, got, want in cases]
    for k, got, ok in res:
        print(("ok  " if ok else "FAIL"), k, "->", got)
    print("SELFTEST", f"{sum(r[2] for r in res)}/{len(res)}")
    return 0 if all(r[2] for r in res) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pick", choices=["lambda", "lr"])
    ap.add_argument("--mve", type=Path)
    ap.add_argument("--dev", type=Path)
    ap.add_argument("--params", type=Path)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--evals", type=Path)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.pick:
        return pick(a.pick, a.inputs, a.mve, a.dev, a.params)
    params = read_json(a.params or a.inputs / "params.json")
    m = measure(a.inputs, a.evals, a.train, params)
    d = decide(m)
    rec = {"cell": "C50 injection verdict", "rule": {"primary_min": PRIMARY_MIN, "open_min": OPEN_MIN,
           "fire_ceiling": FIRE_CEIL, "selective": SELECTIVE, "docs_tol": DOCS_TOL, "calib_max": CALIB_MAX,
           "answer_bag_max": BAG_MAX}, "params": params, "measured": m, "verdict": d}
    a.json.write_text(json.dumps(rec, indent=1, default=float) + "\n", encoding="utf-8")
    print("OUTCOME", d["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
