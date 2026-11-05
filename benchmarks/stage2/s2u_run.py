"""Stage 2 test 2b-bis runner: cell 1 of test 2b with the tail a * preview + b, judged on new text.  It prints numbers; s2u_verdict.py decides.

    sets     the 96 documents of c50_holdout_v1's reserved split into a .tsv; the rendering checked on the panel's 72, overlap with the
             head trainer's corpus counted
    fit      a and b as test 2b's diagnosis found them, and b_mass, the b that gives the tail its true mass on the same dev rows
    dump     P0 on the new documents: records, top-32, hidden states; the records are the new reference
    replay   the new binary with the affine tail off, on the panel's first 2 documents, against test 2b's own records
    panel    the judged cell, on the new documents and on chat
    float    every tail in float, on the new documents, the panel's 72 and ternary-expert states: dBPB, miss, KL, the tail's mass

Usage: python -I benchmarks/stage2/s2u_run.py sets|fit|dump|replay|panel|float --out benchmarks/stage2/results/s2u_run.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
import s2t_head as T  # noqa: E402
import s2t_run as R  # noqa: E402
from s1_run import INPUTS, MAX_PROJECTED_S, STOP, sha256  # noqa: E402

T.CHUNK = 256                                  # beside the trainer on the same card
WORK = Path("E:/SiliconLLM_Artifacts/stage2/s2u")
CORPUS = HERE.parents[0] / "chatbot50" / "corpus" / "c50_holdout_v1"
IDS = WORK / "inputs" / "reserved.tsv"
SCORE, GEN = HERE / "bin" / "s2u_llama_score_avx2.exe", HERE / "bin" / "s2u_llama_generate_avx2.exe"
FIT = HERE / "results" / "s2u_fit.json"
PROBE = HERE / "results" / "s2t_tail_probe.json"
PREVIEW, K, FIT_ROWS, PLANT_B, NGRAM, DRAWS = "tern_tr", 1024, 20000, -3.0, 32, 10000
ARM = "two_pass_a"


def load(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"cell": "S2U-tail", "panels": {}}


def save(out: Path, rep: dict) -> None:
    out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")


def sets(a: argparse.Namespace) -> int:
    rep = load(a.out)
    (WORK / "inputs").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (CORPUS / "reserved_ids.jsonl").read_text(encoding="utf-8").splitlines()]
    with IDS.open("w", encoding="ascii", newline="\n") as f:
        for i, r in enumerate(rows):
            f.write(f"{i}\t{','.join(map(str, r['ids']))}\n")
    # the rendering: fit_ids.jsonl must give back the panel's 72 lines
    fit = {r["item_sha256"]: r for r in map(json.loads, (CORPUS / "fit_ids.jsonl").read_text(encoding="utf-8").splitlines())}
    meta = [json.loads(x) for x in (INPUTS["a4"] / "docs_meta.jsonl").read_text(encoding="utf-8").splitlines()]
    lines = (INPUTS["a4"] / "docs.tsv").read_text(encoding="ascii").splitlines()
    same = sum(ln.split("\t")[1] == ",".join(map(str, fit[m["item_sha256"]]["ids"])) for m, ln in zip(meta, lines))
    # overlap: runs of NGRAM tokens shared with what trained the preview, what fitted a and b, and the panel's documents
    import s1t_train as S
    seen: set = set()
    tr, he = S.load_corpus(1_500_000, 75_000)
    for s in list(tr) + list(he) + [[int(x) for x in ln.split("\t")[1].split(",")] for ln in lines]:
        s = [int(x) for x in s]
        seen.update(hash(tuple(s[i:i + NGRAM])) for i in range(0, max(len(s) - NGRAM, 0)))
    hit = sum(any(hash(tuple(r["ids"][i:i + NGRAM])) in seen for i in range(0, len(r["ids"]) - NGRAM)) for r in rows)
    rep["sets"] = {"documents": len(rows), "by_category": {c: sum(r["category"] == c for r in rows) for c in ("code", "technical_general", "prose")},
                   "tokens": sum(len(r["ids"]) for r in rows), "tsv_sha256": sha256(IDS), "panel_lines_rebuilt": same, "panel_lines": len(lines),
                   "documents_sharing_a_run": hit, "ngram": NGRAM}
    save(a.out, rep)
    print(json.dumps(rep["sets"]))
    return 0


def cell(name: str = PREVIEW) -> dict:
    return {k: v.to(T.DEV).float() for k, v in torch.load(T.WORK / "previews" / f"{name}.pt", weights_only=True).items()}


def fit(a: argparse.Namespace) -> int:
    f = next(r for r in json.loads(PROBE.read_text(encoding="utf-8"))["cells"] if r["preview"] == PREVIEW and r["k"] == K)["fit"]
    W = torch.load(T.WORK / "previews" / "exact.pt", weights_only=True)["w"].to(T.DEV).float()
    p = cell()
    H = T.hidden(T.WORK / "h" / "dev.h.f32", FIT_ROWS)
    lc, lt, le = [], [], []
    for i in range(0, H.shape[0], T.CHUNK):
        h = H[i:i + T.CHUNK].to(T.DEV)
        ze, zp = h @ W.T, T.logits(p, h)
        c = zp.topk(K, 1).indices
        m = torch.ones_like(ze, dtype=torch.bool)
        m.scatter_(1, c, False)
        lc.append(ze.gather(1, c).logsumexp(1).double().cpu())
        le.append(ze.masked_fill(~m, -torch.inf).logsumexp(1).double().cpu())
        lt.append((f["a"] * zp).masked_fill(~m, -torch.inf).logsumexp(1).double().cpu())
    lc, lt, le = torch.cat(lc), torch.cat(lt), torch.cat(le)
    true = float(torch.sigmoid(le - lc).mean())
    lo, hi = -40.0, 40.0
    for _ in range(80):
        b = (lo + hi) / 2
        if float(torch.sigmoid(lt + b - lc).mean()) < true:
            lo = b
        else:
            hi = b
    res = {"preview": PREVIEW, "k": K, "fit_rows": int(H.shape[0]), "a": f["a"], "b": f["b"], "b_mass": b, "dev_tail_mass_true": true,
           "dev_tail_mass_affine": float(torch.sigmoid(lt + f["b"] - lc).mean()), "dev_tail_mass_b_mass": float(torch.sigmoid(lt + b - lc).mean())}
    FIT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0


def dump(a: argparse.Namespace) -> int:
    rep = load(a.out)
    pre = T.WORK / "h" / "reserved"
    cmd = [str(SCORE), "--model", str(R.P0), "--ids", str(IDS), "--out", str(pre), "--threads", "6", "--ubatch", "512", "--ctx", "8192", "--top", "32",
           "--moe-mode", "none", "--dump-h", f"{pre}.h.f32"]
    r = R.launch(cmd, ["two_pass_k=0", "top=32", "moe_mode=none", f"model={R.P0}"])
    rep["dump"] = r
    (WORK / "ref").mkdir(parents=True, exist_ok=True)
    if r["status"] == "COMPLETE":
        for x in ("pos", "top"):
            shutil.copyfile(f"{pre}.{x}.bin", WORK / "ref" / f"docs.{x}.bin")
    save(a.out, rep)
    print(r["status"], r["wall_s"])
    return 0 if r["status"] == "COMPLETE" else 1


def tail_args(off: int = 0) -> list[str]:
    f = json.loads(FIT.read_text(encoding="utf-8"))
    prev = R.PREVIEWS[PREVIEW][0]
    two = ["--preview", str(prev), "--two-pass-k", str(K), "--tail-offset", "1"]
    return two if off else two + ["--tail-affine", "1", "--tail-a", repr(f["a"]), "--tail-b", repr(f["b"])]


def replay(a: argparse.Namespace) -> int:
    rep = load(a.out)
    ids, top = R.subset(2)
    (WORK / "replay2").mkdir(parents=True, exist_ok=True)
    cmd = [str(SCORE), "--model", str(R.P0), "--ids", str(ids), "--ctx", "8192", "--top", "32", "--ref-top", str(top), "--out", str(WORK / "replay2" / "docs"),
           "--threads", "6", "--ubatch", "512", "--max-projected-s", str(MAX_PROJECTED_S), "--moe-mode", "none"] + tail_args(1)
    r = R.launch(cmd, ["two_pass_k=1024", "tail_affine=0"])
    got = (WORK / "replay2" / "docs.pos.bin").read_bytes()
    r["equal_to_two_pass"] = got == (R.STAGE1_WORK / "two_pass" / "docs.pos.bin").read_bytes()[:len(got)] and len(got) > 0
    rep["replay"] = r
    save(a.out, rep)
    print(r["status"], r["equal_to_two_pass"])
    return 0 if r["equal_to_two_pass"] else 1


def panel(a: argparse.Namespace) -> int:
    rep = load(a.out)
    if sha256(R.P0) != R.P0_SHA or sha256(R.PREVIEWS[PREVIEW][0]) != R.PREVIEWS[PREVIEW][1]:
        raise SystemExit("a model differs from its pin")
    (WORK / ARM).mkdir(parents=True, exist_ok=True)
    f = json.loads(FIT.read_text(encoding="utf-8"))
    need = ["two_pass_k=1024", "tail_affine=1", f"preview={R.PREVIEWS[PREVIEW][0]}", f"model={R.P0}"]
    for name, cmd in (("docs", [str(SCORE), "--model", str(R.P0), "--ids", str(IDS), "--ctx", "8192", "--top", "32", "--ref-top", str(WORK / "ref" / "docs.top.bin")]),
                      ("chat", [str(GEN), "--model", str(R.P0), "--prompts", str(INPUTS["a4"] / "chat.tsv"), "--max-new", "256", "--stop", STOP])):
        if rep["panels"].get(name, {}).get("status") == "COMPLETE":
            continue
        cmd += ["--out", str(WORK / ARM / name), "--threads", "6", "--ubatch", "512", "--max-projected-s", str(MAX_PROJECTED_S), "--moe-mode", "none"] + tail_args()
        rep["panels"][name] = R.launch(cmd, need)
        rep["panels"][name]["fit"] = f
        save(a.out, rep)
        print(name, rep["panels"][name]["status"], rep["panels"][name]["wall_s"], rep["panels"][name]["two_pass"])
        if rep["panels"][name]["status"] != "COMPLETE":
            return 1
    return 0


def one_set(name: str, W: torch.Tensor, p: dict, f: dict, blen_of) -> dict:
    pos = np.fromfile(T.WORK / "h" / f"{name}.pos.bin", T.REC)
    H = T.hidden(T.WORK / "h" / f"{name}.h.f32")
    if H.shape[0] != len(pos):
        raise SystemExit(f"{name}: states and records differ")
    tails = {"affine": lambda zp, off: f["a"] * zp + f["b"], "mass": lambda zp, off: f["a"] * zp + f["b_mass"], "offset": lambda zp, off: zp + off,
             "plant": lambda zp, off: f["a"] * zp + f["b"] + PLANT_B}
    tgt = torch.from_numpy(pos["target"].astype(np.int64))
    rows = {"affine": "row_affine", "mass": "row_mass"}
    d = {t: np.zeros(len(pos)) for t in [*tails, *rows.values()]}
    kl = {t: 0.0 for t in tails}
    mass = {t: 0.0 for t in tails}
    et = np.zeros(len(pos))
    miss = 0
    for i in range(0, H.shape[0], T.CHUNK):
        h = H[i:i + T.CHUNK].to(T.DEV)
        t = tgt[i:i + T.CHUNK].to(T.DEV)[:, None]
        ze, zp = h @ W.T, T.logits(p, h)
        c = zp.topk(K, 1).indices
        ex = ze.gather(1, c)
        off = (ex - zp.gather(1, c)).mean(1, keepdim=True)
        lp = ze.log_softmax(1)
        miss += int((c.gather(1, ex.argmax(1, keepdim=True)).squeeze(1) != ze.argmax(1)).sum())
        et[i:i + T.CHUNK] = (1 - lp.gather(1, c).exp().sum(1)).double().cpu().numpy()
        for tail, fn in tails.items():
            z = fn(zp, off)
            z.scatter_(1, c, ex)
            lq = z.log_softmax(1)
            d[tail][i:i + T.CHUNK] = (lp.gather(1, t) - lq.gather(1, t)).squeeze(1).double().cpu().numpy()
            if tail in rows:                   # a caller who asks for one word reads its row: the numerator is exact, the sum still holds the tail
                d[rows[tail]][i:i + T.CHUNK] = (lp.gather(1, t) - ze.gather(1, t) + z.logsumexp(1, keepdim=True)).squeeze(1).double().cpu().numpy()
            kl[tail] += float((lp.exp() * (lp - lq)).sum())
            mass[tail] += float((1 - lq.gather(1, c).exp().sum(1)).sum())
    blen = blen_of[pos["target"]].astype(np.float64)
    docs = np.unique(pos["doc"])
    n = len(pos)
    res = {"positions": n, "documents": len(docs), "miss_per_1000": 1000 * miss / n,
           "exact_tail_mass": {"mean": float(et.mean()), "p50": float(np.percentile(et, 50)), "p99": float(np.percentile(et, 99)), "max": float(et.max())},
           "tails": {}}
    for tail, v in d.items():
        dd = np.array([v[pos["doc"] == x].sum() / np.log(2) / blen[pos["doc"] == x].sum() for x in docs])
        rng = np.random.default_rng(T.SEED)
        bb = dd[rng.integers(0, len(dd), (DRAWS, len(dd)))].mean(1)
        res["tails"][tail] = {"dbpb": float(dd.mean()), "ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))], "kl_full_vocab": kl[tail] / n if tail in kl else None,
                              "tail_mass": mass[tail] / n if tail in mass else None}
    return res


def float_(a: argparse.Namespace) -> int:
    from c50_compare import byte_lengths
    import c50_a4_checks as A4
    rep = load(a.out)
    f = json.loads(FIT.read_text(encoding="utf-8"))
    W = torch.load(T.WORK / "previews" / "exact.pt", weights_only=True)["w"].to(T.DEV).float()
    p = cell()
    blen_of = byte_lengths(Path(A4.TOKENIZER))
    rep["float"] = {}
    for name in a.sets.split(","):
        rep["float"][name] = one_set(name, W, p, f, blen_of)
        print(name, json.dumps(rep["float"][name]), flush=True)
        save(a.out, rep)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("sets", "fit", "dump", "replay", "panel", "float"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="reserved,docs,tern")
    a = ap.parse_args()
    return {"sets": sets, "fit": fit, "dump": dump, "replay": replay, "panel": panel, "float": float_}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
