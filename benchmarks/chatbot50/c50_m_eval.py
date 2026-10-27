"""C50 cell M (the metric): one pass per arm for a temperature-calibrated reading of R. Nothing is trained.

For one arm (S4's t0/a/b/c48/c496, or T2's v0/n/e2 on v0's GGUF) it writes:
- the temperature fit on the validation shards 470-471, at every temperature of the grid, by two rules that are the
  same for every arm: the NLL of the corpus targets (Guo et al. 2017's rule; `tau_nll`), and the 33-bucket KL to F0
  (S4's `kl33`, the training objective; `tau_kl`);
- on S4's 36 documents, the log-probability of each target at every temperature of the grid (`docs.temp.f32`,
  positions x grid, in the order of `docs.pos.bin`), and `docs.pos.bin` itself at temperature 1, which must equal the
  arm's recorded evaluation (the instrument check), and `docs.rank.i32`, each target's rank (0 = top-1) at
  temperature 1, which no temperature changes.

    python -I benchmarks/chatbot50/c50_m_eval.py --family s4 --gguf T0 --arm t0|a|b|c48|c496 [--weights W]
        [--bank DIR --bank-mode full] --teacher DIR --corpus TSV --a4-inputs DIR --split S.json --out DIR [--cpu]
    python -I benchmarks/chatbot50/c50_m_eval.py --family t2 --gguf V0 --arm v0|n|e2 [--weights W] [--n-mode shuffle] ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_eval as E  # noqa: E402
import c50_s4_train as S4  # noqa: E402
from c50_s4_student import kl33  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402
from c50_s4_t0_torch import mm  # noqa: E402
from c50_t1_train import TeacherL, pack  # noqa: E402

TAUS = np.round(np.arange(0.50, 2.5001, 0.025), 3)     # 81 temperatures, 1.0 included
I1 = int(np.flatnonzero(TAUS == 1.0)[0])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build(a, W):
    if a.family == "s4":
        ns = argparse.Namespace(arm=a.arm, weights=a.weights, bank=a.bank, bank_mode=a.bank_mode, a_mode=a.a_mode)
        return E.build(ns, W)
    import c50_t2_n_eval as T
    ns = argparse.Namespace(arm=a.arm, weights=a.weights, n_mode=a.n_mode)
    return T.build(ns, W), None


@torch.no_grad()
def fit(S, bank, V) -> tuple[np.ndarray, np.ndarray]:
    """Mean KL33 to F0 and mean NLL of the corpus targets over the validation positions, at every temperature."""
    taus = torch.tensor(TAUS, dtype=torch.float32)
    tot, nll, npos = np.zeros(len(TAUS)), np.zeros(len(TAUS)), 0
    w = S.W.get("output.weight")
    for batch in V:
        ids, segs, tid, tlp, _ = pack(batch)
        bv, bh = E.bank_for(S, bank, ids.numpy(), segs)
        h = S.hidden(ids, segs, bv, bh, ckpt=False)
        keep = torch.ones(len(ids), dtype=torch.bool)
        for s0, s1 in segs:
            keep[s1 - 1] = False
        h = h[keep.to(h.device)]
        nxt = torch.cat([ids[s0 + 1:s1] for s0, s1 in segs])     # the corpus target of each kept position
        assert len(nxt) == len(h) == len(tid), (len(nxt), len(h), len(tid))
        for c in range(0, len(h), E.HEAD_CHUNK):
            lg = mm(h[c:c + E.HEAD_CHUNK].to(w.device), w).float()
            ti, tl = tid[c:c + E.HEAD_CHUNK].to(lg.device), tlp[c:c + E.HEAD_CHUNK].to(lg.device)
            y = nxt[c:c + E.HEAD_CHUNK].to(lg.device)
            for g, t in enumerate(taus.tolist()):
                tot[g] += float(kl33(lg / t, ti, tl))
                z = lg / t
                nll[g] += float((torch.logsumexp(z, -1) - z.gather(1, y[:, None])[:, 0]).sum())
        npos += len(h)
    return tot / npos, nll / npos


@torch.no_grad()
def docs_temp(S, bank, seqs) -> tuple[np.ndarray, np.ndarray]:
    """E.score's batches and records, plus the target's log-probability at every temperature."""
    taus = torch.tensor(TAUS, dtype=torch.float32)
    recs, temps, ranks = [], [], []
    i = 0
    while i < len(seqs):
        batch, n = [], 0
        while i < len(seqs) and (not batch or n + len(seqs[i][1]) <= E.BATCH_TOKENS):
            batch.append(seqs[i])
            n += len(seqs[i][1])
            i += 1
        ids = np.concatenate([b[1] for b in batch])
        segs, a0 = [], 0
        for b in batch:
            segs.append((a0, a0 + len(b[1])))
            a0 += len(b[1])
        bv, bh = E.bank_for(S, bank, ids, segs)
        h = S.hidden(torch.from_numpy(ids), segs, bv, bh, ckpt=False)
        w = S.W.get("output.weight")
        for (d, x), (s0, s1) in zip(batch, segs):
            for c in range(s0, s1 - 1, E.HEAD_CHUNK):
                c1 = min(c + E.HEAD_CHUNK, s1 - 1)
                lg = mm(h[c:c1].to(w.device), w).float()
                tgt = torch.from_numpy(ids[c + 1:c1 + 1]).to(lg.device)
                lse = torch.logsumexp(lg, -1)
                lt = lg.gather(1, tgt[:, None])[:, 0] - lse
                m, top = lg.max(-1)
                r = np.zeros(c1 - c, dtype=E.REC)
                r["doc"], r["pos"], r["target"] = d, np.arange(c - s0 + 1, c1 - s0 + 1), ids[c + 1:c1 + 1]
                r["top1"], r["logp_target"] = top.cpu().numpy(), lt.cpu().numpy()
                r["logp_top1"], r["lse"] = (m - lse).cpu().numpy(), lse.double().cpu().numpy()
                recs.append(r)
                ranks.append((lg > lg.gather(1, tgt[:, None])).sum(-1).cpu().numpy().astype(np.int32))
                tt = torch.stack([(lg.gather(1, tgt[:, None])[:, 0] / t) - torch.logsumexp(lg / t, -1)
                                  for t in taus.tolist()], 1)
                temps.append(tt.cpu().numpy().astype(np.float32))
    return np.concatenate(recs), np.concatenate(temps), np.concatenate(ranks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["s4", "t2"])
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--bank-mode", default="full")
    ap.add_argument("--a-mode", default="full")
    ap.add_argument("--n-mode", default="full")
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--a4-inputs", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--docs-tsv", type=Path, help="part 2: every document of this file (A4-bis's 72) instead of S4's 36")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: this many validation batches and documents")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    devs = ("cpu", "cpu") if a.cpu else ("cuda:0", "cuda:1")
    cfg = {"cell": "C50 cell M eval", "family": a.family, "arm": a.arm, "n_mode": a.n_mode, "a_mode": a.a_mode,
           "bank_mode": a.bank_mode if a.bank else None, "gguf": str(a.gguf), "gguf_sha256": sha256(a.gguf),
           "weights": str(a.weights) if a.weights else None,
           "weights_sha256": sha256(a.weights) if a.weights else None, "taus": TAUS.tolist(),
           "fit": "sum of kl33 to F0 over val shards 470-471, argmin over the grid", "limit": a.limit,
           "docs_tsv": str(a.docs_tsv) if a.docs_tsv else None,
           "torch": torch.__version__, "devs": list(devs),
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if not a.cpu else []}
    print("CONFIG", json.dumps({k: v for k, v in cfg.items() if k != "taus"}), flush=True)
    if not a.cpu:
        assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    W = T0Quant(a.gguf, devs, wdtype=torch.float32 if a.cpu else torch.float16)
    S, bank = build(a, W)
    a.out.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    V = list(TeacherL(a.teacher, a.corpus, list(S4.VAL_SHARDS)).batches(S4.TOKENS))
    if a.limit:
        V = V[:1]
    kl, nll = fit(S, bank, V)
    gk, gn = int(np.argmin(kl)), int(np.argmin(nll))
    rec = {**cfg, "val_batches": len(V), "val_kl": [round(float(x), 6) for x in kl],
           "val_nll": [round(float(x), 6) for x in nll], "tau_kl": float(TAUS[gk]), "tau_nll": float(TAUS[gn]),
           "val_kl_at_1": float(kl[I1]), "val_kl_at_tau_kl": float(kl[gk]), "val_nll_at_1": float(nll[I1]),
           "val_nll_at_tau_nll": float(nll[gn]), "fit_s": round(time.time() - t1, 1),
           "tau_at_edge": bool({gk, gn} & {0, len(TAUS) - 1})}
    print("FIT", json.dumps({k: rec[k] for k in ("tau_kl", "tau_nll", "val_kl_at_1", "val_nll_at_1", "tau_at_edge")}),
          flush=True)
    split = json.loads(a.split.read_text(encoding="utf-8"))
    rows = E.read_tsv(a.docs_tsv) if a.docs_tsv else E.selected("docs", E.read_tsv(a.a4_inputs / "docs.tsv"), split)
    if a.limit:
        rows = rows[:a.limit]
    t1 = time.time()
    r, tt, rk = docs_temp(S, bank, rows)
    r.tofile(a.out / "docs.pos.bin")
    tt.tofile(a.out / "docs.temp.f32")
    rk.tofile(a.out / "docs.rank.i32")
    rec.update({"docs": len(rows), "positions": int(len(r)), "docs_s": round(time.time() - t1, 1),
                "col_1_vs_logp_target_max": float(np.abs(tt[:, I1] - r["logp_target"]).max()),
                "rank0_vs_top1_mismatch": int(((rk == 0) != (r["top1"] == r["target"])).sum())})   # ties only
    (a.out / "eval.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DONE", json.dumps({k: rec[k] for k in ("positions", "docs_s", "col_1_vs_logp_target_max")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
