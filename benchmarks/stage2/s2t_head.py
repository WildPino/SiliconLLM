"""Stage 2 test 2b: the head in two passes on Ling-mini.  No training.  A cheap preview over the whole vocabulary picks K
candidates; their rows are read from the original head (Q6_K), so their logits are exact.  This tool measures how often
the original's choice falls among the candidates.  The panels (s2t_run.py) judge the composed logits on llama.cpp's own
kernels.

    sets      the token sets: fit (150,000 tokens of the head trainer's training split) and dev (its 75,000 held-out
              tokens) from s3/corpus_v1; gen (prompt + ref's generated tokens of nq, tqa, gsm8k and chat).
    previews  the previews, from the head as stored in P0 (Q6_K, dequantized) unless said otherwise:
                tern_tr   test 2's trained ternary head, from org_head-q4.gguf;
                tern_im   test 1's untrained ternary head, from head_tern_im-q4.gguf;
                sign      s * sign(w), s = mean |w| of the row (1 bit);
                int4      per-row symmetric 4 bits (HiRE-Q);
                svd<r>    the top r singular directions of W, the factor of the rows in int8 per row (SVD-softmax);
                wsvd<r>   the same after whitening by the input's covariance on fit (SVD-LLM's whitening): the top r
                          directions of W S, S the Cholesky factor of the covariance, so the error is the logits' error;
                shuf      tern_tr's rows permuted (seed 20261103): the planted control, which must miss;
              and the two lists that read no weights, from fit:
                freq      one list, the tokens by how often they are the original's choice: the trivial control;
                clus<r>   spherical k-means of the hidden states into r clusters, one list per cluster, backfilled with
                          freq (Learning to Screen without its Gumbel training).
    sweep     --set S: for each preview, the rank in the preview of the original's top-1 and of its top-32, so one pass
              gives every K.  Recall against the panel tool's own top-1 and top-32 (the llama.cpp run's records), never
              against a float recomputation.  Strata: hard positions (the original's p(top-1) < 0.5), rare choices
              (outside freq's first 8,192), generated positions, and by document.
    compose   --set S --preview P: the composed logits against the exact head in float, for K and the tail offset:
              top-1 agreement and KL over the whole vocabulary.  It chooses, on dev, what the panels run.
    materialize --preview P: a low-rank preview as a dense f16 head in a copy of P0, for the panels' second context.

Usage: python -I benchmarks/stage2/s2t_head.py sets | previews | sweep --set dev --json F | compose --set dev --preview P --json F
       python -I benchmarks/stage2/s2t_head.py materialize --preview wsvd256 --json F
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE))
from s2h_head import CAMPAIGN, P0, gguf  # noqa: E402

WORK = Path("C:/SiliconLLM_fast/s2t")
STAGE1_REF = Path("E:/SiliconLLM_Artifacts/stage1/s1a/ref")
ORG_HEAD = Path("E:/SiliconLLM_Artifacts/stage2/s2o/models/org_head-q4.gguf")
TERN_IM = Path("E:/SiliconLLM_Artifacts/stage2/s2h/models/head_tern_im-q4.gguf")
SEED = 20261103
V, D = 157184, 2048
KS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
RANKS = [32, 64, 128, 256, 512]
CLUSTERS = [64, 256]
Q6K_ROW = D // 256 * 210                      # bytes of one exact row: 8 blocks of 210
CHUNK = 1024
BOOT = 2000
GEN_PANELS = ("nq", "tqa", "gsm8k", "chat")
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"), ("lp_t", "<f4"), ("lp_1", "<f4"),
                ("lse", "<f8")])
TOPH = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"), ("lp_t", "<f4"), ("k", "<i4"),
                 ("lse", "<f8")])
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def head_of(path: Path) -> torch.Tensor:
    t = next(x for x in gguf.GGUFReader(str(path)).tensors if x.name == "output.weight")
    return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).reshape(V, D).astype(np.float32))


def read_top(path: Path, k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """A .top.bin: the heads and the [n, k] ids."""
    raw = np.fromfile(path, np.dtype([("h", TOPH), ("e", [("id", "<i4"), ("lp", "<f4")], (k,))]))
    return raw["h"], raw["e"]["id"]


def cmd_sets(_: argparse.Namespace) -> None:
    import s1t_train as T
    (WORK / "sets").mkdir(parents=True, exist_ok=True)
    tr, he = T.load_corpus(1_500_000, 75_000)
    rec = {}
    for name, seqs, limit in (("fit", tr, 150_000), ("dev", he, 75_000)):
        n = 0
        with (WORK / "sets" / f"{name}.tsv").open("w", encoding="ascii", newline="\n") as f:
            for i, s in enumerate(seqs):
                if n >= limit:
                    break
                f.write(f"{i}\t{','.join(str(int(x)) for x in s)}\n")
                n += len(s)
        rec[name] = n
    gen = {}
    with (WORK / "sets" / "gen.tsv").open("w", encoding="ascii", newline="\n") as f:
        for p, (panel, src) in enumerate((("nq", "s2"), ("tqa", "s2"), ("gsm8k", "a4"), ("chat", "a4"))):
            prompts = {}
            for line in (CAMPAIGN / src / "inputs" / f"{panel}.tsv").read_text(encoding="ascii").splitlines():
                i, ids = line.split("\t")
                prompts[int(i)] = ids
            for line in (STAGE1_REF / f"{panel}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                doc = 100000 * (p + 1) + r["item"]
                f.write(f"{doc}\t{prompts[r['item']]},{','.join(map(str, r['gen']))}\n")
                gen[str(doc)] = r["prompt_tokens"]
    (WORK / "sets" / "gen_prompt_tokens.json").write_text(json.dumps(gen), encoding="utf-8")
    print(json.dumps({"tokens": rec, "gen_items": len(gen)}))


def quant_rows(x: torch.Tensor, bits: int) -> torch.Tensor:
    q = 2 ** (bits - 1) - 1
    s = x.abs().amax(1, keepdim=True).clamp_min(1e-12) / q
    return (x / s).round().clamp(-q, q) * s


def hidden(path: Path, rows: int | None = None) -> torch.Tensor:
    m = np.memmap(path, dtype="<f4", mode="r").reshape(-1, D)
    return torch.from_numpy(np.asarray(m[:rows] if rows else m))


def cmd_previews(_: argparse.Namespace) -> None:
    out = WORK / "previews"
    out.mkdir(parents=True, exist_ok=True)
    W = head_of(P0)
    rec = {}

    def save(name: str, obj: dict, nbytes: int) -> None:
        torch.save(obj, out / f"{name}.pt")
        rec[name] = {"bytes_per_token": nbytes}
        print(name, nbytes, flush=True)

    save("exact", {"w": W.half()}, V * Q6K_ROW)              # the parity control: the float head against the panel tool
    tr = head_of(ORG_HEAD)
    save("tern_tr", {"w": tr.half()}, V * D // 4 + 4 * V)
    perm = torch.from_numpy(np.random.default_rng(SEED).permutation(V))
    save("shuf", {"w": tr[perm].half()}, V * D // 4 + 4 * V)
    save("tern_im", {"w": head_of(TERN_IM).half()}, V * D // 4 + 4 * V)
    save("sign", {"w": (W.sign() * W.abs().mean(1, keepdim=True)).half()}, V * D // 8 + 4 * V)
    save("int4", {"w": quant_rows(W, 4).half()}, V * D // 2 + 4 * V)
    Wd = W.to(DEV).double()
    gram = Wd.T @ Wd
    _, vec = torch.linalg.eigh(gram)
    vec = vec.flip(1)                                        # descending
    h = hidden(WORK / "h" / "fit.h.f32").to(DEV).double()
    cov = h.T @ h / h.shape[0]
    S = torch.linalg.cholesky(cov + 1e-6 * cov.diagonal().mean() * torch.eye(D, device=DEV, dtype=torch.float64))
    WS = Wd @ S
    _, wvec = torch.linalg.eigh(WS.T @ WS)
    wvec = wvec.flip(1)
    Sinv = torch.linalg.solve_triangular(S, torch.eye(D, device=DEV, dtype=torch.float64), upper=False)
    for r in RANKS:
        B = vec[:, :r]                                       # [D, r]
        A = quant_rows((Wd @ B).float(), 8)
        save(f"svd{r}", {"a": A.half().cpu(), "b": B.float().cpu()}, V * r + 4 * V + 2 * D * r)
        B = wvec[:, :r]
        A = quant_rows((WS @ B).float(), 8)
        save(f"wsvd{r}", {"a": A.half().cpu(), "b": (Sinv.T @ B).float().cpu()}, V * r + 4 * V + 2 * D * r)
    del Wd, WS, gram
    torch.cuda.empty_cache()
    # the lists: the original's choices on fit
    head, ids = read_top(WORK / "h" / "fit.top.bin")
    counts = np.bincount(head["top1"], minlength=V).astype(np.float64)
    # ties and never-chosen tokens by the top-32 counts, then by id: a total order
    c32 = np.bincount(ids.ravel(), minlength=V).astype(np.float64)
    order = np.lexsort((np.arange(V), -c32, -counts))
    rank = np.empty(V, np.int64)
    rank[order] = np.arange(V)
    torch.save({"rank": torch.from_numpy(rank)}, out / "freq.pt")
    rec["freq"] = {"bytes_per_token": 0}
    hn = torch.nn.functional.normalize(h.float(), dim=1)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    for r in CLUSTERS:
        cen = hn[torch.randperm(hn.shape[0], generator=g, device=DEV)[:r]].clone()
        for _ in range(20):
            lab = torch.cat([(hn[i:i + 8192] @ cen.T).argmax(1) for i in range(0, hn.shape[0], 8192)])
            cen = torch.zeros_like(cen).index_add_(0, lab, hn)
            cen = torch.nn.functional.normalize(cen, dim=1)
        lab = lab.cpu().numpy()
        lists = np.zeros((r, 4096), np.int32)
        for c in range(r):
            m = lab == c
            w = 4096.0 * np.bincount(head["top1"][m], minlength=V) + np.bincount(ids[m].ravel(), minlength=V)
            lists[c] = np.lexsort((rank, -w))[:4096]         # the cluster's choices first, backfilled by freq
        torch.save({"centroids": cen.cpu(), "lists": torch.from_numpy(lists)}, out / f"clus{r}.pt")
        rec[f"clus{r}"] = {"bytes_per_token": r * D * 2}
    (WORK / "previews.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


def logits(p: dict, h: torch.Tensor) -> torch.Tensor:
    if "w" in p:
        return h @ p["w"].T
    return (h @ p["b"]) @ p["a"].T


def strata(s: str) -> dict:
    """The records of a set, position by position, and the masks of its strata."""
    pos = np.fromfile(WORK / "h" / f"{s}.pos.bin", REC)
    head, ids = read_top(WORK / "h" / f"{s}.top.bin")
    if not (np.array_equal(pos["top1"], head["top1"]) and np.array_equal(pos["doc"], head["doc"])):
        raise SystemExit("pos.bin and top.bin differ")
    freq = torch.load(WORK / "previews" / "freq.pt", weights_only=True)["rank"].numpy()
    m = {"all": np.ones(len(pos), bool), "hard": np.exp(pos["lp_1"]) < 0.5, "rare": freq[pos["top1"]] >= 8192}
    if s == "gen":
        pt = json.loads((WORK / "sets" / "gen_prompt_tokens.json").read_text(encoding="utf-8"))
        m["generated"] = pos["pos"] >= np.array([pt[str(d)] for d in pos["doc"]])
        for i, panel in enumerate(GEN_PANELS):
            m[f"generated_{panel}"] = m["generated"] & (pos["doc"] // 100000 == i + 1)
    return {"pos": pos, "ids": ids, "masks": m}


def boot_ci(miss_by_doc: np.ndarray, n_by_doc: np.ndarray) -> list[float]:
    """CI95 of the miss rate per 1,000, resampling documents."""
    idx = np.random.default_rng(SEED).integers(0, len(n_by_doc), (BOOT, len(n_by_doc)))
    r = 1000 * miss_by_doc[idx].sum(1) / n_by_doc[idx].sum(1)
    return [round(float(x), 3) for x in np.percentile(r, [2.5, 97.5])]


def summarize(r1: np.ndarray, r32: np.ndarray, st: dict, nbytes: int) -> dict:
    out = {}
    doc = st["pos"]["doc"]
    docs, inv = np.unique(doc, return_inverse=True)
    n_by_doc = np.bincount(inv)
    for k in KS:
        hit = r1 < k
        row = {"bytes_per_token": nbytes + k * Q6K_ROW,
               "miss_per_1000": {n: round(1000 * float((~hit[m]).mean()), 3) for n, m in st["masks"].items() if m.any()},
               "top32_cover": round(float((r32 < k).mean()), 5), "top32_whole": round(float((r32 < k).all(1).mean()), 5)}
        by_doc = np.bincount(inv, weights=~hit)
        row["miss_ci95"] = boot_ci(by_doc, n_by_doc)
        w = int(np.argmax(by_doc / n_by_doc))
        row["worst_doc_per_1000"], row["worst_doc"] = round(1000 * float(by_doc[w] / n_by_doc[w]), 3), int(docs[w])
        if "generated" in st["masks"]:
            # a greedy answer changes only after a miss on its own path, so the items touched bound the task score's change
            row["items_touched"] = {
                p: [int(np.unique(doc[st["masks"][f"generated_{p}"] & ~hit]).size),
                    int(np.unique(doc[st["masks"][f"generated_{p}"]]).size)] for p in GEN_PANELS}
        out[str(k)] = row
    return out


def cmd_sweep(a: argparse.Namespace) -> None:
    st = strata(a.set)
    top1 = torch.from_numpy(st["pos"]["top1"].astype(np.int64))
    ids = torch.from_numpy(st["ids"].astype(np.int64))
    n = len(top1)
    H = hidden(WORK / "h" / f"{a.set}.h.f32")
    if H.shape[0] != n:
        raise SystemExit("hidden states and records differ")
    meta = json.loads((WORK / "previews.json").read_text(encoding="utf-8"))
    res = {"set": a.set, "positions": n, "strata": {k: int(v.sum()) for k, v in st["masks"].items()}, "ks": KS,
           "exact_row_bytes": Q6K_ROW, "random_miss_per_1000": {str(k): round(1000 * (1 - k / V), 3) for k in KS},
           "previews": {}}
    names = a.names.split(",") if a.names else [x for x in meta if not x.startswith("clus") and x != "freq"]
    for name in names:
        t0 = time.time()
        p = {k: v.to(DEV).float() for k, v in torch.load(WORK / "previews" / f"{name}.pt", weights_only=True).items()}
        r1 = np.empty(n, np.int64)
        r32 = np.empty((n, 32), np.int64)
        for i in range(0, n, CHUNK):
            z = logits(p, H[i:i + CHUNK].to(DEV))
            srt = z.sort(1).values
            for out, want in ((r1, top1[i:i + CHUNK, None]), (r32, ids[i:i + CHUNK])):
                v = z.gather(1, want.to(DEV))
                # the rank: how many logits are strictly above this one
                out[i:i + CHUNK] = (V - torch.searchsorted(srt, v.contiguous(), right=True)).cpu().numpy().reshape(
                    out[i:i + CHUNK].shape)
        res["previews"][name] = summarize(r1, r32, st, meta[name]["bytes_per_token"])
        del p
        torch.cuda.empty_cache()
        print(name, round(time.time() - t0, 1), json.dumps(res["previews"][name]["4096"]["miss_per_1000"]), flush=True)
    fr = torch.load(WORK / "previews" / "freq.pt", weights_only=True)["rank"].numpy()
    res["previews"]["freq"] = summarize(fr[top1.numpy()], fr[ids.numpy()], st, 0)
    for name in [x for x in meta if x.startswith("clus")]:
        c = torch.load(WORK / "previews" / f"{name}.pt", weights_only=True)
        cen, lists = c["centroids"].to(DEV), c["lists"].numpy()
        lab = np.concatenate([(torch.nn.functional.normalize(H[i:i + 8192].to(DEV), dim=1) @ cen.T).argmax(1).cpu().numpy()
                              for i in range(0, n, 8192)])
        r1 = np.full(n, 1 << 30, np.int64)
        r32 = np.full((n, 32), 1 << 30, np.int64)
        for cl in range(lists.shape[0]):
            m = lab == cl
            if not m.any():
                continue
            pos_of = np.full(V, 1 << 30, np.int64)
            pos_of[lists[cl]] = np.arange(lists.shape[1])
            r1[m] = pos_of[top1.numpy()[m]]
            r32[m] = pos_of[ids.numpy()[m]]
        res["previews"][name] = summarize(r1, r32, st, meta[name]["bytes_per_token"])
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def cmd_compose(a: argparse.Namespace) -> None:
    st = strata(a.set)
    H = hidden(WORK / "h" / f"{a.set}.h.f32")
    W = torch.load(WORK / "previews" / "exact.pt", weights_only=True)["w"].to(DEV).float()
    p = {k: v.to(DEV).float() for k, v in torch.load(WORK / "previews" / f"{a.preview}.pt", weights_only=True).items()}
    ks = [int(x) for x in a.ks.split(",")]
    n = H.shape[0]
    acc = {(k, o): [0.0, 0.0, 0, 0.0, 0.0] for k in ks for o in (0, 1)}
    for i in range(0, n, CHUNK):
        h = H[i:i + CHUNK].to(DEV)
        ze, zp = h @ W.T, logits(p, h)
        lp = ze.log_softmax(1)
        ref = ze.argmax(1)
        idx = zp.topk(max(ks), 1).indices
        for k in ks:
            c = idx[:, :k]
            for o in (0, 1):
                off = (ze.gather(1, c) - zp.gather(1, c)).mean(1, keepdim=True) if o else 0.0
                z = zp + off
                z.scatter_(1, c, ze.gather(1, c))
                top = c.gather(1, ze.gather(1, c).argmax(1, keepdim=True)).squeeze(1)
                e = acc[(k, o)]
                lq = z.log_softmax(1)
                e[0] += float((lp.exp() * (lp - lq)).sum())
                e[1] += float((top == ref).sum())
                e[2] += h.shape[0]
                e[3] += float((1 - lp.gather(1, c).exp().sum(1)).sum())      # the tail's mass, exact and composed
                e[4] += float((1 - lq.gather(1, c).exp().sum(1)).sum())
    res = {"set": a.set, "preview": a.preview, "cells": [
        {"k": k, "tail_offset": o, "kl_nats": round(v[0] / v[2], 5), "top1": round(v[1] / v[2], 5),
         "tail_mass_exact": round(v[3] / v[2], 5), "tail_mass_composed": round(v[4] / v[2], 5)}
        for (k, o), v in acc.items()]}
    print(json.dumps(res["cells"]))
    Path(a.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def cmd_materialize(a: argparse.Namespace) -> None:
    """A low-rank preview as a dense f16 output.weight in a copy of P0, for the panels' second context."""
    from s2h_head import BF16, BIN, FIT24, head_bf16, run
    p = torch.load(WORK / "previews" / f"{a.preview}.pt", weights_only=True)
    _, off, _, _ = head_bf16()
    dense = (p["a"].float() @ p["b"].T).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    bf, q4 = WORK / "m" / f"{a.preview}-bf16.gguf", WORK / "m" / f"{a.preview}-q4.gguf"
    shutil.copyfile(BF16, bf)
    with bf.open("r+b") as f:
        f.seek(off)
        f.write(dense.astype("<u2").tobytes())
    run([str(BIN / "llama-quantize.exe"), "--imatrix", FIT24.as_posix(), "--output-tensor-type", "f16",
         bf.as_posix(), q4.as_posix(), "Q4_K_M", "6"], WORK / "m" / f"{a.preview}-quantize.log")
    bf.unlink()
    got = {t.name: t for t in gguf.GGUFReader(str(q4)).tensors}
    ref = {t.name: t for t in gguf.GGUFReader(str(P0)).tensors}
    differ = [n for n in ref if got[n].tensor_type != ref[n].tensor_type
              or np.asarray(got[n].data).tobytes() != np.asarray(ref[n].data).tobytes()]
    o = got["output.weight"]
    d = gguf.quants.dequantize(o.data, o.tensor_type).reshape(V, D).astype(np.float32)
    want = (dense.astype(np.uint32) << 16).view(np.float32)
    rec = {"preview": a.preview, "q4": str(q4), "head_type": o.tensor_type.name, "tensors_differ_from_p0": differ,
           "head_max_abs_err": float(np.abs(d - want).max()), "head_max_abs": float(np.abs(want).max())}
    print(json.dumps(rec))
    Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    if differ != ["output.weight"] or o.tensor_type.name != "F16":
        raise SystemExit("the file differs from P0 in more than its head")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("sets").set_defaults(fn=cmd_sets)
    sp.add_parser("previews").set_defaults(fn=cmd_previews)
    m = sp.add_parser("materialize")
    m.add_argument("--preview", required=True)
    m.add_argument("--json", required=True)
    m.set_defaults(fn=cmd_materialize)
    s = sp.add_parser("sweep")
    s.add_argument("--set", required=True)
    s.add_argument("--names", default="")
    s.add_argument("--json", required=True)
    s.set_defaults(fn=cmd_sweep)
    c = sp.add_parser("compose")
    c.add_argument("--set", required=True)
    c.add_argument("--preview", required=True)
    c.add_argument("--ks", default="256,1024,4096")
    c.add_argument("--json", required=True)
    c.set_defaults(fn=cmd_compose)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
