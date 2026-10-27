"""C50-S4: the PyTorch evaluator (S4 protocol, panels and rules; MVE-5 runs it on T0).

It writes the llama.cpp tools' formats, so S2's and A4's checks score its outputs unchanged:
- score panels (docs, arc, mmlu): `<panel>.pos.bin`, one record per position p >= 1 of each sequence (doc = the input's
  first column, target = token p, its log-probability, the top-1 and its log-probability, the log-sum-exp);
- generation panels (nq, tqa, gsm8k, chat): `<panel>.jsonl`, {"item", "prompt_tokens", "stop", "gen"}, greedy, up to
  32 / 32 / 384 / 256 new tokens; a stop ID (156892, 156895) ends an item as "EOS" and stays in `gen`, else "MAX".
Only the S4 half of the donor-gap panel and the general panel are run (`results/c50_s4_split.json`): the S5 half is
never evaluated. The forward is `c50_s4_student.py`'s, with a KV cache for generation.

    python -I benchmarks/chatbot50/c50_s4_eval.py --gguf G --inputs DIR --split S.json --arm t0|a|b|c48|c496
        [--weights final.pt] [--bank DIR --bank-mode full|remove|shuffle|mask] [--a-mode full|shuffle] --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c50_s4_bank import BankReader  # noqa: E402
from c50_s4_student import BANK_SITES, Student  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402
from c50_s4_t0_torch import HD, N_LAYERS, NH, NKV, mm, rms, rope  # noqa: E402

REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
STOP_IDS = (156892, 156895)
PANELS = {"docs": ("score", 0), "arc": ("score", 0), "mmlu": ("score", 0),
          "nq": ("gen", 32), "tqa": ("gen", 32), "gsm8k": ("gen", 384), "chat": ("gen", 256)}
ARMS = {"t0": ("t0", 0), "a": ("a", 0), "b": ("b", 0), "c48": ("c", 48), "c496": ("c", 496)}
SEED, BATCH_TOKENS, CACHE_POSITIONS, HEAD_CHUNK = 20261016, 8192, 65536, 1024


def read_tsv(p: Path) -> list[tuple[int, np.ndarray]]:
    out = []
    for line in p.read_text(encoding="ascii").splitlines():
        d, s = line.split("\t", 1)
        out.append((int(d), np.array(s.strip().rstrip(",").split(","), dtype=np.int64)))
    return out


def selected(panel: str, rows, split: dict):
    """The panel's sequences that S4 may read: the S5 half of the donor-gap panel is dropped; documents are S4's."""
    if panel == "docs":
        keep = set(split["s4"]["docs"])
        return [r for r in rows if r[0] in keep]
    drop = set(split["s5"].get(panel, []))
    item = (lambda d: d // 8) if PANELS[panel][0] == "score" else (lambda d: d)
    return [r for r in rows if item(r[0]) not in drop]


# ---------------------------------------------------------------- arms
def a_shuffle(S: Student, seed: int = SEED) -> None:
    """Arm (a)'s shuffle control: within each group of each layer, every copy's router row, bias and deltas move to
    another copy's slot; the frozen sources stay. The picks' frequencies are unchanged."""
    rng = np.random.default_rng(seed)
    for i, nx in S.newx.items():
        grp = (nx.new // 67).cpu().numpy()
        perm = np.arange(len(grp))
        for g in np.unique(grp):
            idx = np.flatnonzero(grp == g)
            perm[idx] = idx[rng.permutation(len(idx))]
        p = torch.from_numpy(perm).to(nx.rows.device)
        with torch.no_grad():
            for t in (nx.rows, nx.gA, nx.gB, nx.uA, nx.uB, nx.dA, nx.dB, nx.bias):
                t.copy_(t[p])


def build(a, W: T0Quant) -> tuple[Student, BankReader | None]:
    kind, rank = ARMS[a.arm]
    torch.manual_seed(SEED)
    S = Student(W, kind, rank, seed=SEED)
    if a.weights:
        S.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True), strict=True)
    else:
        assert kind == "t0", "a trained arm needs --weights"
    if kind == "a" and a.a_mode == "shuffle":
        a_shuffle(S)
    bank = BankReader(a.bank, a.bank_mode) if kind == "b" else None
    S.eval()
    return S, bank


def bank_for(S: Student, bank: BankReader | None, ids: np.ndarray, segs):
    if S.arm != "b":
        return None, None
    v, hit, _, _ = bank.read(ids, segs)
    return v, hit


# ---------------------------------------------------------------- score panels
@torch.no_grad()
def score(S: Student, bank, seqs: list[tuple[int, np.ndarray]]) -> np.ndarray:
    out = []
    i = 0
    while i < len(seqs):
        batch, n = [], 0
        while i < len(seqs) and (not batch or n + len(seqs[i][1]) <= BATCH_TOKENS):
            batch.append(seqs[i])
            n += len(seqs[i][1])
            i += 1
        ids = np.concatenate([b[1] for b in batch])
        segs, a0 = [], 0
        for b in batch:
            segs.append((a0, a0 + len(b[1])))
            a0 += len(b[1])
        bv, bh = bank_for(S, bank, ids, segs)
        h = S.hidden(torch.from_numpy(ids), segs, bv, bh, ckpt=False)
        w = S.W.get("output.weight")
        for (d, x), (s0, s1) in zip(batch, segs):
            for c in range(s0, s1 - 1, HEAD_CHUNK):
                c1 = min(c + HEAD_CHUNK, s1 - 1)
                lg = mm(h[c:c1].to(w.device), w).float()
                lse = torch.logsumexp(lg, -1)
                tgt = torch.from_numpy(ids[c + 1:c1 + 1]).to(lg.device)
                lt = lg.gather(1, tgt[:, None])[:, 0] - lse
                m, top = lg.max(-1)
                r = np.zeros(c1 - c, dtype=REC)
                r["doc"], r["pos"], r["target"] = d, np.arange(c - s0 + 1, c1 - s0 + 1), ids[c + 1:c1 + 1]
                r["top1"], r["logp_target"] = top.cpu().numpy(), lt.cpu().numpy()
                r["logp_top1"], r["lse"] = (m - lse).cpu().numpy(), lse.double().cpu().numpy()
                out.append(r)
    return np.concatenate(out)


# ---------------------------------------------------------------- generation with a KV cache
class Cache:
    def __init__(self, S: Student, B: int, L: int):
        dt = S.W.wdtype
        self.k = [torch.zeros(B, NKV, L, HD, dtype=dt, device=S.W.dev(i)) for i in range(N_LAYERS)]
        self.v = [torch.zeros(B, NKV, L, HD, dtype=dt, device=S.W.dev(i)) for i in range(N_LAYERS)]


def qkv(S: Student, x, i):
    W = S.W
    h = rms(x, W.get(f"blk.{i}.attn_norm.weight"))
    t = S.lin(h, i, "attn_qkv")
    T = x.shape[0]
    q = rms(t[:, :NH * HD].view(T, NH, HD), W.get(f"blk.{i}.attn_q_norm.weight"))
    k = rms(t[:, NH * HD:(NH + NKV) * HD].view(T, NKV, HD), W.get(f"blk.{i}.attn_k_norm.weight"))
    return q, k, t[:, (NH + NKV) * HD:].view(T, NKV, HD)


def attn_prefill(S: Student, x, i, segs, rows, cache: Cache):
    q, k, v = qkv(S, x, i)
    wdt = S.W.wdtype
    outs = []
    for (a, b), r in zip(segs, rows):
        pos = torch.arange(b - a, device=x.device)
        qs, ks, vs = rope(q[a:b].transpose(0, 1), pos), rope(k[a:b].transpose(0, 1), pos), v[a:b].transpose(0, 1)
        cache.k[i][r, :, :b - a] = ks.to(wdt)
        cache.v[i][r, :, :b - a] = vs.to(wdt)
        ks, vs = ks.repeat_interleave(NH // NKV, 0), vs.repeat_interleave(NH // NKV, 0)
        o = F.scaled_dot_product_attention(qs[None].to(wdt), ks[None].to(wdt), vs[None].to(wdt), is_causal=True)[0]
        outs.append(o.float().transpose(0, 1).reshape(b - a, NH * HD))
    return x + S.lin(torch.cat(outs), i, "attn_output")


def attn_decode(S: Student, x, i, rows, pos, cache: Cache):
    """One new token per active item, at position pos[b], attending to its cached positions 0..pos[b]."""
    q, k, v = qkv(S, x, i)
    wdt = S.W.wdtype
    p = pos.to(x.device)
    q, k = rope(q.transpose(0, 1), p).transpose(0, 1), rope(k.transpose(0, 1), p).transpose(0, 1)
    r = rows.to(x.device)
    cache.k[i][r, :, p] = k.to(wdt)
    cache.v[i][r, :, p] = v.to(wdt)
    L = int(p.max()) + 1
    K = cache.k[i][r, :, :L].repeat_interleave(NH // NKV, 1)
    V = cache.v[i][r, :, :L].repeat_interleave(NH // NKV, 1)
    mask = (torch.arange(L, device=x.device)[None, :] <= p[:, None])[:, None, None, :]
    o = F.scaled_dot_product_attention(q[:, :, None, :].to(wdt), K, V, attn_mask=mask)[:, :, 0]
    return x + S.lin(o.float().reshape(len(r), NH * HD), i, "attn_output")


def run_layers(S: Student, x, attn, bank_v, bank_hit):
    for i in range(N_LAYERS):
        x = x.to(S.W.dev(i))
        x = attn(x, i)
        if S.arm == "b" and i in BANK_SITES:
            x = S.bank[str(i)](x, bank_v.to(x.device), bank_hit.to(x.device))
        x = S.ffn(x, i)
    return rms(x, S.W.get("output_norm.weight"))


@torch.no_grad()
def generate(S: Student, bank, prompts: list[tuple[int, np.ndarray]], max_new: int, pick=None) -> list[dict]:
    """Greedy (argmax, ties to the lowest id); `pick`, if given, maps a [B, V] logits tensor to B next ids instead."""
    pick = pick or (lambda lg: lg.argmax(-1))
    res = []
    i0 = 0
    while i0 < len(prompts):
        grp = []
        while i0 < len(prompts) and (len(grp) + 1) * (max(len(p[1]) for p in grp + [prompts[i0]]) + max_new) \
                <= CACHE_POSITIONS:
            grp.append(prompts[i0])
            i0 += 1
        B, L = len(grp), max(len(p[1]) for p in grp) + max_new
        cache = Cache(S, B, L)
        seqs = [list(p[1]) for p in grp]
        gen = [[] for _ in grp]
        stop = [None] * B
        nxt = [0] * B
        # prefill, packed in chunks of whole prompts
        j = 0
        while j < B:
            rows, n = [], 0
            while j < B and (not rows or n + len(seqs[j]) <= BATCH_TOKENS):
                rows.append(j)
                n += len(seqs[j])
                j += 1
            ids = np.concatenate([np.array(seqs[r]) for r in rows])
            segs, a0 = [], 0
            for r in rows:
                segs.append((a0, a0 + len(seqs[r])))
                a0 += len(seqs[r])
            bv, bh = bank_for(S, bank, ids, segs)
            x = S.W.embed_rows(torch.from_numpy(ids))
            h = run_layers(S, x, lambda x, i: attn_prefill(S, x, i, segs, rows, cache), bv, bh)
            last = torch.tensor([b - 1 for _, b in segs], device=h.device)
            lg = mm(h[last].to(S.W.get("output.weight").device), S.W.get("output.weight"))
            for r, t in zip(rows, pick(lg).tolist()):
                nxt[r] = t
        for step in range(max_new):
            for b in range(B):
                if stop[b] is None:
                    gen[b].append(nxt[b])
                    seqs[b].append(nxt[b])
                    if nxt[b] in STOP_IDS:
                        stop[b] = "EOS"
                    elif len(gen[b]) == max_new:
                        stop[b] = "MAX"
            act = [b for b in range(B) if stop[b] is None]
            if not act:
                break
            ids = np.array([seqs[b][-1] for b in act])
            pos = torch.tensor([len(seqs[b]) - 1 for b in act])
            if S.arm == "b":
                tails = [np.array(seqs[b][-3:]) for b in act]
                segs, a0 = [], 0
                for t in tails:
                    segs.append((a0, a0 + len(t)))
                    a0 += len(t)
                v, hit = bank_for(S, bank, np.concatenate(tails), segs)
                last = [s1 - 1 for _, s1 in segs]
                bv, bh = v[last], hit[last]
            else:
                bv = bh = None
            x = S.W.embed_rows(torch.from_numpy(ids))
            rows_t = torch.tensor(act)
            h = run_layers(S, x, lambda x, i: attn_decode(S, x, i, rows_t, pos, cache), bv, bh)
            lg = mm(h.to(S.W.get("output.weight").device), S.W.get("output.weight"))
            for b, t in zip(act, pick(lg).tolist()):
                nxt[b] = t
        for (d, p), g, s in zip(grp, gen, stop):
            res.append({"item": d, "prompt_tokens": len(p), "stop": s, "gen": [int(t) for t in g]})
        del cache
    return res


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="the panels' TSV files (A4's and S2's inputs)")
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--bank", type=Path)
    ap.add_argument("--bank-mode", default="full", choices=["full", "remove", "shuffle", "mask"])
    ap.add_argument("--a-mode", default="full", choices=["full", "shuffle"])
    ap.add_argument("--panels", default=",".join(PANELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: this many sequences per panel")
    ap.add_argument("--selfcheck", action="store_true", help="cached decoding against a full recompute, then exit")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    devs = ("cpu", "cpu") if a.cpu else ("cuda:0", "cuda:1")
    cfg = {"cell": "C50-S4 eval", "arm": a.arm, "weights": str(a.weights) if a.weights else None,
           "bank": str(a.bank) if a.bank else None, "bank_mode": a.bank_mode if a.arm == "b" else None,
           "a_mode": a.a_mode if a.arm == "a" else None, "panels": a.panels.split(","), "limit": a.limit,
           "torch": torch.__version__, "devs": list(devs),
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if not a.cpu else []}
    print("CONFIG", json.dumps(cfg), flush=True)
    if not a.cpu:
        assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert (a.arm == "b") == (a.bank is not None)
    split = json.loads(a.split.read_text(encoding="utf-8"))
    W = T0Quant(a.gguf, devs, wdtype=torch.float32 if a.cpu else torch.float16)
    S, bank = build(a, W)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.selfcheck:
        return selfcheck(S, bank, read_tsv(a.inputs / "nq.tsv")[:2], a.out)
    rec = {**cfg, "panels_done": {}}
    if a.arm == "b" and a.bank_mode == "mask":
        rec["mask_lookup_share"] = mask_share(bank, a.inputs, split)
    for p in cfg["panels"]:
        t1 = time.time()
        rows = selected(p, read_tsv(a.inputs / f"{p}.tsv"), split)
        if a.limit:
            rows = rows[:a.limit]
        kind, max_new = PANELS[p]
        if kind == "score":
            r = score(S, bank, rows)
            r.tofile(a.out / f"{p}.pos.bin")
            n = int(len(r))
        else:
            g = generate(S, bank, rows, max_new)
            (a.out / f"{p}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g), encoding="utf-8")
            n = sum(len(x["gen"]) for x in g)
        rec["panels_done"][p] = {"sequences": len(rows), "positions_or_generated": n, "wall_s": round(time.time() - t1, 1)}
        print("PANEL", p, json.dumps(rec["panels_done"][p]), flush=True)
        (a.out / "eval.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DONE", flush=True)
    return 0


def mask_share(bank: BankReader, inputs: Path, split) -> float:
    """The share of the S4 documents' lookups that the mask control hides (protocol: reported)."""
    rows = selected("docs", read_tsv(inputs / "docs.tsv"), split)
    r = np.concatenate([bank.rows(x, [(0, len(x))]) for _, x in rows])
    hit = r >= 0
    return float(bank.masked[r[hit]].sum() / hit.sum())


@torch.no_grad()
def selfcheck(S: Student, bank, prompts, out: Path) -> int:
    """Generate 4 tokens with the cache, then recompute each step's logits from the full sequence."""
    g = generate(S, bank, prompts, 4)
    agree = True
    for (d, p), r in zip(prompts, g):
        seq = np.concatenate([p, np.array(r["gen"], dtype=np.int64)])
        bv, bh = bank_for(S, bank, seq, [(0, len(seq))])
        h = S.hidden(torch.from_numpy(seq), [(0, len(seq))], bv, bh, ckpt=False)
        w = S.W.get("output.weight")
        lg = mm(h[len(p) - 1:len(seq) - 1].to(w.device), w)
        agree &= lg.argmax(-1).tolist() == r["gen"]
    res = {"selfcheck": "cached greedy equals full-recompute greedy", "agree": bool(agree), "items": len(prompts)}
    (out / "selfcheck.json").write_text(json.dumps(res) + "\n", encoding="utf-8")
    print("SELFCHECK", json.dumps(res), flush=True)
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
