"""C50-S4: the student, its three arms, the teacher data and the 33-bucket KL; MVE-3 (memory and throughput per arm).

The base is T0 frozen, from its quantized blocks (`c50_s4_t0_gpu.T0Quant`), with the forward of `c50_s4_t0_torch.py`
rewritten here with the arms' hooks and per-layer activation checkpointing (each layer's weights are dequantized again
in the backward instead of being kept).

- arm "a" (S1's m2 layout, `c50_s1_expand.py`): 536 slots per MoE layer in 8 contiguous groups of 67; slot g*67 + 32k + j
  holds copy k of T0 expert 32g + j. Copy 0 is T0's expert, frozen, with T0's router row and bias. A copy k >= 1 is its
  source's frozen weights plus a trained rank-4 delta on gate, up and down (one factor zero at step 0), with a trained
  router row (S1's: the source row times 1 + 0.05 z, rounded to BF16) and a selection bias that starts below every
  original's (no copy picked at step 0, so the arm starts as T0) and rises by the base's aux-loss-free load rule.
- arm "b": a frozen bank of rows (one per trigram, else bigram, of the training split), read after the attention of
  layers 4 and 12: x += sigmoid(rms(x) . (K v + w) / sqrt(H)) * V v, with V zero at step 0.
- arm "c": LoRA of rank r on the fused QKV, the attention output, the shared expert (layers 1-19) and the dense FFN
  (layer 0); B zero at step 0.

    python -I benchmarks/chatbot50/c50_s4_student.py --mve3 --gguf G --teacher DIR --corpus TSV --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c50_s4_t0_torch as t0  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402
from c50_s4_t0_torch import EPS, H, HD, N_GROUP, N_LAYERS, NH, NKV, SCALE, TOPK, TOPK_GROUP, mm, rms, rope  # noqa: E402

E0, GE0 = 256, 32
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("lpt", "<f4"), ("lse", "<f8"),
                ("top", [("id", "<i4"), ("lp", "<f4")], (32,))])
BANK_SITES = (4, 12)
LORA_MODULES = ("attn_qkv", "attn_output", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
LORA_DENSE = ("attn_qkv", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


# ---------------------------------------------------------------- teacher data
class Teacher:
    """Block-1 shards and the corpus: one item per document, (ids, top-32 ids, top-32 log-probs) for positions 1..L-1."""

    def __init__(self, shard_dir: Path, corpus: Path, shards: list[int]):
        self.recs = [np.fromfile(shard_dir / f"shard_{s:06d}.bin", dtype=REC) for s in shards]
        docs = set()
        for r in self.recs:
            docs.update(np.unique(r["doc"]).tolist())
        self.ids = {}
        with corpus.open(encoding="ascii") as f:
            for line in f:
                d, s = line.split("\t", 1)
                if int(d) in docs:
                    self.ids[int(d)] = np.array(s.strip().rstrip(",").split(","), dtype=np.int64)

    def docs(self):
        for r in self.recs:
            bounds = np.flatnonzero(np.diff(r["doc"])) + 1
            for part in np.split(r, bounds):
                d = int(part["doc"][0])
                ids = self.ids[d]
                assert np.array_equal(part["pos"], np.arange(1, len(ids))), d
                assert np.array_equal(part["target"], ids[1:]), d
                yield d, ids, part["top"]["id"].astype(np.int64), part["top"]["lp"]

    def batches(self, tokens: int):
        """Documents packed in order until `tokens`; a longer document goes alone."""
        cur, n = [], 0
        for item in self.docs():
            L = len(item[1])
            if cur and n + L > tokens:
                yield cur
                cur, n = [], 0
            cur.append(item)
            n += L
        if cur:
            yield cur


def kl33(logits: torch.Tensor, tid: torch.Tensor, tlp: torch.Tensor) -> torch.Tensor:
    """Sum over positions of KL(teacher || student) on 33 buckets: the teacher's top-32 and the rest."""
    ls = torch.log_softmax(logits.float(), -1)
    lq = ls.gather(1, tid)
    q_rest = (1 - lq.exp().sum(-1)).clamp_min(1e-12)
    p = tlp.exp()
    p_rest = (1 - p.sum(-1)).clamp_min(1e-12)
    return ((p * (tlp - lq)).sum(-1) + p_rest * (p_rest.log() - q_rest.log())).sum()


# ---------------------------------------------------------------- arms
class LoRA(nn.Module):
    def __init__(self, n_out: int, n_in: int, r: int, dev):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, n_in, device=dev) / math.sqrt(n_in))
        self.B = nn.Parameter(torch.zeros(n_out, r, device=dev))

    def forward(self, x):
        return (x.float() @ self.A.T) @ self.B.T


class Bank(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.K = nn.Parameter(torch.randn(H, H, device=dev) / math.sqrt(H))
        self.V = nn.Parameter(torch.zeros(H, H, device=dev))
        self.w = nn.Parameter(torch.zeros(H, device=dev))

    def forward(self, x, v, hit):
        """x: [T, H] residual; v: [T, H] bank rows (zeros where there is no row); hit: [T] bool."""
        h = rms(x, torch.ones(H, device=x.device))
        g = torch.sigmoid((h * (v @ self.K.T + self.w)).sum(-1, keepdim=True) / math.sqrt(H))
        return x + (g * (v @ self.V.T)) * hit[:, None]


class NewExperts(nn.Module):
    """The copy >= 1 slots of one MoE layer: rank-4 deltas, router rows and biases."""

    def __init__(self, W: T0Quant, layer: int, rank: int, sigma: float, seed: int, offset: float | None):
        super().__init__()
        from c50_s1_expand import router_rows, slot_copy, slot_sources
        dev = W.dev(layer)
        e = 536
        self.src = torch.from_numpy(slot_sources(e)).to(dev)
        self.copy = torch.from_numpy(slot_copy(e)).to(dev)
        self.new = torch.nonzero(self.copy > 0).flatten()
        n = len(self.new)
        rows = router_rows(W.get(f"blk.{layer}.ffn_gate_inp.weight").cpu().numpy(), e, layer, sigma, seed)
        self.rows = nn.Parameter(torch.from_numpy(rows).to(dev)[self.new])
        # The selection bias gets no gradient (it is not in the weights); it starts `offset` below the source's and
        # follows the base's aux-loss-free load rule (`load_step`). offset None: 1 + the layer's bias spread + 0.01,
        # so that a copy's biased score (< 1 + its bias - offset) is below every original's (> min bias): at step 0 no
        # copy is picked and the model is T0.
        b0 = W.get(f"blk.{layer}.exp_probs_b.bias")
        if offset is None:
            offset = 1.0 + float(b0.max() - b0.min()) + 0.01
        self.offset = offset
        self.register_buffer("bias", b0[self.src[self.new]].clone() - offset)
        self.register_buffer("load", torch.zeros(e, device=dev))
        I = 512
        z = lambda *s: nn.Parameter(torch.zeros(*s, device=dev))
        g = lambda *s: nn.Parameter(torch.randn(*s, device=dev) / math.sqrt(s[-1]))
        self.gA, self.gB = z(n, I, rank), g(n, rank, H)
        self.uA, self.uB = z(n, I, rank), g(n, rank, H)
        self.dA, self.dB = z(n, H, rank), g(n, rank, I)
        self.slot_new = torch.full((e,), -1, dtype=torch.long, device=dev)
        self.slot_new[self.new] = torch.arange(n, device=dev)


# ---------------------------------------------------------------- student
class Student(nn.Module):
    def __init__(self, W: T0Quant, arm: str, rank: int = 0, seed: int = 20261016, offset: float | None = None,
                 bias_rate: float = 5e-3):
        super().__init__()
        self.W, self.arm, self.bias_rate = W, arm, bias_rate
        if arm == "a":
            self.newx = nn.ModuleDict({str(i): NewExperts(W, i, 4, 0.05, seed, offset) for i in range(1, N_LAYERS)})
        if arm == "b":
            self.bank = nn.ModuleDict({str(i): Bank(W.dev(i)) for i in BANK_SITES})
        if arm == "c":
            torch.manual_seed(seed)
            self.lora = nn.ModuleDict()
            for i in range(N_LAYERS):
                for m in (LORA_DENSE if i == 0 else LORA_MODULES):
                    w = W.qt[f"blk.{i}.{m}.weight"][1]
                    self.lora[f"{i}_{m}"] = LoRA(w[0], w[1], rank, W.dev(i))

    @torch.no_grad()
    def load_step(self) -> float:
        """The aux-loss-free rule on the copies' selection bias, after each step: b += rate * sign(mean load - load),
        loads counted over the step's forward (the checkpoint recompute counts again, equally for every slot). Returns
        the copies' share of the picks."""
        share = []
        for nx in self.newx.values():
            ld = nx.load
            if ld.sum() > 0:
                share.append(float(ld[nx.new].sum() / ld.sum()))
                nx.bias += self.bias_rate * torch.sign(ld.mean() - ld[nx.new])
            ld.zero_()
        return float(np.mean(share)) if share else 0.0

    def lin(self, x, i, m):
        y = mm(x, self.W.get(f"blk.{i}.{m}.weight"))
        if self.arm == "c" and f"{i}_{m}" in self.lora:
            y = y + self.lora[f"{i}_{m}"](x)
        return y

    def attention(self, x, i, segs):
        W = self.W
        h = rms(x, W.get(f"blk.{i}.attn_norm.weight"))
        qkv = self.lin(h, i, "attn_qkv")
        T = x.shape[0]
        q = rms(qkv[:, :NH * HD].view(T, NH, HD), W.get(f"blk.{i}.attn_q_norm.weight"))
        k = rms(qkv[:, NH * HD:(NH + NKV) * HD].view(T, NKV, HD), W.get(f"blk.{i}.attn_k_norm.weight"))
        v = qkv[:, (NH + NKV) * HD:].view(T, NKV, HD)
        wdt = W.wdtype
        outs = []
        for a, b in segs:
            pos = torch.arange(b - a, device=x.device)
            qs, ks = rope(q[a:b].transpose(0, 1), pos), rope(k[a:b].transpose(0, 1), pos)
            vs = v[a:b].transpose(0, 1)
            ks, vs = ks.repeat_interleave(NH // NKV, 0), vs.repeat_interleave(NH // NKV, 0)
            o = F.scaled_dot_product_attention(qs[None].to(wdt), ks[None].to(wdt), vs[None].to(wdt), is_causal=True)[0]
            outs.append(o.float().transpose(0, 1).reshape(b - a, NH * HD))
        return x + self.lin(torch.cat(outs), i, "attn_output")

    def swiglu(self, h, i, kind):
        if kind == "dense":
            g, u, d = "ffn_gate", "ffn_up", "ffn_down"
        else:
            g, u, d = "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"
        return self.lin(F.silu(self.lin(h, i, g)) * self.lin(h, i, u), i, d)

    def ffn(self, x, i):
        W = self.W
        h = rms(x, W.get(f"blk.{i}.ffn_norm.weight"))
        if i == 0:
            return x + self.swiglu(h, i, "dense")
        y = self.swiglu(h, i, "shared")
        gw, bias = W.get(f"blk.{i}.ffn_gate_inp.weight"), W.get(f"blk.{i}.exp_probs_b.bias")
        G, U, D = (W.get(f"blk.{i}.ffn_{n}_exps.weight") for n in ("gate", "up", "down"))
        if self.arm == "a":
            nx = self.newx[str(i)]
            rows = torch.cat([gw, nx.rows])           # originals at their T0 index, then the new slots
            b_all = torch.cat([bias, nx.bias])
            perm = torch.where(nx.copy == 0, nx.src, E0 + nx.slot_new)
            s = torch.sigmoid(mm(h, rows[perm]))
            sb = s + b_all[perm]
            n_exp = len(perm)
        else:
            s = torch.sigmoid(mm(h, gw))
            sb = s + bias
            n_exp = E0
        gs = n_exp // N_GROUP
        g = sb.view(-1, N_GROUP, gs).topk(2, -1)[0].sum(-1)
        keep = torch.zeros_like(g).scatter_(1, g.topk(TOPK_GROUP, -1)[1], 1.0)
        mask = keep[:, :, None].expand(-1, N_GROUP, gs).reshape(-1, n_exp).bool()
        idx = sb.masked_fill(~mask, float("-inf")).topk(TOPK, -1)[1]
        if self.arm == "a" and torch.is_grad_enabled():
            nx.load += torch.bincount(idx.flatten(), minlength=n_exp).float()
        w = s.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * SCALE
        for e in idx.unique().tolist():
            tok, slot = (idx == e).nonzero(as_tuple=True)
            ht = h[tok]
            if self.arm == "a":
                src, j = int(nx.src[e]), int(nx.slot_new[e])
            else:
                src, j = e, -1
            gt, ut = mm(ht, G[src]), mm(ht, U[src])
            if j >= 0:
                gt = gt + (ht @ nx.gB[j].T) @ nx.gA[j].T
                ut = ut + (ht @ nx.uB[j].T) @ nx.uA[j].T
            a = F.silu(gt) * ut
            o = mm(a, D[src])
            if j >= 0:
                o = o + (a @ nx.dB[j].T) @ nx.dA[j].T
            y = y.index_add(0, tok, o * w[tok, slot][:, None])
        return x + y

    def layer(self, x, i, segs, bank_v, bank_hit):
        x = self.attention(x, i, segs)
        if self.arm == "b" and i in BANK_SITES:
            x = self.bank[str(i)](x, bank_v.to(x.device), bank_hit.to(x.device))
        return self.ffn(x, i)

    def hidden(self, ids, segs, bank_v=None, bank_hit=None, ckpt=True):
        x = self.W.embed_rows(ids)
        for i in range(N_LAYERS):
            x = x.to(self.W.dev(i))
            if ckpt:
                x = checkpoint(self.layer, x, i, segs, bank_v, bank_hit, use_reentrant=False)
            else:
                x = self.layer(x, i, segs, bank_v, bank_hit)
        return rms(x, self.W.get("output_norm.weight"))

    def loss(self, ids, segs, tid, tlp, bank_v=None, bank_hit=None, chunk=1024):
        """Sum of the 33-bucket KL over every scored position (the last position of each document has no target)."""
        h = self.hidden(ids, segs, bank_v, bank_hit)
        keep = torch.ones(len(ids), dtype=torch.bool)
        for a, b in segs:
            keep[b - 1] = False
        h = h[keep.to(h.device)]
        tid, tlp = tid.to(h.device), tlp.to(h.device)
        out_w = self.W.get("output.weight")
        tot = h.new_zeros(())
        for a in range(0, len(h), chunk):
            tot = tot + checkpoint(lambda hh, ti, tl: kl33(mm(hh, out_w), ti, tl), h[a:a + chunk], tid[a:a + chunk],
                                   tlp[a:a + chunk], use_reentrant=False)
        return tot, int(len(h))


def pack(batch):
    ids = np.concatenate([b[1] for b in batch])
    segs, a = [], 0
    for b in batch:
        segs.append((a, a + len(b[1])))
        a += len(b[1])
    return (torch.from_numpy(ids), segs, torch.from_numpy(np.concatenate([b[2] for b in batch])),
            torch.from_numpy(np.concatenate([b[3] for b in batch]).astype(np.float32)))


# ---------------------------------------------------------------- MVE-3
@torch.no_grad()
def identity(S: Student, ids: torch.Tensor) -> dict:
    """The student at step 0 against T0's forward (`c50_s4_t0_torch.logits`) on one document: every arm starts as T0
    exactly (b: V = 0; c: B = 0; a: no copy can be picked at the starting bias, see `NewExperts`)."""
    a = mm(S.hidden(ids, [(0, len(ids))], *bank_none(S, ids), ckpt=False), S.W.get("output.weight"))
    b = t0.logits(ids, S.W)
    la, lb = torch.log_softmax(a.double(), -1), torch.log_softmax(b.double(), -1)
    return {"tokens": int(len(ids)), "max_abs_logit": float((a - b).abs().max()),
            "mean_abs_dlogp_targets": float((la[:-1].gather(1, ids[1:, None].to(la.device)) -
                                             lb[:-1].gather(1, ids[1:, None].to(lb.device))).abs().mean()),
            "top1_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean())}


def bank_none(S: Student, ids: torch.Tensor):
    if S.arm != "b":
        return None, None
    return torch.zeros(len(ids), H), torch.zeros(len(ids), dtype=torch.bool)


def mve3(gguf_path: Path, teacher: Path, corpus: Path, out: Path, arms: list[str], steps: int, tokens: int,
         cap_min: float, shards: int, devs=("cuda:0", "cuda:1"), bank_rows: int = 1_492_153) -> dict:
    rec = {"cell": "C50-S4 MVE-3", "torch": torch.__version__,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "arms": arms, "steps": steps, "tokens_per_step": tokens, "cap_min_per_arm": cap_min, "shards": shards,
           "optimizer": "AdamW lr 1e-4, fp32 trainables, GradScaler", "verdict": "none (feasibility record)"}
    print("CONFIG", json.dumps(rec), flush=True)
    cuda = devs[0].startswith("cuda")
    if cuda:
        assert rec["gpus"] == ["Tesla T4", "Tesla T4"], rec["gpus"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    sync = (lambda: [torch.cuda.synchronize(d) for d in range(2)]) if cuda else (lambda: None)
    peak = (lambda: [round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(2)]) if cuda else (lambda: [])
    T = Teacher(teacher, corpus, list(range(shards)))
    W = T0Quant(gguf_path, devs, wdtype=torch.float16 if cuda else torch.float32)
    first = next(T.docs())[1][:min(512, tokens)]
    bank = None
    rec["runs"] = []
    for arm in arms:
        spec, _, tok = arm.partition("@")          # kind[rank][@tokens per step]
        kind, rank = spec[0], int(spec[1:] or 0)
        tokens_arm = int(tok or tokens)
        if kind == "b" and bank is None:
            g = torch.Generator().manual_seed(0)
            bank = torch.empty(bank_rows, H, dtype=torch.float16)
            for a in range(0, len(bank), 1 << 16):
                bank[a:a + (1 << 16)] = (torch.randn(min(1 << 16, len(bank) - a), H, generator=g) * 0.02).half()
        if cuda:
            for d in range(2):
                torch.cuda.reset_peak_memory_stats(d)
        torch.manual_seed(0)
        # "A": arm a with the copies' bias at their source's (offset 0), copies picked from step 0: for timing the
        # regime with copies in use only (its step 0 is not T0, and it is recorded as such).
        S = Student(W, "a", offset=0.0) if kind == "A" else Student(W, kind, rank)
        ident = identity(S, torch.from_numpy(first))
        print("IDENTITY", arm, json.dumps(ident), flush=True)
        params = [p for p in S.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        r = {"arm": arm, "tokens_per_step": tokens_arm, "trainable": n_train, "identity_step0": ident, "steps_done": 0,
             "step_s": [], "positions": [], "loss": []}
        if kind in "aA":
            r["copy_offset"] = {i: round(nx.offset, 4) for i, nx in S.newx.items()}
            r["bias_rate"] = S.bias_rate
            r["copy_share"] = []
        print("ARM", arm, "trainable", n_train, flush=True)
        t_arm = time.time()
        try:
            for step, batch in enumerate(T.batches(tokens_arm)):
                if step >= steps or time.time() - t_arm > cap_min * 60:
                    break
                ids, segs, tid, tlp = pack(batch)
                bv = bh = None
                if kind == "b":
                    rows = torch.randint(-1, len(bank), (len(ids),), generator=torch.Generator().manual_seed(step))
                    bh = rows >= 0
                    bv = bank[rows.clamp_min(0)].float() * bh[:, None]
                sync()
                t1 = time.time()
                loss, npos = S.loss(ids, segs, tid, tlp, bv, bh)
                scaler.scale(loss / npos).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if kind in "aA":
                    r["copy_share"].append(round(S.load_step(), 5))
                sync()
                r["step_s"].append(round(time.time() - t1, 3))
                r["positions"].append(npos)
                r["loss"].append(round(float(loss) / npos, 5))
                r["steps_done"] = step + 1
                if step < 3 or step % 20 == 0:
                    print(f"{arm} step {step} pos {npos} loss {r['loss'][-1]} {r['step_s'][-1]}s "
                          f"mem {peak()}" + (f" copy_share {r['copy_share'][-1]}" if kind in "aA" else ""), flush=True)
        except torch.OutOfMemoryError as e:
            r["oom"] = str(e).splitlines()[0]
            print("OOM", arm, r["oom"], flush=True)
            opt.zero_grad(set_to_none=True)
        if r["step_s"]:
            st = np.array(r["step_s"][3:] or r["step_s"])
            ps = np.array(r["positions"][3:] or r["positions"])
            r["median_step_s"] = float(np.median(st))
            r["positions_per_s"] = float(ps.sum() / st.sum())
        r["mem_gib_peak"] = peak()
        r["wall_s"] = round(time.time() - t_arm, 1)
        rec["runs"].append(r)
        print("RESULT", json.dumps({k: v for k, v in r.items() if k not in ("step_s", "positions", "loss")}), flush=True)
        del S, opt, params
        if cuda:
            torch.cuda.empty_cache()
        out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mve3", action="store_true")
    ap.add_argument("--gguf", type=Path, default=t0.T0_GGUF)
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--arms", default="c48@4096,c48@8192,b@8192,c496@8192,a@8192,A@8192")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--cap-min", type=float, default=30.0)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cpu-smoke", action="store_true", help="local: both devices on the CPU, a small bank")
    a = ap.parse_args()
    if a.mve3:
        kw = {"devs": ("cpu", "cpu"), "bank_rows": 4096} if a.cpu_smoke else {}
        mve3(a.gguf, a.teacher, a.corpus, a.out, a.arms.split(","), a.steps, a.tokens, a.cap_min, a.shards, **kw)
