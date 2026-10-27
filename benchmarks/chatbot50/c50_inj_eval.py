"""C50 injection cell: one arm's evaluation on the injection sets (T4 x 2, or --cpu for smokes).

Generation is S4's (`c50_s4_eval.generate`: greedy, KV cache, stops on the chat's end ids). Sets, from
`c50_inj_data.py build` (`eval_<set>.tsv`, one row per fact and variant) and `known.tsv`:
  rew    the two rewordings (F0's, T0's) of each fact's question      32 new tokens
  orig   the original question                                          32
  cloze  F0's first passage up to its gold alias, plain text            16
  open   the original question with the first passage as context       32   (planted positive; v0 only by default)
  known  fresh nq/tqa items v0 already answers right (the guard)        32
For the branch arms the record also holds, per row, the share of (prompt token, layer) pairs where the gate fired
(`<set>.fire.f32`, aligned with the set's rows): one forward over the prompts, no generation, counted only over the
row's own span: the tokens after the prefix every row of the set shares and before the suffix they share (the chat
template and the instruction), so the shared frame cannot dilute the injected-against-twin contrast.

`--sample K --temp T` (arm v0, set qa): K draws per prompt at temperature T, seeded, for the "v0 does not know"
filter; writes `qa.samples.jsonl`.

    python -I c50_inj_eval.py --gguf V0 --arm v0|e|l [--weights PT] [--rot] --inputs DIR --out DIR [--sets ...]
    python -I c50_inj_eval.py --gguf V0 --arm v0 --inputs DIR --out DIR --sample 8 --temp 0.7
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
from c50_s4_student import Student  # noqa: E402
from c50_s4_t0_gpu import T0Quant  # noqa: E402
from c50_t1_train import AnchoredStudent  # noqa: E402
from c50_t2_n import NStudent, shuffle  # noqa: E402

SEED = 20261016
MAX_NEW = {"rew": 32, "orig": 32, "cloze": 16, "open": 32, "open_rew": 32, "known": 32}
DEFAULT_SETS = "rew,orig,cloze,known"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build(a, W: T0Quant):
    torch.manual_seed(SEED)
    if a.arm == "v0":
        S = Student(W, "t0")
    elif a.arm == "l":
        S = AnchoredStudent(W, 48, SEED, "all")
        S.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True), strict=True)
    else:
        S = NStudent(W, SEED)
        S.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True), strict=True)
        if a.rot:
            shuffle(S)
    S.eval()
    return S


def shared_frame(rows) -> tuple[int, int]:
    """The lengths of the prefix and of the suffix that every row shares (0, 0 for a single row)."""
    if len(rows) < 2:
        return 0, 0
    seqs = [list(r[1]) for r in rows]
    m = min(len(x) for x in seqs)
    pre = next((i for i in range(m) if len({x[i] for x in seqs}) > 1), m)
    suf = next((i for i in range(m - pre) if len({x[-1 - i] for x in seqs}) > 1), m - pre)
    return pre, suf


@torch.no_grad()
def prompt_fire(S: NStudent, rows) -> np.ndarray:
    """Per row: the share of (token, layer) pairs where the gate fired over the row's own span (see the docstring),
    from one packed forward per batch."""
    pre, suf = shared_frame(rows)
    out = []
    j = 0
    L = len(S.aux)
    while j < len(rows):
        grp, n = [], 0
        while j < len(rows) and (not grp or n + len(rows[j][1]) <= E.BATCH_TOKENS):
            grp.append(rows[j])
            n += len(rows[j][1])
            j += 1
        ids = np.concatenate([r[1] for r in grp])
        segs, a0 = [], 0
        for r in grp:
            segs.append((a0, a0 + len(r[1])))
            a0 += len(r[1])
        trace = []
        for ax in S.aux.values():
            ax.trace = trace
        S.hidden(torch.from_numpy(ids), segs, ckpt=False)
        for ax in S.aux.values():
            ax.trace = None
        assert len(trace) == L, len(trace)
        tok = torch.stack(trace).sum(0).float().numpy()
        for s0, s1 in segs:
            a_, b_ = s0 + pre, s1 - suf
            out.append(tok[a_:b_].sum() / ((b_ - a_) * L) if b_ > a_ else np.nan)
    return np.array(out, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--arm", required=True, choices=["v0", "e", "l"])
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--rot", action="store_true", help="arm e: expert e moved to slot (e + 1) mod 4")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default=DEFAULT_SETS)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    devs = ("cpu", "cpu") if a.cpu else ("cuda:0", "cuda:1")
    sets = ["qa"] if a.sample else a.sets.split(",")
    files = {s: a.inputs / ("qa.tsv" if s == "qa" else "known.tsv" if s == "known" else f"eval_{s}.tsv") for s in sets}
    cfg = {"cell": "C50 injection eval" + (" sample" if a.sample else ""), "arm": a.arm, "rot": a.rot,
           "gguf": str(a.gguf), "gguf_sha256": sha256(a.gguf), "weights": str(a.weights) if a.weights else None,
           "weights_sha256": sha256(a.weights) if a.weights else None, "sets": sets,
           "inputs_sha256": {s: sha256(p) for s, p in files.items()},
           "manifest_sha256": sha256(a.inputs / "manifest.json") if (a.inputs / "manifest.json").exists() else None,
           "sample": a.sample, "temp": a.temp if a.sample else None, "limit": a.limit, "torch": torch.__version__,
           "devs": list(devs), "gpus": [] if a.cpu else [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    print("CONFIG", json.dumps(cfg), flush=True)
    if not a.cpu:
        assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert (a.arm == "v0") == (a.weights is None) and (not a.rot or a.arm == "e") and (not a.sample or a.arm == "v0")
    W = T0Quant(a.gguf, devs, wdtype=torch.float32 if a.cpu else torch.float16)
    S = build(a, W)
    a.out.mkdir(parents=True, exist_ok=True)
    rec = {**cfg, "done": {}}
    for s in sets:
        t1 = time.time()
        rows = E.read_tsv(files[s])
        if a.limit:
            rows = rows[:a.limit]
        if a.sample:
            draws = {d: [] for d, _ in rows}
            for k in range(a.sample):
                torch.manual_seed(SEED + k)        # seeds the CPU and every CUDA generator

                def pick(lg):
                    return torch.multinomial(torch.softmax(lg.float() / a.temp, -1), 1).squeeze(-1)
                for r in E.generate(S, None, rows, MAX_NEW["orig"], pick=pick):
                    draws[r["item"]].append(r["gen"])
            (a.out / "qa.samples.jsonl").write_text("".join(json.dumps({"item": d, "gens": g}) + "\n"
                                                           for d, g in draws.items()), encoding="utf-8")
            rec["done"][s] = {"rows": len(rows), "draws": a.sample, "wall_s": round(time.time() - t1, 1)}
        else:
            g = E.generate(S, None, rows, MAX_NEW[s])
            (a.out / f"{s}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g), encoding="utf-8")
            done = {"rows": len(rows), "generated": sum(len(x["gen"]) for x in g), "wall_s": round(time.time() - t1, 1)}
            if isinstance(S, NStudent):
                f = prompt_fire(S, rows)
                f.tofile(a.out / f"{s}.fire.f32")
                done["fire_mean"] = round(float(np.nanmean(f)), 5)
                done["fire_frame"] = shared_frame(rows)
            rec["done"][s] = done
        print("SET", s, json.dumps(rec["done"][s]), flush=True)
        (a.out / "eval.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
