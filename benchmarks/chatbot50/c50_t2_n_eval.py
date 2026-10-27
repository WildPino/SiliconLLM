"""C50-T2 option N: the PyTorch evaluator for v0, the N arm (and its shuffle) and the extra-epoch control E2.

S4's evaluator (`c50_s4_eval.score`, `generate`), with v0's GGUF as the base, on these panels:
- docs: S4's 36 documents (A4's inputs, S4's split), for R;
- nq, tqa, mmlu, arc: addendum 2's fresh items (`t1/fresh/inputs`), whole;
- gsm8k, chat: A4's panels, whole (plain draws: no donor-gap item is left out).
Arms: v0 (the base alone), n (`c50_t2_n.NStudent` with --weights; --n-mode shuffle for the control), e2 (S4's LoRA
r48 on v0, with the E2 training's weights). The record holds the GGUF's and the weights' sha256 and, for n, the share
of (token, layer) pairs where the branch fired, per panel and per layer; on the documents it also writes
`docs.fire.u8`, the number of layers (0-19) where the branch fired at each scored position, aligned with `docs.pos.bin`.

    python -I benchmarks/chatbot50/c50_t2_n_eval.py --gguf V0 --arm v0|n|e2 [--weights final.pt] [--n-mode full|shuffle]
        --a4-inputs DIR --fresh DIR --split S.json --out DIR [--panels ...] [--limit N] [--cpu]
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
from c50_t2_n import NStudent, shuffle  # noqa: E402

SEED = 20261016
PANELS = ("docs", "nq", "tqa", "mmlu", "arc", "gsm8k", "chat")
FRESH = ("nq", "tqa", "mmlu", "arc")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build(a, W: T0Quant):
    torch.manual_seed(SEED)
    if a.arm == "v0":
        assert a.weights is None
        S = Student(W, "t0")
    elif a.arm == "e2":
        S = Student(W, "c", 48, seed=SEED)
        S.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True), strict=True)
    else:
        S = NStudent(W, SEED)
        S.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True), strict=True)
        if a.n_mode == "shuffle":
            shuffle(S)
    S.eval()
    return S


def rows_of(a, panel: str, split: dict):
    if panel == "docs":
        return E.selected("docs", E.read_tsv(a.a4_inputs / "docs.tsv"), split)
    src = a.fresh if panel in FRESH else a.a4_inputs
    return E.read_tsv(src / f"{panel}.tsv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--arm", required=True, choices=["v0", "n", "e2"])
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--n-mode", default="full", choices=["full", "shuffle"])
    ap.add_argument("--a4-inputs", type=Path, required=True)
    ap.add_argument("--fresh", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--panels", default=",".join(PANELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: this many sequences per panel")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    devs = ("cpu", "cpu") if a.cpu else ("cuda:0", "cuda:1")
    cfg = {"cell": "C50-T2 option N eval", "arm": a.arm, "n_mode": a.n_mode if a.arm == "n" else None,
           "gguf": str(a.gguf), "gguf_sha256": sha256(a.gguf), "weights": str(a.weights) if a.weights else None,
           "weights_sha256": sha256(a.weights) if a.weights else None, "panels": a.panels.split(","),
           "fresh": str(a.fresh), "fresh_manifest_sha256": sha256(a.fresh / "manifest.json")
           if (a.fresh / "manifest.json").exists() else None, "limit": a.limit, "torch": torch.__version__,
           "devs": list(devs), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
           if not a.cpu else []}
    print("CONFIG", json.dumps(cfg), flush=True)
    if not a.cpu:
        assert cfg["gpus"] == ["Tesla T4", "Tesla T4"], cfg["gpus"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert (a.arm == "v0") == (a.weights is None)
    split = json.loads(a.split.read_text(encoding="utf-8"))
    W = T0Quant(a.gguf, devs, wdtype=torch.float32 if a.cpu else torch.float16)
    S = build(a, W)
    a.out.mkdir(parents=True, exist_ok=True)
    rec = {**cfg, "panels_done": {}}
    for p in cfg["panels"]:
        t1 = time.time()
        rows = rows_of(a, p, split)
        if a.limit:
            rows = rows[:a.limit]
        if a.arm == "n":
            S.fire_by_layer()
        trace = [] if a.arm == "n" and p == "docs" else None
        if trace is not None:
            for ax in S.aux.values():
                ax.trace = trace
        kind, max_new = E.PANELS[p]
        if kind == "score":
            r = E.score(S, None, rows)
            r.tofile(a.out / f"{p}.pos.bin")
            n = int(len(r))
        if trace is not None:
            # one forward per batch, layers 1-19 in order, no checkpoint: 19 masks per batch, over whole sequences
            for ax in S.aux.values():
                ax.trace = None
            L = len(S.aux)
            assert len(trace) % L == 0
            tok = torch.cat([torch.stack(trace[j:j + L]).sum(0) for j in range(0, len(trace), L)]).numpy()
            lens = np.array([len(x[1]) for x in rows])
            assert lens.sum() == len(tok), (lens.sum(), len(tok))
            keep = np.ones(len(tok), dtype=bool)
            keep[np.cumsum(lens) - 1] = False             # a sequence's last token predicts nothing
            fc = tok[keep].astype(np.uint8)
            assert len(fc) == len(r), (len(fc), len(r))
            fc.tofile(a.out / "docs.fire.u8")
        else:
            g = E.generate(S, None, rows, max_new)
            (a.out / f"{p}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g), encoding="utf-8")
            n = sum(len(x["gen"]) for x in g)
        done = {"sequences": len(rows), "positions_or_generated": n, "wall_s": round(time.time() - t1, 1)}
        if a.arm == "n":
            fl = S.fire_by_layer()
            done["fire_by_layer"] = {k: round(v, 5) for k, v in fl.items() if v is not None}
            done["fire"] = round(sum(done["fire_by_layer"].values()) / len(done["fire_by_layer"]), 5)
        rec["panels_done"][p] = done
        print("PANEL", p, json.dumps(done), flush=True)
        (a.out / "eval.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
