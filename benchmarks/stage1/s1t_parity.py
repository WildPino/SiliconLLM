"""Stage 1 (a) part 2: parity of the torch model (s1t_model.py) with llama.cpp, before any training.

On the dry run's documents (part 1b's s1m_dry.py work dir), the torch model runs the forms k = 8, top4, top4m8 and
top4m16 and writes, per scored position, its top-1 token.  It prints:
- top-1 agreement of torch k = 8 with llama.cpp's mode none on the same documents (bf16 here, Q4_K_M there);
- for each k = 4 form, its agreement with torch k = 8, next to llama.cpp's agreement of the same form with mode none
  (part 1b's dry run: top4 0.924, top4m8 0.953, top4m16 0.931): the torch forms must reproduce the hook's;
- the agreement of each torch form with llama.cpp's same form.
Optionally (--llama-bf16 DIR) the same against llama.cpp runs on the bf16 GGUF.
Usage: python -I s1t_parity.py --dry E:/SiliconLLM_Artifacts/stage1/drym --out results/s1t_parity.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1t_model as M  # noqa: E402

REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
FORMS = {"k8": M.Form(8, 0), "top4": M.Form(4, 0), "top4m8": M.Form(4, 8), "top4m16": M.Form(4, 16)}
LLAMA = {"k8": "none", "top4": "top4", "top4m8": "m8_top4", "top4m16": "m16_top4"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-bf16", default="")
    a = ap.parse_args()
    dry = Path(a.dry)
    docs = [[int(t) for t in ln.split("\t")[1].split(",")] for ln in (dry / "docs2.tsv").read_text().splitlines() if ln]
    dev = torch.device("cuda")
    t0 = time.time()
    W = M.Weights()
    g = W.get(["model.word_embeddings.weight", "model.norm.weight", "lm_head.weight"])
    hs = {f: [g["model.word_embeddings.weight"][torch.tensor(d)].to(dev) for d in docs] for f in FORMS}
    css = [M.rope(len(d), dev) for d in docs]
    with torch.no_grad():
        for i in range(M.N_LAYER):
            w = M.to(W.layer(i), dev)
            for f, form in FORMS.items():
                hs[f] = [M.layer_forward(i, w, h, cs, form) for h, cs in zip(hs[f], css)]
            del w
            torch.cuda.empty_cache()
        norm, head = g["model.norm.weight"].to(dev), g["lm_head.weight"].to(dev)
        top1 = {f: [torch.cat([(M.rms(h[s:s + 1024], norm) @ head.t()).float().argmax(-1)
                               for s in range(0, h.shape[0], 1024)]).cpu().numpy() for h in hs[f]] for f in FORMS}
    rec = {n: np.fromfile(dry / f"{n}.pos.bin", dtype=REC) for n in LLAMA.values()}

    def mine(f: str) -> np.ndarray:
        r = rec["none"]
        return np.array([top1[f][d][p - 1] for d, p in zip(r["doc"], r["pos"])])  # pos is the target's index

    agree = lambda x, y: float(np.mean(x == y))  # noqa: E731
    out = {"src": str(M.SRC), "positions": int(len(rec["none"])), "seconds": round(time.time() - t0, 1),
           "torch_k8_vs_llama_none": agree(mine("k8"), rec["none"]["top1"]),
           "torch_form_vs_torch_k8": {f: agree(mine(f), mine("k8")) for f in FORMS if f != "k8"},
           "llama_form_vs_llama_none": {f: agree(rec[LLAMA[f]]["top1"], rec["none"]["top1"]) for f in FORMS if f != "k8"},
           "torch_form_vs_llama_form": {f: agree(mine(f), rec[LLAMA[f]]["top1"]) for f in FORMS},
           "target_hit_torch_k8": agree(mine("k8"), rec["none"]["target"]),
           "target_hit_llama_none": agree(rec["none"]["top1"], rec["none"]["target"])}
    if a.llama_bf16:
        b = {n: np.fromfile(Path(a.llama_bf16) / f"{n}.pos.bin", dtype=REC) for n in LLAMA.values()}
        out["torch_form_vs_llama_bf16_form"] = {f: agree(mine(f), b[LLAMA[f]]["top1"]) for f in FORMS}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
