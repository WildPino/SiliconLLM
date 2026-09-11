#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E26 GATES -- does donor_engine.c's CARVED FFN compute what PyTorch computes?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md,
pushed before any of this existed.

Phase 60's law: kernel-bit-exact does NOT compose to system-correctness, so a new kernel is
gated end to end.  The carve introduces four things at once -- a permutation of the F axis, a
TRANSPOSED packed down_proj, a row-selected matvec, and a router with a top-k -- and each of
them can be wrong in a way that still produces plausible logits.  Three gates separate them:

  G-E26P  the carved artifact at --carve-k E against the DENSE artifact of the same donor, both
          through the engine.  At k = E the carve keeps every neuron, so the two files describe
          the SAME function; what differs is the permutation, the transposed packing, the
          transposed kernel and the row list at full width.  Bar 1e-4: this is one summation
          order against another on the same trits, not a quantization difference.
  G-E26S  the SELECTED GROUPS, engine against reference, every layer and every position.  A
          logits comparison can pass while the router is subtly wrong; a set comparison cannot.
  G-E26R  the carved artifact at k < E against a PyTorch reference that applies the same
          permutation, the same ternarization and the same carve.  Bar 2e-3, parity_gate's.

The reference dequantizes through qwen_export.quantize -- the exporter's OWN quantizer, not a
re-derivation -- and builds the router through carve_common.router_weights, the same function
the exporter called.  Neither side reads the artifact under test to decide what it should be.

  python e26_parity_carve.py --dense D:/_ktmp/e26/qwen15b_dense.bin \
      --carved D:/_ktmp/e26/qwen15b_carve.bin --engine ./donor_engine_e26.exe --k 64
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "ternary")))

import carve_common as CV          # noqa: E402  -- perm/router/top-k, one definition
import qwen_export as QX           # noqa: E402  -- the exporter's own quantizer

TOL_P = 1e-4        # G-E26P: same trits, a different summation order
TOL_R = 2e-3        # G-E26R: parity_gate.py's bar, unchanged and not re-argued
RE_CARVE = re.compile(r"^CARVE l=(\d+) k=(\d+):(.*)$")


def deq(w, rule="R0"):
    """The dequantized ternary form of one matrix, through the EXPORTER's quantizer."""
    q, s = QX.quantize(w.data if hasattr(w, "data") else w, rule, None)
    return (q.float() * s.float()[:, None]).contiguous()


class CarvedMLP(torch.nn.Module):
    """The reference carve: score E groups, keep the top k, zero the rest before down_proj.

    Zeroing is the right REFERENCE even though the engine skips: the non-kept neurons multiply
    columns of down that the engine never reads, so the two agree exactly in exact arithmetic.
    That equivalence is the whole reason a carve can be a speed lever at all, and E19's hook
    priced it without ever testing it in a runtime.
    """

    def __init__(self, gate_w, up_w, down_w, wr, E, k, act_fn, log):
        super().__init__()
        self.gw, self.uw, self.dw, self.wr = gate_w, up_w, down_w, wr
        self.E, self.k, self.act, self.log = E, k, act_fn, log
        self.F = gate_w.shape[0]
        self.GSZ = self.F // E

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        scores = flat @ self.wr.T                      # [T, E]
        out = torch.zeros(flat.shape[0], self.dw.shape[0], dtype=flat.dtype)
        for t in range(flat.shape[0]):
            sel = CV.topk_lower_ties(scores[t].detach().numpy(), self.k)
            self.log.append(sel.tolist())
            h = self.act(flat[t] @ self.gw.T) * (flat[t] @ self.uw.T)
            keep = torch.zeros(self.F, dtype=torch.bool)
            for g in sel:
                keep[g * self.GSZ:(g + 1) * self.GSZ] = True
            out[t] = (h * keep) @ self.dw.T
        return out.reshape(x.shape[:-1] + (self.dw.shape[0],))


def run_engine(engine, weights, idf, n, lof, threads, extra=()):
    cmd = [engine, "--weights", weights, "--threads", str(threads)] + list(extra) + \
          ["--logits", idf, str(n), lof]
    r = subprocess.run(cmd, capture_output=True)
    err = r.stderr.decode(errors="replace")
    if r.returncode != 0:
        print(err[-3000:])
        raise SystemExit("engine failed: " + " ".join(cmd))
    return err


def compare(ours, ref, tok, label, tol):
    print("\n[%s]\n pos |    max|diff|  |  rel l2   | our argmax | ref argmax | top1 tok" % label)
    worst = 0.0
    for t in range(ours.shape[0]):
        d = np.abs(ours[t] - ref[t])
        rel = float(np.linalg.norm(ours[t] - ref[t]) / (np.linalg.norm(ref[t]) + 1e-30))
        worst = max(worst, rel)
        ao, ar = int(ours[t].argmax()), int(ref[t].argmax())
        print(" %3d | %12.6f | %9.2e | %10d | %10d | %s"
              % (t, float(d.max()), rel, ao, ar,
                 (repr(tok.decode([ar]))[:18] if tok else "")))
    top1 = float(np.mean(ours.argmax(1) == ref.argmax(1)))
    print(" worst relative l2 = %.3e   top-1 = %.4f   (bar %.0e)" % (worst, top1, tol))
    return worst, top1, bool(worst < tol and top1 == 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="the dense artifact of the same donor")
    ap.add_argument("--carved", required=True, help="the quant==4 carved artifact")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--model", default=None, help="default: read from the carved sidecar")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--k", type=int, default=64, help="the k for G-E26S and G-E26R")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e26_parity_carve.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()

    side = json.load(open(a.carved + ".json", encoding="utf-8"))
    model_id = a.model or side["model"]
    rev = a.revision or side.get("revision")
    E, seed, rule = int(side["carve_E"]), int(side["carve_seed"]), side["rule"]
    labels = side["carve_labels"]
    print("carved artifact: %s  E=%d  group=%d  rule=%s  router seed=%d"
          % (os.path.basename(a.carved), E, side["carve_group_size"], rule, seed))
    assert side["quant"] == "carved", "not a carved artifact"
    assert side["fold"] == "none", "the reference below does not fold"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    ids = np.array(tok("The capital of France is Paris, and the capital of Italy is")
                   ["input_ids"][: a.n], dtype="<i4")
    n = len(ids)
    tmp = os.path.dirname(a.carved) or "."
    idf = os.path.join(tmp, "_e26_ids.bin")
    ids.tofile(idf)
    print("tokens (%d): %s" % (n, ids.tolist()), flush=True)

    m = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.float32,
                                             attn_implementation="eager").eval()
    V, L, D, F = m.config.vocab_size, m.config.num_hidden_layers, \
        m.config.hidden_size, m.config.intermediate_size
    assert F % E == 0

    # ------------------------------------------------------------------ G-E26P
    la = os.path.join(tmp, "_e26_dense.bin")
    lb = os.path.join(tmp, "_e26_kE.bin")
    print(run_engine(a.engine, a.dense, idf, n, la, a.threads).strip())
    print(run_engine(a.engine, a.carved, idf, n, lb, a.threads,
                     ("--carve-k", str(E))).strip())
    dense = np.fromfile(la, dtype="<f4").reshape(n, V)
    kE = np.fromfile(lb, dtype="<f4").reshape(n, V)
    wP, t1P, okP = compare(kE, dense, tok, "G-E26P  carved --carve-k %d  vs  dense" % E, TOL_P)

    # ------------------------------------------------------------------ the engine at k < E
    lc = os.path.join(tmp, "_e26_k.bin")
    err = run_engine(a.engine, a.carved, idf, n, lc, a.threads,
                     ("--carve-k", str(a.k), "--carve-dump"))
    ours = np.fromfile(lc, dtype="<f4").reshape(n, V)
    eng_sel = []
    for line in err.splitlines():
        mo = RE_CARVE.match(line.strip())
        if mo:
            assert int(mo.group(2)) == a.k
            eng_sel.append([int(x) for x in mo.group(3).split()])
    print("\nengine dumped %d selections (expected %d = %d positions x %d layers)"
          % (len(eng_sel), n * L, n, L))
    assert len(eng_sel) == n * L, "the --carve-dump stream is not one line per layer per token"

    # ------------------------------------------------------------------ the reference
    lab = np.load(labels)
    nq = 0
    for li in range(L):
        lay = m.model.layers[li]
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(lay.self_attn, nm)
            mod.weight.data = deq(mod.weight.data, rule)
            nq += 1
    if side.get("head_ternary"):
        m.lm_head.weight = torch.nn.Parameter(deq(m.lm_head.weight.data, rule),
                                              requires_grad=False)
        nq += 1
    ref_log = []
    for li in range(L):
        lay = m.model.layers[li]
        pm = torch.from_numpy(CV.perm_from_labels(lab["c%d" % li], E))
        gw = deq(lay.mlp.gate_proj.weight.data[pm], rule)
        uw = deq(lay.mlp.up_proj.weight.data[pm], rule)
        dw = deq(lay.mlp.down_proj.weight.data[:, pm], rule)
        wr = deq(torch.from_numpy(CV.router_weights(D, E, seed, li)), rule)
        lay.mlp = CarvedMLP(gw, uw, dw, wr, E, a.k, torch.nn.functional.silu, ref_log)
        nq += 4
    print("reference: %d tensors dequantized through qwen_export.quantize(rule=%s), "
          "%d carved MLPs installed" % (nq, rule, L))
    with torch.no_grad():
        ref = m(torch.tensor(ids.astype(np.int64))[None, :]).logits[0].float().numpy()

    # ------------------------------------------------------------------ G-E26S
    # The reference's log is layer-major over one batched forward (layer 0 for all T, then
    # layer 1 ...); the engine's is position-major.  Reindex rather than compare in order.
    ref_sel = [None] * (n * L)
    for li in range(L):
        for t in range(n):
            ref_sel[t * L + li] = ref_log[li * n + t]
    bad = [i for i in range(n * L) if eng_sel[i] != ref_sel[i]]
    okS = not bad
    print("\n[G-E26S] selected groups, %d comparisons (%d positions x %d layers): %s"
          % (n * L, n, L, "IDENTICAL" if okS else "%d DISAGREE" % len(bad)))
    if bad:
        i = bad[0]
        print("  first disagreement at position %d layer %d:\n    engine %s\n    ref    %s"
              % (i // L, i % L, eng_sel[i][:12], ref_sel[i][:12]))

    # ------------------------------------------------------------------ G-E26R
    wR, t1R, okR = compare(ours, ref, tok, "G-E26R  carved --carve-k %d  vs  PyTorch" % a.k,
                           TOL_R)

    ok = okP and okS and okR
    print("\n GATE E26-P: %s   GATE E26-S: %s   GATE E26-R: %s   [%.0fs]"
          % ("FIRES" if okP else "FAILS", "FIRES" if okS else "FAILS",
             "FIRES" if okR else "FAILS", time.time() - t0))
    print(" " + ("ALL THREE FIRE -- the carved path reproduces PyTorch, the selection is "
                 "exact, and keeping every group reproduces the dense engine. Timing may "
                 "proceed." if ok else
                 "AT LEAST ONE FAILED -- no speed number may be quoted from the carved path."))
    json.dump({"gates": ["E26-P", "E26-S", "E26-R"], "model": model_id, "revision": rev,
               "dense": a.dense, "carved": a.carved, "E": E, "k": a.k, "n": n, "rule": rule,
               "router_seed": seed, "labels": labels,
               "P_worst_rel_l2": wP, "P_top1": t1P, "P_tol": TOL_P, "P_fires": bool(okP),
               "S_comparisons": n * L, "S_disagreements": len(bad), "S_fires": bool(okS),
               "R_worst_rel_l2": wR, "R_top1": t1R, "R_tol": TOL_R, "R_fires": bool(okR),
               "all_fire": bool(ok), "tokens": ids.tolist(), "seconds": time.time() - t0},
              open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
