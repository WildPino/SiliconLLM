# -*- coding: utf-8 -*-
"""E7 -- the largest real weights on this disk, end to end.  See
docs/research/donor_adaptation/briefs/BRIEF_E7_REAL_LARGE_DONOR.md (pushed at cb03580).

  python e7_real7b.py --stage parity     # G-L, G-P: the engine's logits vs PyTorch at 7.6 B
  python e7_real7b.py --stage generate   # G-G, G-C: greedy identity, and the planted control
  python e7_real7b.py --stage bench      # G-S: the pre-registered speed band, 3 reps per cell

The prompts are E6's, imported rather than re-typed: one definition, and they were frozen before
E6 ran, so nothing here can be shopped for.
"""
from __future__ import print_function
import argparse
import json
import os
import statistics
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e6_generate import PROMPTS, N_NEW, run, parse_gen  # noqa: E402

ENG = os.path.join(HERE, "donor_engine.exe")
TMP = r"D:\_ktmp\e7"
RES = os.path.join(HERE, "results", "e7")
HF = "Qwen/Qwen2.5-Coder-7B"
THREADS = 6

F32 = os.path.join(TMP, "qwen25-coder7b_f32.bin")
PACKED = os.path.join(TMP, "qwen25-coder7b_p.bin")

# G-S: the cells and the reps, fixed by the brief
BENCH_CELLS = [300, 800]
BENCH_REPS = 3


def ids_path(i):
    return os.path.join(TMP, "p%d.bin" % i)


def write_prompt_ids():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(HF)
    out = []
    os.makedirs(TMP, exist_ok=True)
    for i, p in enumerate(PROMPTS):
        ids = tk(p)["input_ids"]
        with open(ids_path(i), "wb") as f:
            f.write(struct.pack("<%di" % len(ids), *ids))
        out.append(ids)
    return out


def engine_stderr(cmd):
    """the header line and the 'consumed exactly N bytes' line go to stderr"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    so, se = p.communicate()
    return (p.returncode, so.decode("utf-8", "replace"), se.decode("utf-8", "replace"))


# --------------------------------------------------------------------------- G-L, G-P
def stage_parity():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    os.makedirs(RES, exist_ok=True)
    prompt_ids = write_prompt_ids()
    n = min(8, len(prompt_ids[3]))          # prompt 3 is the longest of the five
    idsf = ids_path(3)
    out = {"model": HF, "n_positions": n}

    # ---- G-L: the engine consumes exactly the file, and its header matches config.json
    cfg = AutoConfig.from_pretrained(HF)
    lg = os.path.join(TMP, "parity_logits.bin")
    rc, so, se = engine_stderr([ENG, "--weights", F32, "--threads", str(THREADS),
                                "--logits", idsf, str(n), lg])
    print(se.strip())
    if rc != 0:
        raise SystemExit("engine failed")
    hdr = {}
    for tok in se.replace("\n", " ").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            hdr[k] = v
    want = {"D": cfg.hidden_size, "F": cfg.intermediate_size, "L": cfg.num_hidden_layers,
            "V": cfg.vocab_size, "hd": cfg.hidden_size // cfg.num_attention_heads}
    gl = all(str(want[k]) == hdr.get(k) for k in want)
    gl = gl and ("consumed exactly" in se) and (str(os.path.getsize(F32)) in se)
    gl = gl and hdr.get("heads") == "%d/%d" % (cfg.num_attention_heads, cfg.num_key_value_heads)
    out["G_L"] = bool(gl)
    out["header"] = hdr
    print("G-L  header matches config.json and the file is consumed exactly : %s"
          % ("PASS" if gl else "FAIL"))

    # ---- G-P: those logits against PyTorch fp32
    ours = np.fromfile(lg, dtype=np.float32).reshape(n, cfg.vocab_size)
    m = AutoModelForCausalLM.from_pretrained(HF, torch_dtype=torch.float32,
                                             attn_implementation="eager")
    m.eval()
    torch.set_num_threads(THREADS)
    with torch.no_grad():
        ref = m(torch.tensor([prompt_ids[3][:n]])).logits[0].float().numpy()
    del m
    rel = float(np.linalg.norm(ours - ref) / np.linalg.norm(ref))
    top1 = float((ours.argmax(1) == ref.argmax(1)).mean())
    out["rel_l2"] = rel
    out["top1"] = top1
    out["G_P"] = bool(rel <= 1e-4 and top1 == 1.0)
    print("G-P  rel L2 %.3e (<= 1e-4), top-1 %.4f (== 1.0000) : %s"
          % (rel, top1, "PASS" if out["G_P"] else "FAIL"))
    json.dump(out, open(os.path.join(RES, "parity.json"), "w"), indent=1)
    os.remove(lg)


# --------------------------------------------------------------------------- G-G, G-C
def stage_generate():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(RES, exist_ok=True)
    prompt_ids = write_prompt_ids()
    out = {"model": HF, "n_new": N_NEW, "arms": {}}

    for name, wp in (("f32", F32), ("packed", PACKED)):
        if not os.path.exists(wp):
            print("  SKIP %s -- not on disk" % name)
            continue
        runs = []
        for i in range(len(PROMPTS)):
            pfx = os.path.join(TMP, "%s_p%d" % (name, i))
            r = parse_gen(run([ENG, "--weights", wp, "--threads", str(THREADS),
                               "--generate", ids_path(i), str(N_NEW), pfx]))
            for ext in (".prefill.bin",):
                if os.path.exists(pfx + ext):
                    os.remove(pfx + ext)
            runs.append({"prompt": i, "ids": r["ids"], "decode_toks": r["decode_toks"],
                         "prefill_toks": r["prefill_toks"]})
            print("  %s p%d  decode %.2f tok/s" % (name, i, r["decode_toks"]))
        out["arms"][name] = runs

    # ---- PyTorch greedy, with the KV cache on: without it a 7.6 B model re-reads the whole
    #      prefix 160 times and the reference alone costs half an hour
    tk = AutoTokenizer.from_pretrained(HF)
    m = AutoModelForCausalLM.from_pretrained(HF, torch_dtype=torch.float32,
                                             attn_implementation="eager")
    m.eval()
    torch.set_num_threads(THREADS)
    ref = []
    for i, p in enumerate(PROMPTS):
        ids = list(prompt_ids[i])
        gaps = []
        with torch.no_grad():
            o = m(torch.tensor([ids]), use_cache=True)
            for _ in range(N_NEW):
                lg = o.logits[0, -1].float()
                top = torch.topk(lg, 2)
                gaps.append(float(top.values[0] - top.values[1]))
                ids.append(int(top.indices[0]))
                o = m(torch.tensor([[ids[-1]]]), past_key_values=o.past_key_values,
                      use_cache=True)
        ref.append({"prompt": i, "ids": ids, "top2_gap": gaps})
        print("  REF p%d done" % i)
    del m
    out["ref"] = ref

    # ---- score
    rr = {x["prompt"]: x for x in ref}
    summary = {}
    for name, runs in out["arms"].items():
        tot = match = 0
        worst = None
        firstdiv = None
        for r in runs:
            ours = r["ids"][-N_NEW:]
            theirs = rr[r["prompt"]]["ids"][-N_NEW:]
            gaps = rr[r["prompt"]]["top2_gap"]
            for k in range(N_NEW):
                tot += 1
                if ours[k] == theirs[k]:
                    match += 1
                else:
                    if firstdiv is None:
                        firstdiv = (r["prompt"], k)
                    worst = gaps[k] if worst is None else max(worst, gaps[k])
        summary[name] = {"agree": match / float(tot), "matched": match, "counted": tot,
                         "first_div": firstdiv, "worst_gap_at_div": worst}
        print("  %-7s agreement %5.1f%%  (%d/%d)"
              % (name, 100.0 * summary[name]["agree"], match, tot))
    a = summary.get("f32")
    gg = bool(a) and a["agree"] >= 0.90 and (a["worst_gap_at_div"] is None
                                             or a["worst_gap_at_div"] < 1e-2)
    gc = ("packed" in summary) and summary["packed"]["agree"] < 0.90
    print("G-G  fp32 vs PyTorch greedy : %s" % ("PASS" if gg else "FAIL"))
    print("G-C  packed must NOT pass   : %s" % ("PASS" if gc else "FAIL -- E7 IS VOID"))
    out["summary"] = summary
    out["G_G"], out["G_C"] = gg, gc
    json.dump(out, open(os.path.join(RES, "generate.json"), "w"), indent=1)

    for name, runs in sorted(out["arms"].items()):
        print("---- %s" % name)
        for r in runs:
            print("  %-46s -> %s" % (json.dumps(PROMPTS[r["prompt"]]),
                                     json.dumps(tk.decode(r["ids"][-N_NEW:]))))


# --------------------------------------------------------------------------- G-S
def stage_bench():
    os.makedirs(RES, exist_ok=True)
    out = {"model": HF, "weights": PACKED, "threads": THREADS, "cells": {}}
    for ctx in BENCH_CELLS:
        rates = []
        for rep in range(BENCH_REPS):
            txt = run([ENG, "--weights", PACKED, "--threads", str(THREADS),
                       "--bench", str(ctx)])
            for line in txt.splitlines():
                if line.startswith("BENCH"):
                    rates.append(float(line.split()[4]))
                    print("  ctx %d rep %d : %s" % (ctx, rep + 1, line.strip()))
        med = statistics.median(rates)
        spread = (max(rates) - min(rates)) / med * 100.0
        out["cells"][str(ctx)] = {"rates": rates, "median": med, "spread_pct": spread}
        print("  ctx %d : median %.3f tok/s, spread %.2f%% over %d reps"
              % (ctx, med, spread, len(rates)))
    m300 = out["cells"]["300"]["median"]
    out["band_300"] = [4.0, 5.4]
    out["G_S"] = bool(4.0 <= m300 <= 5.4)
    print("G-S  300-context median %.3f in the pre-registered band 4.0-5.4 : %s"
          % (m300, "PASS" if out["G_S"] else "FAIL"))
    json.dump(out, open(os.path.join(RES, "bench.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["parity", "generate", "bench"])
    a = ap.parse_args()
    {"parity": stage_parity, "generate": stage_generate, "bench": stage_bench}[a.stage]()
