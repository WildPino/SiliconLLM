# -*- coding: utf-8 -*-
"""E6 -- the donor speaks.  Runs the gates of briefs/BRIEF_E6_GENERATE.md, in the order the brief
fixes them: G-P and G-D BEFORE any text is read, then G-A on the fp32 arm, then G-C, the planted
control, on the ternary arm.

Every other mode in this engine is teacher-forced.  This one lets the model choose the next token,
so it is the first time a donor emits anything of its own here.

  python e6_generate.py --stage engine     # our runtime, all arms, all prompts (G-P, G-D)
  python e6_generate.py --stage ref        # PyTorch greedy reference (needs the HF cache)
  python e6_generate.py --stage score      # G-A / G-C / the transcript
"""
from __future__ import print_function
import argparse
import json
import os
import struct
import subprocess
import sys

ENG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donor_engine.exe")
TMP = r"D:\_ktmp\e6"
RES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e6"))

# frozen in the brief, section 5 -- not editable after the first run without voiding E6
PROMPTS = ["The capital of France is",
           "def fibonacci(n):",
           "Water boils at",
           "The three laws of motion were formulated by",
           "import numpy as np"]
N_NEW = 32
THREADS = 6

ARMS = [
    # name, weights, the HF model it is a conversion of, role
    ("A1", r"D:\_ktmp\e1\qwen25-05b_f32.bin", "Qwen/Qwen2.5-0.5B", "known-positive (fp32)"),
    ("A2", r"D:\_ktmp\e1\qwen25-05b_tqh.bin", "Qwen/Qwen2.5-0.5B", "planted control (ternary+head)"),
    ("A3", r"D:\_ktmp\e1\qwen25-15b_tqh.bin", "Qwen/Qwen2.5-1.5B", "planted control at 1.5B"),
]


def ids_path(i):
    return os.path.join(TMP, "p%d.bin" % i)


def write_prompt_ids():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    out = []
    for i, p in enumerate(PROMPTS):
        ids = tk(p)["input_ids"]
        with open(ids_path(i), "wb") as f:
            f.write(struct.pack("<%di" % len(ids), *ids))
        out.append(ids)
    return out


def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    so, se = p.communicate()
    if p.returncode != 0:
        sys.stderr.write(se.decode("utf-8", "replace"))
        raise SystemExit("engine failed: %s" % " ".join(cmd))
    return so.decode("utf-8", "replace")


def parse_gen(txt):
    d = {}
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "GEN_IDS":
            d["ids"] = [int(x) for x in f[1:]]
        elif f[0].startswith("GEN_"):
            d[f[0][4:].lower()] = float(f[1])
    return d


def stage_engine():
    os.makedirs(RES, exist_ok=True)
    prompt_ids = write_prompt_ids()
    out = {"n_new": N_NEW, "threads": THREADS, "prompts": PROMPTS,
           "prompt_ids": prompt_ids, "arms": {}}
    for name, wp, hf, role in ARMS:
        if not os.path.exists(wp):
            print("  SKIP %s -- %s not on disk" % (name, wp))
            continue
        rec = {"weights": wp, "hf": hf, "role": role, "runs": []}
        for i in range(len(PROMPTS)):
            pfx = os.path.join(TMP, "%s_p%d" % (name, i))
            base = [ENG, "--weights", wp, "--threads", str(THREADS)]
            r1 = parse_gen(run(base + ["--generate", ids_path(i), str(N_NEW), pfx]))
            # G-D: the same arm, twice, must give byte-identical ids
            r2 = parse_gen(run(base + ["--generate", ids_path(i), str(N_NEW), pfx + "_b"]))
            gd = (open(pfx + ".ids.bin", "rb").read() ==
                  open(pfx + "_b.ids.bin", "rb").read())
            # G-P: the prefill logits must be byte-identical to what --logits writes for the
            # same ids -- i.e. generation is the forward pass E1 already gated, not a second one
            lg = pfx + ".ref_logits.bin"
            run(base + ["--logits", ids_path(i), str(len(prompt_ids[i])), lg])
            gp = (open(pfx + ".prefill.bin", "rb").read() == open(lg, "rb").read())
            os.remove(lg)
            os.remove(pfx + "_b.ids.bin")
            os.remove(pfx + "_b.prefill.bin")
            os.remove(pfx + ".prefill.bin")
            rec["runs"].append({"prompt": i, "ids": r1["ids"],
                                "prefill_s": r1["prefill_s"], "decode_s": r1["decode_s"],
                                "decode_toks": r1["decode_toks"],
                                "prefill_toks": r1["prefill_toks"],
                                "G_P": bool(gp), "G_D": bool(gd)})
            print("  %s p%d  G-P %s  G-D %s  decode %.2f tok/s"
                  % (name, i, "PASS" if gp else "FAIL", "PASS" if gd else "FAIL",
                     r1["decode_toks"]))
        out["arms"][name] = rec
    with open(os.path.join(RES, "engine.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote " + os.path.join(RES, "engine.json"))


def stage_ref():
    """PyTorch greedy, fp32, on CPU -- the same five prompts, the same 32 steps.  Records the
    top-2 logit gap at every step, because the brief allows a divergence only where that gap is
    below 1e-2 (a tie-break rather than a different model)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(RES, exist_ok=True)
    prompt_ids = write_prompt_ids()
    out = {}
    for hf in sorted(set(a[2] for a in ARMS)):
        tk = AutoTokenizer.from_pretrained(hf)
        m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
        m.eval()
        torch.set_num_threads(THREADS)
        runs = []
        for i, p in enumerate(PROMPTS):
            ids = list(prompt_ids[i])
            gaps = []
            with torch.no_grad():
                for _ in range(N_NEW):
                    lg = m(torch.tensor([ids])).logits[0, -1].float()
                    top = torch.topk(lg, 2)
                    gaps.append(float(top.values[0] - top.values[1]))
                    ids.append(int(top.indices[0]))
            runs.append({"prompt": i, "ids": ids, "top2_gap": gaps})
            print("  REF %s p%d done" % (hf, i))
        out[hf] = runs
        del m
    with open(os.path.join(RES, "ref.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote " + os.path.join(RES, "ref.json"))


def stage_score():
    from transformers import AutoTokenizer
    eng = json.load(open(os.path.join(RES, "engine.json")))
    ref = json.load(open(os.path.join(RES, "ref.json")))
    tks = {}
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    w("E6 -- a real pretrained donor generating text on donor_engine.c")
    w("n_new=%d, threads=%d, greedy argmax, five prompts frozen in the brief"
      % (eng["n_new"], eng["threads"]))
    w("")

    # ---- G-P and G-D first, before any text is read
    gp_all = gd_all = True
    for name, rec in sorted(eng["arms"].items()):
        for r in rec["runs"]:
            gp_all &= r["G_P"]
            gd_all &= r["G_D"]
    w("G-P  prefill logits byte-identical to --logits : %s" % ("PASS" if gp_all else "FAIL"))
    w("G-D  same arm twice, byte-identical ids        : %s" % ("PASS" if gd_all else "FAIL"))
    w("")

    # ---- agreement
    w("%-4s %-30s %8s %8s %10s %s" % ("arm", "role", "agree", "first-div", "worst gap", "decode"))
    summary = {}
    for name, rec in sorted(eng["arms"].items()):
        hf = rec["hf"]
        rr = {x["prompt"]: x for x in ref[hf]}
        # the brief fixes the denominator at 5 prompts x n_new positions; every position is
        # counted, including those after a divergence has already split the two contexts
        tot = match = 0
        firstdiv = None
        worst_gap = None
        rates = []
        for r in rec["runs"]:
            ours = r["ids"][-eng["n_new"]:]
            theirs = rr[r["prompt"]]["ids"][-eng["n_new"]:]
            gaps = rr[r["prompt"]]["top2_gap"]
            for k in range(eng["n_new"]):
                tot += 1
                if ours[k] == theirs[k]:
                    match += 1
                else:
                    if firstdiv is None:
                        firstdiv = (r["prompt"], k)
                    # the REF top-2 gap at this position: a divergence is allowed only where
                    # PyTorch itself was choosing between two nearly equal logits
                    worst_gap = gaps[k] if worst_gap is None else max(worst_gap, gaps[k])
            rates.append(r["decode_toks"])
        share = match / float(tot) if tot else 0.0
        summary[name] = {"agree": share, "matched": match, "counted": tot,
                         "first_div": firstdiv, "worst_gap_at_div": worst_gap,
                         "decode_toks": rates}
        w("%-4s %-30s %7.1f%% %8s %10s %.2f tok/s"
          % (name, rec["role"], 100.0 * share,
             ("p%d/%d" % firstdiv) if firstdiv else "none",
             ("%.4f" % worst_gap) if worst_gap is not None else "-",
             sum(rates) / len(rates)))
    w("")

    a1 = summary.get("A1")
    ga = bool(a1) and a1["agree"] >= 0.90 and (a1["worst_gap_at_div"] is None
                                               or a1["worst_gap_at_div"] < 1e-2)
    controls = [k for k in summary if k != "A1"]
    gc = all(summary[k]["agree"] < 0.90 for k in controls) if controls else False
    w("G-A  A1 vs PyTorch greedy, >=90%% and any divergence a tie-break : %s"
      % ("PASS" if ga else "FAIL"))
    w("G-C  planted control: the ternary arms must NOT also pass       : %s"
      % ("PASS" if gc else "FAIL -- E6 IS VOID"))
    w("")
    w("VERDICT: %s" % ("the donor speaks, and it is the same model PyTorch runs"
                       if (ga and gc and gp_all and gd_all) else "gates failed -- see above"))
    w("")

    # ---- the transcripts, last
    for name, rec in sorted(eng["arms"].items()):
        hf = rec["hf"]
        if hf not in tks:
            tks[hf] = AutoTokenizer.from_pretrained(hf)
        tk = tks[hf]
        rr = {x["prompt"]: x for x in ref[hf]}
        w("---- %s  %s  (%s)" % (name, os.path.basename(rec["weights"]), rec["role"]))
        for r in rec["runs"]:
            p = eng["prompts"][r["prompt"]]
            ours = tk.decode(r["ids"][-eng["n_new"]:])
            theirs = tk.decode(rr[r["prompt"]]["ids"][-eng["n_new"]:])
            w("  prompt : %s" % json.dumps(p))
            w("  ours   : %s" % json.dumps(ours))
            if name == "A1":
                w("  pytorch: %s" % json.dumps(theirs))
        w("")

    open(os.path.join(RES, "score.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["engine", "ref", "score"])
    a = ap.parse_args()
    {"engine": stage_engine, "ref": stage_ref, "score": stage_score}[a.stage]()
