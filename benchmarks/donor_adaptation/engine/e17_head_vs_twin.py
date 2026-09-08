# -*- coding: utf-8 -*-
"""E17 follow-up: does removing the ternary HEAD change the arm's own output at all?

G-H1/G-H2 compare each arm to PyTorch.  This compares the head-fp32 arm to its head-ternary
TWIN -- same body, same rule, same calibration -- so it reads the head's effect directly rather
than through the reference.  Descriptive; no gate, nothing registered rides on it.
"""
import json, os, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e6_generate import PROMPTS, N_NEW, run, parse_gen

ENGINE = os.path.join(HERE, "donor_engine.exe")
E1 = r"D:\_ktmp\e1"
TMP = r"D:\_ktmp\e17"
e6 = json.load(open(os.path.join(HERE, "results", "e6", "engine.json"), encoding="utf-8"))
ref = json.load(open(os.path.join(HERE, "results", "e6", "ref.json"), encoding="utf-8"))

# E6's stored head-TERNARY ids, by prompt
TWIN = {"H1": {r["prompt"]: r["ids"][-N_NEW:] for r in e6["arms"]["A3"]["runs"]},
        "H2": {r["prompt"]: r["ids"][-N_NEW:] for r in e6["arms"]["A2"]["runs"]}}
ARMS = [("H1", "qwen25-15b_tq.bin", "Qwen/Qwen2.5-1.5B", "A3"),
        ("H2", "qwen25-05b_tq.bin", "Qwen/Qwen2.5-0.5B", "A2")]

from transformers import AutoTokenizer
out = {}
for tag, wf, hf, twin in ARMS:
    tk = AutoTokenizer.from_pretrained(hf)
    same = tot = 0
    firstdiff = None
    texts = []
    for i, p in enumerate(PROMPTS):
        pids = tk(p)["input_ids"]
        ip = os.path.join(TMP, "d_p%d.bin" % i)
        open(ip, "wb").write(struct.pack("<%di" % len(pids), *pids))
        pfx = os.path.join(TMP, "%s_diff_p%d" % (tag, i))
        r = parse_gen(run([ENGINE, "--weights", os.path.join(E1, wf), "--threads", "6",
                           "--generate", ip, str(N_NEW), pfx]))
        ours = r["ids"][-N_NEW:]
        theirs = TWIN[tag][i]
        s = sum(1 for a, b in zip(ours, theirs) if a == b)
        d = next((k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
        if d is not None and firstdiff is None:
            firstdiff = [i, d]
        same += s; tot += N_NEW
        texts.append(tk.decode(ours))
        for ext in (".prefill.bin", ".ids.bin"):
            if os.path.exists(pfx + ext):
                os.remove(pfx + ext)
    out[tag] = {"twin": twin, "same_as_twin": same, "of": tot,
                "frac": same / float(tot), "first_diff": firstdiff, "texts": texts}
    print("%s vs %s (head-ternary twin): %d/%d identical (%.2f%%), first diff %s"
          % (tag, twin, same, tot, 100.0 * same / tot, firstdiff))
    for i, t in enumerate(texts):
        print("   p%d ours   : %r" % (i, t))
        print("   p%d twin   : %r" % (i, tk.decode(TWIN[tag][i])))
json.dump(out, open(os.path.join(HERE, "results", "e17_head_vs_twin.json"), "w",
                    encoding="utf-8"), indent=1)
print("wrote results/e17_head_vs_twin.json")
