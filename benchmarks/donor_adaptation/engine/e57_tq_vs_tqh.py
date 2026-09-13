"""Post-E57 check: do 05b_tq and 05b_tqh actually emit DIFFERENT tokens?

E57's quality phase returned 3/160 for BOTH, with the same first-divergence positions.
At 1.5B the two arms differ (12 vs 10), so they are not the same code path -- but "the
counts coincided" is not the same statement as "the sequences coincided", and only one of
those is a measurement.  This makes it one.

Run only when the machine is idle of E57's timing phase.
"""
import json, os, subprocess, sys
HERE = r"D:\_THINGS\Progetti\SiliconLLM\benchmarks\donor_adaptation\engine"
sys.path.insert(0, HERE)
import e51_math_errno as E51

ENGINE = os.path.join(HERE, "donor_engine_e53.exe")
PAIRS = [("05b_tq", r"D:\_ktmp\e1\qwen25-05b_tq.bin"),
         ("05b_tqh", r"D:\_ktmp\e1\qwen25-05b_tqh.bin"),
         ("15b_tq", r"D:\_ktmp\e1\qwen25-15b_tq.bin"),
         ("15b_tqh", r"D:\_ktmp\e1\qwen25-15b_tqh.bin")]

e6 = json.load(open(os.path.join(E51.E6_RES, "engine.json")))
n_new = e6["n_new"]
E51.write_prompt_ids(e6["prompt_ids"])
seqs = {}
for tag, wp in PAIRS:
    seqs[tag] = []
    for i in range(len(e6["prompt_ids"])):
        pfx = os.path.join(E51.TMP, "e57chk_%s_p%d" % (tag, i))
        ids = os.path.join(E51.TMP, "p%d.bin" % i)
        r = E51.parse_gen(E51.run([ENGINE, "--weights", wp, "--threads", "6",
                                   "--generate", ids, str(n_new), pfx], "gen")[0])
        seqs[tag].append(r["ids"][-n_new:])
        for f in (pfx + ".prefill.bin", pfx + ".ids.bin"):
            if os.path.exists(f):
                os.remove(f)

for a, b in (("05b_tq", "05b_tqh"), ("15b_tq", "15b_tqh")):
    same = sum(1 for x, y in zip(seqs[a], seqs[b]) if x == y)
    tokw = sum(1 for x, y in zip(seqs[a], seqs[b]) for p, q in zip(x, y) if p == q)
    print("%-8s vs %-8s : %d of %d prompt sequences IDENTICAL, %d of %d tokens equal"
          % (a, b, same, len(seqs[a]), tokw, len(seqs[a]) * n_new))
