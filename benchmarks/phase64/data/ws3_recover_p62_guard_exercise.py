"""Both-direction exercise of the architecture guard in ws3_recover_p62.py.

No BPB is computed here: the adopted numbers are NOT re-derived. This exercises the LOAD PATH only --
must-pass on the three real artefacts, must-refuse on five planted ones.
"""
import copy, io, os, sys, zipfile, torch

ROOT = os.path.abspath(".")
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase64", "data"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase64", "mve"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
import ws3_recover_p62 as T

RES = os.path.join(ROOT, "kaggle_rung1", "results")
ARMS = {"arm1_V2048": (2048, 0), "arm2_V4096": (4096, 0), "arm3_xproj26_V2048": (2048, 26)}
npass = nfail = 0


def attempt(label, ck, arm, table_rank, V, expect):
    """expect='pass' -> must build and load; expect='refuse' -> must raise SystemExit."""
    global npass, nfail
    try:
        model, swapped, allowed = T.build_from_cfg(ck, arm, table_rank, V, "cpu")
        T.load_or_refuse(model, ck, arm, allowed)
        got, msg = "pass", ""
    except SystemExit as e:
        got, msg = "refuse", str(e).splitlines()[0]
    ok = (got == expect)
    npass, nfail = npass + ok, nfail + (not ok)
    print(f"  [{'OK ' if ok else 'XX '}] atteso {expect:6s} -> ottenuto {got:6s}   {label}")
    if msg:
        print(f"           {msg}")
    return ok


print("=" * 96)
print("ESERCIZIO NELLE DUE DIREZIONI -- guardia architetturale di ws3_recover_p62.py")
print("Nessuna BPB calcolata: i numeri adottati NON vengono ri-derivati.")
print("=" * 96)

cks = {}
print("\nMUST-PASS -- i tre artefatti veri, l'architettura derivata dal loro cfg\n")
for arm, (V, rank) in ARMS.items():
    cks[arm] = T.load_extracted(os.path.join(RES, arm))
    cfg = cks[arm]["cfg"]
    attempt(f"{arm}  (cfg: stages={cfg['stages']!r} xproj_rank={cfg['xproj_rank']} "
            f"qat_alpha={cfg['qat_alpha']} recall={cfg['recall']!r})", cks[arm], arm, rank, V, "pass")

print("\nMUST-REFUSE -- cinque difetti piantati, uno per ogni via da cui la guardia puo' perdere\n")

# P1: the table contradicts the artefact. This is arm3's specific risk: the low-rank swap.
attempt("P1  la tabella ARMS dice xproj_rank=0, il cfg dice 26 (rischio specifico di arm3)",
        cks["arm3_xproj26_V2048"], "arm3_xproj26_V2048", 0, 2048, "refuse")

# P2: the weights do not match the architecture the cfg declares -> key-level mismatch, complete counts.
c2 = copy.copy(cks["arm1_V2048"]); c2["model"] = dict(c2["model"])
_gone = [k for k in c2["model"] if k.startswith("blocks.3.")]
for k in _gone:
    del c2["model"][k]
# The count is DERIVED, not typed. The first version of this label said 15 because a block carries 15 keys
# when x_proj is low-rank (Vl + Ul); arm1 is dense, so it carries 14. The guard reported 14 and the prose
# said 15 -- an error message asserting a number it took from the wrong assumption, which is the exact
# defect already on the record from the "Stage E runs at seq 1024" refusal.
attempt(f"P2  pesi incoerenti col cfg: {len(_gone)} chiavi di blocks.3 rimosse", c2, "arm1_V2048", 0, 2048, "refuse")

# P3: a configuration the tool cannot rebuild exactly.
c3 = copy.copy(cks["arm1_V2048"]); c3["cfg"] = dict(c3["cfg"]); c3["cfg"]["stages"] = "CDEF"
attempt("P3  cfg stages='CDEF': richiederebbe di rigiocare l'upcycle MoE", c3, "arm1_V2048", 0, 2048, "refuse")

# P4: no cfg at all -> the artefact cannot be the authority, and the table must not become it.
c4 = copy.copy(cks["arm1_V2048"]); c4.pop("cfg")
attempt("P4  cfg assente: l'artefatto non puo' essere l'autorita'", c4, "arm1_V2048", 0, 2048, "refuse")

# P5: the recall exception, which is the point a guard loses. recall=on with stages that never insert it.
c5 = copy.copy(cks["arm1_V2048"]); c5["cfg"] = dict(c5["cfg"]); c5["cfg"]["recall"] = "on"
attempt("P5  cfg recall='on' con stages='C': lo slot non e' mai inserito", c5, "arm1_V2048", 0, 2048, "refuse")

print("\n" + "=" * 96)
print(f"VERDETTO: {npass} corrette, {nfail} sbagliate  ->  {'GUARDIA ESERCITATA' if nfail == 0 else 'GUARDIA DIFETTOSA'}")
print("=" * 96)
sys.exit(1 if nfail else 0)
