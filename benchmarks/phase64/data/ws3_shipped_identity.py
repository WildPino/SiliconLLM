#!/usr/bin/env python3
"""Check (a): the bytes that SHIP are the bytes a gate certified.

WHY THIS EXISTS AND WHY IT IS SEPARATE. The epsilon-identity gate proves the trainer's behaviour is
unchanged. It proves it about a file in the working tree. Between that proof and the upload there is a
packaging step -- and on 2026-07-31 that step was edited after the gate had run (SRC_TAG collapsed back
into TAG, which renamed shipped files and moved DATA_SHA). "The artefact that runs is the artefact that
was gated" is exactly the invariant the stale packages of 2026-07-22 violated, and it is not a suspicion
to be reasoned about. It is a sha comparison.

It is verified against the GATE LOG, not against a remembered value: the gate prints the sha256 of the
file it certified, and this reads that line. A check whose expected value is typed in by whoever runs it
is a check against their memory.

WHAT IS COMPARED
  repo         benchmarks/phase64/mve/mve_train.py and every other CODE_FILES entry
  bundle       kaggle_rung1/code_bundle/code/<same paths>      (what the cell pins by CODE_SHA)
  packages     kaggle_rung1/*/code/<same paths>                (what rides with the data)
All three must agree, and the trainer must additionally equal the gate's CERTIFIED FILE sha.

Run: python benchmarks/phase64/data/ws3_shipped_identity.py
"""
import glob, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from pack_kaggle import CODE_FILES                       # one source of truth  # noqa: E402

OUT = os.path.join(ROOT, "kaggle_rung1")
GATELOG = os.path.join(ROOT, "results", "phase64_rung1", "epsilon_identity_uint16_ce.txt")
TRAINER = "benchmarks/phase64/mve/mve_train.py"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def certified_sha(path):
    """The sha the gate printed for the file it certified. Absent => this check cannot conclude."""
    if not os.path.isfile(path):
        return None, f"no gate log at {path}"
    txt = open(path, encoding="utf-8", errors="replace").read()
    if "VERDICT: PASS" not in txt:
        return None, "the gate log does not carry a PASS verdict"
    m = re.search(r"CERTIFIED FILE\s+(\S+)\s*\n\s*sha256\s+([0-9a-f]{64})", txt)
    if not m:
        return None, ("the gate log carries no 'CERTIFIED FILE ... sha256' line -- it was produced by a "
                      "gate generation that did not print what it certified, so this check has nothing "
                      "to compare against and must NOT be reported as passing")
    return (m.group(1), m.group(2)), None


def main():
    print("=" * 78)
    print("shipped-identity check -- the bytes that ship vs the bytes that were gated")
    print("=" * 78)

    cert, why = certified_sha(GATELOG)
    if cert is None:
        print(f"\n  CANNOT CONCLUDE: {why}")
        print("\n  Refusing to print a verdict. An unverified upload and an upload whose check could not\n"
              "  run are the same thing, and only one of them looks alarming.")
        sys.exit(2)
    cfile, csha = cert
    print(f"\n  gate log   {os.path.relpath(GATELOG, ROOT)}   (VERDICT: PASS)")
    print(f"  certified  {cfile}\n             sha256 {csha}")

    # Every place a copy of the shipped code exists.
    trees = [("repo", ROOT)]
    b = os.path.join(OUT, "code_bundle", "code")
    if os.path.isdir(b):
        trees.append(("code_bundle", b))
    for d in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        cd = os.path.join(OUT, d, "code")
        if os.path.isdir(cd):
            trees.append((f"package:{d}", cd))

    # A tree may legitimately hold a DIFFERENT generation: the relaytest bundle is pinned to CODE_SHA
    # 27b22582 on purpose, because that throwaway validates the resume chain against the trainer that is
    # already uploaded. The first version of this check called that FAIL, which is the wrong
    # classification -- and a check that goes red for a legitimate reason gets overridden, after which its
    # red means nothing. But "differs" must not become a free pass either, so a divergent tree has to earn
    # it on three counts: it carries its own CODE_MANIFEST, it REPRODUCES that manifest's code_sha256, and
    # that sha is pinned by a generated notebook cell. Then it is a declared alternate generation with a
    # consumer. Anything else is stale.
    pinned = set()
    for nb in sorted(glob.glob(os.path.join(OUT, "NOTEBOOK_*.py"))):
        m = re.search(r"^CODE_SHA\s*=\s*'([0-9a-f]{64})'", open(nb, encoding="utf-8").read(), re.M)
        if m:
            pinned.add(m.group(1))

    def declared_alternate(tree):
        """(True, sha) if `tree` is a self-consistent bundle of another, still-referenced generation."""
        mp = os.path.join(os.path.dirname(tree), "CODE_MANIFEST.json")
        if not os.path.isfile(mp):
            return False, None
        man = json.load(open(mp))
        want = man.get("code_sha256")
        h = hashlib.sha256()
        for rel in sorted(man.get("files", {})):
            p = os.path.join(os.path.dirname(tree), rel) if rel == "assert_package.py" \
                else os.path.join(tree, rel)
            h.update(f"{rel} {sha(p) if os.path.isfile(p) else 'MISSING'}\n".encode())
        return (h.hexdigest() == want and want in pinned), want

    bad, rows, alt = [], [], {}
    for name, tree in trees:
        # ALTERNATE only applies to a tree whose trainer actually DIFFERS from the certified one. The
        # first version tested self-consistency first, which made the CURRENT bundle -- whose sha is of
        # course pinned by the main-run cell -- classify as an alternate generation, and the summary then
        # printed "It is NOT for the main run" about the bundle the main run uses. Flatly false, and
        # dangerous in the direction that matters. Third defect in this checker inside ten minutes: the
        # repair keeps being the thing that breaks.
        tp = os.path.join(tree, TRAINER)
        if name == "repo" or not os.path.isfile(tp) or sha(tp) == csha:
            continue
        ok, want = declared_alternate(tree)
        if ok:
            alt[name] = want
    for rel in CODE_FILES:
        base = sha(os.path.join(ROOT, rel))
        for name, tree in trees:
            p = os.path.join(tree, rel)
            if not os.path.isfile(p):
                if name != "repo":
                    bad.append(f"{name}: {rel} MISSING")
                continue
            s = sha(p)
            if s != base and name not in alt:
                bad.append(f"{name}: {rel} differs from the repo copy")
            rows.append((name, rel, s))

    print(f"\n  trees compared: {', '.join(n for n, _ in trees)}")
    for name, tree in trees:
        got = [s for n, r, s in rows if n == name and r == TRAINER]
        if not got:
            print(f"    {name:24s} {TRAINER}  {'-':<20s} absent"); continue
        if got[0] == csha:
            mark = "== certified"
        elif name in alt:
            mark = f"declared alternate generation {alt[name][:12]}..., self-consistent and pinned by a cell"
        else:
            mark = "DIFFERS from certified"
            bad.append(f"{name}: the trainer is NOT the file the gate certified, and the tree is not a "
                       f"self-consistent bundle of a generation any cell pins")
        print(f"    {name:24s} {TRAINER}  {got[0][:16] + '...':<20s} {mark}")

    print("\n" + "=" * 78)
    if bad:
        print("VERDICT: FAIL")
        for b_ in bad:
            print(f"  {b_}")
        print("\n  Do not upload. Rebuild the bundle and the packages from the current repo, then re-run\n"
              "  the gate if the trainer itself moved.")
        print("=" * 78)
        sys.exit(1)
    n = len({r for _, r, _ in rows})
    nmain = len(trees) - len(alt)
    print("VERDICT: PASS")
    print(f"  {n} module(s) checked across {len(trees)} tree(s).")
    print(f"  {nmain} tree(s) carry the certified trainer byte-for-byte ({csha[:16]}...).")
    if alt:
        # Said explicitly, because "every shipped tree is the certified one" would be FALSE here and a
        # summary line that contradicts its own table is how a green check stops being read.
        for k, v in sorted(alt.items()):
            print(f"  {k} is a DECLARED ALTERNATE generation ({v[:16]}...): self-consistent with its own")
            print(f"    CODE_MANIFEST and pinned by a generated cell. It is NOT for the main run.")
    print("=" * 78)


if __name__ == "__main__":
    main()
