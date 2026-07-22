#!/usr/bin/env python3
"""WS3 smoke -- every stage downstream of the fetch, exercised today, with PLANTED positives.

The pipeline is a set of filters, and a filter that has never been seen to remove something it should
remove is an assumption. Same doctrine as the MQAR calibration: this smoke plants documents that MUST
die and asserts they did.

  * an exact duplicate                 -> must die at exact-hash
  * a near-duplicate (~5% of lines edited) -> must survive exact-hash and die at MinHash J=0.7
  * a verbatim slice of the P62 code-val   -> must die at decontamination J=0.5 -- and this one is the
    load-bearing assertion, because it is the leak that would make a rung-1 BPB gate read good
  * an unrelated document              -> must SURVIVE all three (a filter that removes everything is
    not a passing filter)

Corpus: this repository's own Python. It is real code, it is local, and it costs nothing.

Run: python benchmarks/phase64/data/ws3_smoke.py
"""
import glob, hashlib, json, os, random, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import corpus_pipeline as cp   # noqa: E402


def doc(text, tag):
    return dict(blob_id=tag, path=tag, language="Python", src_encoding="UTF-8",
                length=len(text), license_type="permissive",
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), text=text)


def main():
    rng = random.Random(20260719)
    files = sorted(glob.glob(os.path.join(ROOT, "benchmarks", "**", "*.py"), recursive=True))
    docs = []
    for p in files:
        t = open(p, encoding="utf-8", errors="replace").read()
        if len(t) > 400:
            docs.append(doc(t, os.path.relpath(p, ROOT).replace("\\", "/")))
    if len(docs) < 10:
        sys.exit("not enough local source to smoke with")
    print(f"WS3 smoke   base corpus: {len(docs)} files from benchmarks/**.py\n")

    base = docs[0]["text"]
    lines = base.splitlines()
    near = "\n".join(l + "  # edited" if rng.random() < 0.05 else l for l in lines)
    val = open(cp.P62_VAL, encoding="utf-8", errors="replace").read()[20000:44000]
    other = "\n".join(f"def f{i}(x{i}): return x{i} * {i} + {i*7}" for i in range(400))

    planted = {"PLANT/exact-dup": base, "PLANT/near-dup": near,
               "PLANT/p62-leak": val, "PLANT/unrelated": other}
    for tag, t in planted.items():
        docs.append(doc(t, tag))
    print(f"  planted 4 documents: {', '.join(planted)}\n")

    out = os.path.join(tempfile.gettempdir(), "ws3_smoke_out")
    kept, man = cp.run(docs, out, seed=20260719)
    alive = {d["path"] for d in kept}

    expect = {"PLANT/exact-dup": False, "PLANT/near-dup": False,
              "PLANT/p62-leak": False, "PLANT/unrelated": True}
    print(f"\n  {'planted doc':22s} {'expected':10s} {'actual':10s}  verdict")
    ok = True
    for tag, want in expect.items():
        got = tag in alive
        good = got == want
        ok &= good
        print(f"  {tag:22s} {'survive' if want else 'removed':10s} "
              f"{'survived' if got else 'removed':10s}  {'ok' if good else 'FAIL'}")

    leak = "PLANT/p62-leak" in alive
    print(f"\n  decontamination is MANDATORY and this is its calibration: a verbatim P62 val slice "
          f"{'SURVIVED -- the filter is blind' if leak else 'was removed'}.")
    print("\n==== WS3 filter calibration: " + ("PASS" if ok else "FAIL") + " ====")
    print("\nSTOP. WS3 smoke above. No commit.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
