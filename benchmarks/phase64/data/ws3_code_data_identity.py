#!/usr/bin/env python3
"""Gate (b) for the code_data.py memory rewrite: the student side must be BIT-IDENTICAL, old vs new.

WHY THIS GATE EXISTS. The 5.5 GB build died twice, and the diagnosis -- Python list accumulators at 76 B
per token -- was correct but too narrow: the first remedy would have left a 41 GiB array and a 77 GiB one
standing, twelve lines below. The rewrite that followed touches the code that produces the artifact the
whole rung is measured on. A refactor of a data builder is exactly the kind of change that is invisible
when it is wrong: the training runs, the loss curve looks ordinary, and the number at the end is the
number of a different corpus.

WHAT IS COMPARED, and against what. Not a re-derivation -- the ACTUAL s0 artifacts, on disk since
2026-07-21, produced by the pre-rewrite code at 0.5 GB, the scale where that code ran to completion.

raw_s0.bin and docbound_s0.i64 are INPUTS here, seeded into the scratch tree so build_raw reuses them
rather than re-enumerating the shards. That is not a convenience, it is a correctness requirement, and
the first attempt at this gate discovered why: `--gb 0.5` today does NOT reproduce raw_s0.bin. On
2026-07-21 the shard set ran out at 483,190,011 B over 95,564 documents; today the same command reaches
the full 512 MiB target over 105,920, because the WS3 corpus finished downloading afterwards. A tag is
therefore not a reproducible specification on its own -- raw_sha256 is -- and a gate that re-derived the
corpus would have reported a difference that had nothing to do with the code under test. The seeded
corpus is sha-checked against meta_s0.json's raw_sha256 before anything runs, so the input is pinned to
the byte.

That also fixes where the P62 claim lives. `fetch_corpus.is_val` decides which documents are held out and
the corpus bytes ARE that decision; pinning raw to the sha of the build the whole rung was measured on
asserts the held-out set is the same one. build_raw and is_val are untouched by this rewrite -- the diff
is confined to teacher_tokens, build_vocab and main -- so there is nothing about the split for the gate
to re-prove, only to pin.

  ids_V2048/V4096     the rewritten code path: pool.imap streamed to disk instead of pool.map into a
                      flattening comprehension. THE claim of this gate.
  teacher_*, anchors_*, t2s_*, decomp_*   the KD apparatus, under --kd-apparatus. Not needed by any arm
                      at this rung, compared anyway: it proves the flag REMOVES work rather than CHANGING
                      it, which is what makes the default path's silence trustworthy.

THREE RUNS.
  A  new code, --kd-apparatus  -> every s0 artifact must reproduce, byte for byte.
  B  new code, default (no KD) -> the student files must reproduce and the KD files must be ABSENT.
  C  PLANTED CONTROL. One byte of the corpus is changed and the ids must come out DIFFERENT. A
     comparison that cannot fail is not evidence; this rung has already been burned once by a control
     that could never fire (the epsilon-identity VOID check, blind to CRLF).

Run: python benchmarks/phase64/data/ws3_code_data_identity.py
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REAL = os.path.join(ROOT, "results", "phase64", "rung1")
P62 = os.path.join(ROOT, "results", "phase62", "code_val.txt")
CD = os.path.join(HERE, "code_data.py")
REF = "s0"
GB = 0.5
VOCABS = (2048, 4096)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def same_npz(a, b):
    """decomp_*.npz is a zip container; comparing raw bytes would compare archive metadata too. The
    claim is about the ARRAYS."""
    za, zb = np.load(a), np.load(b)
    if sorted(za.files) != sorted(zb.files): return False
    return all(np.array_equal(za[k], zb[k]) for k in za.files)


def run(tag, out, kd, vocabs):
    cmd = [sys.executable, CD, "--gb", str(GB), "--tag", tag, "--out", out,
           "--vocabs", ",".join(str(v) for v in vocabs)] + (["--kd-apparatus"] if kd else [])
    print(f"\n  $ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"  -> exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
    if r.returncode != 0:
        sys.exit(f"GATE ABORTED: the builder failed for run tag {tag}.")


def compare(tag, out, names, expect_same=True):
    """Returns (n_ok, failures). `names` are the s0-relative basenames with {tag} substituted."""
    ok, bad = 0, []
    for pat in names:
        new = os.path.join(out, pat.format(tag=tag))
        ref = os.path.join(REAL, pat.format(tag=REF))
        if not os.path.isfile(new):
            bad.append((pat, "ABSENT in the new build")); continue
        if not os.path.isfile(ref):
            bad.append((pat, "no s0 reference on disk")); continue
        eq = same_npz(new, ref) if new.endswith(".npz") else sha(new) == sha(ref)
        mark = "IDENTICAL" if eq else "DIFFERS"
        print(f"    {mark:<10} {pat.format(tag=tag)}")
        if eq == expect_same: ok += 1
        else: bad.append((pat, mark))
    return ok, bad


def seed(out, tag):
    """Copy the s0 corpus in under the run's tag so build_raw reuses it. Returns the sha it pinned."""
    r = os.path.join(out, f"raw_{tag}.bin")
    shutil.copyfile(os.path.join(REAL, f"raw_{REF}.bin"), r)
    shutil.copyfile(os.path.join(REAL, f"docbound_{REF}.i64"), os.path.join(out, f"docbound_{tag}.i64"))
    return sha(r)


STUDENT = [f"ids_V{v}_{{tag}}.u16" for v in VOCABS]
KDFILES = ["teacher_ids_{tag}.i32", "teacher_start_{tag}.i64"] + \
          [f"anchors_V{v}_{{tag}}.i32" for v in VOCABS] + \
          [f"t2s_V{v}_{{tag}}.i32" for v in VOCABS] + \
          [f"decomp_V{v}_{{tag}}.npz" for v in VOCABS]


def main():
    fail = []
    print("=" * 78)
    print("code_data.py memory rewrite -- bit-identity gate against the s0 artifacts")
    print("=" * 78)
    print(f"\n  reference tree : {REAL}")
    print(f"  P62 decider    : {P62}")
    print(f"                   sha {sha(P62)[:32]}...  ({os.path.getsize(P62)} bytes)")
    print("                   read-only by law; recorded here because raw_*.bin is the val split's")
    print("                   footprint and this gate pins that file by sha.")
    want_raw = json.load(open(os.path.join(REAL, f"meta_{REF}.json")))["raw_sha256"]
    print(f"  input pin      : raw_{REF}.bin must be sha {want_raw[:32]}...")
    for pat in STUDENT + KDFILES:
        p = os.path.join(REAL, pat.format(tag=REF))
        if not os.path.isfile(p):
            sys.exit(f"GATE ABORTED: no s0 reference for {pat.format(tag=REF)}. Nothing to compare to.")

    tmp = tempfile.mkdtemp(prefix="cdid_")
    try:
        # ---- A: new code, KD apparatus on. Everything s0 has must come back.
        print("\n" + "-" * 78)
        print("RUN A  new code, --kd-apparatus  (expect: every s0 artifact reproduces)")
        print("-" * 78)
        oa = os.path.join(tmp, "a"); os.makedirs(oa)
        got = seed(oa, "g0")
        print(f"    input pinned: raw sha {got[:32]}...  {'OK' if got == want_raw else 'MISMATCH'}")
        if got != want_raw:
            sys.exit("GATE ABORTED: the seeded corpus is not the one s0 was built from.")
        run("g0", oa, True, VOCABS)
        na, ba = compare("g0", oa, STUDENT + KDFILES, expect_same=True)
        print(f"  {na}/{len(STUDENT+KDFILES)} identical")
        fail += [("A", p, w) for p, w in ba]

        # ---- B: default path. Student identical, KD absent BY DESIGN.
        print("\n" + "-" * 78)
        print("RUN B  new code, default  (expect: student identical, KD apparatus absent)")
        print("-" * 78)
        ob = os.path.join(tmp, "b"); os.makedirs(ob)
        seed(ob, "g1")
        run("g1", ob, False, VOCABS)
        nb, bb = compare("g1", ob, STUDENT, expect_same=True)
        print(f"  {nb}/{len(STUDENT)} identical")
        fail += [("B", p, w) for p, w in bb]
        leaked = [p for p in KDFILES if os.path.isfile(os.path.join(ob, p.format(tag="g1")))]
        print(f"    KD apparatus files present: {len(leaked)} of {len(KDFILES)}  "
              f"({'as designed' if not leaked else 'LEAK'})")
        if leaked: fail.append(("B", ", ".join(leaked), "built despite the flag being off"))

        # ---- C: planted control. Corrupt one corpus byte; the ids MUST diverge.
        print("\n" + "-" * 78)
        print("RUN C  PLANTED CONTROL: one corpus byte changed  (expect: ids DIFFER)")
        print("-" * 78)
        oc = os.path.join(tmp, "c"); os.makedirs(oc)
        raw = bytearray(open(os.path.join(REAL, f"raw_{REF}.bin"), "rb").read())
        off = next(i for i in range(1 << 16) if 97 <= raw[i] <= 121)   # ASCII a..y, stays valid UTF-8
        was = raw[off]; raw[off] = was + 1
        print(f"    corpus byte {off}: {chr(was)!r} -> {chr(was+1)!r}  (1 byte in {len(raw)})")
        open(os.path.join(oc, "raw_g2.bin"), "wb").write(bytes(raw))
        shutil.copyfile(os.path.join(REAL, f"docbound_{REF}.i64"), os.path.join(oc, "docbound_g2.i64"))
        run("g2", oc, False, (2048,))
        nc, bc = compare("g2", oc, ["ids_V2048_{tag}.u16"], expect_same=False)
        if nc == 1:
            print("    control FIRED: the comparison can distinguish a one-byte corpus change.")
        else:
            fail.append(("C", "ids_V2048", "control did NOT fire -- run A proves nothing"))

        print("\n" + "=" * 78)
        if fail:
            print("VERDICT: FAIL")
            for r_, p, w in fail:
                print(f"  [{r_}] {p}: {w}")
            print("\nThe rewrite does NOT reproduce the s0 artifacts. Do not build the main-run corpus.")
            print("=" * 78)
            sys.exit(1)
        print("VERDICT: PASS")
        print(f"  A: {len(STUDENT+KDFILES)}/{len(STUDENT+KDFILES)} artifacts bit-identical to s0 under")
        print("     --kd-apparatus -- the rewrite changed no output, only how it is accumulated.")
        print(f"  B: {len(STUDENT)}/{len(STUDENT)} student artifacts bit-identical with the flag off, and")
        print("     0 KD files built. The flag removes work; it does not alter the artifact.")
        print("  C: the control fired on a single changed byte, so A's and B's nulls are informative.")
        print("\n  The P62 val split is pinned, not re-derived: the corpus fed to all three runs is")
        print("  raw_s0.bin by sha, so the held-out document set is the one the rung was measured on.")
        print("  build_raw and fetch_corpus.is_val are outside this rewrite's diff. code_val.txt untouched.")
        print("\n  NOTE on the domain pin. Under the default path there are no anchors, so there is no")
        print("  teacher-scored slice and SLICE_SHA is empty for the main run -- correctly, not by")
        print("  omission. --expect-slice-sha pinned the KD packages' sampling domain; a CE run over the")
        print("  whole corpus has no such restriction to pin.")
        print("=" * 78)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
