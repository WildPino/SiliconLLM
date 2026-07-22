#!/usr/bin/env python3
"""WS3 -- the rung-1 corpus fetch: resumable, heartbeat, target-driven, monotone.

SCOPE (adjudicated): Python only, STRICT-PERMISSIVE by an enumerated licence whitelist. Rung-1 claims
are Python-scoped.

WHY THE WHITELIST IS ENUMERATED RATHER THAN INHERITED. Measured on 4789 sampled files, BigCode's
`license_type == "permissive"` admits files whose detected licences include GPL-1.0-or-later, MPL-2.0,
CC-BY-4.0 and LicenseRef-scancode-proprietary-license. The class is a per-file aggregate, not a
guarantee. A file is kept only if `detected_licenses` is NON-EMPTY and EVERY entry is in the whitelist
(all, not any). This filter is load-bearing twice over: it sets the corpus size AND it is the licence
claim, so the whitelist is written into the manifest verbatim.

SELECTION IS UNIFORM AND MONOTONE, which are two separate requirements:

  uniform   The Stack is ordered BY REPOSITORY, so a prefix of the stream is a sample of a few repos
            and their licences, not of the corpus. Selection is by HASH THRESHOLD --
            sha256(seed || blob_id) < p -- which is independent of stream position, plus a shuffle
            buffer so the traversal itself is decorrelated.

  monotone  Raising the target later must leave the earlier file-set an EXACT SUBSET, so earlier
            manifests stay valid. This is why the traversal length R is a FIXED, DECLARED parameter
            and p is the knob: selected = {first R rows in seed order : h(blob_id) < p}. Raising p with
            R and the seed unchanged yields a strict superset; so does raising R at fixed p.
            NOTE why "stop when the target is reached" was rejected: a larger p reaches the target
            EARLIER in the stream, so the covered prefix SHRINKS and the superset property breaks. The
            byte target therefore sets p up front; it never terminates the traversal.

QUALITY FILTERS. Metadata-level ones run before any byte is fetched. Content-level ones run after.
Each filter has its own counter, because the post-filter corpus size now drives a pre-registered
trigger (if it falls under 2x the S2 requirement, the multi-language expansion gate fires early) --
the rejection rates are a measurement, not hygiene.

Removed on review: a "printable-ASCII fraction" filter. It was backwards -- base64 is 100% printable
ASCII and would sail through, while legitimate Python with Chinese or Cyrillic comments would fail. It
discarded corpus at 1.0x margin with a systematic bias against non-anglophone authors, and strict UTF-8
decoding already covers binaries. Replaced by a whitespace-fraction floor: data blobs have almost none,
code has plenty.

Run:
    python benchmarks/phase64/data/fetch_corpus.py --target-gb 6 --workers 64
Resume: re-run the identical command. Completed blobs are skipped.
"""
import argparse, gzip, hashlib, json, os, queue, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from swh_fetch import BUCKET, KEY_FMT, make_client   # noqa: E402

OUT_DEF = os.path.join(ROOT, "data", "phase64", "raw_python")
SEED = 20260719
N_PYTHON = 47_272_886      # exact, from the dataset info -- the traversal-coverage denominator

# Enumerated, conservative, and quoted verbatim in the manifest. Weak-copyleft (MPL), attribution
# (CC-BY) and every LicenseRef-scancode-* pseudo-licence are deliberately absent: "unknown licence
# reference" is not permission.
LICENSE_WHITELIST = ("MIT", "MIT-0", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "BSD-1-Clause",
                     "0BSD", "ISC", "Unlicense", "CC0-1.0", "Python-2.0", "Zlib", "PostgreSQL", "BSL-1.0")

MIN_BYTES, MAX_BYTES = 200, 1 << 20
MAX_LINE, MEAN_LINE = 1000, 200
MIN_WS = 0.08          # whitespace fraction floor: data blobs have almost none, source has ~15-25%


VAL_BAND_HI = 0.01     # h(blob_id) < this  =>  PERMANENTLY held out of training. Never widen it.


def h01(blob_id, seed=SEED):
    """Uniform in [0,1), independent of stream position. The selection key."""
    d = hashlib.sha256(f"{seed}:{blob_id}".encode()).digest()[:8]
    return int.from_bytes(d, "big") / 2**64


def is_val(blob_id):
    """The RESERVED BAND: an in-corpus validation set that no future corpus expansion can ever absorb.

    The pinned P62 val is temporally held out (it postdates the 2023 Stack snapshot), which makes rung-1
    BPB a measure of temporal generalization. That is the right metric, but it confounds two things if
    the number disappoints: has the model failed to model code, or is the val simply harder because it
    is from the future? This band answers that -- same distribution, same snapshot, never trained on.
    Record-only, never gated.

    Why a hash band and not a random split: the selection threshold p grows monotonically as later rungs
    raise the byte target, so ANY val defined by "files we happened not to fetch" would be swallowed the
    first time p rises. A band at the BOTTOM of the hash range is inside every selection that will ever
    be made, so it must be carved out explicitly -- and then it is the SAME file set forever, at every p
    and every traversal length. This function is the single definition; the fetcher and the data prep
    both import it rather than re-deriving the rule."""
    return h01(blob_id) < VAL_BAND_HI


def meta_ok(r, ctr):
    if (r.get("extension") or "").lower() != "py": ctr["ext"] += 1; return False
    if r.get("is_generated"): ctr["generated"] += 1; return False
    if r.get("is_vendor"): ctr["vendor"] += 1; return False
    L = int(r.get("length_bytes", 0) or 0)
    if not (MIN_BYTES <= L <= MAX_BYTES): ctr["size"] += 1; return False
    dl = r.get("detected_licenses") or []
    if not dl or not all(x in LICENSE_WHITELIST for x in dl): ctr["license"] += 1; return False
    return True


def content_ok(text, ctr):
    lines = text.splitlines() or [""]
    if max(len(l) for l in lines) > MAX_LINE: ctr["longline"] += 1; return False
    if sum(len(l) for l in lines) / len(lines) > MEAN_LINE: ctr["meanline"] += 1; return False
    if sum(c.isspace() for c in text) / max(len(text), 1) < MIN_WS: ctr["whitespace"] += 1; return False
    return True


# ------------------------------------------------------------------ metadata harvest -----------------
def harvest(out, scan_rows, p, ctr):
    """Stream `scan_rows` metadata rows in seeded shuffle order, keep those passing metadata filters AND
    the hash gate. Resumable: the harvest is written once and reused."""
    path = os.path.join(out, "candidates.jsonl")
    if os.path.isfile(path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        print(f"  harvest: reusing {len(rows)} candidates from {path}")
        return rows
    from datasets import load_dataset
    ds = load_dataset("bigcode/the-stack-v2-dedup", "Python", split="train", streaming=True,
                      token=os.environ.get("HF_TOKEN")).shuffle(seed=SEED, buffer_size=10_000)
    rows, seen, t0 = [], 0, time.time()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in ds:
            seen += 1
            if seen > scan_rows: break
            if not meta_ok(r, ctr): continue
            if h01(r["blob_id"]) >= p: ctr["not_selected"] += 1; continue
            row = dict(blob_id=r["blob_id"], path=r.get("path", ""),
                       src_encoding=r.get("src_encoding", "UTF-8"),
                       length=int(r.get("length_bytes", 0) or 0),
                       licenses=list(r.get("detected_licenses") or []))
            rows.append(row); f.write(json.dumps(row) + "\n")
            if seen % 50_000 == 0:
                print(f"    scanned {seen:9d}/{scan_rows}  kept {len(rows):7d}  "
                      f"{sum(x['length'] for x in rows)/2**30:6.2f} GB  "
                      f"{seen/max(time.time()-t0,1e-9):7.0f} rows/s", flush=True)
    os.replace(tmp, path)
    print(f"  harvest: scanned {seen}, kept {len(rows)} candidates "
          f"({sum(x['length'] for x in rows)/2**30:.2f} GB nominal)")
    return rows


# ------------------------------------------------------------------ resumable fetch -------------------
def fetch(rows, out, workers, anon, ctr, beat=10.0, shard_docs=20_000):
    done_path = os.path.join(out, "done.txt")
    done = set()
    if os.path.isfile(done_path):
        done = set(l.strip() for l in open(done_path) if l.strip())
    todo = [r for r in rows if r["blob_id"] not in done]
    print(f"  fetch: {len(todo)} to go, {len(done)} already done")
    if not todo: return 0

    from botocore.config import Config
    cfg = Config(max_pool_connections=max(workers, 10), retries={"max_attempts": 3, "mode": "standard"})
    q = queue.Queue()
    for r in todo: q.put(r)
    lock = threading.Lock()
    st = dict(ok=0, lost=0, filtered=0, bytes=0)
    shard = [len(done) // shard_docs]
    fh = [open(os.path.join(out, f"shard{shard[0]:04d}.jsonl"), "a", encoding="utf-8")]
    df = open(done_path, "a")

    def worker():
        s3 = make_client(anon, cfg)
        while True:
            try: r = q.get_nowait()
            except queue.Empty: return
            try:
                raw = gzip.decompress(s3.get_object(Bucket=BUCKET,
                                                    Key=KEY_FMT.format(blob_id=r["blob_id"]))["Body"].read())
                text = raw.decode(r["src_encoding"] or "UTF-8", errors="strict")
            except Exception:
                with lock:
                    st["lost"] += 1; df.write(r["blob_id"] + "\n")
                continue
            with lock:
                keep = content_ok(text, ctr)
                df.write(r["blob_id"] + "\n")
                if not keep:
                    st["filtered"] += 1; continue
                st["ok"] += 1; st["bytes"] += len(raw)
                fh[0].write(json.dumps(dict(**r, sha256=hashlib.sha256(raw).hexdigest(),
                                            text=text)) + "\n")
                if st["ok"] % shard_docs == 0:
                    fh[0].close(); shard[0] += 1
                    fh[0] = open(os.path.join(out, f"shard{shard[0]:04d}.jsonl"), "a", encoding="utf-8")

    t0 = time.time()
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in ts: t.start()
    # Polling and reporting are separate intervals. Collapsing them once put this loop to sleep while
    # every worker had already exited: a live process at 2% CPU is indistinguishable from slow work.
    last = 0.0
    while any(t.is_alive() for t in ts):
        time.sleep(0.5)
        el = time.time() - t0
        if el - last < beat: continue
        last = el
        with lock:
            n = st["ok"] + st["lost"] + st["filtered"]
            rate = n / max(el, 1e-9)
            eta = (len(todo) - n) / max(rate, 1e-9) / 60
            print(f"    [{el/60:6.1f} min] {n:8d}/{len(todo)}  kept {st['ok']:8d}  "
                  f"{st['bytes']/2**30:6.3f} GB  {st['bytes']/2**20/max(el,1e-9):6.2f} MiB/s  "
                  f"ETA {eta:6.1f} min", flush=True)
            json.dump(dict(elapsed_s=el, **st, todo=len(todo)),
                      open(os.path.join(out, "status.json"), "w"))
    for t in ts: t.join()
    fh[0].close(); df.close()
    el = time.time() - t0
    print(f"  fetch: {st['ok']} kept, {st['filtered']} content-filtered, {st['lost']} lost in "
          f"{el/60:.1f} min = {st['bytes']/2**20/max(el,1e-9):.2f} MiB/s")
    return st["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb", type=float, default=6.0)
    ap.add_argument("--pool-gb", type=float, default=39.8, help="measured strict-permissive Python pool")
    ap.add_argument("--scan-rows", type=int, default=6_000_000,
                    help="DECLARED traversal length; monotonicity is defined against it")
    ap.add_argument("--select-p", type=float, default=0.0, help="override the hash threshold directly")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=OUT_DEF)
    a = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN absent from the environment.")
    os.makedirs(a.out, exist_ok=True)

    # p is derived from the byte target, then FIXED and recorded. It never depends on progress.
    # The traversal only SEES the fraction R/N of the pool, so the threshold must be scaled by it:
    # expected bytes = pool * (R/N) * p. Writing p = target/pool (the obvious form) silently assumes a
    # full traversal and under-delivers by exactly R/N.
    cover = min(1.0, a.scan_rows / N_PYTHON)
    p = a.select_p or a.target_gb / max(a.pool_gb * cover, 1e-9)
    if p > 1.0:
        sys.exit(f"ERROR: {a.target_gb:.1f} GB is not reachable inside R = {a.scan_rows} rows "
                 f"({100*cover:.1f}% of the corpus holds only {a.pool_gb*cover:.2f} GB). "
                 f"Raise --scan-rows to at least {int(N_PYTHON * a.target_gb / a.pool_gb):d}, or lower "
                 f"the target. Refusing to run a traversal that cannot deliver what was asked.")
    ctr = dict(ext=0, generated=0, vendor=0, size=0, license=0, not_selected=0,
               longline=0, meanline=0, whitespace=0)

    print(f"WS3 corpus fetch   Python, strict-permissive   seed={SEED}")
    print(f"  whitelist ({len(LICENSE_WHITELIST)}): {', '.join(LICENSE_WHITELIST)}")
    print(f"  target {a.target_gb:.1f} GB of a {a.pool_gb:.1f} GB pool -> hash threshold p = {p:.4f}")
    print(f"  traversal R = {a.scan_rows} rows (declared; raising p or R keeps this file-set a subset)")
    print(f"  out {a.out}\n")

    rows = harvest(a.out, a.scan_rows, p, ctr)
    if not rows: sys.exit("no candidates -- check the filters before touching the target")
    got = fetch(rows, a.out, a.workers, True, ctr)

    print(f"\n  filter counters (the corpus-size trigger reads these):")
    for k, v in ctr.items():
        print(f"    {k:14s} {v:9d}")
    man = dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scope="Python",
               seed=SEED, select_p=p, scan_rows=a.scan_rows, coverage=cover, target_gb=a.target_gb,
               pool_gb=a.pool_gb, license_whitelist=list(LICENSE_WHITELIST),
               license_rule="detected_licenses non-empty AND every entry in whitelist",
               filters=dict(min_bytes=MIN_BYTES, max_bytes=MAX_BYTES, max_line=MAX_LINE,
                            mean_line=MEAN_LINE, min_whitespace_fraction=MIN_WS),
               counters=ctr, fetched_bytes=got)
    json.dump(man, open(os.path.join(a.out, "fetch_manifest.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(a.out, 'fetch_manifest.json')}")
    print(f"  post-filter corpus: {got/2**30:.2f} GB")
    print("\nSTOP. WS3 fetch above. No commit.")


if __name__ == "__main__":
    main()
