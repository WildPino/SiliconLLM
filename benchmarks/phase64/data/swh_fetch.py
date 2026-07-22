#!/usr/bin/env python3
"""WS3 stage 0 -- the content path: The Stack v2 metadata (HF) + file contents (Software Heritage S3).

THIS IS THE CORRECTION THAT MADE WS3 A WORKSTREAM. The original brief's scripts read a `content` column
straight out of the HF parquets; that is Stack-*v1*-shaped. v2 ships file IDs and metadata only, and the
bytes live in the Software Heritage S3 bucket, one object per blob, gzip-compressed. So the pipeline has
a network stage that v1 did not have, and its sustained throughput is a rung-1 schedule risk: the
download must not become the bottleneck.

The purpose of the one-shard smoke is to answer three questions with measurements, not documentation:
  1. does the key layout `s3://softwareheritage/content/{blob_id}` resolve, and with which compression;
  2. what sustained MB/s do N parallel workers reach from this machine;
  3. what fraction of blobs are missing or undecodable (it is never zero).

CREDENTIALS come from the environment ONLY -- never a file, never a literal, never committed:
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   (and HF_TOKEN for the gated metadata)
The same rule that governs HF_TOKEN. This script reads them and never prints them.

Run (once the owner's credentials exist):
    python benchmarks/phase64/data/swh_fetch.py --smoke --n 2000 --workers 64
Run today, with no credentials and no network:
    python benchmarks/phase64/data/swh_fetch.py --dry-run
"""
import argparse, gzip, hashlib, io, json, os, queue, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DEF = os.path.join(ROOT, "data", "phase64", "raw")

BUCKET = "softwareheritage"
KEY_FMT = "content/{blob_id}"          # to be CONFIRMED by the smoke, not assumed
PERMISSIVE = ("permissive", "no_license")


def env_creds():
    """Read credentials from the environment and report only their PRESENCE."""
    need = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    have = {k: bool(os.environ.get(k)) for k in need}
    return have, all(have.values())


# ------------------------------------------------------------------ metadata ------------------------
def load_shard(languages, limit):
    """the-stack-v2-dedup metadata, license-filtered. Returns (rows, rows_scanned).

    Language selection uses the dataset CONFIG, not a post-hoc filter on a mixed stream: the config is a
    separate set of parquet files, so filtering by config reads only the bytes we want, while filtering
    a mixed stream downloads every language and throws most of it away.

    The license filter also runs HERE, before any blob is fetched. Fetching a blob we would then discard
    is the one cost in this pipeline that is pure waste."""
    from datasets import load_dataset
    rows, seen = [], 0
    per = max(1, limit // max(len(languages), 1))
    for lang in languages:
        ds = load_dataset("bigcode/the-stack-v2-dedup", lang, split="train", streaming=True,
                          token=os.environ.get("HF_TOKEN"))
        n0 = len(rows)
        for r in ds:
            seen += 1
            if r.get("license_type") not in PERMISSIVE:
                continue
            rows.append(dict(blob_id=r["blob_id"], path=r.get("path", ""), language=r.get("language", ""),
                             src_encoding=r.get("src_encoding", "UTF-8"),
                             length=int(r.get("length_bytes", 0) or 0),
                             license_type=r.get("license_type", "")))
            if len(rows) - n0 >= per:
                break
    return rows, seen


def make_client(anon, cfg):
    """Anonymous first. Software Heritage is on the AWS Open Data registry, so the content bucket reads
    without credentials -- verified, not assumed (see the key probe in the WS3 return). Credentials are
    the FALLBACK, not the default: not needing them removes a blocking owner action."""
    import boto3
    from botocore import UNSIGNED
    if anon:
        from botocore.config import Config
        return boto3.client("s3", config=cfg.merge(Config(signature_version=UNSIGNED)))
    return boto3.session.Session().client("s3", config=cfg)


def probe_access(blob_id):
    """Resolve the access mode on ONE object before a thread pool is spent on it: anonymous first, then
    credentials. Returns (anon, note) or exits. The key layout is confirmed here too -- it was marked
    'to be confirmed', and this is the confirmation."""
    from botocore.config import Config
    cfg = Config(retries={"max_attempts": 2, "mode": "standard"})
    for anon in (True, False):
        if not anon and not env_creds()[1]:
            break
        try:
            o = make_client(anon, cfg).get_object(Bucket=BUCKET, Key=KEY_FMT.format(blob_id=blob_id))
            raw = o["Body"].read()
            n = len(gzip.decompress(raw))
            return anon, (f"{'anonymous' if anon else 'credentialed'} read OK on "
                          f"s3://{BUCKET}/{KEY_FMT.format(blob_id=blob_id)}: "
                          f"{len(raw)} B on the wire -> {n} B gzip-decompressed")
        except Exception as e:
            print(f"    {'anonymous' if anon else 'credentialed'}: {type(e).__name__} {str(e)[:70]}")
    sys.exit("ERROR: could not read a single blob either anonymously or with credentials. The key layout "
             f"({KEY_FMT}) or the access path is wrong -- do NOT report a throughput number from here.")


# ------------------------------------------------------------------ parallel fetch -------------------
def fetch_many(rows, workers, anon, report_every=5.0):
    """N threads, one boto3 client each. Returns (stats, list of (row, bytes)).

    A client per thread rather than one shared: botocore connection pools are per-client, and a shared
    client silently serialises at pool_maxsize -- which would make the throughput number a measurement
    of the default pool size instead of of the link."""
    from botocore.config import Config
    cfg = Config(max_pool_connections=max(workers, 10), retries={"max_attempts": 3, "mode": "standard"})
    q = queue.Queue()
    for r in rows: q.put(r)
    lock = threading.Lock()
    st = dict(ok=0, missing=0, undecodable=0, bytes_raw=0, bytes_text=0)
    got = []

    def worker():
        s3 = make_client(anon, cfg)
        while True:
            try: r = q.get_nowait()
            except queue.Empty: return
            try:
                blob = s3.get_object(Bucket=BUCKET, Key=KEY_FMT.format(**r))["Body"].read()
                raw = gzip.decompress(blob)
                text = raw.decode(r["src_encoding"] or "UTF-8", errors="strict")
            except Exception as e:
                with lock:
                    st["missing" if "NoSuchKey" in type(e).__name__ or "404" in str(e)
                       else "undecodable"] += 1
                continue
            with lock:
                st["ok"] += 1; st["bytes_raw"] += len(blob); st["bytes_text"] += len(raw)
                got.append((r, text))

    t0 = time.time()
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in ts: t.start()
    # POLLING interval and REPORTING interval are separate on purpose. They used to be the same value,
    # and passing a huge report_every to silence the progress lines then put this loop to sleep for
    # ~31 years while every worker had already finished. The symptom was a process at 2% CPU with one
    # thread and zero open sockets -- alive, doing nothing, indistinguishable from slow work.
    last = 0.0
    while any(t.is_alive() for t in ts):
        time.sleep(0.5)
        el = time.time() - t0
        if el - last < report_every: continue
        last = el
        with lock:
            print(f"    {st['ok']:7d} ok  {st['missing']+st['undecodable']:5d} lost  "
                  f"{st['bytes_text']/2**20:8.1f} MiB  {st['bytes_text']/2**20/max(el,1e-9):6.2f} MiB/s "
                  f"(wire {st['bytes_raw']/2**20/max(el,1e-9):5.2f})", flush=True)
    for t in ts: t.join()
    st["seconds"] = time.time() - t0
    st["mib_s"] = st["bytes_text"] / 2**20 / max(st["seconds"], 1e-9)
    return st, got


def write_shard(got, out_dir, name):
    """One JSONL per shard, with the content sha256 alongside. The sha is what the exact-hash dedup pass
    consumes, so it is computed once here and never recomputed from re-read bytes."""
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.jsonl")
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r, text in got:
            f.write(json.dumps(dict(**r, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                    text=text)) + "\n")
            n += 1
    return p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="blobs to fetch in the smoke")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--languages", default="Python,C,C++,Rust,Go,JavaScript,Java,Shell")
    ap.add_argument("--out", default=OUT_DEF)
    ap.add_argument("--name", default="shard0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check imports, credentials and paths; no network")
    a = ap.parse_args()

    have, ok = env_creds()
    print("WS3 stage 0 -- Stack v2 metadata + Software Heritage content")
    print(f"  credentials (presence only): " + "  ".join(f"{k}={'set' if v else 'ABSENT'}"
                                                         for k, v in have.items())
          + f"  HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'ABSENT'}")
    import importlib.util as _iu
    missing = [m for m in ("boto3", "datasets") if _iu.find_spec(m) is None]
    print(f"  packages: " + ("all present" if not missing else f"MISSING {missing} -> pip install {' '.join(missing)}"))
    print(f"  bucket s3://{BUCKET}/{KEY_FMT}  (layout to be CONFIRMED by the first live fetch)")
    print(f"  license filter: {PERMISSIVE}   languages: {a.languages}")
    print(f"  output: {a.out}")

    if a.dry_run:
        print("\n  --dry-run: nothing fetched. Blocking items above are the owner's §0 actions.")
        return
    if missing:
        sys.exit("\nERROR: packages missing (above). Refusing to run a partial smoke that would report a "
                 "throughput number for a path it never exercised.")

    langs = [x for x in a.languages.split(",") if x]
    print("\n  listing metadata (streaming, per-language config, license-filtered before any fetch)...")
    rows, seen = load_shard(langs, a.n)
    print(f"    kept {len(rows)} of {seen} rows scanned "
          f"({100*len(rows)/max(seen,1):.1f}% survive the license filter)")
    if not rows:
        sys.exit("no rows survived the filter -- nothing to fetch")

    print("\n  resolving access on one object (anonymous first, credentials only as fallback)...")
    anon, note = probe_access(rows[0]["blob_id"])
    print(f"    CONFIRMED: {note}")

    print(f"\n  fetching {len(rows)} blobs, {a.workers} workers, "
          f"{'ANONYMOUS' if anon else 'credentialed'}...")
    st, got = fetch_many(rows, a.workers, anon)
    p, n = write_shard(got, a.out, a.name)

    lost = st["missing"] + st["undecodable"]
    print(f"\n  RESULT  {st['ok']} ok / {lost} lost ({st['missing']} missing, {st['undecodable']} undecodable)"
          f" = {100*lost/max(len(rows),1):.2f}% loss")
    print(f"          {st['bytes_text']/2**20:.1f} MiB text in {st['seconds']:.1f}s = "
          f"{st['mib_s']:.2f} MiB/s sustained at {a.workers} workers")
    print(f"          compression on the wire: {st['bytes_raw']/max(st['bytes_text'],1):.3f}x")
    print(f"          wrote {p} ({n} docs)")
    tgt_gb = 40
    print(f"\n  EXTRAPOLATION to the plan's ~{tgt_gb} GB source: "
          f"{tgt_gb*1024/max(st['mib_s'],1e-9)/3600:.1f} h at this rate. "
          f"Scale --workers until this stops improving; that is the number the schedule needs.")
    print("\nSTOP. WS3 stage-0 numbers above. No commit.")


if __name__ == "__main__":
    main()
