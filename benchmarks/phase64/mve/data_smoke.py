#!/usr/bin/env python3
# Phase 64.4 / stage-0 tail — the DATA SMOKE + the two license live-checks (plan §10, corrections 2 and 4).
#   magic: "MVEZ" 0x4D56455A
#
# (1) The Stack v2 CONTENT PATH. The Z.ai data brief's scripts read a `content` column from the HF parquets. That is
#     Stack-*v1*-shaped. v2 is believed to ship file IDs + metadata, with the actual bytes fetched from Software
#     Heritage S3. This is load-bearing: if true, every ingestion script in the brief has to be rewritten, and the
#     pipeline acquires an S3 dependency (credentials, egress, rate limits) that is nowhere in the cost model.
#     -> resolve it by LOOKING, on one shard, not by arguing.
#
# (2) LICENSES, live: Qwen2.5-Coder-1.5B (Apache-2.0 expected -> the sealed teacher) and the 3B (believed to sit under
#     the Qwen *Research* non-commercial licence -> the "only defensible upgrade" would carry a restriction the 1.5B
#     does not). Plus the brief's post-cutoff claim that Gemma 4 is Apache-2.0.
#
# Read-only, no downloads of corpus bytes. Run: .venv/Scripts/python.exe benchmarks/phase64/mve/data_smoke.py
import json, sys, urllib.request, urllib.error

HF = "https://huggingface.co/api"
MODELS = ["Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-0.5B", "Qwen/Qwen2.5-Coder-7B"]
GEMMA = ["google/gemma-3-12b-pt", "google/gemma-2-9b"]
DSETS = ["bigcode/the-stack-v2-dedup", "bigcode/the-stack-v2", "bigcode/the-stack-dedup"]


def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "siliconllm-stage0"}), timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:
        return -1, str(e)


print("==== (1) licence live-check: the sealed teacher and its upgrade path ====")
print(f"{'model':32s} {'licence':22s} {'gated':7s} {'status'}")
for m in MODELS + GEMMA:
    st, j = get(f"{HF}/models/{m}")
    if st != 200:
        print(f"{m:32s} {'-':22s} {'-':7s} HTTP {st}"); continue
    cd = j.get("cardData") or {}
    lic = cd.get("license") or "?"
    ln = cd.get("license_name") or ""
    gated = str(j.get("gated", False))
    print(f"{m:32s} {(lic + (' / ' + ln if ln else '')):22s} {gated:7s} ok")

print("\n==== (2) The Stack v2: is there a `content` column, or only IDs? ====")
for d in DSETS:
    st, j = get(f"{HF}/datasets/{d}")
    if st != 200:
        print(f"  {d:28s} HTTP {st}  (gated? needs token + accepted terms)"); continue
    gated = j.get("gated", False)
    print(f"  {d:28s} gated={gated}")
    st2, j2 = get(f"https://datasets-server.huggingface.co/info?dataset={d}")
    if st2 != 200:
        print(f"    datasets-server info: HTTP {st2} (gated datasets are not served publicly)"); continue
    for cfg, info in (j2.get("dataset_info") or {}).items():
        feats = list((info.get("features") or {}).keys())
        has = "content" in feats
        print(f"    config {cfg}: {len(feats)} cols | content column: {'YES' if has else 'NO'}")
        print(f"      {feats[:14]}")
        break

print("\n==== (2b) the dataset CARD is public even when the data is gated -> read the schema from it ====")
def raw(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "siliconllm-stage0"}), timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:
        return f"__ERR__ {e}"

for d in ["bigcode/the-stack-v2-dedup"]:
    md = raw(f"https://huggingface.co/datasets/{d}/raw/main/README.md")
    if md.startswith("__ERR__"):
        print(f"  {d}: {md[:80]}"); continue
    low = md.lower()
    for probe in ["blob_id", "softwareheritage", "s3", "content", "download_contents", "boto3", "src_encoding"]:
        n = low.count(probe)
        print(f"    {probe:18s} mentioned {n:3d}x in the card")
    i = low.find("download_contents")
    if i > 0:
        print("\n    --- the card's own content-fetch recipe (excerpt) ---")
        print("    " + "\n    ".join(md[max(0, i-320):i+260].splitlines()[:16]))

print("\n==== (2c) does a Gemma 4 even exist on the Hub? (the brief's Apache-2.0 claim is post-cutoff) ====")
for m in ["google/gemma-4-12b-pt", "google/gemma-4-12b", "google/gemma-4-9b", "google/gemma-4"]:
    st, j = get(f"{HF}/models/{m}")
    lic = ((j.get("cardData") or {}).get("license") if st == 200 and j else "-")
    print(f"  {m:24s} HTTP {st}  licence={lic}")

print("\nReading:")
print("  - if `content` is absent and blob_id/src_encoding/... are present, the brief's ingestion scripts are")
print("    Stack-v1-shaped and must be rewritten against the Software Heritage S3 fetch path (credentials + egress).")
print("  - if the dataset is gated, stage-0 also needs an HF token with the terms accepted, before ANY bulk pull.")
print("STOP. No commit.")
