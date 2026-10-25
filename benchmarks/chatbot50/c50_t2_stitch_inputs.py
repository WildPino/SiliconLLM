"""C50-T2 stitching cell: the documents both models read, from S3 corpus v1's teacher ids (never trained on).

S4 and T1 train on teacher shards 0-469; shards 470-493 are the validation slice, whose training curves used only
470-471. So:
  dev  = 4 documents of shards 470-471, for checking the tools (never the cell's numbers);
  main = 128 documents of shards 472-493, split by document into fit 64 / select 32 / test 32.
A document qualifies if it has at least LEN tokens; it is cut to its first LEN tokens. Draws use SEED.

    python -I benchmarks/chatbot50/c50_t2_stitch_inputs.py OUT_DIR
writes OUT_DIR/{dev,main}.ids (c50_llama_dump's input) and OUT_DIR/manifest.json.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

TEACHER = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s3/corpus_v1/teacher.tsv")
SHARD_DOCS, LEN, SEED = 48, 288, 20261026
DEV_SHARDS, MAIN_SHARDS = (470, 471), tuple(range(472, 494))
N_DEV, SPLIT = 4, (("fit", 64), ("select", 32), ("test", 32))


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    raw = TEACHER.read_bytes()
    assert len(raw) == 221584962, len(raw)  # the teacher run's ids_bytes
    lines = raw.decode("utf-8").split("\n")
    lo, hi = DEV_SHARDS[0] * SHARD_DOCS, (MAIN_SHARDS[-1] + 1) * SHARD_DOCS
    docs = {}
    for i in range(lo, hi):
        d, ids = lines[i].split("\t")
        assert int(d) == i
        docs[i] = [int(x) for x in ids.split(",")]
    ok = lambda shards: [i for i in range(shards[0] * SHARD_DOCS, (shards[-1] + 1) * SHARD_DOCS) if len(docs[i]) >= LEN]
    rng = random.Random(SEED)
    dev = sorted(rng.sample(ok(DEV_SHARDS), N_DEV))
    pool = ok(MAIN_SHARDS)
    main_docs = rng.sample(pool, sum(n for _, n in SPLIT))
    split, k = {}, 0
    for name, n in SPLIT:
        split[name] = sorted(main_docs[k:k + n])
        k += n
    order = split["fit"] + split["select"] + split["test"]
    man = {"teacher_tsv": str(TEACHER), "teacher_bytes": len(raw), "teacher_sha256": hashlib.sha256(raw).hexdigest(),
           "len": LEN, "seed": SEED, "dev_shards": list(DEV_SHARDS), "main_shards": [MAIN_SHARDS[0], MAIN_SHARDS[-1]],
           "qualifying": {"dev": len(ok(DEV_SHARDS)), "main": len(pool)}, "dev": dev, "split": split, "main_order": order}
    for name, idx in (("dev", dev), ("main", order)):
        text = "".join(f"{i}\t{','.join(map(str, docs[i][:LEN]))}\n" for i in idx)
        (out / f"{name}.ids").write_bytes(text.encode("utf-8"))
        man[f"{name}_ids_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k not in ("split", "main_order")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
