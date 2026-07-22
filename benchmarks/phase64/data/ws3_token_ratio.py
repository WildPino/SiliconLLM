#!/usr/bin/env python3
"""WS3 -- re-measure the teacher/student token ratio on the REAL corpus, before any 3060 time.

WHY THIS GATES THE LOGIT PRODUCTION. The screening-block logit budget (~6-12 h of 3060) was computed
from a teacher/student token ratio of 0.479. That number is from TinyStories with a V=1024 student. On
code, against Qwen's code-efficient BPE and a V=2048/4096 student, it will be different -- Qwen packs
code into fewer, larger tokens, and our larger vocab packs the student side tighter too. Both move the
ratio, and the ratio is the multiplier on the hours. Committing 3060 time on the stale figure is
spending the budget on an unmeasured assumption.

The ratio that sizes the run is teacher_tokens / student_tokens: for every student position we store a
teacher distribution, so the teacher-token count is what the logit store and the scoring pass cost.

METHOD: a uniform sample of the DECONTAMINATED corpus (same seed), encoded three ways -- Qwen teacher,
student BPE V=2048, student BPE V=4096 -- counting tokens and bytes. No training, minutes.

Run (after corpus_pipeline.py has produced the tokenizers):
    python benchmarks/phase64/data/ws3_token_ratio.py --corpus data/phase64/corpus
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
TEACHER = "Qwen/Qwen2.5-Coder-1.5B"
MVE_RATIO = 0.479          # TinyStories, V=1024 -- the figure being replaced


def sample_text(corpus_dir, raw_dir, n_bytes, seed):
    """Prefer the decontaminated corpus; fall back to the raw fetch shards if the pipeline has not
    written kept-doc text (the manifest alone is not text). Uniform over docs under the seed."""
    shards = sorted(glob.glob(os.path.join(corpus_dir, "*.jsonl"))) or \
             sorted(glob.glob(os.path.join(raw_dir, "shard*.jsonl")))
    if not shards:
        sys.exit(f"no shard text found under {corpus_dir} or {raw_dir}")
    rng = np.random.default_rng(seed)
    rng.shuffle(shards)
    out, got = [], 0
    for sh in shards:
        for line in open(sh, encoding="utf-8"):
            d = json.loads(line)
            t = d.get("text")
            if not t: continue
            out.append(t); got += len(t.encode("utf-8"))
            if got >= n_bytes:
                return out, got, shards[0]
    return out, got, shards[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "phase64", "corpus"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "phase64", "raw_python"))
    ap.add_argument("--mb", type=float, default=64.0, help="MiB of text to sample")
    ap.add_argument("--seed", type=int, default=20260719)
    a = ap.parse_args()

    texts, nbytes, src = sample_text(a.corpus, a.raw, int(a.mb * 2**20), a.seed)
    blob = "".join(texts)
    nbytes = len(blob.encode("utf-8"))
    print(f"WS3 teacher/student token ratio   sample {nbytes/2**20:.1f} MiB, {len(texts)} docs, seed={a.seed}")
    print(f"  source: {os.path.dirname(src)}\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER, token=os.environ.get("HF_TOKEN"))
    tt = len(tok(blob, add_special_tokens=False)["input_ids"])
    print(f"  {'tokenizer':22s} {'tokens':>12s} {'bytes/tok':>10s}")
    print(f"  {TEACHER:22s} {tt:12d} {nbytes/tt:10.3f}")

    from cartography import Bpe
    rows = []
    for V in (2048, 4096):
        p = os.path.join(a.corpus, f"bpe{V}_code.bin")
        if not os.path.isfile(p):
            print(f"  bpe{V}_code.bin  NOT FOUND -- run corpus_pipeline.py first"); continue
        bpe = Bpe.load(p)
        st = len(bpe.encode(blob.encode("utf-8")))
        rows.append((V, st))
        print(f"  {'student V='+str(V):22s} {st:12d} {nbytes/st:10.3f}")

    print(f"\n  teacher/student ratio (the multiplier on the logit-production hours):")
    for V, st in rows:
        r = tt / st
        dv = 100 * (r / MVE_RATIO - 1)
        print(f"    V={V}: {r:.3f}   vs MVE {MVE_RATIO} = {dv:+.1f}%   "
              f"-> screening hours scale by {r/MVE_RATIO:.2f}x")
    if rows:
        print(f"\n  Read: the screening logit budget was 6-12 h at ratio {MVE_RATIO}. Multiply by the factor "
              f"above for the real figure. The chosen vocab arm's ratio is the one that binds.")
    print("\nSTOP. WS3 token ratio above. No commit.")


if __name__ == "__main__":
    main()
