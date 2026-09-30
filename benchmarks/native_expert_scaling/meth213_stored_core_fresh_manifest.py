#!/usr/bin/env python3
"""Model-free, source-disjoint selection after the METH-212 stored loader pass."""
import argparse
import json
from pathlib import Path
import sys
import time

import meth212_stored_core_loader as L

sys.path.insert(0, str(L.P.ROOT / "benchmarks/donor_adaptation/s1"))
import meth176_long_external_manifest as M

M.SEED = "meth213-stored-core-quality-213213"
M.PG19_REL = "data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet"
M.PRIOR = M.PRIOR + ((M.DOC / "meth176_long_external_manifest.json",
    "9ac93c3b6e7eeaf67e928b3973f7042ad6ad002f2f5eed7f810046f8242d9029"),)
LOADER = M.DOC / "meth212_stored_loader_result.json"
LOADER_SHA = "cbc971d4850aa6532c31c7911c0af10001c7df2aee79a5fb2480027b707e89b1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start, stage = time.monotonic(), "bindings"
    try:
        assert M.digest(LOADER) == LOADER_SHA
        record = json.loads(LOADER.read_text(encoding="utf-8"))
        assert record["decision"] == "stored_loader_pass_fresh_quality_next"
        assert M.digest(L.CORE) == L.CORE_SHA
        assert M.digest(L.EXPORT) == L.EXPORT_SHA
        assert M.digest(L.P.M122.SPECIALIZED) == L.P.M122.SPECIALIZED_SHA
        assert M.digest(L.P.M122.TRAINING) == L.P.M57.TRAINING_SHA
        parent = json.loads(L.P.M122.TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
        assert M.digest(parent["path"]) == L.P.M57.CHECKPOINT_SHA
        stage = "parquet_binding"
        pg_sha = M.digest(M.ROOT / M.PG19_REL)
        tokenizer = M.AutoTokenizer.from_pretrained(M.M42.MODEL, revision=M.M42.REV,
                                                    local_files_only=True)
        assert M.M15.M13.C.tok_fingerprint(tokenizer) == M.M15.M13.TOK_FP
        stage = "prior_context"
        excluded, base, priors = M.prior_context(tokenizer)
        initial_excluded = len(excluded)
        print(json.dumps({"stage": stage, "excluded_sources": initial_excluded,
                          "runtime": M.budget(start)}), flush=True)
        stage = "selection"
        items, provenance, row_count = M.select(tokenizer, excluded, base, start)
        for item in items:
            if item["category"] == "prose":
                item["source_kind"] = "pg19_train14"
        result = {"experiment": "METH-213-stored-core-fresh-manifest",
            "core_sha256": L.CORE_SHA, "export_sha256": L.EXPORT_SHA,
            "loader_result_sha256": LOADER_SHA,
            "donor_sha256": L.P.M57.MODEL_SHA,
            "parent_checkpoint_sha256": L.P.M57.CHECKPOINT_SHA,
            "child_checkpoint_sha256": L.P.M122.SPECIALIZED_SHA,
            "model": M.M42.MODEL, "revision": M.M42.REV,
            "tokenizer_fingerprint": M.M15.M13.TOK_FP,
            "source_commit": M.REF, "pg19_parquet": M.PG19_REL,
            "pg19_parquet_sha256": pg_sha, "pg19_rows": row_count,
            "h0_calib_sha256": M.M17.CALIB_SHA, "prior_manifest_sha256": priors,
            "excluded_source_count": initial_excluded,
            "seed": M.SEED, "selected_counts": M.COUNTS,
            "chat_template": M.M42.REQUEST, "excerpt_characters": 384,
            "selection_provenance": provenance, "items": items,
            "runtime": M.budget(start),
            "scope": "Model-free source/fragment-disjoint selection; no quality inference"}
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        assert args.out.stat().st_size < 1_000_000_000
        print(json.dumps({"sha256": M.digest(args.out), "counts": M.COUNTS,
            "source_ids": [r["source_id"] for r in items], "runtime": result["runtime"]}), flush=True)
    except BaseException as error:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"stage": stage,
            "error": repr(error), "elapsed_seconds": time.monotonic()-start}, indent=2)+"\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
