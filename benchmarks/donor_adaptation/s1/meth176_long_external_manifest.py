#!/usr/bin/env python3
"""Select untouched METH-176 quality sources only after both long arms pass."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import psutil
import pyarrow.parquet as pq
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth41_fresh_c96_manifest as M41
import meth42_instruct_prompt_manifest as M42
import meth56_product_key_retention as M56
import meth57_product_key_external_manifest as M57M
import meth57_product_key_external_audit as M57
import meth107_long_chat_child_retention as M107
import meth121_zero_mean_child_external_manifest as M121
import meth133_q15_external_manifest as M133
import meth136_matched_sparse_train as M136
import meth156_factorized_external_manifest as M156


ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
REF = M121.REF
SEED = "meth176-long-quality-176176"
COUNTS = {"code": 8, "prose": 8, "technical_general": 8}
PG19_REL = "data/external/pg19/data/train-00009-of-00023-f07a29af90b71085.parquet"
CONTROL = DOC / "meth175_control_long_result.json"
CANDIDATE = DOC / "meth175_candidate_long_result.json"
TABLE = DOC / "meth172_threshold8_shared_table.json"
TABLE_SHA = "e89ff2cecbf887b85b5e361d47a8b918d8ccb93bd5c385a580ba993b93ac72db"
ROUTE = DOC / "meth173_fresh_recurrence_route_result.json"
ROUTE_SHA = "8cf875ade5a2996e7f2e9db352f5dde1afb65efe537e0038a0e61033c97770e3"
DRAWS_SHA = "9d4e06eced3cc1c51bac4e6219013f37f4495acd10c87827b567f0966f5c490c"
EXTRA_PRIOR = (
    (DOC / "meth156_factorized_external_manifest.json", "a3c370f285dce386e94570ac889c7941d9181a87122a2491e9c47905ff832ca1"),
    (DOC / "meth158_independent_training_manifest.json", "09086fc7c27877e12ae7c122365d67eef5d78d3cda442e5c3a5d36e8d452ed69"),
    (DOC / "meth164_cross_source_route_manifest.json", "d8420488104990ab63338721837c4f100135c45a07e787f2708eb741dfb63c9b"),
    (DOC / "meth165_expanded_training_manifest.json", "2875ac50a4de005b56ad6403c6bd30e358b87eb21d7dbb1918738957855068ba"),
    (DOC / "meth169_fresh_boundary_route_manifest.json", "8b2e3a545f33dbd31283fd173b9044e2c87bd55463e168d6b2f8c513cde4e468"),
    (DOC / "meth173_fresh_recurrence_route_manifest.json", "6ebfc36fd452f2a20cfcc9e8210505e36b725ef95992949655c19f3c94eaee20"),
)
PRIOR = M156.PRIOR + EXTRA_PRIOR
PRIOR_TEACHERS = (
    (DOC / "meth158_independent_teacher_merged.json", "cdcb22a4273148dede97a17eac81345c4fc5f8e40f09cb18ecac4e35115e437c"),
    (DOC / "meth164_cross_source_teacher_merged.json", "73bd039595682a268244bfc1d3df8c0e9bb005f5afe0cdc3c43315b90e131125"),
    (DOC / "meth165_expanded_teacher_merged.json", "9843b8d98d0bbe322cbbf2567f72d5c3b09a34bfa774b9079076ed9"),
    (DOC / "meth169_fresh_boundary_teacher.json", "b051d9cc45bac62bf55d599ea5f0a34ec08d3dbe015483d7818d59997f5140ce"),
    (DOC / "meth173_fresh_recurrence_teacher.json", "c1ce0318ee62484a650d4f7825c0fc417aa59c2606c73226c54eef0a47cf6e53"),
)
MAX_SECONDS = 50 * 60
MAX_RSS = 12 * (1 << 30)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rank(category, source_id):
    return M17.sha(f"{SEED}|{category}|{source_id}".encode("utf-8"))


def budget(start):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-176 manifest resource stop: {result}")
    return result


def bind_training(control_sha, candidate_sha):
    """The reserved PG19 shard must not be opened before this returns."""
    assert digest(CONTROL) == control_sha
    assert digest(CANDIDATE) == candidate_sha
    assert digest(TABLE) == TABLE_SHA and digest(ROUTE) == ROUTE_SHA
    assert digest(M136.EXACT_BANK) == M136.EXACT_SHA
    assert digest(M136.CHILD) == M136.CHILD_SHA
    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    bindings = {}
    for arm, result, expected in (
            ("control", control, "matched_long_control_complete"),
            ("candidate", candidate, "matched_long_candidate_pass_fresh_quality_pending")):
        assert result["decision"] == expected and result["updates"] == 3840
        assert len(result["training"]["records"]) == 3840
        assert [row["update"] for row in result["training"]["records"]] == list(range(1, 3841))
        assert all(result["gates"].values())
        assert result["training"]["initial_parity"]["logit_max_abs_error"] == 0
        assert result["draws_sha256"] == DRAWS_SHA
        assert result["table_sha256"] == TABLE_SHA
        assert result["route_replay_sha256"] == "3e71f5df77c54081e7d39b106c2cc1138902e9b91cf5b836d18eee350714dba9"
        bank = Path(result["artifact"]["path"])
        assert bank.is_file() and bank.stat().st_size == result["artifact"]["bytes"]
        assert result["artifact"]["readback_exact"]
        assert digest(bank) == result["artifact"]["sha256"]
        assert result["runtime"]["seconds"] <= (4.5 if arm == "control" else 12) * 3600
        bindings[arm] = {"result_sha256": control_sha if arm == "control" else candidate_sha,
                         "bank_path": str(bank.resolve()),
                         "bank_sha256": result["artifact"]["sha256"],
                         "bank_bytes": result["artifact"]["bytes"]}
    return bindings


def prior_context(tokenizer):
    assert M17.sha(M57M.OLD.read_bytes()) == M57M.OLD_SHA
    assert M17.sha(M57M.M45.read_bytes()) == M57M.M45_SHA
    assert M17.sha(M57.EXTERNAL.read_bytes()) == M57.EXTERNAL_SHA
    old = json.loads(M57M.OLD.read_text(encoding="utf-8"))
    rebuilt, old_items = M41.select(tokenizer)
    assert rebuilt == old
    m45 = json.loads(M57M.M45.read_text(encoding="utf-8"))
    m57 = json.loads(M57.EXTERNAL.read_text(encoding="utf-8"))
    excluded, base = M57M.prior_context(m57["items"], old_items, m45["items"])
    calib = (ROOT / "benchmarks/donor_adaptation/density/corpus/calib.txt").read_bytes()
    assert M17.sha(calib) == M17.CALIB_SHA
    base.append(calib)
    prior_sha = {}
    for path, expected in PRIOR:
        assert digest(path) == expected, path
        prior_sha[path.name] = expected
        previous = json.loads(path.read_text(encoding="utf-8"))
        for row in previous.get("items", []):
            excluded.add(row["source_id"])
            base.append(row["text"].encode("utf-8"))
        for key in ("chat_rows", "raw_rows", "document_rows"):
            for row in previous.get(key, []):
                for ids_key in ("prompt_ids", "window_ids", "span_ids"):
                    if ids_key in row:
                        base.append(tokenizer.decode(row[ids_key]).encode("utf-8"))
                source = previous.get("source")
                if source is not None and "source_row" in row:
                    excluded.add(f"pg19:{source}:row={row['source_row']}")
    for path, expected in PRIOR_TEACHERS:
        assert digest(path) == expected, path
        prior_sha[path.name] = expected
        teacher = json.loads(path.read_text(encoding="utf-8"))
        for row in teacher["rows"]:
            base.append(tokenizer.decode(row["continuation_ids"]).encode("utf-8"))
    assert M17.sha(M56.TRAIN_CHAT_PATH.read_bytes()) == M56.TRAIN_CHAT_SHA
    assert M17.sha(M107.LONG_TEACHER.read_bytes()) == M107.LONG_TEACHER_SHA
    train_chat = json.loads(M56.TRAIN_CHAT_PATH.read_text(encoding="utf-8"))["rows"]
    long_teacher = json.loads(M107.LONG_TEACHER.read_text(encoding="utf-8"))["rows"]
    base.extend(tokenizer.decode(row["prompt_ids"]).encode("utf-8") for row in train_chat)
    base.extend(row["continuation_text"].encode("utf-8") for row in long_teacher)
    return excluded, base, prior_sha


def select(tokenizer, excluded, base, start):
    selected, provenance = [], {}

    def choose(category, candidates):
        found = []
        considered = 0
        for source_id, source_kind, source_ref, raw in candidates:
            considered += 1
            budget(start)
            if source_id in excluded:
                continue
            hit = M121.span(raw, rank(category, source_id))
            if hit is None:
                continue
            offset, text, data = hit
            words = re.findall(r"[A-Za-z0-9_]{2,}", text[:384])
            if len(words) < 20 or len({word.lower() for word in words}) < 12:
                continue
            if M17.fragment_overlap(data, base):
                continue
            if M17.fragment_overlap(data, [row["text"].encode("utf-8") for row in found]):
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 256:
                continue
            found.append({"category": category, "source_id": source_id,
                          "source_kind": source_kind, "source_ref": source_ref,
                          "source_sha256": M17.sha(raw), "source_bytes": len(raw),
                          "span_start_byte": offset, "text": text,
                          "text_sha256": M17.sha(data), "bytes": len(data),
                          "answerability_screen": {
                              "lexical_tokens": len(words),
                              "distinct_lexical_tokens": len({word.lower() for word in words}),
                              "passed": True},
                          "document_ids": ids,
                          "document_ids_sha256": M17.sha(np.asarray(ids, dtype=np.int32).tobytes())})
            if len(found) == COUNTS[category]:
                break
        if len(found) != COUNTS[category]:
            raise RuntimeError(f"METH-176 insufficient {category}: {len(found)} of {COUNTS[category]} after {considered}")
        selected.extend(found)
        excluded.update(row["source_id"] for row in found)
        base.extend(row["text"].encode("utf-8") for row in found)
        provenance[category] = {"considered": considered,
                                "chosen_source_ids": [row["source_id"] for row in found]}

    def git_candidates(paths):
        for path in paths:
            yield path, "git", REF, subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=ROOT)

    code_paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only",
                                          REF, "benchmarks"], cwd=ROOT, text=True).splitlines()
    code_paths = [path for path in code_paths if path.endswith((".py", ".c", ".h"))]
    choose("code", git_candidates(sorted(code_paths, key=lambda path: rank("code", path))))

    prose = []
    row_index = 0
    for batch in pq.ParquetFile(ROOT / PG19_REL).iter_batches(batch_size=8, columns=["text"]):
        for value in batch.column("text").to_pylist():
            source_id = f"pg19:{PG19_REL}:row={row_index}"
            row_index += 1
            if isinstance(value, str):
                prose.append((source_id, "pg19_train9", PG19_REL, value.encode("utf-8")))
    assert row_index >= 100
    choose("prose", sorted(prose, key=lambda item: rank("prose", item[0])))

    tech_paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only",
                                          REF, "docs"], cwd=ROOT, text=True).splitlines()
    tech_paths = [path for path in tech_paths if path.endswith(".md") and
                  not path.startswith("docs/research/NATIVE_EXPERT_SCALING_20260925/")]
    choose("technical_general", git_candidates(sorted(
        tech_paths, key=lambda path: rank("technical_general", path))))

    for row in selected:
        excerpt = row["text"][:384]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": M42.REQUEST.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(prompt) <= 1024
        row["excerpt"] = excerpt
        row["prompt_ids"] = prompt
        row["prompt_ids_sha256"] = M17.sha(np.asarray(prompt, dtype=np.int32).tobytes())
    assert len(selected) == 24 and len({row["source_id"] for row in selected}) == 24
    return selected, provenance, row_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    start = time.monotonic()
    stage = "training_bindings"
    try:
        bindings = bind_training(args.control_sha, args.candidate_sha)
        budget(start)
        # The reserved shard is first read below, after the training gate above.
        stage = "reserved_shard_sha256"
        pg19_sha = digest(ROOT / PG19_REL)
        budget(start)
        stage = "tokenizer_and_prior_context"
        assert subprocess.check_output(["git", "rev-parse", REF], cwd=ROOT,
                                       text=True).strip() == REF
        tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                                  local_files_only=True)
        assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
        excluded, base, prior_sha = prior_context(tokenizer)
        budget(start)
        stage = "deterministic_selection"
        selected, provenance, row_count = select(tokenizer, excluded, base, start)
        stage = "manifest_write"
        result = {"experiment": "METH-176-matched-long-fresh-external-manifest",
              "training_bindings": bindings,
              "exact_bank_sha256": M136.EXACT_SHA,
              "child_checkpoint_sha256": M136.CHILD_SHA,
              "route_result_sha256": ROUTE_SHA,
              "route_table_sha256": TABLE_SHA,
              "draws_sha256": DRAWS_SHA,
              "donor_sha256": M133.M57.MODEL_SHA,
              "parent_checkpoint_sha256": M133.M57.CHECKPOINT_SHA,
              "source_commit": REF, "pg19_train9_parquet_sha256": pg19_sha,
              "pg19_train9_rows": row_count,
              "h0_calib_sha256": M17.CALIB_SHA,
              "prior_manifest_sha256": prior_sha,
              "seed": SEED, "selected_counts": COUNTS,
              "model": M42.MODEL, "revision": M42.REV,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "chat_template": M42.REQUEST, "excerpt_characters": 384,
              "selection_provenance": provenance, "items": selected,
              "runtime": budget(start)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        assert args.out.stat().st_size < 10_000_000_000
        print(json.dumps({"manifest_sha256": digest(args.out),
                      "counts": COUNTS,
                      "source_ids": [row["source_id"] for row in selected],
                      "max_document_tokens": max(len(row["document_ids"]) for row in selected),
                      "max_prompt_tokens": max(len(row["prompt_ids"]) for row in selected)},
                         indent=2), flush=True)
    except BaseException as error:
        failure = args.out.with_name(args.out.stem + ".failure.json")
        failure.write_text(json.dumps({"experiment": "METH-176-manifest-failure",
            "stage": stage, "error": repr(error),
            "elapsed_seconds": time.monotonic() - start,
            "rss_bytes": psutil.Process().memory_info().rss},
            indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
