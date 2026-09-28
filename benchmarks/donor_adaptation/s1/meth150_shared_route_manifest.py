"""Select unseen source/fragment-disjoint documents for the shared route."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth41_fresh_c96_manifest as M41
import meth42_instruct_prompt_manifest as M42
import meth57_product_key_external_manifest as M57M
import meth57_product_key_external_audit as M57
import meth121_zero_mean_child_external_manifest as M121
import meth133_q15_external_manifest as M133
import meth56_product_key_retention as M56
import meth107_long_chat_child_retention as M107


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
REF = M121.REF
SEED = "meth150-shared-structure-route-150000"
COUNTS = {"code": 8, "prose": 8, "technical_general": 8}
PG19_REL = "data/external/pg19/data/train-00005-of-00023-87dece6a0ab708e8.parquet"
PG19_SHA = "047e8d8d9609a57bff14be4091d9226cebe5fab59f504eb3e5a917c8f69df5ac"
PG19_FIRST_REL = "data/external/pg19/data/train-00000-of-00023-5263dc4323e881d2.parquet"
M133_PATH = DIR / "meth133_q15_external_manifest.json"
M133_SHA = "ec6f839a2c9b1803dc8d5658aafa8fd9191b59fa2607961850f5a8666bc35674"
M143_PATH = DIR / "meth143_fresh_hash_route_manifest.json"
M143_SHA = "e6c4fb6a92929eb29332c55a0887adb6627c9528f80b21c09f7759139070ccff"
ROUTE_RESULT = DIR / "meth149_recurrent_context_route_result.json"
ROUTE_RESULT_SHA = "cf8087f1677bf9c60076e055f35ac95cdb2e7f426ab5dce3d54ad441ed2e28e2"
ROUTE_TABLE = DIR / "meth149_recurrent_context_table.json"
ROUTE_TABLE_SHA = "93b6ca77ece874889e83ae368b9f63d78bef8e90a9fbda37cb9e1a871cb0503e"
EXACT_BANK = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth126_shared_a_factor_bank.bin"
EXACT_SHA = "1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1"
CHILD = ROOT / "benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth107_long_chat_e1280.pt"
CHILD_SHA = "15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520"
PRIOR = M133.PRIOR + ((M133_PATH, M133_SHA), (M143_PATH, M143_SHA))


def rank(category, source_id):
    return M17.sha((SEED + "|" + category + "|" + source_id).encode("utf-8"))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M17.sha(M57M.OLD.read_bytes()) == M57M.OLD_SHA
    assert M17.sha(M57M.M45.read_bytes()) == M57M.M45_SHA
    assert M17.sha(M57.EXTERNAL.read_bytes()) == M57.EXTERNAL_SHA
    assert M17.sha((ROOT / PG19_REL).read_bytes()) == PG19_SHA
    # The first shard satisfies H0's PG19 byte budget, so shard 5 is new.
    pg19_budget = int(320 * 1024 * 1024 * 0.40)
    first_bytes = 0
    for batch in pq.ParquetFile(ROOT / PG19_FIRST_REL).iter_batches(
            batch_size=8, columns=["text"]):
        for value in batch.column("text").to_pylist():
            if value:
                first_bytes += len(value.encode("utf-8", "ignore"))
            if first_bytes >= pg19_budget:
                break
        if first_bytes >= pg19_budget:
            break
    assert first_bytes >= pg19_budget
    assert file_sha(EXACT_BANK) == EXACT_SHA
    assert file_sha(CHILD) == CHILD_SHA
    assert file_sha(ROUTE_RESULT) == ROUTE_RESULT_SHA
    assert file_sha(ROUTE_TABLE) == ROUTE_TABLE_SHA
    route_result = json.loads(ROUTE_RESULT.read_text(encoding="utf-8"))
    route_table = json.loads(ROUTE_TABLE.read_text(encoding="utf-8"))
    assert route_result["decision"] == "diagnostic_only_meth145_rejected"
    assert route_result["recurrent_context_table_sha256"] == ROUTE_TABLE_SHA
    assert route_table["recurrent_tuple_count"] == 75
    calib = (ROOT / "benchmarks/donor_adaptation/density/corpus/calib.txt").read_bytes()
    assert M17.sha(calib) == M17.CALIB_SHA
    assert subprocess.check_output(["git", "rev-parse", REF], cwd=ROOT,
                                    text=True).strip() == REF
    for path, digest in PRIOR:
        assert M17.sha(path.read_bytes()) == digest, path
    tok = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                        local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tok) == M15.M13.TOK_FP
    old = json.loads(M57M.OLD.read_text(encoding="utf-8"))
    rebuilt_old, old_items = M41.select(tok)
    assert rebuilt_old == old
    m45 = json.loads(M57M.M45.read_text(encoding="utf-8"))
    m57 = json.loads(M57.EXTERNAL.read_text(encoding="utf-8"))
    excluded, base = M57M.prior_context(m57["items"], old_items, m45["items"])
    base.append(calib)
    for path, _ in PRIOR:
        previous = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(row["source_id"] for row in previous["items"])
        base.extend(row["text"].encode("utf-8") for row in previous["items"])
    assert M17.sha(M56.TRAIN_CHAT_PATH.read_bytes()) == M56.TRAIN_CHAT_SHA
    assert M17.sha(M107.LONG_TEACHER.read_bytes()) == M107.LONG_TEACHER_SHA
    train_chat = json.loads(M56.TRAIN_CHAT_PATH.read_text(encoding="utf-8"))["rows"]
    long_teacher = json.loads(M107.LONG_TEACHER.read_text(encoding="utf-8"))["rows"]
    assert len(train_chat) == len(long_teacher) == 256
    base.extend(tok.decode(row["prompt_ids"]).encode("utf-8") for row in train_chat)
    base.extend(row["continuation_text"].encode("utf-8") for row in long_teacher)
    selected = []
    provenance = {}

    def choose(category, candidates):
        found = []
        considered = 0
        for source_id, source_kind, source_ref, raw in candidates:
            considered += 1
            if source_id in excluded:
                continue
            hit = M121.span(raw, rank(category, source_id))
            if hit is None:
                continue
            offset, text, data = hit
            if M17.fragment_overlap(data, base):
                continue
            if M17.fragment_overlap(data, [r["text"].encode("utf-8") for r in found]):
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) < 256:
                continue
            found.append({"category": category, "source_id": source_id,
                          "source_kind": source_kind, "source_ref": source_ref,
                          "source_sha256": M17.sha(raw), "source_bytes": len(raw),
                          "span_start_byte": offset, "text": text,
                          "text_sha256": M17.sha(data), "bytes": len(data),
                          "document_ids": ids,
                          "document_ids_sha256": M17.sha(np.asarray(ids, dtype=np.int32).tobytes())})
            if len(found) == COUNTS[category]:
                break
        if len(found) != COUNTS[category]:
            raise RuntimeError(f"insufficient {category} source pool: {len(found)} of {COUNTS[category]} after {considered} candidates")
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
    code_paths = [p for p in code_paths if p.endswith((".py", ".c", ".h"))]
    choose("code", git_candidates(sorted(code_paths, key=lambda p: rank("code", p))))

    prose = []
    row_index = 0
    for batch in pq.ParquetFile(ROOT / PG19_REL).iter_batches(batch_size=8, columns=["text"]):
        for value in batch.column("text").to_pylist():
            sid = f"pg19:{PG19_REL}:row={row_index}"
            row_index += 1
            if isinstance(value, str):
                prose.append((sid, "pg19_train5", PG19_REL, value.encode("utf-8")))
    assert row_index >= 100
    choose("prose", sorted(prose, key=lambda item: rank("prose", item[0])))

    tech_paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only",
                                          REF, "docs"], cwd=ROOT, text=True).splitlines()
    tech_paths = [p for p in tech_paths if p.endswith(".md") and
                  not p.startswith("docs/research/NATIVE_EXPERT_SCALING_20260925/")]
    choose("technical_general", git_candidates(sorted(tech_paths,
                      key=lambda p: rank("technical_general", p))))

    for row in selected:
        excerpt = row["text"][:384]
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": M42.REQUEST.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(prompt) <= 1024
        row["excerpt"] = excerpt
        row["prompt_ids"] = prompt
        row["prompt_ids_sha256"] = M17.sha(np.asarray(prompt, dtype=np.int32).tobytes())
    assert len(selected) == 24 and len({r["source_id"] for r in selected}) == 24
    result = {"experiment": "METH-150-shared-route-new-source-manifest",
              "meth149_result_sha256": ROUTE_RESULT_SHA,
              "meth149_table_sha256": ROUTE_TABLE_SHA,
              "exact_bank_sha256": EXACT_SHA,
              "child_checkpoint_sha256": CHILD_SHA,
              "donor_sha256": M133.M57.MODEL_SHA,
              "parent_checkpoint_sha256": M133.M57.CHECKPOINT_SHA,
              "source_commit": REF, "pg19_train5_parquet_sha256": PG19_SHA,
              "pg19_train5_rows": row_index,
              "h0_calib_sha256": M17.CALIB_SHA,
              "h0_pg19_first_shard_satisfies_budget": True,
              "h0_pg19_first_shard_bytes_scanned": first_bytes,
              "h0_pg19_budget_bytes": pg19_budget,
              "train_chat_sha256": M56.TRAIN_CHAT_SHA,
              "long_teacher_sha256": M107.LONG_TEACHER_SHA,
              "prior_manifest_sha256": {path.name: digest for path, digest in PRIOR},
              "seed": SEED, "selected_counts": COUNTS,
              "model": M42.MODEL, "revision": M42.REV,
              "tokenizer_fingerprint": M15.M13.TOK_FP,
              "chat_template": M42.REQUEST, "excerpt_characters": 384,
              "selection_provenance": provenance, "items": selected}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "counts": COUNTS, "source_ids": [r["source_id"] for r in selected],
                      "max_document_tokens": max(len(r["document_ids"]) for r in selected),
                      "max_prompt_tokens": max(len(r["prompt_ids"]) for r in selected)}, indent=2))


if __name__ == "__main__":
    main()
