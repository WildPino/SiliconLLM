"""Freeze source-disjoint METH-86 grouped-R8 development documents."""

import argparse
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


ROOT = Path(__file__).resolve().parents[3]
REF = "882bb43f9118df1e4f79f85a6072111897bede9e"
SEED = "meth86-group64-r8-development-86086"
COUNTS = {"code": 8, "prose": 8, "technical_general": 8}
M62_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth62_mixed_external_manifest.json"
M62_SHA = "56237bcf2e83ea2e743ac0db2475bff299deed988b7cb75cf31cebcad8b4b2aa"
M72_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth72_e1280_external_manifest.json"
M72_SHA = "11621dce9638aae01cc3bd8c76730de7676e4bbb704271056f164b9ad4d5b437"
M83_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth83_q4_core_dev_manifest.json"
M83_SHA = "3ecc2193d8c5436b73a43fd959a99e6b8551f48d92392bbc92dad992a81d85fb"
PG19_VALIDATION_REL = "data/external/pg19/data/validation-00000-of-00001-0f92e2337f79aeac.parquet"
PG19_VALIDATION_SHA = "81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef"


def rank(category, source_id):
    return M17.sha((SEED + "|" + category + "|" + source_id).encode("utf-8"))


def span(raw, digest):
    if len(raw) < M17.SPAN:
        return None
    start = int(digest[:16], 16) % (len(raw) - M17.SPAN + 1)
    text = raw[start:start + M17.SPAN].decode("utf-8", errors="ignore")
    data = text.encode("utf-8")
    return (start, text, data) if len(data) >= 4000 else None


def build():
    assert M17.sha(M57M.OLD.read_bytes()) == M57M.OLD_SHA
    assert M17.sha(M57M.M45.read_bytes()) == M57M.M45_SHA
    assert M17.sha(M57.EXTERNAL.read_bytes()) == M57.EXTERNAL_SHA
    assert M17.sha(M62_PATH.read_bytes()) == M62_SHA
    assert M17.sha(M72_PATH.read_bytes()) == M72_SHA
    assert M17.sha(M83_PATH.read_bytes()) == M83_SHA
    assert M17.sha((ROOT / M57M.PARQUET_REL).read_bytes()) == M57M.PARQUET_SHA
    assert M17.sha((ROOT / PG19_VALIDATION_REL).read_bytes()) == PG19_VALIDATION_SHA
    assert subprocess.check_output(["git", "rev-parse", REF], cwd=ROOT,
                                    text=True).strip() == REF
    old = json.loads(M57M.OLD.read_text(encoding="utf-8"))
    m45 = json.loads(M57M.M45.read_text(encoding="utf-8"))
    m57 = json.loads(M57.EXTERNAL.read_text(encoding="utf-8"))
    m62 = json.loads(M62_PATH.read_text(encoding="utf-8"))
    m72 = json.loads(M72_PATH.read_text(encoding="utf-8"))
    m83 = json.loads(M83_PATH.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV,
                                               local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    rebuilt_old, old_items = M41.select(tokenizer)
    assert rebuilt_old == old
    excluded_ids, base = M57M.prior_context(m57["items"], old_items, m45["items"])
    excluded_ids.update(row["source_id"] for row in m62["items"])
    base.extend(row["text"].encode("utf-8") for row in m62["items"])
    excluded_ids.update(row["source_id"] for row in m72["items"])
    base.extend(row["text"].encode("utf-8") for row in m72["items"])
    excluded_ids.update(row["source_id"] for row in m83["items"])
    base.extend(row["text"].encode("utf-8") for row in m83["items"])
    selected = []
    provenance = {}

    def choose(category, candidates):
        chosen = []
        considered = 0
        for source_id, source_kind, source_ref, raw in candidates:
            considered += 1
            if source_id in excluded_ids:
                continue
            digest = rank(category, source_id)
            hit = span(raw, digest)
            if hit is None:
                continue
            offset, text, data = hit
            if M17.fragment_overlap(data, base):
                continue
            if M17.fragment_overlap(data, [r["text"].encode("utf-8") for r in chosen]):
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 256:
                continue
            row = {"category": category, "source_id": source_id,
                   "source_kind": source_kind, "source_ref": source_ref,
                   "source_sha256": M17.sha(raw), "source_bytes": len(raw),
                   "span_start_byte": offset, "text": text,
                   "text_sha256": M17.sha(data), "bytes": len(data),
                   "document_ids": ids,
                   "document_ids_sha256": M17.sha(np.asarray(ids, dtype=np.int32).tobytes())}
            chosen.append(row)
            if len(chosen) == COUNTS[category]:
                break
        assert len(chosen) == COUNTS[category], (category, len(chosen), considered)
        selected.extend(chosen)
        excluded_ids.update(row["source_id"] for row in chosen)
        base.extend(row["text"].encode("utf-8") for row in chosen)
        provenance[category] = {"considered": considered,
                                "chosen_source_ids": [r["source_id"] for r in chosen]}

    code_paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REF, "benchmarks"],
        cwd=ROOT, text=True).splitlines()
    code_paths = [p for p in code_paths if p.endswith((".py", ".c", ".h"))]
    code_paths.sort(key=lambda p: rank("code", p))
    def git_candidates(paths):
        for path in paths:
            raw = subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=ROOT)
            yield path, "git", REF, raw
    choose("code", git_candidates(code_paths))

    parquet = ROOT / M57M.PARQUET_REL
    prose_raw = []
    row_index = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8, columns=["text"]):
        for value in batch.column("text").to_pylist():
            source_id = f"pg19:{M57M.PARQUET_REL}:row={row_index}"
            row_index += 1
            if isinstance(value, str):
                prose_raw.append((source_id, "pg19", M57M.PARQUET_REL,
                                  value.encode("utf-8")))
    assert row_index == 100
    validation_index = 0
    for batch in pq.ParquetFile(ROOT / PG19_VALIDATION_REL).iter_batches(
            batch_size=8, columns=["text"]):
        for value in batch.column("text").to_pylist():
            source_id = f"pg19:{PG19_VALIDATION_REL}:row={validation_index}"
            validation_index += 1
            if isinstance(value, str):
                prose_raw.append((source_id, "pg19_validation", PG19_VALIDATION_REL,
                                  value.encode("utf-8")))
    prose_raw.sort(key=lambda item: rank("prose", item[0]))
    choose("prose", prose_raw)

    tech_paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REF, "docs"],
        cwd=ROOT, text=True).splitlines()
    tech_paths = [p for p in tech_paths if p.endswith(".md") and
                  not p.startswith("docs/research/NATIVE_EXPERT_SCALING_20260925/")]
    tech_paths.sort(key=lambda p: rank("technical_general", p))
    choose("technical_general", git_candidates(tech_paths))

    for row in selected:
        excerpt = row["text"][:384]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": M42.REQUEST.format(excerpt=excerpt)}],
            tokenize=True, add_generation_prompt=True)["input_ids"]
        assert 1 <= len(prompt) <= 1024
        row["excerpt"] = excerpt
        row["prompt_ids"] = prompt
        row["prompt_ids_sha256"] = M17.sha(np.asarray(prompt, dtype=np.int32).tobytes())
    assert len(selected) == 24
    assert len({r["source_id"] for r in selected}) == 24
    return {"experiment": "METH-86-group64-R8-core-development-manifest",
            "source_commit": REF, "parquet_sha256": M57M.PARQUET_SHA,
            "pg19_validation_parquet_sha256": PG19_VALIDATION_SHA,
            "prior_m57_manifest_sha256": M57.EXTERNAL_SHA,
            "prior_m62_manifest_sha256": M62_SHA,
            "prior_m72_manifest_sha256": M72_SHA,
            "prior_m83_manifest_sha256": M83_SHA,
            "seed": SEED, "selected_counts": COUNTS,
            "model": M42.MODEL, "revision": M42.REV,
            "tokenizer_fingerprint": M15.M13.TOK_FP,
            "chat_template": M42.REQUEST,
            "excerpt_characters": 384,
            "selection_provenance": provenance,
            "items": selected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"sha256": M17.sha(args.out.read_bytes()),
                      "counts": COUNTS,
                      "max_document_tokens": max(len(r["document_ids"]) for r in manifest["items"]),
                      "max_prompt_tokens": max(len(r["prompt_ids"]) for r in manifest["items"]),
                      "source_ids": [r["source_id"] for r in manifest["items"]]}, indent=2))


if __name__ == "__main__":
    main()
