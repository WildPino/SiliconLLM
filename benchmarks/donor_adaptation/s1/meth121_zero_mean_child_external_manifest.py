"""Freeze source-disjoint METH-121 zero-mean-child external documents."""

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
SEED = "meth121-zero-mean-child-external-121000"
COUNTS = {"code": 8, "prose": 8, "technical_general": 8}
M62_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth62_mixed_external_manifest.json"
M62_SHA = "56237bcf2e83ea2e743ac0db2475bff299deed988b7cb75cf31cebcad8b4b2aa"
M72_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth72_e1280_external_manifest.json"
M72_SHA = "11621dce9638aae01cc3bd8c76730de7676e4bbb704271056f164b9ad4d5b437"
M83_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth83_q4_core_dev_manifest.json"
M83_SHA = "3ecc2193d8c5436b73a43fd959a99e6b8551f48d92392bbc92dad992a81d85fb"
PG19_VALIDATION_REL = "data/external/pg19/data/validation-00000-of-00001-0f92e2337f79aeac.parquet"
PG19_VALIDATION_SHA = "81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef"
PG19_TRAIN3_REL = "data/external/pg19/data/train-00003-of-00023-e7f0ba0d3c2eb6c0.parquet"
PG19_TRAIN3_SHA = "a8667487ba7cce1894211efde87b343f438ae5fb7da590ca4280bddab2a333d0"
M86_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth86_group64_r8_dev_manifest.json"
M86_SHA = "0d4ac34beea84ecbcaf92bac088345ccab5cc3a2e3f7898710018248bf4bca0c"
M89_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth89_quant_aware_dev_manifest.json"
M89_SHA = "6330eb7e0683bcfdc89da6712d98ba0e994dbea56c91b26230f5ad756ec060f8"
M90_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth90_quant_aware_external_manifest.json"
M90_SHA = "70d48b30de55c41283374223ba2b61edaacad396e6f3410c45c166bbaa885937"
M92_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth92_grouped_head_dev_manifest.json"
M92_SHA = "b07a40ce4739c943b22cdb984238a09ef9142910c83ed0369e2945e4b3262556"
M97_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth97_hierarchical_dev_manifest.json"
M97_SHA = "e5914ce5af584b0774f463cf8d80973f91c63d964f480382b29f65cfbd9422b5"
M100_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth100_hierarchical_retention_manifest.json"
M100_SHA = "40cfc1e5c6d5073b40f522e98f5cde8447905c201dbb5f68c27c453e15571ab0"
M102_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth102_hierarchical_external_manifest.json"
M102_SHA = "a5d1f6ef484eca561336d4625dd09db695034a139130fac490048da5a6650764"
M108_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth108_long_chat_dev_manifest.json"
M108_SHA = "dd0969e77853fc91cf12950da302c272d9d94f736945ed920ae0df5edea1a0b4"
M110_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth110_long_chat_external_manifest.json"
M110_SHA = "f2c30361c3379053712acce27e9083fa14e4ab3baa3dfad367a37129a85f4bda"
M113_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth113_child_scale_dev_manifest.json"
M113_SHA = "41f2a3f98ea95ab025280e2380b9793d4d6c46737f71b619dbdeb84c4bcbb271"
M115_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth115_alpha075_external_manifest.json"
M115_SHA = "965193d46bce1e9bd4ff8ce08b3339e8b19785bdcd3fa5cdf8c929004617fc05"
M118_PATH = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth118_zero_mean_child_dev_manifest.json"
M118_SHA = "1c89a4b0479d891e0441b152048c27947b12309c3d1062c9982b062c9e3fbae1"


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
    assert M17.sha(M86_PATH.read_bytes()) == M86_SHA
    assert M17.sha(M89_PATH.read_bytes()) == M89_SHA
    assert M17.sha(M90_PATH.read_bytes()) == M90_SHA
    assert M17.sha(M92_PATH.read_bytes()) == M92_SHA
    assert M17.sha(M97_PATH.read_bytes()) == M97_SHA
    assert M17.sha(M100_PATH.read_bytes()) == M100_SHA
    assert M17.sha(M102_PATH.read_bytes()) == M102_SHA
    assert M17.sha(M108_PATH.read_bytes()) == M108_SHA
    assert M17.sha(M110_PATH.read_bytes()) == M110_SHA
    assert M17.sha(M113_PATH.read_bytes()) == M113_SHA
    assert M17.sha(M115_PATH.read_bytes()) == M115_SHA
    assert M17.sha(M118_PATH.read_bytes()) == M118_SHA
    assert M17.sha((ROOT / M57M.PARQUET_REL).read_bytes()) == M57M.PARQUET_SHA
    assert M17.sha((ROOT / PG19_VALIDATION_REL).read_bytes()) == PG19_VALIDATION_SHA
    assert M17.sha((ROOT / PG19_TRAIN3_REL).read_bytes()) == PG19_TRAIN3_SHA
    assert subprocess.check_output(["git", "rev-parse", REF], cwd=ROOT,
                                    text=True).strip() == REF
    old = json.loads(M57M.OLD.read_text(encoding="utf-8"))
    m45 = json.loads(M57M.M45.read_text(encoding="utf-8"))
    m57 = json.loads(M57.EXTERNAL.read_text(encoding="utf-8"))
    m62 = json.loads(M62_PATH.read_text(encoding="utf-8"))
    m72 = json.loads(M72_PATH.read_text(encoding="utf-8"))
    m83 = json.loads(M83_PATH.read_text(encoding="utf-8"))
    m86 = json.loads(M86_PATH.read_text(encoding="utf-8"))
    m89 = json.loads(M89_PATH.read_text(encoding="utf-8"))
    m90 = json.loads(M90_PATH.read_text(encoding="utf-8"))
    m92 = json.loads(M92_PATH.read_text(encoding="utf-8"))
    m97 = json.loads(M97_PATH.read_text(encoding="utf-8"))
    m100 = json.loads(M100_PATH.read_text(encoding="utf-8"))
    m102 = json.loads(M102_PATH.read_text(encoding="utf-8"))
    m108 = json.loads(M108_PATH.read_text(encoding="utf-8"))
    m110 = json.loads(M110_PATH.read_text(encoding="utf-8"))
    m113 = json.loads(M113_PATH.read_text(encoding="utf-8"))
    m115 = json.loads(M115_PATH.read_text(encoding="utf-8"))
    m118 = json.loads(M118_PATH.read_text(encoding="utf-8"))
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
    excluded_ids.update(row["source_id"] for row in m86["items"])
    base.extend(row["text"].encode("utf-8") for row in m86["items"])
    excluded_ids.update(row["source_id"] for row in m89["items"])
    base.extend(row["text"].encode("utf-8") for row in m89["items"])
    excluded_ids.update(row["source_id"] for row in m90["items"])
    base.extend(row["text"].encode("utf-8") for row in m90["items"])
    excluded_ids.update(row["source_id"] for row in m92["items"])
    base.extend(row["text"].encode("utf-8") for row in m92["items"])
    excluded_ids.update(row["source_id"] for row in m97["items"])
    base.extend(row["text"].encode("utf-8") for row in m97["items"])
    excluded_ids.update(row["source_id"] for row in m100["items"])
    base.extend(row["text"].encode("utf-8") for row in m100["items"])
    excluded_ids.update(row["source_id"] for row in m102["items"])
    base.extend(row["text"].encode("utf-8") for row in m102["items"])
    excluded_ids.update(row["source_id"] for row in m108["items"])
    base.extend(row["text"].encode("utf-8") for row in m108["items"])
    excluded_ids.update(row["source_id"] for row in m110["items"])
    base.extend(row["text"].encode("utf-8") for row in m110["items"])
    excluded_ids.update(row["source_id"] for row in m113["items"])
    base.extend(row["text"].encode("utf-8") for row in m113["items"])
    excluded_ids.update(row["source_id"] for row in m115["items"])
    base.extend(row["text"].encode("utf-8") for row in m115["items"])
    excluded_ids.update(row["source_id"] for row in m118["items"])
    base.extend(row["text"].encode("utf-8") for row in m118["items"])
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

    parquet = ROOT / PG19_TRAIN3_REL
    prose_raw = []
    row_index = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8, columns=["text"]):
        for value in batch.column("text").to_pylist():
            source_id = f"pg19:{PG19_TRAIN3_REL}:row={row_index}"
            row_index += 1
            if isinstance(value, str):
                prose_raw.append((source_id, "pg19_train3", PG19_TRAIN3_REL,
                                  value.encode("utf-8")))
    assert row_index >= 100
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
    return {"experiment": "METH-121-zero-mean-E1280-external-manifest",
            "source_commit": REF, "parquet_sha256": M57M.PARQUET_SHA,
            "pg19_validation_parquet_sha256": PG19_VALIDATION_SHA,
            "pg19_train3_parquet_sha256": PG19_TRAIN3_SHA,
            "pg19_train3_rows": row_index,
            "prior_m57_manifest_sha256": M57.EXTERNAL_SHA,
            "prior_m62_manifest_sha256": M62_SHA,
            "prior_m72_manifest_sha256": M72_SHA,
            "prior_m83_manifest_sha256": M83_SHA,
            "prior_m86_manifest_sha256": M86_SHA,
            "prior_m89_manifest_sha256": M89_SHA,
            "prior_m90_manifest_sha256": M90_SHA,
            "prior_m92_manifest_sha256": M92_SHA,
            "prior_m97_manifest_sha256": M97_SHA,
            "prior_m100_manifest_sha256": M100_SHA,
            "prior_m102_manifest_sha256": M102_SHA,
            "prior_m108_manifest_sha256": M108_SHA,
            "prior_m110_manifest_sha256": M110_SHA,
            "prior_m113_manifest_sha256": M113_SHA,
            "prior_m115_manifest_sha256": M115_SHA,
            "prior_m118_manifest_sha256": M118_SHA,
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
