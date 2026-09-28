"""Select new source/fragment-disjoint documents for the Q15 quality audit."""

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
import meth121_zero_mean_child_external_manifest as M121


ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
REF = M121.REF
SEED = "meth133-q15-full-model-133000"
COUNTS = {"code": 8, "prose": 8, "technical_general": 8}
PG19_REL = "data/external/pg19/data/train-00005-of-00023-87dece6a0ab708e8.parquet"
PG19_SHA = "047e8d8d9609a57bff14be4091d9226cebe5fab59f504eb3e5a917c8f69df5ac"
M121_PATH = DIR / "meth121_zero_mean_child_external_manifest.json"
M121_SHA = "7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366"
M131_PATH = DIR / "meth131_q7_external_manifest.json"
M131_SHA = "84d8dbb400be133b6d1e440d55da5d09afd8ebbe0bd659474bdad8e708868ba9"
PRIOR = (
    (M121.M62_PATH, M121.M62_SHA), (M121.M72_PATH, M121.M72_SHA),
    (M121.M83_PATH, M121.M83_SHA), (M121.M86_PATH, M121.M86_SHA),
    (M121.M89_PATH, M121.M89_SHA), (M121.M90_PATH, M121.M90_SHA),
    (M121.M92_PATH, M121.M92_SHA), (M121.M97_PATH, M121.M97_SHA),
    (M121.M100_PATH, M121.M100_SHA), (M121.M102_PATH, M121.M102_SHA),
    (M121.M108_PATH, M121.M108_SHA), (M121.M110_PATH, M121.M110_SHA),
    (M121.M113_PATH, M121.M113_SHA), (M121.M115_PATH, M121.M115_SHA),
    (M121.M118_PATH, M121.M118_SHA), (M121_PATH, M121_SHA),
    (M131_PATH, M131_SHA),
)


def rank(category, source_id):
    return M17.sha((SEED + "|" + category + "|" + source_id).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert M17.sha(M57M.OLD.read_bytes()) == M57M.OLD_SHA
    assert M17.sha(M57M.M45.read_bytes()) == M57M.M45_SHA
    assert M17.sha(M57.EXTERNAL.read_bytes()) == M57.EXTERNAL_SHA
    assert M17.sha((ROOT / PG19_REL).read_bytes()) == PG19_SHA
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
    for path, _ in PRIOR:
        previous = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(row["source_id"] for row in previous["items"])
        base.extend(row["text"].encode("utf-8") for row in previous["items"])
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
    result = {"experiment": "METH-133-Q15-fresh-external-manifest",
              "source_commit": REF, "pg19_train5_parquet_sha256": PG19_SHA,
              "pg19_train5_rows": row_index,
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
