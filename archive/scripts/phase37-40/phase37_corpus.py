"""
Phase 37: Build multi-domain training corpora for static prior tribunal.

Produces 4 balanced corpora in data/:
  phase37_multidomain.bin      — 7 domains × 128KB
  phase37_loo_no_md.bin        — LOO: no markdown
  phase37_loo_no_code.bin      — LOO: no c_code
  phase37_loo_no_prose.bin     — LOO: no natural_text + no lit_prose

Shuffled is intentionally excluded from all training corpora.
"""

import os
import json

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")

SLICE_BYTES = 128_000  # 128 KB per domain

DOMAINS = {
    "c_code":       os.path.join(DATA_DIR, "c_code.c"),
    "markdown":     os.path.join(DATA_DIR, "markdown_docs.md"),
    "natural_text": os.path.join(DATA_DIR, "natural_text.txt"),
    "lit_prose":    os.path.join(DATA_DIR, "promessi_sposi.txt"),
    "json":         os.path.join(DATA_DIR, "json_synth.json"),
    "log":          os.path.join(DATA_DIR, "log_synth.log"),
    "notes_it":     os.path.join(DATA_DIR, "project_notes_it.txt"),
}

PROSE_DOMAINS = {"natural_text", "lit_prose"}


def load_slice(path, nbytes):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read(min(nbytes, size))
    print(f"  {os.path.basename(path)}: loaded {len(data):,} bytes")
    return data


def build_corpus(name, domain_keys):
    out_path = os.path.join(DATA_DIR, name)
    manifest = {}
    offset = 0
    with open(out_path, "wb") as f:
        for key in domain_keys:
            chunk = load_slice(DOMAINS[key], SLICE_BYTES)
            f.write(chunk)
            manifest[key] = {"start": offset, "end": offset + len(chunk)}
            offset += len(chunk)
    man_path = out_path + ".manifest.json"
    with open(man_path, "w") as f:
        json.dump({"file": name, "total_bytes": offset, "domains": manifest}, f, indent=2)
    print(f"-> {name}: {offset:,} bytes, {len(domain_keys)} domains")
    return out_path


def main():
    all_domains = list(DOMAINS.keys())

    print("Building phase37_multidomain.bin (all 7 domains)...")
    build_corpus("phase37_multidomain.bin", all_domains)

    print("\nBuilding phase37_loo_no_md.bin (no markdown)...")
    build_corpus("phase37_loo_no_md.bin",
                 [k for k in all_domains if k != "markdown"])

    print("\nBuilding phase37_loo_no_code.bin (no c_code)...")
    build_corpus("phase37_loo_no_code.bin",
                 [k for k in all_domains if k != "c_code"])

    print("\nBuilding phase37_loo_no_prose.bin (no natural_text, no lit_prose)...")
    build_corpus("phase37_loo_no_prose.bin",
                 [k for k in all_domains if k not in PROSE_DOMAINS])

    print("\nAll corpora built.")


if __name__ == "__main__":
    main()
