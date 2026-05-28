"""
Phase 41a.4: Unit x Domain Cartography Tribunal.

For each (corpus, unit) cell measure entropy_per_byte, cycles_per_byte,
cache_footprint. Compute composite_cost = entropy * cycles (lower=better).
Output: JSON raw results + Markdown report with dual-axis verdict.

Run: .venv\\Scripts\\python.exe experiments\\phase41a\\cartography.py
Expected duration: 5-15 min (BPE training dominates).
"""
import os, json, time, hashlib
from collections import Counter
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

CORPORA = {
    "tinystories": ("experiments/phase41a/corpora/tinystories_64mb.txt", "C"),
    "python_code": ("experiments/phase41a/corpora/python_code_64mb.txt", "A"),
    "logs":        ("experiments/phase41a/corpora/logs_64mb.txt",        "A"),
    "markdown":    ("experiments/phase41a/corpora/markdown_64mb.txt",    "B"),
}
RESULTS_JSON = "experiments/phase41a/results/cartography_raw.json"
REPORT_MD    = "docs/phases/phase41a_cartography.md"

def shannon(counts):
    total = sum(counts.values())
    return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)

def time_lookup_loop(token_iter, V):
    """Time a 'dummy predict' loop: read token, lookup table entry, accumulate."""
    table = np.zeros(V, dtype=np.float32)
    t0 = time.perf_counter_ns()
    s = 0.0
    for tk in token_iter:
        s += table[tk]
    return time.perf_counter_ns() - t0

# ── 6 candidate units ─────────────────────────────────────────────────────────

def u_byte(data):
    counts = Counter(data)
    H = shannon(counts)
    ns = time_lookup_loop(data, 256)
    return {"V": 256, "L": 1, "n_tokens": len(data),
            "H_per_byte": H, "ns_per_byte": ns/len(data),
            "footprint": 256*4}

def u_byte_tuple2(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    if len(arr) % 2: arr = np.concatenate([arr, [0]])
    toks = (arr[0::2].astype(np.uint16) << 8) | arr[1::2].astype(np.uint16)
    counts = Counter(toks.tolist())
    H = shannon(counts) / 2
    ns = time_lookup_loop(toks, 65536)
    return {"V": 65536, "L": 2, "n_tokens": len(toks),
            "H_per_byte": H, "ns_per_byte": ns/len(data),
            "footprint": 65536*4}

def u_hash_vocab(data, V=8192, L=4):
    n = len(data) // L
    toks = np.zeros(n, dtype=np.int32)
    for i in range(n):
        h = 0x811c9dc5
        for j in range(L):
            h = ((h ^ data[i*L+j]) * 0x01000193) & 0xFFFFFFFF
        toks[i] = h % V
    counts = Counter(toks.tolist())
    H = shannon(counts) / L
    ns = time_lookup_loop(toks, V)
    return {"V": V, "L": L, "n_tokens": n,
            "H_per_byte": H, "ns_per_byte": ns/len(data),
            "footprint": V*4}

def u_multi_stride(data, L=4):
    n = len(data) // L
    arr = np.frombuffer(data[:n*L], dtype=np.uint8).reshape(n, L)
    H_total = sum(shannon(Counter(arr[:, p].tolist())) for p in range(L))
    H = H_total / L
    # cycles: 4 independent lookups per group
    tables = [np.zeros(256, dtype=np.float32) for _ in range(L)]
    t0 = time.perf_counter_ns()
    s = 0.0
    for g in arr:
        for p in range(L):
            s += tables[p][g[p]]
    ns = time.perf_counter_ns() - t0
    return {"V": 256, "L": L, "n_tokens": n,
            "H_per_byte": H, "ns_per_byte": ns/len(data),
            "footprint": 256*L*4}

def u_bpe(corpus_path, data, vocab_size):
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    # Train on file directly (avoids UTF-8 decode on mixed-encoding corpora)
    tk.train(files=[corpus_path], trainer=trainer)
    # Use ByteLevel encoding which handles non-UTF8 natively
    text = data.decode("utf-8", errors="replace")  # For metrics only
    t0 = time.perf_counter_ns()
    enc = tk.encode(text)
    ns = time.perf_counter_ns() - t0
    ids = enc.ids
    counts = Counter(ids)
    L = len(data) / len(ids)
    H = shannon(counts) / L
    V = tk.get_vocab_size()
    return {"V": V, "L": round(L, 3), "n_tokens": len(ids),
            "H_per_byte": H, "ns_per_byte": ns/len(data),
            "footprint": V*32}  # vocab + merges estimate

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    results = {}
    for name, (path, cat) in CORPORA.items():
        print(f"\n=== {name} ({cat}) ===")
        data = open(path, "rb").read()
        print(f"  {len(data)} bytes loaded")
        results[name] = {"category": cat, "size": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()[:16],
                         "units": {}}
        for label, fn in [
            ("byte",          lambda: u_byte(data)),
            ("byte_tuple2",   lambda: u_byte_tuple2(data)),
            ("hash_vocab",    lambda: u_hash_vocab(data, 8192, 4)),
            ("multi_stride",  lambda: u_multi_stride(data, 4)),
            ("bpe_8k",        lambda: u_bpe(path, data, 8192)),
            ("bpe_16k",       lambda: u_bpe(path, data, 16384)),
        ]:
            print(f"  {label}...", end="", flush=True)
            t0 = time.time()
            try:
                results[name]["units"][label] = fn()
                print(f" done in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f" SKIP ({type(e).__name__}: {str(e)[:60]})")
                results[name]["units"][label] = {"error": str(e)}
    json.dump(results, open(RESULTS_JSON, "w"), indent=2)
    write_report(results)
    print(f"\nResults  : {RESULTS_JSON}\nReport   : {REPORT_MD}")

def write_report(results):
    units = ["byte", "byte_tuple2", "hash_vocab", "multi_stride", "bpe_8k", "bpe_16k"]
    corpora = list(results.keys())
    lines = ["# Phase 41a — Unit x Domain Cartography\n"]

    def fmt_table(metric, fmt="{:.4f}"):
        out = [f"\n## {metric}\n",
               "| Unit \\ Corpus | " + " | ".join(f"{c} ({results[c]['category']})" for c in corpora) + " |",
               "|" + "---|" * (len(corpora)+1)]
        for u in units:
            row = [u] + [fmt.format(results[c]["units"][u][metric]) for c in corpora]
            out.append("| " + " | ".join(row) + " |")
        return out

    lines += fmt_table("H_per_byte", "{:.4f}")
    lines += fmt_table("ns_per_byte", "{:.2f}")
    lines += fmt_table("footprint", "{:d}")

    # composite cost
    lines.append("\n## composite_cost = H_per_byte * ns_per_byte (lower = better)\n")
    lines.append("| Unit \\ Corpus | " + " | ".join(f"{c} ({results[c]['category']})" for c in corpora) + " | mean |")
    lines.append("|" + "---|" * (len(corpora)+2))
    unit_means = {}
    for u in units:
        costs = [results[c]["units"][u]["H_per_byte"] * results[c]["units"][u]["ns_per_byte"] for c in corpora]
        unit_means[u] = np.mean(costs)
        lines.append("| " + " | ".join([u] + [f"{x:.2f}" for x in costs] + [f"{unit_means[u]:.2f}"]) + " |")

    # axes
    lines.append("\n## Domain axis (mean composite across all units)\n")
    cat_means = {}
    for c in corpora:
        m = np.mean([results[c]["units"][u]["H_per_byte"] * results[c]["units"][u]["ns_per_byte"] for u in units])
        cat_means[c] = (results[c]["category"], m)
        lines.append(f"- **{c}** ({results[c]['category']}): {m:.2f}")
    a_vals = [m for _, (cat, m) in cat_means.items() if cat == "A"]
    c_vals = [m for _, (cat, m) in cat_means.items() if cat == "C"]
    a_mean = np.mean(a_vals); c_mean = np.mean(c_vals)
    ratio = c_mean / a_mean if a_mean > 0 else float("inf")
    lines.append(f"\n- **Mean A**: {a_mean:.2f}  |  **Mean C**: {c_mean:.2f}  |  **C/A ratio**: {ratio:.2f}")

    if ratio >= 1.5:
        verdict = "DECISIVE — silicon prefers category A. **Target pivot to A-dominant recommended.**"
    elif ratio < 1.2:
        verdict = "WEAK — prose remains viable."
    else:
        verdict = "AMBIGUOUS — additional corpora needed."
    lines.append(f"\n### Domain verdict: {verdict}\n")

    lines.append("\n## Unit axis (best unit per corpus)\n")
    lines.append("| Corpus | Winner | 2nd | Winner cost |")
    lines.append("|---|---|---|---|")
    for c in corpora:
        ranked = sorted(units, key=lambda u: results[c]["units"][u]["H_per_byte"] * results[c]["units"][u]["ns_per_byte"])
        wc = results[c]["units"][ranked[0]]["H_per_byte"] * results[c]["units"][ranked[0]]["ns_per_byte"]
        lines.append(f"| {c} | {ranked[0]} | {ranked[1]} | {wc:.2f} |")

    winners = [sorted(units, key=lambda u: results[c]["units"][u]["H_per_byte"] * results[c]["units"][u]["ns_per_byte"])[0] for c in corpora]
    consistency = len(set(winners))
    if consistency == 1:
        unit_verdict = f"UNANIMOUS — **{winners[0]}** wins on all corpora."
    elif consistency == 2:
        unit_verdict = f"MOSTLY CONSISTENT — 2 distinct winners: {set(winners)}."
    else:
        unit_verdict = f"DOMAIN-DEPENDENT — winners differ ({set(winners)}); multi-unit architecture may be warranted."
    lines.append(f"\n### Unit verdict: {unit_verdict}\n")

    # Mean-rank overall winner
    overall_winner = min(unit_means, key=unit_means.get)
    lines.append(f"\n### Overall best unit (min mean composite): **{overall_winner}** ({unit_means[overall_winner]:.2f})\n")

    open(REPORT_MD, "w", encoding="utf-8").write("\n".join(lines))

if __name__ == "__main__":
    run()
