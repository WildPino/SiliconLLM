"""C50-A5 part 5: head restricted to the most-used vocabulary (protocol C50_A5_PART5_USED_VOCAB_HEAD_PROTOCOL).

--select   frequency of target IDs over SEL (per-category normalized, each category one third), kept set
           K_N = top-N (ties to the lower ID) + 256 single-byte tokens + 262 added tokens; FIT12 top-1 recall
           of the exact restricted scan for each grid N (engine top-1 from a5/FIT12.pos.bin); selection = the
           smallest N with recall >= 0.9995.  Control: the full vocabulary must give recall 1.0 (else VOID).
           Writes the frozen kept set (.npy, sha256 in the JSON) and the FIT record.
--confirm  reads the frozen kept set (sha256 must match the FIT record) and the engine scoring of DEV-B once:
           prints the gate line for the selected N first, then the other grid N, categories and missed tokens.

Usage:
    python -I benchmarks/chatbot50/c50_a5_used_vocab.py --select --sel <sel_ids.jsonl> --out results/c50_a5_used_vocab_fit.json
    python -I benchmarks/chatbot50/c50_a5_used_vocab.py --confirm --fit-record results/c50_a5_used_vocab_fit.json \
        --devb-pos <DEVB.pos.bin> --devb-ids <devb_ids.jsonl> --out results/c50_a5_used_vocab_devb.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

CAMP = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
TOKENIZER = CAMP / "sources/Ling-mini-2.0_a810f64/tokenizer.json"
FIT12 = CAMP / "a5/FIT12.pos.bin"
FROZEN = CAMP / "a5/a5_part5_kept.npy"
V, ROW_BYTES = 157184, 1680
GRID = [8192, 16384, 24576, 32768, 40960, 49152, 65536, 81920, 98304]
SELECT_BAR, GATE = 0.9995, 0.999
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])


def always_kept() -> np.ndarray:
    tok = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    single = [vocab[chr(c)] for c in cs]
    added = [a["id"] for a in tok["added_tokens"]]
    assert len(single) == 256 and len(added) == 262
    return np.array(sorted(set(single) | set(added)), dtype=np.int64)


def ranking(sel_rows) -> np.ndarray:
    by_cat = {}
    for r in sel_rows:
        by_cat.setdefault(r["category"], Counter()).update(r["ids"])
    score = np.zeros(V, dtype=np.float64)
    for c, cnt in by_cat.items():
        tot = sum(cnt.values())
        for i, k in cnt.items():
            score[i] += k / tot
    return np.lexsort((np.arange(V), -score))      # descending score, ties to the lower ID


def kept_mask(order, n, extra) -> np.ndarray:
    m = np.zeros(V, dtype=bool)
    m[order[:n]] = True
    m[extra] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--sel")
    ap.add_argument("--fit-record")
    ap.add_argument("--devb-pos")
    ap.add_argument("--devb-ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extra = always_kept()
    if args.select:
        sel = [json.loads(x) for x in Path(args.sel).read_text(encoding="utf-8").splitlines()]
        order = ranking(sel)
        fit = np.fromfile(FIT12, dtype=REC)
        full = kept_mask(order, V, extra)
        control = float(full[fit["top1"]].mean())
        out = {"cell": "C50-A5 part 5 FIT", "sel_sha256": hashlib.sha256(Path(args.sel).read_bytes()).hexdigest(),
               "sel_documents": len(sel), "fit12_positions": int(len(fit)), "control_full_vocab_recall": control,
               "void": control != 1.0, "grid": []}
        if control != 1.0:
            Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
            raise SystemExit("VOID: the full vocabulary does not give recall 1.0")
        for n in GRID:
            m = kept_mask(order, n, extra)
            rec = float(m[fit["top1"]].mean())
            out["grid"].append({"N": n, "kept_rows": int(m.sum()), "mb_per_token": m.sum() * ROW_BYTES / 1e6,
                                "fit12_recall": rec, "fit12_misses": int((~m[fit["top1"]]).sum())})
            print(f"N={n:6d} rows={m.sum():6d} {m.sum() * ROW_BYTES / 1e6:6.1f} MB  FIT12 recall {rec:.5f}")
        q = [g for g in out["grid"] if g["fit12_recall"] >= SELECT_BAR]
        out["selected"] = q[0] if q else None
        if q:
            m = kept_mask(order, q[0]["N"], extra)
            np.save(FROZEN, np.flatnonzero(m).astype(np.int32))
            out["frozen_path"] = str(FROZEN)
            out["frozen_sha256"] = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
            out["order_sha256"] = hashlib.sha256(order.astype(np.int32).tobytes()).hexdigest()
            np.save(FROZEN.with_name("a5_part5_order.npy"), order.astype(np.int32))
        print("SELECTED", out["selected"]["N"] if q else "none (part 5 FAILS on FIT; DEV-B is not read)")
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 0
    if args.confirm:
        fitrec = json.loads(Path(args.fit_record).read_text(encoding="utf-8"))
        if not fitrec.get("selected"):
            raise SystemExit("no selection on FIT: DEV-B is not read")
        if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != fitrec["frozen_sha256"]:
            raise SystemExit("frozen kept set changed since the FIT record")
        kept = np.zeros(V, dtype=bool)
        kept[np.load(FROZEN)] = True
        dev = np.fromfile(args.devb_pos, dtype=REC)
        hit = kept[dev["top1"]]
        rec = float(hit.mean())
        out = {"cell": "C50-A5 part 5 DEV-B confirmation", "devb_positions": int(len(dev)), "selected_N": fitrec["selected"]["N"],
               "devb_recall": rec, "gate": GATE, "verdict": "PASS" if rec >= GATE else "FAIL"}
        print(f"GATE N={fitrec['selected']['N']} DEV-B recall {rec:.5f} vs {GATE}: {out['verdict']}")
        order = np.load(FROZEN.with_name("a5_part5_order.npy"))
        out["reported_grid"] = []
        for n in GRID:
            m = kept_mask(order, n, extra)
            out["reported_grid"].append({"N": n, "devb_recall": float(m[dev["top1"]].mean())})
        cats = [json.loads(x)["category"] for x in Path(args.devb_ids).read_text(encoding="utf-8").splitlines()]
        dcat = np.array([cats[d] for d in dev["doc"]])
        out["reported_by_category"] = {c: float(hit[dcat == c].mean()) for c in sorted(set(cats))}
        tok = json.loads(TOKENIZER.read_text(encoding="utf-8"))
        inv = {i: s for s, i in tok["model"]["vocab"].items()}
        miss = Counter(int(t) for t in dev["top1"][~hit])
        out["reported_top_missed"] = [{"id": i, "token": ascii(inv.get(i, "?")), "count": k} for i, k in miss.most_common(25)]
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 0
    raise SystemExit("--select or --confirm")


if __name__ == "__main__":
    raise SystemExit(main())
