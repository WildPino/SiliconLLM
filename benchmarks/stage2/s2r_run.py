"""Stage 2, routing by conversation: records the experts each token chose, on P0, without changing the model.  Numbers only.

    pairs                              writes the planted set: 24 pairs, the first 1,024 tokens of a document and of a document of
                                       another category, one after the other.
    record --set S --k K --out F       S: docs (the 72 documents), gen (prompts and answers of the panels, as test 2b's set),
                                       pairs.  K: 8, as trained; 4, top 4 with k2 = 16 on every position.
Records go to WORK/<set>_k<K>.route, one per run; CONFIG, MOE and ROUTE lines must confirm the cell.

Usage: python -I benchmarks/stage2/s2r_run.py pairs
       python -I benchmarks/stage2/s2r_run.py record --set docs --k 8 --out benchmarks/stage2/results/s2r_run.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
from s1_run import INPUTS, sha256  # noqa: E402

P0 = Path("C:/SiliconLLM_fast/s2t/m/p0.gguf")
P0_SHA = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
SCORE = HERE / "bin" / "s2r_llama_score_avx2.exe"
WORK = Path("E:/SiliconLLM_Artifacts/stage2/s2r")
SETS = {"docs": INPUTS["a4"] / "docs.tsv", "gen": Path("C:/SiliconLLM_fast/s2t/sets/gen.tsv"), "pairs": WORK / "pairs.tsv",
        "tiny": Path("C:/SiliconLLM_fast/s2t/sets/tiny.tsv")}
HALF, PAIRS_PER = 1024, 8


def pairs(_: argparse.Namespace) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    ids = [ln.split("\t")[1].strip().split(",") for ln in SETS["docs"].read_text(encoding="ascii").splitlines()]
    cats: dict[str, list[int]] = {}
    for ln in (INPUTS["a4"] / "docs_meta.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(ln)
        cats.setdefault(m["category"], []).append(m["index"])
    names = sorted(cats)
    rows, meta = [], []
    for c, a in enumerate(names):
        b = names[(c + 1) % len(names)]
        for i in range(PAIRS_PER):
            x, y = cats[a][i], cats[b][i]
            rows.append(f"{len(rows)}\t{','.join(ids[x][:HALF] + ids[y][:HALF])}\n")
            meta.append({"first": x, "second": y, "first_category": a, "second_category": b})
    SETS["pairs"].write_text("".join(rows), encoding="ascii", newline="\n")
    (WORK / "pairs_meta.json").write_text(json.dumps({"half": HALF, "pairs": meta}, indent=1) + "\n", encoding="utf-8")
    print(len(rows), "pairs")
    return 0


def record(a: argparse.Namespace) -> int:
    if sha256(P0) != P0_SHA:
        raise SystemExit("P0 differs from its pin")
    report = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {"cell": "S2R-routing-by-conversation", "runs": {}}
    WORK.mkdir(parents=True, exist_ok=True)
    name = f"{a.set}_k{a.k}"
    route = WORK / f"{name}.route"
    cmd = [str(SCORE), "--model", str(P0), "--ids", str(SETS[a.set]), "--out", str(WORK / name), "--threads", "6", "--ubatch", "512",
           "--ctx", "8192", "--max-projected-s", "100000", "--route-out", str(route)]
    cmd += ["--moe-mode", "none"] if a.k == 8 else ["--moe-mode", "top", "--moe-k", str(a.k), "--moe-mass", "16"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    config = [x for x in lines if x.startswith("CONFIG ")]
    moe = [x for x in lines if x.startswith("MOE ")]
    rt = [x for x in lines if x.startswith("ROUTE ")]
    ok = proc.returncode == 0 and len(config) == 1 and len(moe) == 1 and len(rt) == 1 and f"route_out={route}" in config[0]
    ok = ok and (f"mode=none" in moe[0] if a.k == 8 else f"mode=top k={a.k} " in moe[0])
    report["runs"][name] = {"cmd": cmd, "exe_sha256": sha256(SCORE), "returncode": proc.returncode, "wall_s": round(time.time() - t0, 1),
                            "config": config, "moe": moe, "route": rt, "stderr_tail": proc.stderr[-400:],
                            "route_sha256": sha256(route) if ok else None, "status": "COMPLETE" if ok else "STOPPED"}
    a.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(name, report["runs"][name]["status"], report["runs"][name]["wall_s"], rt, flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("pairs").set_defaults(fn=pairs)
    r = sp.add_parser("record")
    r.add_argument("--set", required=True, choices=sorted(SETS))
    r.add_argument("--k", type=int, required=True, choices=[8, 4])
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(fn=record)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
