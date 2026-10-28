"""C50 cell M part 2, F0's side (local, CPU): F0's temperature and its records on A4-bis's 72 post-release documents.

Two runs of the pinned scorer (`bin/c50_llama_score_avx2.exe`, as S2's F0 arm: threads 6, ubatch 512, ctx 8192) with
`--dump-logprobs` (the full log-softmax row per position, from which any temperature is exact):
- `val`: the validation segments of shards 470-471 that `c50_m_eval.py` fits every arm on (`TeacherL(...).batches(S4.
  TOKENS)`, each segment one document; the scored positions are the same: 1..n-1 of each segment). F0's NLL of the
  corpus targets at every temperature of the grid; `tau_nll` is its argmin (Guo et al. 2017's rule, as for the arms).
- `docs`: A4-bis's 72 documents (`a4bis_post/inputs/docs.tsv`): `docs.pos.bin` (the scorer's records), and from the
  dump `docs.temp.f32` (positions x grid, the target's log-probability at every temperature) and `docs.rank.i32`
  (each target's rank, 0 = top-1), in the format `c50_m_eval.py` writes for the arms.
The dumps (about 30 and 95 GB) are deleted once read; `f0.json` records their sizes and the checks.

    python -I benchmarks/chatbot50/c50_m_f0.py val|docs [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c50_a4_checks as A4  # noqa: E402
import c50_s4_train as S4  # noqa: E402
from c50_m_eval import I1, TAUS  # noqa: E402
from c50_t1_train import TeacherL, pack  # noqa: E402

C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
MODEL = Path("C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf")
PIN = "5348c73fe6497c5152e16fd792d6c7b5aeefa15de8b15b8c8e18bef691704b8e"
EXE = HERE / "bin" / "c50_llama_score_avx2.exe"
DOCS = C / "a4bis_post/inputs/docs.tsv"
NV, ROWS = 157184, 256


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def score(ids: Path, prefix: Path) -> dict:
    cmd = [str(EXE), "--model", str(MODEL), "--ids", str(ids), "--out", str(prefix), "--threads", "6", "--ubatch", "512",
           "--ctx", "8192", "--dump-logprobs"]
    print("CMD", json.dumps(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    config = [ln for ln in p.stdout.splitlines() if ln.startswith("CONFIG ")]
    c = config[0] if len(config) == 1 else ""
    ok = (p.returncode == 0 and "AVX2 = 1" in c and "threads=6 " in c and "ubatch=512 " in c and "ctx=8192 " in c
          and "flash_attn=disabled" in c and "dump=1 " in c)
    rec = {"cmd": cmd, "returncode": p.returncode, "config": c, "wall_s": round(time.time() - t0, 1),
           "stderr_tail": p.stderr[-400:]}
    if not ok:
        raise SystemExit("VOID: the scorer run failed its CONFIG check: " + json.dumps(rec))
    return rec


@torch.no_grad()
def read_dump(prefix: Path, want_temp: bool):
    """From the records and the dump: NLL per temperature, and (want_temp) the target's log-probability per temperature
    and its rank, per position; the check that the dump's target column equals the record."""
    r = np.fromfile(prefix.with_suffix(".pos.bin"), dtype=A4.REC)
    dump = prefix.with_suffix(".logp.f32")
    assert dump.stat().st_size == len(r) * NV * 4, (dump.stat().st_size, len(r))
    lp = np.memmap(dump, dtype=np.float32, mode="r", shape=(len(r), NV))
    taus = torch.tensor(TAUS, dtype=torch.float32)
    nll = np.zeros(len(TAUS))
    temp = np.zeros((len(r), len(TAUS)), dtype=np.float32) if want_temp else None
    rank = np.zeros(len(r), dtype=np.int32)
    col1, top_bad = 0.0, 0
    for c in range(0, len(r), ROWS):
        x = torch.from_numpy(np.array(lp[c:c + ROWS]))
        t = torch.from_numpy(r["target"][c:c + ROWS].astype(np.int64))
        xt = x.gather(1, t[:, None])[:, 0]
        col1 = max(col1, float((xt - torch.from_numpy(r["logp_target"][c:c + ROWS])).abs().max()))
        top_bad += int((x.argmax(1).numpy() != r["top1"][c:c + ROWS]).sum())
        rank[c:c + ROWS] = (x > xt[:, None]).sum(1).numpy()
        tt = torch.stack([xt / tau - torch.logsumexp(x / tau, 1) for tau in taus.tolist()], 1)
        nll += -tt.double().sum(0).numpy()
        if want_temp:
            temp[c:c + ROWS] = tt.numpy()
        if c % (ROWS * 200) == 0:
            print("READ", c, len(r), flush=True)
    return r, nll / len(r), temp, rank, {"dump_vs_record_logp_max": col1, "dump_argmax_vs_record_top1": top_bad,
                                         "dump_bytes": dump.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["val", "docs"])
    ap.add_argument("--out", type=Path, default=C / "m2/f0")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rp = a.out / "f0.json"
    rec = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    if "model_sha256" not in rec:
        print("hashing", MODEL, flush=True)
        rec = {"cell": "C50 cell M part 2, F0", "model": str(MODEL), "model_sha256": sha256(MODEL), "exe": str(EXE),
               "exe_sha256": sha256(EXE), "taus": TAUS.tolist()}
        if rec["model_sha256"] != PIN:
            raise SystemExit("F0's sha256 differs from its pin")
    if a.what == "val":
        V = list(TeacherL(C / "s3/teacher_v1", C / "s3/corpus_v1/teacher.tsv", list(S4.VAL_SHARDS)).batches(S4.TOKENS))
        lines, k = [], 0
        for batch in V:
            ids, segs, *_ = pack(batch)
            for s0, s1 in segs:
                lines.append(f"{k}\t" + ",".join(str(int(x)) for x in ids[s0:s1].tolist()))
                k += 1
        ids_path = a.out / "val_ids.tsv"
        ids_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run = score(ids_path, a.out / "val")
        r, nll, _, _, chk = read_dump(a.out / "val", False)
        g = int(np.argmin(nll))
        rec["val"] = {"batches": len(V), "segments": k, "positions": int(len(r)), "ids_sha256": sha256(ids_path),
                      "nll": [round(float(x), 6) for x in nll], "tau_nll": float(TAUS[g]), "nll_at_1": float(nll[I1]),
                      "nll_at_tau": float(nll[g]), "tau_at_edge": g in (0, len(TAUS) - 1), "run": run, **chk}
        (a.out / "val.logp.f32").unlink()
    else:
        run = score(DOCS, a.out / "docs")
        r, nll, temp, rank, chk = read_dump(a.out / "docs", True)
        temp.tofile(a.out / "docs.temp.f32")
        rank.tofile(a.out / "docs.rank.i32")
        rec["docs"] = {"file": str(DOCS), "file_sha256": sha256(DOCS), "docs": int(len(np.unique(r["doc"]))),
                       "positions": int(len(r)), "col_1_vs_logp_target_max": float(np.abs(temp[:, I1] - r["logp_target"]).max()),
                       "rank0_vs_top1_mismatch": int(((rank == 0) != (r["top1"] == r["target"])).sum()), "run": run, **chk}
        (a.out / "docs.logp.f32").unlink()
    rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DONE", a.what, json.dumps({k: v for k, v in rec[a.what].items() if k not in ("nll", "run")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
