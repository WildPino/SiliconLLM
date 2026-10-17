"""C50-A6 runner: build the precision-ladder GGUFs and score each arm on the A4 document panel (numbers only).

Arms (C50_A6_PRECISION_LADDER_PROTOCOL_20261017.md); one fixed runtime for all of them: the A6 scorer
(c50_llama_score_a6_avx2), flash attention on, 6 threads, ubatch 512, ctx 8192:
    R     P0 Q4_K_M (A2 pin)                                   KV f16    reference
    K8    P0 Q4_K_M                                            KV q8_0   P1
    E3    Q4_K_M, routed experts (gate/up/down) IQ3_S          KV q8_0   P3
    E4    Q4_K_M, routed experts IQ4_XS                        KV q8_0   P3 fallback (risk register)
    E3B   E3 + router (ffn_gate_inp) BF16                      KV q8_0   P4
    E4B   E4 + router BF16                                     KV q8_0   P4 on the fallback
Every built GGUF comes from the A2 BF16 GGUF with the FIT imatrix (fit24), base type Q4_K_M.
Resumable: an arm with a completed record and its outputs is skipped.  The verdict is c50_a6_verdict.py.

Usage:
    python -I benchmarks/chatbot50/c50_a6_run.py --work E:/.../a6 --out benchmarks/chatbot50/results/c50_a6_run.json \
        [--arms R,K8,...] [--budget-h 6] [--build-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

CAMP = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
GG = CAMP / "gguf"
BF16 = GG / "ling-mini-2.0-a810f64-bf16.gguf"
P0 = GG / "ling-mini-2.0-a810f64-q4_k_m.gguf"
P0_SHA = "44de7315"
IMATRIX = GG / "ling-mini-2.0-a810f64-imatrix-fit24.gguf"
QUANT = CAMP / "tools/llama.cpp-5b335f4/build-avx2/bin/llama-quantize.exe"
HERE = Path(__file__).resolve().parent
SCORER = HERE / "bin" / "c50_llama_score_a6_avx2.exe"
DOCS = CAMP / "a4" / "inputs" / "docs.tsv"
TOKENIZER = CAMP / "sources/Ling-mini-2.0_a810f64/tokenizer.json"
EXPS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
BUILDS = {
    "exps_iq3s": [f"{t}=iq3_s" for t in EXPS],
    "exps_iq4xs": [f"{t}=iq4_xs" for t in EXPS],
    "exps_iq3s_rbf16": [f"{t}=iq3_s" for t in EXPS] + ["ffn_gate_inp=bf16"],
    "exps_iq4xs_rbf16": [f"{t}=iq4_xs" for t in EXPS] + ["ffn_gate_inp=bf16"],
}
ARMS = {"R": (None, "f16"), "K8": (None, "q8_0"), "E3": ("exps_iq3s", "q8_0"), "E4": ("exps_iq4xs", "q8_0"),
        "E3B": ("exps_iq3s_rbf16", "q8_0"), "E4B": ("exps_iq4xs_rbf16", "q8_0")}
MAX_PROC_S = 3 * 3600


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def gguf_path(build: str | None) -> Path:
    return P0 if build is None else GG / f"ling-mini-2.0-a810f64-q4_k_m-{build.replace('_', '-')}.gguf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--budget-h", type=float, default=6.0)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    rep = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {
        "cell": "C50-A6 FIT", "docs_tsv_sha256": sha256(DOCS), "scorer_sha256": sha256(SCORER), "builds": {}, "arms": {}}
    if rep["docs_tsv_sha256"] != sha256(DOCS) or rep["scorer_sha256"] != sha256(SCORER):
        raise SystemExit("inputs or scorer changed since the first run of this report")
    arms = args.arms.split(",")
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}")

    def save():
        out.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    for build in sorted({ARMS[a][0] for a in arms} - {None}):
        dst = gguf_path(build)
        if build in rep["builds"] and dst.exists():
            continue
        cmd = [str(QUANT), "--imatrix", str(IMATRIX)]
        for tt in BUILDS[build]:
            cmd += ["--tensor-type", tt]
        cmd += [str(BF16), str(dst), "Q4_K_M", "6"]
        t0 = time.time()
        log = work / f"quantize_{build}.log"
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise SystemExit(f"quantize {build} failed, see {log}")
        chk = work / f"gguf_check_{build}.json"
        c = subprocess.run([sys.executable, "-I", str(HERE / "c50_a2_gguf_check.py"), "--tokenizer", str(TOKENIZER),
                            "--gguf", str(dst), "--out", str(chk)], capture_output=True, text=True)
        if c.returncode != 0:
            raise SystemExit(f"gguf check {build} failed: {c.stdout} {c.stderr}")
        art = json.loads(chk.read_text(encoding="utf-8"))["artifacts"][0]
        rep["builds"][build] = {"cmd": cmd, "wall_s": time.time() - t0, "sha256": art["sha256"],
                                "per_token_weight_bytes": art["per_token_weight_bytes"],
                                "bytes_by_type": art["bytes_by_type"], "log": str(log)}
        save()
        print("built", build, art["sha256"][:16], round(art["per_token_weight_bytes"] / 1e6, 1), "MB/token", flush=True)
    if args.build_only:
        return 0

    for a in arms:
        build, kv = ARMS[a]
        d = work / a
        if a in rep["arms"] and rep["arms"][a].get("ok") and (d / "docs.pos.bin").exists():
            continue
        left = args.budget_h * 3600 - (time.time() - t_start)
        if left < 600:
            print("budget exhausted before arm", a, flush=True)
            break
        model = gguf_path(build)
        msha = sha256(model)
        if build is None and not msha.startswith(P0_SHA):
            raise SystemExit("P0 pin mismatch")
        if build is not None and msha != rep["builds"][build]["sha256"]:
            raise SystemExit(f"{a}: GGUF changed since it was built")
        d.mkdir(exist_ok=True)
        cmd = [str(SCORER), "--model", str(model), "--ids", str(DOCS), "--out", str(d / "docs"), "--threads", "6",
               "--ubatch", "512", "--ctx", "8192", "--fa", "1", "--kv", kv,
               "--max-projected-s", str(int(min(MAX_PROC_S, left)))]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        cfg = next((x for x in proc.stdout.splitlines() if x.startswith("CONFIG")), "")
        ok = (proc.returncode == 0 and "AVX2 = 1" in cfg and "threads=6 " in cfg and "ubatch=512 " in cfg
              and "flash_attn=enabled " in cfg and f"kv={kv} " in cfg and f"model={model.as_posix()}" in cfg.replace("\\", "/"))
        rep["arms"][a] = {"cmd": cmd, "model_sha256": msha, "kv": kv, "build": build, "returncode": proc.returncode,
                          "wall_s": time.time() - t0, "config": cfg, "ok": ok, "stderr_tail": proc.stderr[-2000:]}
        save()
        print(a, "ok" if ok else "NOT OK", f"{time.time() - t0:.0f}s", flush=True)
        if not ok:
            return 1
    rep["wall_s_last_invocation"] = time.time() - t_start
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
