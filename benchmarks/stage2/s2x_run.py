"""Stage 2 test 3b runner: stage 1 (b)'s ternary experts with the chosen slots read from the original Q4_K_M (s2x_mix_hook.h),
on stage 1's panels, into its work directory.  It prints numbers, never a verdict: s2x_verdict.py decides.

    arm --arm NAME     the seven panels of one registered arm.  A panel already COMPLETE is skipped.
    controls           C1-C3 on the panel's first 2 documents.
Each process's CONFIG and MIX lines must confirm the cell, or the run stops.

Usage: python -I benchmarks/stage2/s2x_run.py controls --out benchmarks/stage2/results/s2x_run.json
       python -I benchmarks/stage2/s2x_run.py arm --arm x15 --out benchmarks/stage2/results/s2x_run.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
from s1_run import INPUTS, MAX_PROJECTED_S, PANELS, STOP, check_inputs, sha256  # noqa: E402

P0 = Path("C:/SiliconLLM_fast/s2t/m/p0.gguf")
P0_SHA = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
TERN = Path("C:/SiliconLLM_fast/export/tern-q4.gguf")
TERN_SHA = "f45014038c21f6210b227a57e3f7a919a08c026feed0ee8000b35cf6963fa699"
WORK = Path("E:/SiliconLLM_Artifacts/stage1/s1a")
SCORE = HERE / "bin" / "s2x_llama_score_avx2.exe"
GEN = HERE / "bin" / "s2x_llama_generate_avx2.exe"
SUB = Path("C:/SiliconLLM_fast/s2t/sets")
PREFIX = 2
POS, TOP = 32, 32 + 32 * 8
# arm -> (mode, layers, k = 4 on generated tokens)
ARMS = {"x15": ("top", "15-19", False), "r15": ("random", "15-19", False), "x19": ("top", "1-19", False),
        "x15k4": ("top", "15-19", True), "x9": ("top", "9-19", False), "r9": ("random", "9-19", False),
        # amendment 2, reported; "off" = no exact slot.  The fourth code's arms read --model.
        "x19k4": ("top", "1-19", True), "r19": ("random", "1-19", False), "t4": ("off", "-", True),
        "bal": ("off", "-", False), "bal4": ("off", "-", True)}
# k = 4 acts on generated tokens only, so an arm's score panels are its k = 8 arm's, byte for byte (x15k4 against x15: arc, mmlu, docs).  Copied.
BASE = {"x19k4": "x19", "t4": "tern", "bal4": "bal"}


def launch(cmd: list[str], need: list[str]) -> dict:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    config = [x for x in lines if x.startswith("CONFIG ")]
    mix = [x for x in lines if x.startswith("MIX ")]
    moe = [x for x in lines if x.startswith("MOE ")]
    ok = proc.returncode == 0 and len(config) == 1 and all(x in config[0] for x in need)
    return {"cmd": cmd, "exe_sha256": sha256(Path(cmd[0])), "returncode": proc.returncode, "wall_s": round(time.time() - t0, 1),
            "config": config, "moe": moe, "mix": mix, "stdout_tail": lines[-3:], "stderr_tail": proc.stderr[-600:],
            "status": "COMPLETE" if ok else "STOPPED"}


def load(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"cell": "S2X-exact-where-it-weighs", "arms": {}}


def save(out: Path, report: dict) -> None:
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def pins() -> None:
    if sha256(P0) != P0_SHA or sha256(TERN) != TERN_SHA:
        raise SystemExit("a model differs from its pin")


def exact(mode: str, layers: str) -> list[str]:
    return [] if mode == "off" else ["--exact-from", str(P0), "--exact-mode", mode, "--exact-layers", layers]


def arm(a: argparse.Namespace) -> int:
    report = load(a.out)
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    report.setdefault("inputs_manifest_sha256", man)
    if report["inputs_manifest_sha256"] != man:
        raise SystemExit("inputs changed since the run began")
    pins()
    model = TERN
    if a.arm in ("bal", "bal4"):
        if not a.model or sha256(a.model) != a.model_sha:
            raise SystemExit("the fourth code's arms need --model and its --model-sha")
        model = a.model
    mode, layers, k4 = ARMS[a.arm]
    rec = report["arms"].setdefault(a.arm, {"panels": {}, "mode": mode, "layers": layers, "k4": k4, "model": str(model)})
    (WORK / a.arm).mkdir(parents=True, exist_ok=True)
    for panel, src, tool, extra in PANELS:
        if rec["panels"].get(panel, {}).get("status") == "COMPLETE":
            continue
        if tool == "score" and a.arm in BASE:
            got = list((WORK / BASE[a.arm]).glob(f"{panel}.*"))
            if not got:
                raise SystemExit(f"{BASE[a.arm]} has no {panel}")
            for f in got:
                shutil.copyfile(f, WORK / a.arm / f.name)
            rec["panels"][panel] = {"status": "COMPLETE", "copied_from": BASE[a.arm], "files": sorted(f.name for f in got)}
            save(a.out, report)
            continue
        exe, flag = (SCORE, "--ids") if tool == "score" else (GEN, "--prompts")
        if a.arm in ("bal", "bal4"):    # S2K, amendment 3: Q4_0's repacked path needs the build with unaligned moves
            exe = exe.with_name(exe.name.replace("_avx2", "_ua_avx2"))
        cmd = [str(exe), "--model", str(model), flag, str(INPUTS[src] / f"{panel}.tsv"), "--out", str(WORK / a.arm / panel),
               "--threads", "6", "--ubatch", "512", "--max-projected-s", str(3 * MAX_PROJECTED_S)] + extra + exact(mode, layers)
        if k4 and tool == "generate":
            cmd += ["--moe-mode", "top", "--moe-k", "4", "--moe-mass", "16", "--moe-decode-only"]
        else:
            cmd += ["--moe-mode", "none"]
        if panel == "docs":
            cmd += ["--ref-top", str(WORK / "ref" / "docs.top.bin")]
        if tool == "generate":
            cmd += ["--stop", STOP]
        need = [f"model={model} ", "flash_attn=disabled"] + ([] if mode == "off" else [f"exact_mode={mode} ", f"exact_layers={layers} ", f"exact_from={P0} ", "hook=1 "])
        r = launch(cmd, need)
        if mode != "off":
            if r["status"] == "COMPLETE" and len(r["mix"]) != 1:
                r["status"] = "STOPPED: no MIX line"
            if r["status"] == "COMPLETE" and f"mode={mode} " not in r["mix"][0]:
                r["status"] = "STOPPED: MIX does not confirm the mode"
        rec["panels"][panel] = r
        save(a.out, report)
        print(a.arm, panel, r["status"], r["wall_s"], r["mix"], flush=True)
        if r["status"] != "COMPLETE":
            return 1
    return 0


def controls(a: argparse.Namespace) -> int:
    report = load(a.out)
    pins()
    ids, top = SUB / f"docs{PREFIX}.tsv", SUB / f"docs{PREFIX}.top.bin"
    lines = ids.read_text(encoding="ascii").splitlines()
    n = sum(len(x.split("\t")[1].split(",")) - 1 for x in lines)
    d = Path("C:/SiliconLLM_fast/s2t/x/controls")
    d.mkdir(parents=True, exist_ok=True)
    runs = {"off": (TERN, []), "p0_all": (P0, exact("all", "1-19")), "tern_all": (TERN, exact("all", "1-19")),
            "wrong": (TERN, exact("wrong", "1-19"))}
    out = {"prefix_documents": PREFIX, "positions": n, "runs": {}}
    for name, (model, args) in runs.items():
        cmd = [str(SCORE), "--model", str(model), "--ids", str(ids), "--out", str(d / name), "--threads", "6", "--ubatch", "512",
               "--ctx", "8192", "--top", "32", "--moe-mode", "none", "--ref-top", str(top)] + args
        out["runs"][name] = launch(cmd, [f"model={model} "])
        if out["runs"][name]["status"] != "COMPLETE":
            raise SystemExit(f"{name}: stopped")

    def read(p: Path, size: int) -> bytes:
        b = p.read_bytes()
        return b[:n * size]

    def top1(b: bytes) -> list[bytes]:
        return [b[i * POS + 12:i * POS + 16] for i in range(n)]

    ref, tern = read(WORK / "ref" / "docs.pos.bin", POS), read(WORK / "tern" / "docs.pos.bin", POS)
    off = (d / "off.pos.bin").read_bytes()
    out["off_equals_tern"] = off == tern and (d / "off.top.bin").read_bytes() == read(WORK / "tern" / "docs.top.bin", TOP)
    out["tern_all_equals_p0_all"] = (d / "tern_all.pos.bin").read_bytes() == (d / "p0_all.pos.bin").read_bytes()
    mix = out["runs"]["p0_all"]["mix"][0]
    out["p0_all_slot_rel_err"] = float(re.search(r" slot_rel_err=([\d.]+)", mix).group(1))
    out["p0_all_mean_slot_rel_err"] = float(re.search(r"mean_slot_rel_err=([\d.]+)", mix).group(1))
    r1 = top1(ref)

    def agree(b: bytes) -> float:
        return sum(x == y for x, y in zip(top1(b), r1)) / n
    out["top1"] = {"tern": agree(tern), "p0_all": agree((d / "p0_all.pos.bin").read_bytes()),
                   "wrong": agree((d / "wrong.pos.bin").read_bytes())}
    report["controls"] = out
    save(a.out, report)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("arm")
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, help="the fourth code's arms only")
    p.add_argument("--model-sha", default="")
    p.set_defaults(fn=arm)
    c = sp.add_parser("controls")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(fn=controls)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
