"""C50-S0 driver: v0 of T0 in the engine at contexts 512, 2048 and 4096, the organ breakdown and its planted control.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_S0_V0_PROTOCOL_20261017.md.
Runs after A1b in the same empty-machine session.  The memory mode and priority are the A1b adopted arm's; if A1b
failed, the complete arm with the smallest max/min (labelled UNSTABLE).  Order:
    pre-flight (45 s, mean foreign occupancy < 4.39%)
    planted control: 3 pairs at context 512, (--organs) then (plant_head --organs), fresh processes
    10 rounds; round r visits the contexts rotated by r, and each visit is one --organs-off process, followed in
    rounds 0-4 (2048) or 0-2 (512, 4096) by one --organs process at the same context
Each process is split into own / foreign occupancy (E44); a process at or above the bar is VOID and replaced (at most
5 replacements per arm; an arm is the binary, the context and the organs flag).  Numbers only: c50_s0_report.py.

Usage:
    python -I benchmarks/chatbot50/c50_s0_run.py --a1b-verdict benchmarks/chatbot50/results/c50_a1b_verdict_1.json \
        --out benchmarks/chatbot50/results/c50_s0_run_1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "donor_adaptation" / "engine"))
import e44_interval as E44  # noqa: E402

EXE = {"s0": HERE / "bin" / "c50_engine_s0.exe", "plant": HERE / "bin" / "c50_engine_s0_plant_head.exe"}
EXE_SHA256_PIN = {"s0": "786e1c6dfb58de5ef3f92a28dfce24abedcc1f4be686fc8ceaa4ae52f953e758", "plant": "f2cd3401a9daf698d8f09e619431e0b8465bc5607aa898f6452dc7dd2a23331b"}
# Addendum 1 (17 October, before any S0 process): S1 extends c50_engine.c, so the S0 source is checked as the git blob
# of commit 78356877 (LF blob sha256 1ec8a12e...; the CRLF working copy the binaries were built from was 47736ad3...).
SOURCE_COMMIT = "78356877e5a3a4ee06cdbec4ccfe6f8826a12042"
SOURCE_PATH = "benchmarks/chatbot50/c50_engine.c"
SOURCE_SHA256_PIN = "1ec8a12ebaab71b0ac51613f78adc121532196be9751481bec22ae8afb458aaf"
MODEL = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/ling-mini-2.0-a810f64-q4_k_m.gguf")
MODEL_SHA256 = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
IDS = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs/docs.tsv")
A1B_ARMS = {"B0": ("map", "normal"), "B1": ("private", "high"), "B2": ("large", "high")}
CTXS, STEPS, ROUNDS, PLANT_PAIRS, PLANT_CTX = (512, 2048, 4096), 128, 10, 3, 512
ORGAN_ROUNDS = {512: 3, 2048: 5, 4096: 3}
OCC_BAR, MAX_REPLACEMENTS, PREFLIGHT_SECONDS = 4.39, 5, 45


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def source_blob_sha256() -> str:
    blob = subprocess.run(["git", "-C", str(HERE), "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}"], capture_output=True, check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def adopted(verdict_path: Path) -> dict:
    v = json.loads(verdict_path.read_text(encoding="utf-8"))
    if v.get("verdict") == "PASS":
        return {"arm": v["adopted_arm"], "label": "STABLE", "a1b_verdict": "PASS"}
    if v.get("verdict") == "FAIL":
        done = {a: r for a, r in v["arms"].items() if r["processes"] == 10}
        if done:
            arm = min(done, key=lambda a: (done[a]["max_over_min"], list(A1B_ARMS).index(a)))
            return {"arm": arm, "label": "UNSTABLE", "a1b_verdict": "FAIL"}
    return {"arm": None, "label": "VOID", "a1b_verdict": v.get("verdict")}


def one_process(kind: str, ctx: int, organs: bool, mem: str, prio: str) -> dict:
    split = E44.Split()
    t0 = time.time()
    cmd = [str(EXE[kind]), "--model", str(MODEL), "--bench", str(IDS), "--ctx", str(ctx), "--steps", str(STEPS),
           "--mem", mem, "--prio", prio] + (["--organs"] if organs else [])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
    text = out.decode(errors="replace")
    config = [ln for ln in text.splitlines() if ln.startswith("CONFIG ")]
    rows = [json.loads(ln) for ln in text.splitlines() if ln.startswith("{")]
    rec = {"kind": kind, "ctx": ctx, "organs": organs, "cmd": cmd, "returncode": proc.returncode, "config": config,
           "stderr_tail": err.decode(errors="replace")[-400:], "system_pct": system, "foreign_pct": foreign,
           "wall_s": time.time() - t0, "timing": split.record()}
    want = [f"ctx={ctx} steps={STEPS} threads=6 ", f"mem={mem} prio={prio} organs={int(organs)} ",
            f"plant={'head_twice' if kind == 'plant' else 'none'}", f"model={MODEL.as_posix()} "]
    if proc.returncode != 0 or len(config) != 1 or len(rows) != 1 or \
            not all(w in config[0].replace("\\", "/") + " " for w in want):
        rec["status"] = "FAILED"
        return rec
    rec.update(rows[0])
    rec["status"] = "VOID" if not (foreign == foreign and foreign < OCC_BAR) else "OK"
    return rec


def schedule() -> list[tuple[str, str, int, bool, int]]:
    """(phase, binary, context, organs, round) in run order."""
    out = []
    for i in range(PLANT_PAIRS):
        out += [("plant", "s0", PLANT_CTX, True, i), ("plant", "plant", PLANT_CTX, True, i)]
    for r in range(ROUNDS):
        for k in range(len(CTXS)):
            ctx = CTXS[(k + r) % len(CTXS)]
            out.append(("rate", "s0", ctx, False, r))
            if r < ORGAN_ROUNDS[ctx]:
                out.append(("organs", "s0", ctx, True, r))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1b-verdict", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    report = {"cell": "C50-S0", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "occ_bar_pct": OCC_BAR,
              "source": f"{SOURCE_PATH} @ {SOURCE_COMMIT}", "source_sha256": source_blob_sha256(), "ids_sha256": sha256(IDS),
              "exe_sha256": {k: sha256(p) for k, p in EXE.items()}, "model": MODEL.as_posix()}

    def save():
        out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    if report["exe_sha256"] != EXE_SHA256_PIN or report["source_sha256"] != SOURCE_SHA256_PIN:
        report["verdict_input"] = "VOID: binaries or source differ from the pinned S0 build"
        save()
        print(report["verdict_input"])
        return 1
    arm = adopted(Path(args.a1b_verdict))
    report["a1b"] = dict(arm, verdict_file=args.a1b_verdict, verdict_sha256=sha256(Path(args.a1b_verdict)))
    if arm["arm"] is None:
        report["verdict_input"] = "VOID: no complete A1b arm to adopt"
        save()
        print(report["verdict_input"])
        return 1
    mem, prio = A1B_ARMS[arm["arm"]]
    report["mem"], report["prio"] = mem, prio
    print(f"A1b {arm['a1b_verdict']}: arm {arm['arm']} (mem={mem} prio={prio}) label {arm['label']}", flush=True)
    report["model_sha256"] = sha256(MODEL)
    if report["model_sha256"] != MODEL_SHA256:
        report["verdict_input"] = "VOID: the model is not T0"
        save()
        return 1
    pre = E44.watch(PREFLIGHT_SECONDS)
    mean = sum(pre) / len(pre) if pre else float("nan")
    report["preflight"] = {"seconds": PREFLIGHT_SECONDS, "samples": pre, "rule": "mean < bar (A1 addendum 2)",
                           "mean": mean, "max": max(pre) if pre else None, "admissible": bool(pre) and mean < OCC_BAR}
    print(f"preflight mean={mean:.2f}% admissible={report['preflight']['admissible']}", flush=True)
    report["processes"] = []
    if not report["preflight"]["admissible"]:
        report["verdict_input"] = "VOID: pre-flight occupancy at or above the bar"
        save()
        return 1
    replacements: dict[str, int] = {}
    for phase, kind, ctx, organs, rnd in schedule():
        key = f"{kind}/{ctx}/organs={int(organs)}"
        while True:
            rec = one_process(kind, ctx, organs, mem, prio)
            rec["phase"], rec["round"] = phase, rnd
            report["processes"].append(rec)
            save()
            print(f"{phase} r{rnd} {key} {rec['status']} foreign={rec['foreign_pct']:.2f}% "
                  f"median_tok_s={rec.get('median_tok_s', float('nan')):.3f} wall={rec['wall_s']:.0f}s", flush=True)
            if rec["status"] == "FAILED":
                report["verdict_input"] = f"VOID: process failed ({key})"
                save()
                return 1
            if rec["status"] == "OK":
                break
            replacements[key] = replacements.get(key, 0) + 1
            if replacements[key] > MAX_REPLACEMENTS:
                report["verdict_input"] = f"VOID: more than {MAX_REPLACEMENTS} replacements ({key})"
                save()
                return 1
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["verdict_input"] = "COMPLETE"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
