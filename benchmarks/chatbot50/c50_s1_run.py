"""C50-S1 driver: the engine's rate on T0 and on its byte-copy expansions, in both router formats (arm E) and with
hashed copies (arm H), plus the arm-M memory-bank bench.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md.
Every process runs with the A1b adopted arm B0 (--mem map --prio normal).  Order:
    model hashes; pre-flight (45 s, mean foreign occupancy < 4.39%)
    K1 / K2: --score --dump-logprobs on par1.tsv, cmp against the A3 binary on T0
    20 rounds; round r visits the artifacts rotated by r, and at each the round's arms rotated by r
    arm M: the c50_s1_bank lookup bench (committed 7a89b83), 3 fresh processes per (table, k)
Each rate process is split into own / foreign occupancy (E44); a process at or above the bar is VOID and replaced (at
most 5 replacements per arm).  Numbers only: c50_s1_report.py.

Usage:
    python -I benchmarks/chatbot50/c50_s1_run.py --out benchmarks/chatbot50/results/c50_s1_run_1.json
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

EXE = {"s1": HERE / "bin" / "c50_engine_s1.exe", "a3": HERE / "bin" / "c50_engine.exe",
       "bank": HERE / "bin" / "c50_s1_bank.exe"}
EXE_SHA256_PIN = {"s1": "79e2e4f4dd6bf9854a257d8cf1cfec46e2279b4486b2b8d28722d48ddb1a3de1",
                  "a3": "d14ea9a5903557342763c5ebb7eb18897333c86a4484c24566b42c8627c9035d",
                  "bank": "b076ddae7576d98ca760a37759f7bcb93c4571c598d4b06c22d70aac07248eeb"}
# the engine source is the git blob of the S1 tools commit (its binary was rebuilt from it on 18 October)
SOURCE_COMMIT, SOURCE_PATH = "1eb08f9", "benchmarks/chatbot50/c50_engine.c"
SOURCE_SHA256_PIN = "68c947d9c7da6fa5db123570c37605cfffdb5b912853820e0610e2f5f1b6c627"
STAGE = Path("C:/SiliconLLM_stage/s1")
# name: (path, experts, sha256 of the GGUF; the expansions are pinned from their expander records at build)
ARTIFACTS = {"m1": (STAGE / "ling-mini-2.0-a810f64-q4_k_m.gguf", 256,
                    "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"),
             "m2": (STAGE / "t0-e536.gguf", 536, "PIN_M2"),
             "m4": (STAGE / "t0-e1088.gguf", 1088, "PIN_M4"),
             "m6": (STAGE / "t0-e1600.gguf", 1600, "PIN_M6")}
IDS = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs/docs.tsv")
SEL_IDS = HERE / "corpus" / "c50_holdout_ext_v1" / "sel_ids.tsv"
PAR1_SHA256 = "373b4886c82eb9a98e203d2e2dd52ab4d8860c64f205057e9a4e244a183cb08a"
MEM, PRIO, CTX, STEPS, ROUNDS = "map", "normal", 2048, 128, 20
OCC_BAR, MAX_REPLACEMENTS, PREFLIGHT_SECONDS = 4.39, 5, 45
# arm M (addendum 1): k rows per layer over 19 layers, rows of 4,096 bytes, pageable memory
BANK = [(gb, k) for gb in (1, 4, 16, 48) for k in (2, 8, 32)]
BANK_REPS, BANK_TOKENS, BANK_WARM, BANK_LAYERS = 3, 256, 16, 19


def arms() -> list[dict]:
    """Every rate arm: artifact, router format, hash flag, organs flag, and the rounds it runs in."""
    out = []
    for m in ("m1", "m2", "m4"):
        for r in ("f32", "bf16"):
            out.append({"m": m, "router": r, "hash": False, "organs": False, "rounds": 20})
            out.append({"m": m, "router": r, "hash": False, "organs": True, "rounds": 6 if m == "m1" else 3})
            if m != "m1":
                out.append({"m": m, "router": r, "hash": True, "organs": False, "rounds": 10})
    for r in ("f32", "bf16"):
        out += [{"m": "m6", "router": r, "hash": False, "organs": False, "rounds": 10},
                {"m": "m6", "router": r, "hash": True, "organs": False, "rounds": 10},
                {"m": "m6", "router": r, "hash": False, "organs": True, "rounds": 3}]
    return out


def arm_key(a: dict) -> str:
    return f"{a['m']}/{'H' if a['hash'] else 'E'}-{a['router']}/organs={int(a['organs'])}"


def schedule() -> list[tuple[int, dict]]:
    order = ["m1", "m2", "m4", "m6"]
    all_arms = arms()
    out = []
    for r in range(ROUNDS):
        for k in range(len(order)):
            m = order[(k + r) % len(order)]
            here = [a for a in all_arms if a["m"] == m and r < a["rounds"]]
            here = here[r % len(here):] + here[:r % len(here)] if here else []
            out += [(r, a) for a in here]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def source_blob_sha256() -> str:
    blob = subprocess.run(["git", "-C", str(HERE), "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}"], capture_output=True, check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def engine_cmd(m: str, extra: list[str], router: str, hashed: bool) -> list[str]:
    cmd = [str(EXE["s1"]), "--model", str(ARTIFACTS[m][0])] + extra + ["--mem", MEM, "--prio", PRIO]
    if router == "bf16":
        cmd += ["--router", "bf16"]
    if hashed:
        cmd += ["--hash-copies"]
    return cmd


def rate_process(a: dict) -> dict:
    split = E44.Split()
    t0 = time.time()
    cmd = engine_cmd(a["m"], ["--bench", str(IDS), "--ctx", str(CTX), "--steps", str(STEPS)], a["router"], a["hash"])
    if a["organs"]:
        cmd.append("--organs")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
    text = out.decode(errors="replace")
    config = [ln for ln in text.splitlines() if ln.startswith("CONFIG ")]
    rows = [json.loads(ln) for ln in text.splitlines() if ln.startswith("{")]
    rec = {"arm": arm_key(a), "cmd": cmd, "returncode": proc.returncode, "config": config,
           "stderr_tail": err.decode(errors="replace")[-400:], "system_pct": system, "foreign_pct": foreign,
           "wall_s": time.time() - t0, "timing": split.record()}
    want = [f"ctx={CTX} steps={STEPS} threads=6 ", f"mem={MEM} prio={PRIO} organs={int(a['organs'])} ",
            f"experts={ARTIFACTS[a['m']][1]} router={a['router']} hash={int(a['hash'])} plant=none",
            f"model={ARTIFACTS[a['m']][0].as_posix()} "]
    if proc.returncode != 0 or len(config) != 1 or len(rows) != 1 or \
            not all(w in config[0].replace("\\", "/") + " " for w in want):
        rec["status"] = "FAILED"
        return rec
    rec.update(rows[0])
    rec["status"] = "VOID" if not (foreign == foreign and foreign < OCC_BAR) else "OK"
    return rec


def known_positives(work: Path) -> dict:
    """K1 (BF16 router at m1) and K2 (arm H at m2, m4, m6, both formats): byte-identical to the A3 binary on T0."""
    work.mkdir(parents=True, exist_ok=True)
    line = SEL_IDS.read_text(encoding="utf-8").splitlines()[0].split("\t")
    par1 = work / "par1.tsv"
    par1.write_text("0\t" + ",".join(line[1].strip().split(",")[:200]) + "\n", encoding="utf-8")
    out = {"par1_sha256": sha256(par1)}
    if out["par1_sha256"] != PAR1_SHA256:
        out["status"] = "VOID: par1.tsv differs from its pin"
        return out
    ref = work / "a3"
    subprocess.run([str(EXE["a3"]), "--model", str(ARTIFACTS["m1"][0]), "--score", str(par1), "--out", str(ref),
                    "--dump-logprobs"], capture_output=True, check=True)
    checks = [("K1", "m1", "bf16", False)] + [("K2", m, r, True) for m in ("m2", "m4", "m6") for r in ("f32", "bf16")]
    out["checks"] = []
    for name, m, r, h in checks:
        prefix = work / f"{name}_{m}_{r}"
        p = subprocess.run(engine_cmd(m, ["--score", str(par1), "--out", str(prefix), "--dump-logprobs"], r, h),
                           capture_output=True, text=True)
        same = p.returncode == 0 and all(
            Path(f"{prefix}{ext}").read_bytes() == Path(f"{ref}{ext}").read_bytes() for ext in (".pos.bin", ".logp.f32"))
        out["checks"].append({"check": name, "m": m, "router": r, "hash": h, "returncode": p.returncode,
                              "config": [ln for ln in p.stdout.splitlines() if ln.startswith("CONFIG ")],
                              "byte_identical": same})
        print(f"{name} {m} {r} hash={int(h)} byte_identical={same}", flush=True)
    out["status"] = "COMPLETE"
    return out


def bank() -> list[dict]:
    recs = []
    for rep in range(BANK_REPS):
        for gb, k in BANK[rep % len(BANK):] + BANK[:rep % len(BANK)]:
            cmd = [str(EXE["bank"]), "--table-gb", str(gb), "--row-bytes", "4096", "--k", str(k),
                   "--layers", str(BANK_LAYERS), "--tokens", str(BANK_TOKENS), "--warm", str(BANK_WARM), "--mem", "pageable"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            rows = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = {"cmd": cmd, "rep": rep, "gb": gb, "k": k, "returncode": p.returncode,
                   "config": [ln for ln in p.stdout.splitlines() if ln.startswith("CONFIG ")]}
            rec.update(rows[0] if p.returncode == 0 and len(rows) == 1 else {"status": "FAILED"})
            recs.append(rec)
            print(f"bank rep{rep} gb={gb} k={k} {rec.get('median_token_s', 'FAILED')}", flush=True)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s1/known_positives")
    args = ap.parse_args()
    out_path = Path(args.out)
    report = {"cell": "C50-S1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "occ_bar_pct": OCC_BAR,
              "mem": MEM, "prio": PRIO, "ctx": CTX, "steps": STEPS, "ids_sha256": sha256(IDS),
              "exe_sha256": {k: sha256(p) for k, p in EXE.items()}, "source": f"{SOURCE_PATH} @ {SOURCE_COMMIT}",
              "source_sha256": source_blob_sha256()}

    def save():
        out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    def void(why: str) -> int:
        report["verdict_input"] = why
        save()
        print(why)
        return 1

    if report["exe_sha256"] != EXE_SHA256_PIN or report["source_sha256"] != SOURCE_SHA256_PIN:
        return void("VOID: binaries differ from the pinned S1 build")
    report["model_sha256"] = {m: sha256(p) for m, (p, _, _) in ARTIFACTS.items()}
    if any(report["model_sha256"][m] != pin for m, (_, _, pin) in ARTIFACTS.items()):
        return void("VOID: an artifact differs from its pin")
    pre = E44.watch(PREFLIGHT_SECONDS)
    mean = sum(pre) / len(pre) if pre else float("nan")
    report["preflight"] = {"seconds": PREFLIGHT_SECONDS, "samples": pre, "rule": "mean < bar (A1 addendum 2)",
                           "mean": mean, "max": max(pre) if pre else None, "admissible": bool(pre) and mean < OCC_BAR}
    print(f"preflight mean={mean:.2f}% admissible={report['preflight']['admissible']}", flush=True)
    if not report["preflight"]["admissible"]:
        return void("VOID: pre-flight occupancy at or above the bar")
    report["known_positives"] = known_positives(Path(args.work))
    save()
    if report["known_positives"]["status"] != "COMPLETE":
        return void(report["known_positives"]["status"])
    k1 = [c for c in report["known_positives"]["checks"] if c["check"] == "K1"][0]["byte_identical"]
    k2 = {c["m"] for c in report["known_positives"]["checks"] if c["check"] == "K2" and not c["byte_identical"]}
    report["void_arms"] = sorted({arm_key(a) for a in arms() if (a["router"] == "bf16" and not k1) or
                                  (a["hash"] and a["m"] in k2)})
    report["processes"] = []
    replacements: dict[str, int] = {}
    for rnd, a in schedule():
        key = arm_key(a)
        if key in report["void_arms"]:
            continue
        while True:
            rec = rate_process(a)
            rec["round"] = rnd
            report["processes"].append(rec)
            save()
            print(f"r{rnd} {key} {rec['status']} foreign={rec['foreign_pct']:.2f}% "
                  f"median_tok_s={rec.get('median_tok_s', float('nan')):.3f} wall={rec['wall_s']:.0f}s", flush=True)
            if rec["status"] == "FAILED":
                return void(f"VOID: process failed ({key})")
            if rec["status"] == "OK":
                break
            replacements[key] = replacements.get(key, 0) + 1
            if replacements[key] > MAX_REPLACEMENTS:
                return void(f"VOID: more than {MAX_REPLACEMENTS} replacements ({key})")
    report["bank"] = bank()
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["verdict_input"] = "COMPLETE"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
