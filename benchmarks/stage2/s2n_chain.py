"""S2N's chains, resumable.  A step whose record exists is skipped.  GPU work runs one job at a time, in this chain's order.

dry  (GPU, before the push): the teacher pass; mini's expert load for cell F; parity at k = 8, k = 4 mass 16 and the planted k = 4 mass 0 (llama.cpp's
     k = 4 side from s2n_llama_docs.py), and the two subtle plants at k = 8; ks_stk4 held out as it is; the damage test.  Stops if parity fails.
gpu  (after the push): the lr pick, then e2e_row, e2e_grp, glob, one at a time; after each, its export and the export check.
cpu  (after the push): each arm's seven panels once its export check is in.
second: e2e_row2, e2e_row's second data order: training, export, check, panels; run only if s2n_verdict.py says SECOND ORDER NEEDED.

Usage: python -I benchmarks/stage2/s2n_chain.py dry | gpu | cpu | second
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = HERE / "results"
S = Path("E:/SiliconLLM_Artifacts/stage2/s2n")
SCORES = "E:/SiliconLLM_Artifacts/stage1/s1a"
A4, S2 = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs", "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s2/inputs"
LLAMA_K8 = "E:/SiliconLLM_Artifacts/stage1/s1a/ks_stk4/docs.pos.bin"
LLAMA_K4 = S / "llama/ks_stk4_k4m16/docs.pos.bin"
PY = sys.executable
sys.path.insert(0, str(HERE))


def step(*cmd: str) -> None:
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run([PY, "-I", *cmd], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"stopped ({r.returncode}): {' '.join(cmd)}")


def wait(p: Path) -> None:
    while not p.exists():
        time.sleep(120)


def rd(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def dry() -> None:
    launches = S / "teacher/launches.jsonl"
    if not (launches.exists() and json.loads(launches.read_text(encoding="utf-8").splitlines()[-1])["exit"] == 0):
        step("benchmarks/stage2/s2n_teach.py")
    if not (R / "s2n_route_hist.json").exists():
        step("benchmarks/stage2/s2n_route_hist.py", "--json", str(R / "s2n_route_hist.json"))
    wait(LLAMA_K4)
    runs = [("k8", "k8", LLAMA_K8, None), ("k4m16", "k4m16", str(LLAMA_K4), None), ("k4m0", "k4m0", str(LLAMA_K4), None),
            ("k8_noexc", "k8", LLAMA_K8, "noexc"), ("k8_shexp097", "k8", LLAMA_K8, "shexp097")]
    for name, form, src, plant in runs:
        js = R / f"s2n_parity_{name}.json"
        if not js.exists():
            step("benchmarks/stage2/s2n_train.py", "--parity", "--form", form, "--llama-pos", src, "--json", str(js), *(["--plant", plant] if plant else []))
    import s2n_verdict as V
    if not (V.parity(rd("s2n_parity_k8.json")) and V.parity(rd("s2n_parity_k4m16.json"))):
        raise SystemExit("stopped: parity fails (VOID)")
    if not (R / "s2n_dry_none.json").exists():
        step("benchmarks/stage2/s2n_train.py", "--arm", "none", "--out", str(S / "dry/none"), "--json", str(R / "s2n_dry_none.json"))
    for arm in ("damage", "damage_ref"):
        if not (R / f"s2n_dry_{arm}.json").exists():
            step("benchmarks/stage2/s2n_train.py", "--arm", arm, "--out", str(S / f"dry/{arm}"), "--json", str(R / f"s2n_dry_{arm}.json"))
    ok = V.damage(rd("s2n_dry_damage.json"), rd("s2n_dry_damage_ref.json"), rd("s2n_dry_none.json"))
    print("S2N DRY DONE", "damage repaired" if ok else "damage NOT repaired: no arm runs", flush=True)


def lr() -> float:
    return float(rd("s2n_pick.json")["pick"])


def trained(arm: str) -> None:
    """Training, export and export check of one arm (GPU)."""
    js = R / f"s2n_train_{arm}.json"
    if not js.exists():
        step("benchmarks/stage2/s2n_train.py", "--arm", arm, "--lr", f"{lr():g}", "--out", str(S / "train" / arm), "--json", str(js))
    ex, q4 = R / f"s2n_export_{arm}.json", S / "q4" / f"{arm}.gguf"
    if not ex.exists():
        step("benchmarks/stage2/s2n_export.py", "--arm", arm, "--scales", str(S / "train" / arm / "scales.pt"), "--out", str(q4), "--json", str(ex))
    ec = R / f"s2n_export_check_{arm}.json"
    if not ec.exists():
        step("benchmarks/stage2/s2n_train.py", "--export-check", "--gguf", str(q4), "--train-json", str(js), "--json", str(ec))


def gpu() -> None:
    import s2n_verdict as V
    if not (V.parity(rd("s2n_parity_k8.json")) and V.parity(rd("s2n_parity_k4m16.json"))
            and V.damage(rd("s2n_dry_damage.json"), rd("s2n_dry_damage_ref.json"), rd("s2n_dry_none.json"))):
        raise SystemExit("stopped: the dry run's parity or damage control failed, as registered: no arm runs")
    if not (R / "s2n_pick.json").exists():
        step("benchmarks/stage2/s2n_train.py", "--pick", "--out", str(S / "pick"), "--json", str(R / "s2n_pick.json"))
    for arm in ("e2e_row", "e2e_grp", "glob"):
        trained(arm)
    print("S2N GPU DONE", flush=True)


def panels(arm: str) -> None:
    wait(R / f"s2n_export_check_{arm}.json")
    sha = rd(f"s2n_export_{arm}.json")["out_sha256"]
    step("benchmarks/stage2/s2n_run.py", "--arm", arm, "--model", str(S / "q4" / f"{arm}.gguf"), "--model-sha", sha, "--out", str(R / "s2n_run.json"))
    step("benchmarks/chatbot50/c50_s2_checks.py", "--a4-inputs", A4, "--s2-inputs", S2, "--arm-a4", f"{SCORES}/{arm}", "--arm-s2", f"{SCORES}/{arm}",
         "--out", f"{SCORES}/scores/{arm}.json")


def cpu() -> None:
    for arm in ("e2e_row", "e2e_grp", "glob"):
        panels(arm)
    print("S2N CPU DONE", flush=True)


def second() -> None:
    trained("e2e_row2")
    panels("e2e_row2")
    print("S2N SECOND DONE", flush=True)


if __name__ == "__main__":
    {"dry": dry, "gpu": gpu, "cpu": cpu, "second": second}[sys.argv[1]]()
