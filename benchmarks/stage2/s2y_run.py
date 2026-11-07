"""S2L's panels, numbers only: s2y_verdict.py decides.  The tools are test 3b's with unaligned moves (S2K, amendment 3).  Routing k = 8, mode none, as test 2c, but for stk4, whose generated
tokens run at k = 4 as test 3b's; stk4 scores its own score panels, which do not depend on k.  Resumable: a COMPLETE panel is skipped.

Usage: python -I benchmarks/stage2/s2y_run.py --arm A --model F --model-sha S --out benchmarks/stage2/results/s2y_run.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s2x_run import GEN, INPUTS, MAX_PROJECTED_S, PANELS, SCORE, STOP, WORK, check_inputs, launch, sha256  # noqa: E402

FULL = ("nq", "tqa", "gsm8k", "arc", "mmlu", "docs", "chat")
ARMS = {"as_both": FULL, "st_both2": FULL, "stk4": FULL, "ff_nodense": ("docs", "chat"), "ff_noearly": ("docs", "chat"), "ff_nolate": ("docs", "chat"),
        "ff_keepout": ("docs", "chat"), "ff_keeprnd": ("docs", "chat"), "ks_org": ("docs", "chat"),
        # amendment 1: the same values in another container.  tq404 is tq40 at k = 4; its score panels are copied, as test 3b's.
        "tq40": FULL, "tq404": ("nq", "tqa", "gsm8k", "chat"), "sb_q40": FULL, "sb_bf": ("docs", "chat"), "ff_bf": ("docs", "chat"), "ff_keeprnd2": ("docs", "chat")}
K4 = ("stk4", "tq404")
UA = {t: x.with_name(x.name.replace("_avx2", "_ua_avx2")) for t, x in (("score", SCORE), ("generate", GEN))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--model-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    report = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {"cell": "S2L-organs-first-test", "arms": {}}
    man = {k: check_inputs(p) for k, p in INPUTS.items()}
    if report.setdefault("inputs_manifest_sha256", man) != man:
        raise SystemExit("inputs changed since the run began")
    if sha256(a.model) != a.model_sha:
        raise SystemExit("the model differs from its pin")
    rec = report["arms"].setdefault(a.arm, {"panels": {}, "panel_set": list(ARMS[a.arm]), "model": str(a.model), "model_sha256": a.model_sha})
    (WORK / a.arm).mkdir(parents=True, exist_ok=True)
    if a.arm == "tq404":
        for p in ("arc", "mmlu", "docs"):
            for x in (WORK / "tq40").glob(f"{p}.*"):
                shutil.copyfile(x, WORK / a.arm / x.name)
    for panel, src, tool, extra in PANELS:
        if panel not in ARMS[a.arm] or rec["panels"].get(panel, {}).get("status") == "COMPLETE":
            continue
        flag = "--ids" if tool == "score" else "--prompts"
        cmd = [str(UA[tool]), "--model", str(a.model), flag, str(INPUTS[src] / f"{panel}.tsv"), "--out", str(WORK / a.arm / panel), "--threads", "6", "--ubatch", "512",
               "--max-projected-s", str(3 * MAX_PROJECTED_S)] + extra
        k4 = a.arm in K4 and tool == "generate"
        cmd += ["--moe-mode", "top", "--moe-k", "4", "--moe-mass", "16", "--moe-decode-only"] if k4 else ["--moe-mode", "none"]
        if panel == "docs":
            cmd += ["--ref-top", str(WORK / "ref" / "docs.top.bin")]
        if tool == "generate":
            cmd += ["--stop", STOP]
        r = launch(cmd, [f"model={a.model} ", "flash_attn=disabled", "moe_k=4 " if k4 else "moe_mode=none "])
        rec["panels"][panel] = r
        a.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(a.arm, panel, r["status"], r["wall_s"], flush=True)
        if r["status"] != "COMPLETE":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
