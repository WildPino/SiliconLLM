"""Stage 2 part 1, test 3 verdict: what the jump at layer 6 of stage 1 (b) is (it chooses nothing beyond the
registered rule).

Controls, before any label:
    C1 planted: at layer 3 of plant, the own error's top expert and top dimension are the planted ones, each with a
                share >= 0.9;
    C2 replay:  tern's raw jump ratio, carried at layer 6 over carried at layer 5, within 15% of (b)'s 4.44
                (33.46 / 7.53, benchmarks/stage1/results/s1b_train.json).
A failed control makes the run VOID.
Label, fix5 against tern:
    "THE JUMP IS A MASSIVE DIMENSION THE NORMS REMOVE": the raw carried error at layer 6 cut by >= 50%, the normalized
        carried error at layer 6 and the head's KL each moved by < 10%;
    "THE JUMP CARRIES THE LOSS": the head's KL cut by >= 25%;
    "PARTLY" otherwise.
Reported, not judged: fix15, ph, the normalized own errors, the routing agreement.
--selftest runs planted inputs through the rule.

Usage: python -I benchmarks/stage2/s2c_verdict.py --run benchmarks/stage2/results/s2c_layer6.json \
           --json benchmarks/stage2/results/s2c_verdict.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

B_JUMP = 33.46 / 7.53
REPLAY_TOL, PLANT_SHARE, CUT_RAW, MOVE_MAX, CUT_KL = 0.15, 0.9, 0.5, 0.10, 0.25


def decide(run: dict) -> dict:
    pl = run["arms"]["plant"]["layers"][str(run["plant"]["layer"])]
    c1 = (pl["experts_top"][0][0] == run["plant"]["expert"] and pl["experts_top"][0][1] >= PLANT_SHARE
          and pl["dims_own_top"][0][0] == run["plant"]["dim"] and pl["dims_own_top"][0][1] >= PLANT_SHARE)
    t, f = run["arms"]["tern"], run["arms"]["fix5"]
    jump = t["layers"]["6"]["carried"] / t["layers"]["5"]["carried"]
    c2 = abs(jump / B_JUMP - 1) <= REPLAY_TOL
    out = {"controls": {"C1": c1, "C2": c2, "jump_ratio": jump, "b_jump_ratio": B_JUMP}}
    if not (c1 and c2):
        out["outcome"] = "VOID"
        return out
    raw = 1 - f["layers"]["6"]["carried"] / t["layers"]["6"]["carried"]
    nrm = f["layers"]["6"]["carried_n"] / t["layers"]["6"]["carried_n"] - 1
    kl = f["head"]["kl_nats"] / t["head"]["kl_nats"] - 1
    out["fix5"] = {"raw_cut_l6": raw, "nrm_move_l6": nrm, "kl_move": kl}
    if raw >= CUT_RAW and abs(nrm) < MOVE_MAX and abs(kl) < MOVE_MAX:
        out["outcome"] = "THE JUMP IS A MASSIVE DIMENSION THE NORMS REMOVE"
    elif kl <= -CUT_KL:
        out["outcome"] = "THE JUMP CARRIES THE LOSS"
    else:
        out["outcome"] = "PARTLY"
    return out


def selftest() -> int:
    def run(raw_f=11.0, nrm_f=0.069, kl_f=0.446, jump=4.5, plant=(8, 0.99, 918, 0.99)):
        lay = lambda c, n: {"carried": c, "carried_n": n}  # noqa: E731
        return {"plant": {"layer": 3, "expert": 8, "dim": 918},
                "arms": {"plant": {"layers": {"3": {"experts_top": [[plant[0], plant[1]]],
                                                    "dims_own_top": [[plant[2], plant[3]]]}}},
                         "tern": {"layers": {"5": lay(11.5, 0.06), "6": lay(11.5 * jump, 0.069)},
                                  "head": {"kl_nats": 0.452}},
                         "fix5": {"layers": {"5": lay(11.5, 0.06), "6": lay(raw_f, nrm_f)}, "head": {"kl_nats": kl_f}}}}

    cases = {
        "as in the dry run": (run(), "THE JUMP IS A MASSIVE DIMENSION THE NORMS REMOVE"),
        "carries the loss": (run(kl_f=0.30, nrm_f=0.05), "THE JUMP CARRIES THE LOSS"),
        "partly": (run(kl_f=0.40), "PARTLY"),
        "raw not cut": (run(raw_f=40.0), "PARTLY"),
        "plant wrong expert": (run(plant=(9, 0.99, 918, 0.99)), "VOID"),
        "plant weak dim": (run(plant=(8, 0.99, 918, 0.5)), "VOID"),
        "replay off": (run(jump=3.0), "VOID"),
    }
    bad = 0
    for name, (r, want) in cases.items():
        got = decide(r)["outcome"]
        bad += got != want
        print("selftest", "ok  " if got == want else "FAIL", name, got)
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    out = decide(run)
    out["arms"] = {arm: {"head": r.get("head"), "restored": r.get("restored"),
                         "carried_l6": r["layers"].get("6", {}).get("carried"),
                         "carried_n_l6": r["layers"].get("6", {}).get("carried_n"),
                         "carried_n_l19": r["layers"].get("19", {}).get("carried_n"),
                         "route_same_l10": r["layers"].get("10", {}).get("route_same")}
                   for arm, r in run["arms"].items() if arm != "plant"}
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
