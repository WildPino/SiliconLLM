"""State of a stage 1 (a) run report, for s1_autoresume.ps1. Prints one line:

DONE                    every arm has every panel COMPLETE
RESUME <done>/<total>   only missing, interrupted or budget-stopped panels are left: a relaunch resumes them
HOLD <done>/<total> ..  a VOID or STOPPED panel is left: it waits for a person, not for a relaunch
CORRUPT                 the report does not parse (e.g. the machine went down while it was being written)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_run  # noqa: E402  (constants only; main() is guarded)


def main() -> int:
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"RESUME 0/{len(s1_run.ARMS) * len(s1_run.PANELS)}")
        return 0
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        print("CORRUPT")
        return 0
    done, hold = 0, []
    for arm in s1_run.ARMS:
        panels = report.get("arms", {}).get(arm, {}).get("panels", {})
        for panel, *_ in s1_run.PANELS:
            status = panels.get(panel, {}).get("status", "")
            if status == "COMPLETE":
                done += 1
            elif status.startswith(("VOID", "STOPPED")):
                hold.append(f"{arm}/{panel}")
    total = len(s1_run.ARMS) * len(s1_run.PANELS)
    if done == total:
        print("DONE")
    elif hold:
        print(f"HOLD {done}/{total} {','.join(hold)}")
    else:
        print(f"RESUME {done}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
