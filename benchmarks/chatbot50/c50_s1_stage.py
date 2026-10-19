"""C50-S1 staging: copy T0 and the three pinned expansions from E: (their build location) to the NVMe staging
directory the runner reads, and check every staged copy against its pin.  Idempotent: a staged file with the right
size is only re-hashed.  It reads no rate.  Needs ~129 GB free on C:.

Usage:
    python -I benchmarks/chatbot50/c50_s1_stage.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
STAGE = Path("C:/SiliconLLM_stage/s1")  # c50_s1_run.STAGE
T0 = (C / "gguf" / "ling-mini-2.0-a810f64-q4_k_m.gguf", "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    items = [T0]
    for e in (536, 1088, 1600):
        rec = json.loads((HERE / "results" / f"c50_s1_expand_e{e}.json").read_text(encoding="utf-8"))
        items.append((C / "s1" / f"t0-e{e}.gguf", rec["dst_sha256"]))
    need = sum(src.stat().st_size for src, _ in items if not (STAGE / src.name).exists())
    free = shutil.disk_usage(STAGE.anchor).free
    if need > free:
        raise SystemExit(f"need {need / 1e9:.1f} GB on {STAGE.anchor}, {free / 1e9:.1f} GB free")
    STAGE.mkdir(parents=True, exist_ok=True)
    for src, pin in items:
        dst = STAGE / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            print(time.strftime("%H:%M:%S"), "copy", src.name, flush=True)
            shutil.copyfile(src, dst)
        got = sha256(dst)
        if got != pin:
            raise SystemExit(f"{dst}: sha256 {got} != pin {pin}")
        print(time.strftime("%H:%M:%S"), "staged", dst.name, "=", pin, flush=True)
    print("S1 STAGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
