#!/usr/bin/env python3
"""Export and byte-audit the exact METH-85 grouped-R8 FFN matrices."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import psutil
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "results/native_expert_scaling/meth85_qwen05b_instruct_group64_r8_core.safetensors"
CORE_SHA = "c484a1130e495342d1b8644ff156cc002d0fd68fe12230604e59f35d3af6433a"
MAGIC = b"M182FFN1"
HEADER = "<8sIIII"
LAYERS, WIDTH, INTERMEDIATE, GROUP = 24, 896, 4864, 64
ORGANS = ("gate_proj", "up_proj", "down_proj")
MAX_SECONDS = 15 * 60
MAX_RSS = 8 * (1 << 30)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def budget(start):
    result = {"seconds": time.monotonic() - start,
              "rss_bytes": psutil.Process().memory_info().rss}
    if result["seconds"] > MAX_SECONDS or result["rss_bytes"] > MAX_RSS:
        raise RuntimeError(f"METH-182 export resource stop: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.report.exists()
    start = time.monotonic()
    assert digest(CORE) == CORE_SHA
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(CORE, framework="pt", device="cpu") as archive, args.out.open("xb") as output:
        assert archive.metadata()["format"] == "QWEN25_INSTRUCT_R8H_GROUP64_R8FFN_BF16ATTN_V1"
        output.write(struct.pack(HEADER, MAGIC, LAYERS, WIDTH, INTERMEDIATE, GROUP))
        for layer in range(LAYERS):
            for organ in ORGANS:
                name = f"model.layers.{layer}.mlp.{organ}.weight"
                q = archive.get_tensor(name + ".q")
                scale = archive.get_tensor(name + ".scale")
                rows_count = WIDTH if organ == "down_proj" else INTERMEDIATE
                columns = INTERMEDIATE if organ == "down_proj" else WIDTH
                assert q.dtype == torch.int8 and tuple(q.shape) == (rows_count, columns)
                assert scale.dtype == torch.float16 and tuple(scale.shape) == (rows_count, columns // GROUP)
                assert q.is_contiguous() and scale.is_contiguous()
                assert torch.isfinite(scale).all() and (scale > 0).all()
                qb = q.numpy().tobytes()
                sb = scale.view(torch.uint16).numpy().tobytes()
                offset = output.tell()
                output.write(qb)
                output.write(sb)
                rows.append({"layer": layer, "projection": organ, "offset": offset,
                             "q_bytes": len(qb), "scale_bytes": len(sb),
                             "q_sha256": hashlib.sha256(qb).hexdigest(),
                             "scale_sha256": hashlib.sha256(sb).hexdigest()})
                budget(start)
    assert len(rows) == LAYERS * len(ORGANS)
    with args.out.open("rb") as stored:
        assert stored.read(struct.calcsize(HEADER)) == struct.pack(
            HEADER, MAGIC, LAYERS, WIDTH, INTERMEDIATE, GROUP)
        for row in rows:
            assert stored.tell() == row["offset"]
            assert hashlib.sha256(stored.read(row["q_bytes"])).hexdigest() == row["q_sha256"]
            assert hashlib.sha256(stored.read(row["scale_bytes"])).hexdigest() == row["scale_sha256"]
        assert stored.read(1) == b""
    report = {"experiment": "METH-182-exact-stored-group64-R8-FFN-export",
              "source_core_sha256": CORE_SHA, "layers": LAYERS, "width": WIDTH,
              "intermediate": INTERMEDIATE, "group": GROUP,
              "order": list(ORGANS), "rows": rows,
              "binary": {"path": str(args.out.resolve()), "bytes": args.out.stat().st_size,
                         "sha256": digest(args.out)},
              "runtime": budget(start), "scope": "Stored-code export and byte audit only"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"binary": report["binary"], "runtime": report["runtime"]}), flush=True)


if __name__ == "__main__":
    main()
