#!/usr/bin/env python3
"""Freeze token-context inputs for the actual-state native route timing."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np

import meth136_sparse_experts  # inserts the S1 import path
import meth136_matched_sparse_train as M136


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth143_fresh_hash_route_manifest.json"
MANIFEST_SHA = "e6c4fb6a92929eb29332c55a0887adb6627c9528f80b21c09f7759139070ccff"
HEADER = struct.Struct("<8sI")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.report.exists()
    assert M136.digest(MANIFEST) == MANIFEST_SHA
    item = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"][0]
    ids = np.asarray(item["document_ids"][:256], dtype="<u4")
    assert ids.shape == (256,) and int(ids.min()) >= 0
    data = np.stack((ids, np.r_[np.uint32(0), ids[:-1]],
                     np.arange(256, dtype="<u4")), axis=1).astype("<u4")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as file:
        file.write(HEADER.pack(b"M144TX01", 256))
        file.write(data.tobytes())
    readback = args.out.read_bytes()
    assert readback == HEADER.pack(b"M144TX01", 256) + data.tobytes()
    report = {"experiment": "METH-144-native-context-fixture",
              "manifest_sha256": MANIFEST_SHA, "source_id": item["source_id"],
              "tokens": 256, "bytes": args.out.stat().st_size,
              "sha256": M136.digest(args.out), "readback_exact": True}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
