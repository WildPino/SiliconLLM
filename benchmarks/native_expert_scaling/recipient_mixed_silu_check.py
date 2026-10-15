#!/usr/bin/env python3
"""Check scalar SiLU arithmetic against every saved native diagnostic site.

This reads the retained native trace only. It does not load model weights,
execute a decoder, change the frozen auditor, or infer source-model quality.

CLI: --trace NATIVE.TRACE --result DIAGNOSTIC-RESULT.JSON --out REPORT.JSON
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


HEADER = struct.Struct("<4I")
F32 = struct.Struct("<f")
EXPECTED_TRACE_SHA = "bb965c40a21648b677f6b10f5ee780d15c07cc03a5ea47df4f759baa732c5db8"
POSITIONS, LAYERS, WIDTH = 8, 24, 4864
MAX_TRACE_BYTES = 64 << 20
TAGS = (52, 62, 92)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def f32(value):
    return F32.unpack(F32.pack(value))[0]


def native_scalar_active(gate, up):
    # Match the C expression's F32 rounding after expf, arithmetic, and output.
    if gate >= 0:
        exponential = f32(math.exp(float(-gate)))
        denominator = f32(1.0 + exponential)
        silu = f32(gate / denominator)
    else:
        exponential = f32(math.exp(float(gate)))
        numerator = f32(gate * exponential)
        denominator = f32(1.0 + exponential)
        silu = f32(numerator / denominator)
    return f32(silu * up)


def read_sites(path):
    require(path.is_file() and path.stat().st_size <= MAX_TRACE_BYTES,
            "native trace missing or exceeds 64 MiB cap")
    sites = {}
    records = 0
    with path.open("rb") as stream:
        while True:
            header = stream.read(HEADER.size)
            if not header:
                break
            require(len(header) == HEADER.size, "truncated trace header")
            position, layer, tag, count = HEADER.unpack(header)
            require(position < POSITIONS and layer <= LAYERS and
                    0 < count <= 151936, "trace record geometry")
            raw = stream.read(count * F32.size)
            require(len(raw) == count * F32.size, "truncated trace record")
            records += 1
            if tag in TAGS:
                require(layer < LAYERS and count == WIDTH,
                        "SiLU trace site geometry")
                key = (position, layer)
                site = sites.setdefault(key, {})
                require(tag not in site, "duplicate SiLU trace tag")
                site[tag] = raw
    require(len(sites) == POSITIONS * LAYERS and
            set(sites) == {(p, l) for p in range(POSITIONS) for l in range(LAYERS)},
            "incomplete SiLU trace site census")
    require(all(set(site) == set(TAGS) for site in sites.values()),
            "incomplete gate/up/active trace triple")
    return sites, records


def site_check(position, layer, site):
    gate = [x[0] for x in struct.iter_unpack("<f", site[52])]
    up = [x[0] for x in struct.iter_unpack("<f", site[62])]
    actual = site[92]
    require(len(gate) == len(up) == WIDTH and len(actual) == WIDTH * 4,
            "SiLU trace width")
    require(all(math.isfinite(x) for x in gate + up), "nonfinite SiLU inputs")
    computed = bytearray(WIDTH * 4)
    mismatches = 0
    first = None
    negative = 0
    for index, (g, u) in enumerate(zip(gate, up)):
        negative += g < 0
        value = native_scalar_active(g, u)
        F32.pack_into(computed, index * 4, value)
        if computed[index * 4:index * 4 + 4] != actual[index * 4:index * 4 + 4]:
            mismatches += 1
            if first is None:
                first = {"index": index, "gate": g, "up": u,
                         "computed": value,
                         "native": F32.unpack_from(actual, index * 4)[0]}
    return {"position": position, "layer": layer, "elements": WIDTH,
            "negative_gate_count": negative, "bitwise_mismatches": mismatches,
            "first_mismatch": first,
            "gate_sha256": hashlib.sha256(site[52]).hexdigest(),
            "up_sha256": hashlib.sha256(site[62]).hexdigest(),
            "native_active_sha256": hashlib.sha256(actual).hexdigest(),
            "computed_active_sha256": hashlib.sha256(computed).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and args.out.parent.is_dir(),
            "exclusive output path or parent absent")
    require(args.trace.name == "native.trace" and
            args.trace.parent.resolve() == args.result.parent.resolve(),
            "trace and diagnostic result location")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    require(result.get("schema") == "RECIPIENT_MIXED_DIAGNOSTIC_V1" and
            result.get("status") == "TRACE_COMPARISON_COMPLETE", "diagnostic result identity")
    before = {"trace": digest(args.trace), "result": digest(args.result),
              "reducer": digest(Path(__file__))}
    require(before["trace"] == result["native_trace_sha256"] == EXPECTED_TRACE_SHA,
            "native trace source seal")
    sites, record_count = read_sites(args.trace)
    rows = [site_check(position, layer, sites[(position, layer)])
            for position in range(POSITIONS) for layer in range(LAYERS)]
    by_layer = [{"layer": layer,
                 "sites": POSITIONS,
                 "elements": POSITIONS * WIDTH,
                 "bitwise_mismatches": sum(row["bitwise_mismatches"] for row in rows
                                           if row["layer"] == layer)}
                for layer in range(LAYERS)]
    after = {"trace": digest(args.trace), "result": digest(args.result),
             "reducer": digest(Path(__file__))}
    require(after == before, "input or reducer changed during check")
    mismatch_count = sum(row["bitwise_mismatches"] for row in rows)
    report = {"schema": "RECIPIENT_MIXED_STORED_SILU_ALL_SITES_V1",
              "status": "SCALAR_EXP_ALL_NATIVE_SITES_MATCH" if mismatch_count == 0 else
                        "SCALAR_EXP_MISMATCH",
              "scope": "192 stored native gate/up/active triples; no model replay or parity admission",
              "input_sha256_before_after": before,
              "native_trace_records": record_count,
              "sites": len(rows), "elements": len(rows) * WIDTH,
              "bitwise_mismatches": mismatch_count,
              "by_layer": by_layer, "site_rows": rows}
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if mismatch_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
