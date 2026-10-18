"""C50-H2 topology: physical cores, L3 domains (CCX/CCD), NUMA nodes, packages and RAM of the machine it runs on.

ROADMAP section 8, H2: the first item of the probe pack on every machine, and the input a run-time thread map will be
chosen from.  It reports numbers only and changes no binary; the engine's --threads keeps taking the first N physical
cores.  Pure Python (ctypes on GetLogicalProcessorInformationEx), so it runs natively on x64 and on ARM64 Windows.

    known positive    --expect-r0 asserts the reference box: 6 cores, 12 logical processors, two 16 MiB L3 domains of
                      3 cores each, one NUMA node, one package.  It must pass on R0 before a laptop's output is read.
    consistency       always asserted: every logical processor is in exactly one core, every core in exactly one L3
                      domain (when L3 exists) and one NUMA node.

Usage:
    python -I benchmarks/chatbot50/c50_topology.py --json topo.json [--expect-r0]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import struct
import sys
import winreg
from ctypes import wintypes
from pathlib import Path

REL_CORE, REL_NUMA, REL_CACHE, REL_PACKAGE, REL_ALL = 0, 1, 2, 3, 0xFFFF
CACHE_TYPES = {0: "unified", 1: "instruction", 2: "data", 3: "trace"}
ERROR_INSUFFICIENT_BUFFER = 122


def group_masks(rec: bytes, count_off: int, masks_off: int) -> list[tuple[int, int]]:
    """(group, mask) pairs of a record's GROUP_AFFINITY array (16 bytes each: KAFFINITY, WORD group, WORD[3])."""
    (n,) = struct.unpack_from("<H", rec, count_off)
    return [struct.unpack_from("<QH", rec, masks_off + 16 * i)[::-1] for i in range(max(n, 1))]


def logical(masks: list[tuple[int, int]]) -> list[str]:
    return [f"{g}:{b}" for g, m in masks for b in range(64) if m >> b & 1]


def records() -> list[tuple[int, bytes]]:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetLogicalProcessorInformationEx
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    fn.restype = wintypes.BOOL
    size = wintypes.DWORD(0)
    assert not fn(REL_ALL, None, ctypes.byref(size)) and ctypes.get_last_error() == ERROR_INSUFFICIENT_BUFFER
    buf = ctypes.create_string_buffer(size.value)
    assert fn(REL_ALL, buf, ctypes.byref(size)), ctypes.get_last_error()
    raw, out, off = buf.raw[: size.value], [], 0
    while off < len(raw):
        rel, rsize = struct.unpack_from("<II", raw, off)
        out.append((rel, raw[off + 8: off + rsize]))
        off += rsize
    assert off == len(raw)
    return out


def ram_bytes() -> int:
    class MS(ctypes.Structure):
        _fields_ = [("len", wintypes.DWORD), ("load", wintypes.DWORD), ("total", ctypes.c_uint64),
                    ("avail", ctypes.c_uint64), ("pt", ctypes.c_uint64), ("pa", ctypes.c_uint64),
                    ("vt", ctypes.c_uint64), ("va", ctypes.c_uint64), ("ve", ctypes.c_uint64)]
    ms = MS()
    ms.len = ctypes.sizeof(MS)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return int(ms.total)


def cpu_name() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except OSError:
        return "unknown"


def topology() -> dict:
    cores, packages, numa, caches = [], [], [], []
    for rel, rec in records():
        if rel in (REL_CORE, REL_PACKAGE):
            flags, eff = struct.unpack_from("<BB", rec, 0)
            item = {"logical": logical(group_masks(rec, 22, 24))}
            if rel == REL_CORE:
                item.update(smt=bool(flags & 1), efficiency_class=eff)
                cores.append(item)
            else:
                packages.append(item)
        elif rel == REL_NUMA:
            (node,) = struct.unpack_from("<I", rec, 0)
            numa.append({"node": node, "logical": logical(group_masks(rec, 22, 24))})
        elif rel == REL_CACHE:
            level, assoc, line, size, ctype = struct.unpack_from("<BBHII", rec, 0)
            caches.append({"level": level, "type": CACHE_TYPES.get(ctype, str(ctype)), "bytes": size,
                           "line_bytes": line, "associativity": assoc, "logical": logical(group_masks(rec, 30, 32))})
    cores.sort(key=lambda c: (int(c["logical"][0].split(":")[0]), int(c["logical"][0].split(":")[1])))
    all_lp = [lp for c in cores for lp in c["logical"]]
    assert len(all_lp) == len(set(all_lp)), "a logical processor in two cores"
    core_of = {lp: i for i, c in enumerate(cores) for lp in c["logical"]}

    def core_set(lps: list[str]) -> list[int]:
        return sorted({core_of[lp] for lp in lps if lp in core_of})

    l3 = [c for c in caches if c["level"] == 3 and c["type"] in ("unified", "data")]
    l3_domains = [{"bytes": c["bytes"], "cores": core_set(c["logical"])} for c in l3]
    numa_nodes = [{"node": n["node"], "cores": core_set(n["logical"])} for n in sorted(numa, key=lambda n: n["node"])]
    for name, doms in (("L3 domain", l3_domains), ("NUMA node", numa_nodes)):
        if doms:
            seen = [i for d in doms for i in d["cores"]]
            assert sorted(seen) == list(range(len(cores))), f"cores not partitioned by {name}: {seen}"
    per_level = {}
    for c in caches:
        key = f"L{c['level']}_{c['type']}"
        per_level.setdefault(key, {"instances": 0, "bytes_each": set()})
        per_level[key]["instances"] += 1
        per_level[key]["bytes_each"].add(c["bytes"])
    for v in per_level.values():
        v["bytes_each"] = sorted(v["bytes_each"])
    first_lp = [c["logical"][0] for c in cores]
    return {"cell": "C50-H2 topology", "cpu": cpu_name(), "machine": platform.machine(),
            "windows": platform.version(), "ram_bytes": ram_bytes(),
            "physical_cores": len(cores), "logical_processors": len(all_lp),
            "efficiency_classes": sorted({c["efficiency_class"] for c in cores}),
            "packages": len(packages), "numa_nodes": numa_nodes, "l3_domains": l3_domains,
            "caches": per_level, "cores": cores,
            "engine_threads_first_n": {"rule": "--threads N takes the first N cores in this order, first logical each",
                                       "logical_by_core": first_lp,
                                       "l3_domain_by_core": [next((j for j, d in enumerate(l3_domains) if i in d["cores"]), None)
                                                             for i in range(len(cores))]}}


def expect_r0(t: dict) -> list[str]:
    bad = []
    if t["physical_cores"] != 6: bad.append(f"physical_cores {t['physical_cores']} != 6")
    if t["logical_processors"] != 12: bad.append(f"logical_processors {t['logical_processors']} != 12")
    if [(d["bytes"], len(d["cores"])) for d in t["l3_domains"]] != [(16 << 20, 3)] * 2:
        bad.append(f"l3_domains {[(d['bytes'], d['cores']) for d in t['l3_domains']]} != two 16 MiB domains of 3 cores")
    if len(t["numa_nodes"]) != 1: bad.append(f"numa_nodes {len(t['numa_nodes'])} != 1")
    if t["packages"] != 1: bad.append(f"packages {t['packages']} != 1")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--expect-r0", action="store_true")
    args = ap.parse_args()
    t = topology()
    if args.expect_r0:
        bad = expect_r0(t)
        t["known_positive_r0"] = "PASS" if not bad else {"FAIL": bad}
    Path(args.json).write_text(json.dumps(t, indent=1), encoding="utf-8")
    print(f"TOPOLOGY cpu={t['cpu']!r} machine={t['machine']} ram_gib={t['ram_bytes'] / 2**30:.1f} "
          f"cores={t['physical_cores']} logical={t['logical_processors']} packages={t['packages']} "
          f"numa={len(t['numa_nodes'])} l3={[(d['bytes'] >> 20, d['cores']) for d in t['l3_domains']]} "
          f"eff_classes={t['efficiency_classes']}")
    if args.expect_r0:
        print("KNOWN POSITIVE R0", "PASS" if not bad else f"FAIL {bad}")
        return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
