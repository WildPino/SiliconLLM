#!/usr/bin/env python3
"""Receipt-backed D: to E: artifact migration with guarded junction cutover.

Windows only. `prepare` is read-only for source units; `run` is a single
explicit attempt. An interrupted copy can continue only with
`run --resume-verified`, which revalidates every accepted file receipt.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import threading
import time
import uuid


DROOT = Path("D:/_THINGS/Progetti/SiliconLLM")
EROOT = Path("E:/SiliconLLM_Artifacts/workspace")
CONTROL = Path("E:/SiliconLLM_Artifacts/migrations/20261015")
BACKUP = DROOT / ".codex/storage_migration/20261015/original"
PLAN_SCHEMA = "STORAGE_MIGRATION_PLAN_V1"
MIGRATION_SCHEMA = "STORAGE_MIGRATION_V1"
COPY_TIME_CAP = 72 * 3600
STATUS_INTERVAL = 10.0
FILE_ATTRIBUTE_READONLY = 0x1
FILE_ATTRIBUTE_DIRECTORY = 0x10
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003
IO_REPARSE_TAG_SYMLINK = 0xA000000C
FSCTL_GET_REPARSE_POINT = 0x000900A8
FSCTL_SET_REPARSE_POINT = 0x000900A4
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_READ_ATTRIBUTES = 0x80
FILE_WRITE_ATTRIBUTES = 0x100
GENERIC_WRITE = 0x40000000
FILE_SHARE_ALL = 7
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def fail(message):
    raise RuntimeError(message)


def require(ok, message):
    if not ok:
        fail(message)


def sha_file(path, *, limit=None):
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb", buffering=0) as stream:
        while True:
            block = stream.read(8 << 20)
            if not block:
                break
            digest.update(block)
            total += len(block)
            if limit is not None:
                require(total <= limit, f"file grew while hashing: {path}")
    return digest.hexdigest(), total


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def write_atomic(path, value):
    data = json_bytes(value)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_jsonl(path, value):
    line = json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("ab", buffering=0) as stream:
        stream.write(line.encode("utf-8"))
        os.fsync(stream.fileno())


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lexical(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def guard_under(path, root, *, allow_root=False):
    candidate, base = lexical(path), lexical(root)
    require(os.path.commonpath((candidate, base)) == base and
            (allow_root or candidate != base), f"path outside guarded root: {path}")
    return Path(candidate)


def is_reparse(path):
    try:
        return bool(os.stat(path, follow_symlinks=False).st_file_attributes &
                    FILE_ATTRIBUTE_REPARSE_POINT)
    except FileNotFoundError:
        return False


def guard_no_reparse_ancestors(path, root):
    guard_under(path, root, allow_root=True)
    if lexical(root) == lexical(BACKUP):
        guard_under(BACKUP, DROOT)
        parent = DROOT
        for part in BACKUP.relative_to(DROOT).parts:
            parent /= part
            require(not is_reparse(parent), f"backup ancestor is reparse: {parent}")
    current = Path(root)
    require(not is_reparse(current), f"guard root is reparse: {current}")
    relative = Path(path).relative_to(root)
    for part in relative.parts[:-1]:
        current /= part
        require(not is_reparse(current), f"reparse ancestor forbidden: {current}")


def validate_relative(text):
    require(isinstance(text, str) and text and "\\" not in text,
            f"invalid relative path: {text!r}")
    path = PureWindowsPath(text)
    parts = path.parts
    require(not path.is_absolute() and not path.drive and
            all(p not in ("", ".", "..") and not p.endswith((".", " ")) and
                ":" not in p for p in parts), f"unsafe relative path: {text}")
    require(parts[0].casefold() not in (".git", ".codex") and
            all(p.casefold() != ".git" for p in parts),
            f"protected source path: {text}")
    return Path(*parts)


def frozen_plan(path):
    raw = path.read_bytes()
    plan = json.loads(raw)
    require(plan.get("schema") == PLAN_SCHEMA, "plan schema")
    require(set(plan) == {"schema", "purpose", "source_root", "destination_root",
                          "control_root", "policy", "units"} and
            plan["source_root"] == DROOT.as_posix() and
            plan["destination_root"] == EROOT.as_posix() and
            plan["control_root"] == CONTROL.as_posix() and
            isinstance(plan["units"], list) and
            plan["units"], "plan keys/units")
    require(plan["policy"] == {
        "preserve_original_paths": True, "preserve_source_hardlink_groups": True,
        "verify_sha256_before_source_release": True,
        "edit_historical_receipts": False, "move_git_database": False,
        "rewrite_git_history": False, "run_model_experiments": False}, "plan policy changed")
    units = []
    for entry in plan["units"]:
        require(isinstance(entry, dict) and set(entry) == {"name", "relative_path"},
                "unit keys")
        name = entry["name"]
        require(isinstance(name, str) and name and name.replace("-", "").replace("_", "").isalnum(),
                f"unsafe unit name: {name}")
        relative = validate_relative(entry["relative_path"])
        units.append((name, relative))
    require(len({name.casefold() for name, _ in units}) == len(units), "duplicate unit name")
    names = [str(rel).casefold() for _, rel in units]
    require(len(set(names)) == len(names), "duplicate unit path")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            require(not (left.startswith(right + os.sep) or right.startswith(left + os.sep)),
                    "overlapping units")
    return plan, units, sha_bytes(raw)


if os.name == "nt":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("creation", FILETIME),
                    ("access", FILETIME), ("write", FILETIME),
                    ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
                    ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                    ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]

    class WIN32_FIND_STREAM_DATA(ctypes.Structure):
        _fields_ = [("size", ctypes.c_longlong), ("name", wintypes.WCHAR * 296)]

    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE,
                                                   ctypes.POINTER(BY_HANDLE_FILE_INFORMATION)]
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.CopyFileW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.BOOL]
    kernel.CopyFileW.restype = wintypes.BOOL
    kernel.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,
                                       wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                       ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    kernel.DeviceIoControl.restype = wintypes.BOOL
    kernel.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    kernel.GetFileAttributesW.restype = wintypes.DWORD
    kernel.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel.SetFileAttributesW.restype = wintypes.BOOL
    kernel.FindFirstStreamW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                       ctypes.POINTER(WIN32_FIND_STREAM_DATA), wintypes.DWORD]
    kernel.FindFirstStreamW.restype = wintypes.HANDLE
    kernel.FindNextStreamW.argtypes = [wintypes.HANDLE, ctypes.POINTER(WIN32_FIND_STREAM_DATA)]
    kernel.FindNextStreamW.restype = wintypes.BOOL
    kernel.FindClose.argtypes = [wintypes.HANDLE]
    kernel.FindClose.restype = wintypes.BOOL
    kernel.SetFileTime.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME),
                                   ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
    kernel.SetFileTime.restype = wintypes.BOOL


def windows_only():
    require(os.name == "nt", "migration requires Windows")


def winerror(what):
    raise ctypes.WinError(ctypes.get_last_error(), what)


def open_handle(path, access=FILE_READ_ATTRIBUTES):
    windows_only()
    handle = kernel.CreateFileW(str(path), access, FILE_SHARE_ALL, None, OPEN_EXISTING,
                                FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                                None)
    if handle == INVALID_HANDLE_VALUE:
        winerror(f"CreateFileW: {path}")
    return handle


def file_identity(path):
    handle = open_handle(path)
    try:
        info = BY_HANDLE_FILE_INFORMATION()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
            winerror(f"GetFileInformationByHandle: {path}")
        return {"volume": int(info.volume),
                "file_index": (int(info.index_high) << 32) | int(info.index_low),
                "links": int(info.links), "attributes": int(info.attributes),
                "creation_filetime": (int(info.creation.high) << 32) | int(info.creation.low)}
    finally:
        kernel.CloseHandle(handle)


def streams(path):
    windows_only()
    data = WIN32_FIND_STREAM_DATA()
    handle = kernel.FindFirstStreamW(str(path), 0, ctypes.byref(data), 0)
    if handle == INVALID_HANDLE_VALUE:
        winerror(f"FindFirstStreamW: {path}")
    result = []
    try:
        while True:
            name = str(data.name)
            if name != "::$DATA":
                require(name.startswith(":") and name.endswith(":$DATA"),
                        f"unsupported alternate stream name: {name}")
                stream_name = name[:-6]
                digest, length = sha_file(str(path) + stream_name, limit=int(data.size))
                require(length == int(data.size), f"alternate stream length changed: {path}{name}")
                result.append({"name": name, "bytes": length, "sha256": digest})
            if not kernel.FindNextStreamW(handle, ctypes.byref(data)):
                code = ctypes.get_last_error()
                require(code == 38, f"FindNextStreamW error {code}: {path}")
                break
    finally:
        kernel.FindClose(handle)
    return sorted(result, key=lambda x: x["name"])


def raw_reparse(path):
    handle = open_handle(path)
    try:
        output = ctypes.create_string_buffer(16 << 10)
        count = wintypes.DWORD()
        if not kernel.DeviceIoControl(handle, FSCTL_GET_REPARSE_POINT, None, 0,
                                      output, len(output), ctypes.byref(count), None):
            winerror(f"FSCTL_GET_REPARSE_POINT: {path}")
        raw = output.raw[:count.value]
        tag = struct_unpack_u32(raw[:4])
        require(tag in (IO_REPARSE_TAG_MOUNT_POINT, IO_REPARSE_TAG_SYMLINK),
                f"unsupported reparse tag {tag:#x}: {path}")
        return tag, raw
    finally:
        kernel.CloseHandle(handle)


def struct_unpack_u32(raw):
    return int.from_bytes(raw, "little")


def set_raw_reparse(path, raw):
    handle = open_handle(path, GENERIC_WRITE)
    try:
        input_buffer = ctypes.create_string_buffer(raw)
        count = wintypes.DWORD()
        if not kernel.DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT,
                                      input_buffer, len(raw), None, 0,
                                      ctypes.byref(count), None):
            winerror(f"FSCTL_SET_REPARSE_POINT: {path}")
    finally:
        kernel.CloseHandle(handle)


def make_junction(path, target):
    windows_only()
    require(not path.exists() and not is_reparse(path), f"junction path exists: {path}")
    absolute = str(Path(target).absolute()).replace("/", "\\")
    require(len(absolute) >= 3 and absolute[1:3] == ":\\", "junction target must be drive-absolute")
    substitute = "\\??\\" + absolute
    print_name = absolute
    sub = substitute.encode("utf-16-le")
    shown = print_name.encode("utf-16-le")
    pathname = sub + b"\x00\x00" + shown + b"\x00\x00"
    body = ((0).to_bytes(2, "little") + len(sub).to_bytes(2, "little") +
            (len(sub) + 2).to_bytes(2, "little") + len(shown).to_bytes(2, "little") + pathname)
    raw = (IO_REPARSE_TAG_MOUNT_POINT.to_bytes(4, "little") +
           len(body).to_bytes(2, "little") + b"\x00\x00" + body)
    os.mkdir(path)
    try:
        set_raw_reparse(path, raw)
        tag, actual = raw_reparse(path)
        require(tag == IO_REPARSE_TAG_MOUNT_POINT and actual == raw,
                f"junction reparse readback differs: {path}")
    except BaseException:
        if path.exists() or is_reparse(path):
            os.rmdir(path)
        raise


def normalized_junction_target(path):
    target = os.readlink(path)
    for prefix in ("\\??\\", "\\\\?\\"):
        if target.startswith(prefix):
            target = target[len(prefix):]
            break
    return lexical(target)


def classify(path, info=None):
    st = os.stat(path, follow_symlinks=False)
    info = info or file_identity(path)
    attrs = info["attributes"]
    if attrs & FILE_ATTRIBUTE_REPARSE_POINT:
        tag, raw = raw_reparse(path)
        return "junction" if tag == IO_REPARSE_TAG_MOUNT_POINT else "symlink"
    return "directory" if attrs & FILE_ATTRIBUTE_DIRECTORY else "file"


def metadata(path, kind=None):
    st = os.stat(path, follow_symlinks=False)
    info = file_identity(path)
    kind = kind or classify(path, info)
    result = {"kind": kind, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
              "ctime_ns": int(st.st_ctime_ns), "attributes": info["attributes"],
              "volume": info["volume"], "file_index": info["file_index"],
              "links": info["links"], "creation_filetime": info["creation_filetime"]}
    if kind in ("junction", "symlink"):
        tag, raw = raw_reparse(path)
        result.update(reparse_tag=tag, reparse_sha256=sha_bytes(raw),
                      reparse_bytes=len(raw), raw_target=os.readlink(path))
    if kind == "file":
        result["alternate_streams"] = streams(path)
    return result


def inventory_unit(source, *, logical_junction=False):
    if logical_junction:
        require(os.path.isjunction(source), f"logical root not a junction: {source}")
    else:
        require(source.is_dir() and not is_reparse(source), f"source unit not ordinary directory: {source}")
    records = []

    def walk(directory, relative):
        for entry in sorted(os.scandir(directory), key=lambda x: x.name.casefold()):
            rel = relative / entry.name
            path = Path(entry.path)
            item = {"relative_path": rel.as_posix(), "metadata": metadata(path)}
            records.append(item)
            if item["metadata"]["kind"] == "directory":
                walk(path, rel)

    records.append({"relative_path": ".", "metadata":
                    metadata(source, "junction" if logical_junction else "directory")})
    walk(source, Path())
    require(len({x["relative_path"].casefold() for x in records}) == len(records),
            f"duplicate inventory name: {source}")
    return records


def validate_roots():
    windows_only()
    require(DROOT.is_dir() and EROOT.is_dir() and CONTROL.parent.is_dir(),
            "literal roots not present")
    require(not is_reparse(DROOT) and not is_reparse(EROOT) and
            not is_reparse(CONTROL.parent), "root is reparse")
    require(file_identity(DROOT)["volume"] != file_identity(EROOT)["volume"],
            "D and E roots must be different physical volumes")
    guard_under(BACKUP, DROOT)
    guard_under(CONTROL, CONTROL.parent)


def prepare(plan_path):
    validate_roots()
    plan, units, plan_sha = frozen_plan(plan_path)
    require(not CONTROL.exists(), f"control namespace already exists: {CONTROL}")
    require(not BACKUP.exists(), f"backup namespace already exists: {BACKUP}")
    inventories = {}
    totals = {"files": 0, "bytes": 0, "directories": 0, "links": 0}
    has_symlink = False
    for name, relative in units:
        source, target = DROOT / relative, EROOT / relative
        guard_no_reparse_ancestors(source, DROOT)
        guard_no_reparse_ancestors(target, EROOT)
        require(not target.exists() and not is_reparse(target), f"target exists: {target}")
        inventory = inventory_unit(source)
        require(not any(".git" in (p.casefold() for p in
                                   PureWindowsPath(item["relative_path"]).parts)
                        for item in inventory),
                f"nested Git database is outside migration scope: {source}")
        inventories[name] = {"relative_path": relative.as_posix(), "entries": inventory}
        for item in inventory:
            kind = item["metadata"]["kind"]
            if kind == "file":
                require(not (item["metadata"]["attributes"] & FILE_ATTRIBUTE_READONLY and
                             item["metadata"]["links"] > 1),
                        f"readonly external hardlink needs manual qualification: {source}/{item['relative_path']}")
                totals["files"] += 1
                totals["bytes"] += item["metadata"]["size"]
            elif kind == "directory":
                totals["directories"] += 1
            else:
                totals["links"] += 1
                has_symlink |= kind == "symlink"
    if has_symlink:
        fail("symlink-bearing units require a separately qualified raw-symlink preflight; no cutover prepared")
    CONTROL.mkdir(parents=True, exist_ok=False)
    write_exclusive(CONTROL / "frozen_plan.json", {"plan": plan, "raw_sha256": plan_sha,
                                                   "original_path": str(plan_path)})
    write_exclusive(CONTROL / "inventory.json", {"schema": MIGRATION_SCHEMA,
                                                  "units": inventories, "totals": totals})
    write_exclusive(CONTROL / "state.json", {"schema": MIGRATION_SCHEMA,
                                              "phase": "PREPARED", "units": {n: "PREPARED" for n, _ in units},
                                              "plan_sha256": plan_sha})
    (CONTROL / "receipts.jsonl").open("xb").close()
    print(json.dumps({"status": "PREPARED", "plan_sha256": plan_sha,
                      "units": len(units), "totals": totals}), flush=True)


class Heartbeat:
    def __init__(self, totals):
        self.state = {"phase": "COPYING", "copied_files": 0, "verified_files": 0,
                      "copied_bytes": 0, "verified_bytes": 0, "current_file": None,
                      "total_files": totals["files"], "total_bytes": totals["bytes"],
                      "pid": os.getpid(), "started_wall_time": time.time(),
                      "updated_utc": None}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def snapshot(self):
        with self.lock:
            data = dict(self.state)
        data["verified_percent"] = (100.0 * data["verified_bytes"] /
                                    data["total_bytes"] if data["total_bytes"] else 100.0)
        data["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data["wall_time"] = time.time()
        return data

    def loop(self):
        while not self.stop.wait(STATUS_INTERVAL):
            try:
                write_atomic(CONTROL / "status.json", self.snapshot())
            except BaseException:
                self.stop.set()

    def update(self, **changes):
        with self.lock:
            self.state.update(changes)

    def start(self):
        write_atomic(CONTROL / "status.json", self.snapshot())
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()
        write_atomic(CONTROL / "status.json", self.snapshot())


def receipts():
    result = {}
    with (CONTROL / "receipts.jsonl").open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                fail("incomplete durable file receipt")
            item = json.loads(line)
            key = (item["unit"], item["relative_path"])
            require(key not in result, f"duplicate file receipt: {key}")
            result[key] = item
    return result


def copy_file_native(source, target):
    windows_only()
    require(not target.exists() and not is_reparse(target), f"target already exists: {target}")
    if not kernel.CopyFileW(str(source), str(target), True):
        winerror(f"CopyFileW failed: {source} -> {target}")


def verify_metadata_unchanged(source, expected):
    now = metadata(source)
    # Hardlink counts can fall after an earlier unit's verified D: backup is
    # removed. A rename may also change creation/change timestamps.
    keys = ("kind", "size", "mtime_ns", "attributes", "volume", "file_index",
            "alternate_streams", "reparse_tag", "reparse_sha256", "raw_target")
    require(all(now.get(key) == expected.get(key) for key in keys),
            f"source metadata changed: {source}")


def verify_copied_file(source, target, expected, source_sha=None):
    verify_metadata_unchanged(source, expected)
    require(target.is_file() and not is_reparse(target), f"target file missing: {target}")
    src_hash, src_size = sha_file(source, limit=expected["size"])
    dst_hash, dst_size = sha_file(target, limit=expected["size"])
    require(src_size == dst_size == expected["size"] and src_hash == dst_hash and
            (source_sha is None or src_hash == source_sha), f"copy hash differs: {source}")
    require(streams(target) == expected["alternate_streams"],
            f"alternate stream copy differs: {target}")
    dest = metadata(target)
    require(dest["size"] == expected["size"] and
            dest["mtime_ns"] == expected["mtime_ns"] and
            dest["attributes"] == expected["attributes"] and
            dest["creation_filetime"] == expected["creation_filetime"],
            f"copied file metadata differs: {target}")
    return src_hash


def clone_nested_junction(source, target, expected):
    tag, raw = raw_reparse(source)
    require(tag == IO_REPARSE_TAG_MOUNT_POINT and sha_bytes(raw) == expected["reparse_sha256"],
            f"nested junction changed: {source}")
    require(not target.exists() and not is_reparse(target), f"nested target exists: {target}")
    os.mkdir(target)
    try:
        set_raw_reparse(target, raw)
        got_tag, got = raw_reparse(target)
        require(got_tag == tag and got == raw and os.readlink(target) == expected["raw_target"],
                f"nested junction readback differs: {target}")
    except BaseException:
        os.rmdir(target)
        raise


def unit_paths(name, inventories):
    relative = Path(inventories[name]["relative_path"])
    source, target, backup = DROOT / relative, EROOT / relative, BACKUP / relative
    for path, root in ((source, DROOT), (target, EROOT), (backup, BACKUP)):
        guard_under(path, root)
        guard_no_reparse_ancestors(path, root)
    return source, target, backup


def all_entries(inventories):
    for unit, value in inventories.items():
        for entry in value["entries"]:
            yield unit, entry


def verify_tree(target, entries, receipts_map, unit, *, logical_junction=False,
                hash_files=True):
    actual = inventory_unit(target, logical_junction=logical_junction)
    expected_names = {x["relative_path"] for x in entries}
    require({x["relative_path"] for x in actual} == expected_names,
            f"target missing/extra entries: {target}")
    actual_map = {x["relative_path"]: x["metadata"] for x in actual}
    for entry in entries:
        rel, expected = entry["relative_path"], entry["metadata"]
        got = actual_map[rel]
        if rel == "." and logical_junction:
            require(got["kind"] == "junction", f"logical root kind: {target}")
            continue
        require(got["kind"] == expected["kind"], f"target kind: {target}/{rel}")
        if expected["kind"] == "file":
            receipt = receipts_map[(unit, rel)]
            destination = target if rel == "." else target / rel
            require(got["size"] == expected["size"] and
                    got["alternate_streams"] == expected["alternate_streams"] and
                    got["mtime_ns"] == expected["mtime_ns"] and
                    got["attributes"] == expected["attributes"] and
                    got["creation_filetime"] == expected["creation_filetime"] and
                    got["volume"] == receipt["target_volume"] and
                    got["file_index"] == receipt["target_file_index"],
                    f"target file verification: {destination}")
            if hash_files:
                digest, size = sha_file(destination, limit=expected["size"])
                require(size == expected["size"] and digest == receipt["sha256"],
                        f"target payload hash changed: {destination}")
        elif expected["kind"] == "directory":
            require(got["mtime_ns"] == expected["mtime_ns"] and
                    got["attributes"] == expected["attributes"],
                    f"target directory verification: {target}/{rel}")
        elif expected["kind"] in ("junction", "symlink"):
            require(got["reparse_sha256"] == expected["reparse_sha256"] and
                    got["raw_target"] == expected["raw_target"],
                    f"nested link verification: {target}/{rel}")


def state_update(state, phase=None, unit=None, value=None):
    if phase:
        state["phase"] = phase
    if unit:
        state["units"][unit] = value
    write_atomic(CONTROL / "state.json", state)


def set_times_and_attrs(path, expected):
    # Directory metadata is restored after all children have been created.
    handle = open_handle(path, FILE_WRITE_ATTRIBUTES)
    try:
        raw = expected["creation_filetime"]
        created = FILETIME(raw & 0xffffffff, raw >> 32)
        write_ticks = (expected["mtime_ns"] // 100) + 116444736000000000
        modified = FILETIME(write_ticks & 0xffffffff, write_ticks >> 32)
        if not kernel.SetFileTime(handle, ctypes.byref(created), None, ctypes.byref(modified)):
            winerror(f"SetFileTime: {path}")
    finally:
        kernel.CloseHandle(handle)
    if not kernel.SetFileAttributesW(str(path), expected["attributes"]):
        winerror(f"SetFileAttributesW: {path}")


def guarded_delete_backup(backup, entries, receipts_map, unit):
    guard_under(backup, BACKUP)
    guard_no_reparse_ancestors(backup, BACKUP)
    # Validate the complete, still-intact backup before deleting its first
    # child. Directory mtimes change during the postorder deletion below.
    actual = inventory_unit(backup)
    require({x["relative_path"] for x in actual} ==
            {x["relative_path"] for x in entries}, "backup inventory changed")
    by_rel = {x["relative_path"]: x["metadata"] for x in actual}
    for entry in entries:
        rel, expected = entry["relative_path"], entry["metadata"]
        keys = ("kind", "size", "mtime_ns", "attributes", "volume", "file_index",
                "alternate_streams", "reparse_tag", "reparse_sha256", "raw_target")
        require(all(by_rel[rel].get(k) == expected.get(k) for k in keys),
                f"backup entry changed: {backup}/{rel}")
    # No recursive filesystem API is used: every unlink comes from the frozen inventory.
    for entry in sorted(entries, key=lambda x: (x["relative_path"].count("/"),
                                                x["relative_path"]), reverse=True):
        rel, expected = entry["relative_path"], entry["metadata"]
        path = backup if rel == "." else backup / rel
        guard_under(path, BACKUP)
        guard_no_reparse_ancestors(path, BACKUP)
        require(path.exists() or is_reparse(path), f"backup entry missing: {path}")
        current = metadata(path)
        require(current["kind"] == expected["kind"] and
                current["volume"] == expected["volume"] and
                current["file_index"] == expected["file_index"],
                f"backup identity changed before delete: {path}")
        kind = expected["kind"]
        if kind == "file":
            receipt = receipts_map[(unit, rel)]
            digest, size = sha_file(path, limit=expected["size"])
            require(size == expected["size"] and digest == receipt["sha256"],
                    f"backup content changed: {path}")
            if expected["attributes"] & FILE_ATTRIBUTE_READONLY:
                require(current["links"] == 1,
                        f"readonly hardlink retained for manual cleanup: {path}")
                if not kernel.SetFileAttributesW(str(path),
                                                 expected["attributes"] & ~FILE_ATTRIBUTE_READONLY):
                    winerror(f"clear verified readonly backup: {path}")
            os.unlink(path)
        elif kind == "directory" or (expected["attributes"] & FILE_ATTRIBUTE_DIRECTORY):
            os.rmdir(path)
        else:
            os.unlink(path)


def rollback_cutover(source, target, backup):
    """Restore the original name after a pre-cleanup cutover fault only."""
    guard_no_reparse_ancestors(source, DROOT)
    guard_no_reparse_ancestors(backup, BACKUP)
    require(backup.is_dir() and not is_reparse(backup),
            f"rollback backup missing or not ordinary: {backup}")
    if is_reparse(source):
        require(os.path.isjunction(source) and
                normalized_junction_target(source) == lexical(target),
                f"rollback refuses changed junction: {source}")
        os.rmdir(source)  # remove the link itself; never traverse its target
    require(not source.exists() and not is_reparse(source),
            f"rollback original name occupied: {source}")
    os.rename(backup, source)
    require(source.is_dir() and not is_reparse(source),
            f"rollback original not restored: {source}")


def run(plan_path, resume_verified=False):
    validate_roots()
    _, units, plan_sha = frozen_plan(plan_path)
    frozen = load(CONTROL / "frozen_plan.json")
    require(frozen["raw_sha256"] == plan_sha, "plan changed since prepare")
    inventory_doc = load(CONTROL / "inventory.json")
    inventories = inventory_doc["units"]
    state = load(CONTROL / "state.json")
    require(state["plan_sha256"] == plan_sha and
            set(state["units"]) == {n for n, _ in units}, "prepared state/plan mismatch")
    completed = {"BACKUP_REMOVED"}
    incomplete = {"PREPARED", "COPYING", "VERIFIED", "ROLLED_BACK"}
    require(set(state["units"].values()) <= completed | incomplete,
            "partial cutover requires manual review; no automatic resume")
    if resume_verified:
        require(state["phase"] in ("PREPARED", "COPYING", "FAILED"),
                "resume state not eligible")
    else:
        require(state["phase"] == "PREPARED" and
                all(value == "PREPARED" for value in state["units"].values()) and
                not any((EROOT / rel).exists() or is_reparse(EROOT / rel)
                        for _, rel in units), "fresh run requires absent targets")
        require((CONTROL / "receipts.jsonl").stat().st_size == 0,
                "fresh run has earlier file receipts")
    require(not (CONTROL / "started.json").exists() or resume_verified,
            "run already started; explicit --resume-verified required")
    if not (CONTROL / "started.json").exists():
        write_exclusive(CONTROL / "started.json",
                        {"pid": os.getpid(), "wall_time": time.time(),
                         "plan_sha256": plan_sha})
    started = time.monotonic()
    heartbeat = Heartbeat(inventory_doc["totals"])
    heartbeat.start()
    done = receipts()
    unit_rank = {name: i for i, (name, _) in enumerate(units)}
    hardlink_groups = {}
    for unit, entry in all_entries(inventories):
        meta = entry["metadata"]
        if meta["kind"] == "file" and meta["links"] > 1:
            hardlink_groups.setdefault((meta["volume"], meta["file_index"]), []).append(
                (unit, entry["relative_path"]))
    canonical = {member: min(group, key=lambda x: (unit_rank[x[0]], x[1]))
                 for group in hardlink_groups.values() for member in group}
    verified_files = verified_bytes = 0
    copied_files = copied_bytes = 0
    try:
        state_update(state, phase="COPYING")
        for unit, _ in units:
            source, target, backup = unit_paths(unit, inventories)
            entries = inventories[unit]["entries"]
            files = [e for e in entries if e["metadata"]["kind"] == "file"]
            files.sort(key=lambda e: e["relative_path"])
            if state["units"][unit] == "BACKUP_REMOVED":
                require(os.path.isjunction(source) and
                        normalized_junction_target(source) == lexical(target),
                        f"completed logical junction changed: {source}")
                require(not backup.exists() and not is_reparse(backup),
                        f"completed backup reappeared: {backup}")
                verify_tree(target, entries, done, unit)
                verify_tree(source, entries, done, unit, logical_junction=True)
                verified_files += len(files)
                verified_bytes += sum(e["metadata"]["size"] for e in files)
                copied_files, copied_bytes = verified_files, verified_bytes
                heartbeat.update(copied_files=copied_files, copied_bytes=copied_bytes,
                                 verified_files=verified_files, verified_bytes=verified_bytes)
                continue
            require(source.is_dir() and not is_reparse(source), f"source not ordinary: {source}")
            require(not backup.exists() and not is_reparse(backup),
                    f"unexpected backup before cutover: {backup}")
            state_update(state, unit=unit, value="COPYING")
            target.parent.mkdir(parents=True, exist_ok=True)
            guard_no_reparse_ancestors(target, EROOT)
            if not target.exists():
                os.mkdir(target)
            require(target.is_dir() and not is_reparse(target),
                    f"target root not ordinary: {target}")
            for entry in entries:
                rel, meta = entry["relative_path"], entry["metadata"]
                if rel == ".":
                    continue
                dst = target / rel
                guard_no_reparse_ancestors(dst, EROOT)
                if meta["kind"] == "directory":
                    if not dst.exists():
                        os.mkdir(dst)
                    require(dst.is_dir() and not is_reparse(dst),
                            f"target directory differs: {dst}")
                elif meta["kind"] == "junction":
                    if not is_reparse(dst):
                        clone_nested_junction(source / rel, dst, meta)
                    require(metadata(dst)["reparse_sha256"] == meta["reparse_sha256"],
                            f"existing nested junction differs: {dst}")
                elif meta["kind"] == "symlink":
                    fail(f"symlink unit was not qualified: {unit}/{rel}")
            for entry in files:
                rel, meta = entry["relative_path"], entry["metadata"]
                src, dst = source / rel, target / rel
                guard_no_reparse_ancestors(src, DROOT)
                guard_no_reparse_ancestors(dst, EROOT)
                heartbeat.update(current_file=f"{unit}/{rel}", phase="COPYING")
                receipt_key = (unit, rel)
                if receipt_key in done:
                    require(resume_verified, "receipt exists in fresh run")
                    require(verify_copied_file(src, dst, meta, done[receipt_key]["sha256"]) ==
                            done[receipt_key]["sha256"], f"receipt revalidation: {unit}/{rel}")
                else:
                    require(not dst.exists() and not is_reparse(dst),
                            f"unreceipted partial target requires manual review: {dst}")
                    first = canonical.get((unit, rel))
                    method = "CopyFileW"
                    if first and first != (unit, rel):
                        first_unit, first_rel = first
                        first_target = unit_paths(first_unit, inventories)[1] / first_rel
                        require((first_unit, first_rel) in done,
                                f"canonical hardlink not verified: {first_unit}/{first_rel}")
                        os.link(first_target, dst)
                        method = "hardlink"
                        require(file_identity(first_target)["file_index"] ==
                                file_identity(dst)["file_index"], "target hardlink identity")
                    else:
                        copy_file_native(src, dst)
                        set_times_and_attrs(dst, meta)
                    copied_files += 1
                    copied_bytes += meta["size"]
                    heartbeat.update(copied_files=copied_files, copied_bytes=copied_bytes)
                    digest = verify_copied_file(src, dst, meta)
                    item = {"unit": unit, "relative_path": rel, "bytes": meta["size"],
                            "sha256": digest, "method": method,
                            "source_volume": meta["volume"],
                            "source_file_index": meta["file_index"],
                            "target_volume": file_identity(dst)["volume"],
                            "target_file_index": file_identity(dst)["file_index"],
                            "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    append_jsonl(CONTROL / "receipts.jsonl", item)
                    done[receipt_key] = item
                verified_files += 1
                verified_bytes += meta["size"]
                heartbeat.update(verified_files=verified_files,
                                 verified_bytes=verified_bytes)
                require(time.monotonic() - started <= COPY_TIME_CAP, "copy time cap")
            heartbeat.update(current_file=None, phase="VERIFYING")
            directories = [e for e in entries if e["metadata"]["kind"] == "directory"]
            for entry in sorted(directories, key=lambda e: e["relative_path"].count("/"),
                                reverse=True):
                dst = target if entry["relative_path"] == "." else target / entry["relative_path"]
                set_times_and_attrs(dst, entry["metadata"])
            actual_source = inventory_unit(source)
            require({e["relative_path"] for e in actual_source} ==
                    {e["relative_path"] for e in entries},
                    f"source gained/lost entries: {source}")
            for entry in entries:
                src = source if entry["relative_path"] == "." else source / entry["relative_path"]
                verify_metadata_unchanged(src, entry["metadata"])
            verify_tree(target, entries, done, unit, hash_files=False)
            state_update(state, unit=unit, value="VERIFIED")
            guard_no_reparse_ancestors(source, DROOT)
            guard_no_reparse_ancestors(backup, BACKUP)
            backup.parent.mkdir(parents=True, exist_ok=True)
            require(not backup.exists() and not is_reparse(backup),
                    f"backup path occupied: {backup}")
            os.rename(source, backup)
            try:
                state_update(state, unit=unit, value="RENAMED")
                make_junction(source, target)
                state_update(state, unit=unit, value="JUNCTIONED")
                require(os.path.isjunction(source) and
                        normalized_junction_target(source) == lexical(target),
                        f"cutover junction target: {source}")
                verify_tree(source, entries, done, unit, logical_junction=True,
                            hash_files=False)
                state_update(state, unit=unit, value="LOGICAL_VERIFIED")
            except BaseException as cutover_error:
                try:
                    rollback_cutover(source, target, backup)
                    state_update(state, unit=unit, value="ROLLED_BACK")
                except BaseException as rollback_error:
                    fail(f"cutover failed: {cutover_error}; rollback failed: {rollback_error}")
                raise
            heartbeat.update(phase="CLEANING")
            guarded_delete_backup(backup, entries, done, unit)
            state_update(state, unit=unit, value="BACKUP_REMOVED")
            heartbeat.update(phase="COPYING")
        require(len(done) == inventory_doc["totals"]["files"],
                "incomplete global file receipts")
        state_update(state, phase="COMPLETE")
        heartbeat.update(phase="COMPLETE", current_file=None)
        manifest = {"schema": MIGRATION_SCHEMA, "status": "COMPLETE",
                    "plan_sha256": plan_sha, "source_root": str(DROOT),
                    "target_root": str(EROOT), "control_root": str(CONTROL),
                    "backup_root": str(BACKUP), "units": [n for n, _ in units],
                    "files": verified_files, "logical_file_bytes": verified_bytes,
                    "receipts_sha256": sha_file(CONTROL / "receipts.jsonl")[0],
                    "inventory_sha256": sha_file(CONTROL / "inventory.json")[0],
                    "source_resolve_changes": True,
                    "scope": "Verified copied payloads, preserved hardlinks and junction path access; Path.resolve may change across cutover."}
        write_exclusive(CONTROL / "manifest.json", manifest)
        print(json.dumps({"status": "COMPLETE", "units": len(units),
                          "files": verified_files, "bytes": verified_bytes}), flush=True)
    except BaseException as error:
        state_update(state, phase="FAILED")
        failure = CONTROL / "failure.json"
        if failure.exists():
            failure = CONTROL / f"failure.{int(time.time())}.json"
        write_exclusive(failure, {"error_type": type(error).__name__,
                                  "error": str(error), "state": state})
        raise
    finally:
        heartbeat.close()

def status(plan_path):
    _, units, digest = frozen_plan(plan_path)
    if not CONTROL.exists():
        print(json.dumps({"status": "NOT_PREPARED", "plan_sha256": digest}))
        return
    frozen = load(CONTROL / "frozen_plan.json")
    require(frozen["raw_sha256"] == digest, "status plan mismatch")
    result = {"plan_sha256": digest, "units": len(units),
              "state": load(CONTROL / "state.json")}
    if (CONTROL / "status.json").is_file():
        result["heartbeat"] = load(CONTROL / "status.json")
    if (CONTROL / "manifest.json").is_file():
        result["manifest"] = load(CONTROL / "manifest.json")
    if (CONTROL / "failure.json").is_file():
        result["failure"] = load(CONTROL / "failure.json")
    print(json.dumps(result, indent=2))


def selftest():
    windows_only()
    scratch = CONTROL.parent / ("scratch-selftest-20261015-" + uuid.uuid4().hex[:12])
    require(not scratch.exists() and not is_reparse(scratch),
            f"scratch namespace already exists: {scratch}")
    scratch.mkdir(parents=True, exist_ok=False)
    external = scratch / "external.txt"
    external.write_bytes(b"external-must-survive\n")
    source = scratch / "source"
    target = scratch / "target"
    target_second_unit = scratch / "target-second-unit"
    source.mkdir()
    target.mkdir()
    target_second_unit.mkdir()
    (source / "payload.txt").write_bytes(b"hardlink-payload\n")
    os.utime(source / "payload.txt", ns=(1_700_000_000_000_000_000,
                                          1_700_000_000_000_000_000))
    os.link(source / "payload.txt", source / "alias.txt")
    make_junction(source / "nested", scratch)
    copy_file_native(source / "payload.txt", target / "payload.txt")
    set_times_and_attrs(target / "payload.txt", metadata(source / "payload.txt"))
    os.link(target / "payload.txt", target / "alias.txt")
    os.link(target / "payload.txt", target_second_unit / "cross-unit-alias.txt")
    make_junction(target / "nested", scratch)
    before = sha_file(external)[0]
    require(sha_file(source / "payload.txt")[0] == sha_file(target / "payload.txt")[0] and
            file_identity(target / "payload.txt")["file_index"] ==
            file_identity(target / "alias.txt")["file_index"] and
            file_identity(target / "payload.txt")["file_index"] ==
            file_identity(target_second_unit / "cross-unit-alias.txt")["file_index"] and
            metadata(target / "payload.txt")["mtime_ns"] ==
            metadata(source / "payload.txt")["mtime_ns"], "scratch hardlink/hash/mtime")
    os.rmdir(target / "nested")
    require(external.is_file() and sha_file(external)[0] == before,
            "nested junction removal touched external file")
    git_init = subprocess.run(["git", "-C", str(scratch), "init", "-q"],
                              capture_output=True, text=True)
    require(git_init.returncode == 0, f"scratch git init: {git_init.stderr}")
    git_add = subprocess.run(["git", "-C", str(scratch), "add", "source/payload.txt"],
                             capture_output=True, text=True)
    require(git_add.returncode == 0, f"scratch git add: {git_add.stderr}")
    git_tracked = subprocess.run(["git", "-C", str(scratch), "ls-files",
                                  "--error-unmatch", "source/payload.txt"],
                                 capture_output=True, text=True)
    require(git_tracked.returncode == 0 and
            git_tracked.stdout.strip() == "source/payload.txt",
            "scratch tracked file not in index")
    backup = scratch / "backup"
    os.rename(source, backup)
    make_junction(source, target)
    require(os.path.isjunction(source) and
            normalized_junction_target(source) == lexical(target),
            "scratch cutover junction")
    git = subprocess.run(["git", "-C", str(scratch), "diff", "--exit-code",
                          "--", "source/payload.txt"],
                         capture_output=True, text=True)
    require(git.returncode == 0 and not git.stdout,
            f"scratch tracked subtree through junction: {git.returncode} {git.stderr} {git.stdout}")
    require((source / "payload.txt").read_bytes() == b"hardlink-payload\n",
            "scratch junction traversal")
    print(json.dumps({"status": "SCRATCH_PASS", "scratch": str(scratch),
                      "external_sha256": before,
                      "git_junction_result": "tracked subtree diff --exit-code 0 after cutover",
                      "resolve_change": str(source.resolve()) != str(source)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("prepare", "run", "status"):
        p = sub.add_parser(action)
        p.add_argument("--plan", required=True, type=Path)
        if action == "run":
            p.add_argument("--resume-verified", action="store_true")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.plan)
    elif args.command == "run":
        run(args.plan, args.resume_verified)
    elif args.command == "status":
        status(args.plan)
    else:
        selftest()


if __name__ == "__main__":
    main()
