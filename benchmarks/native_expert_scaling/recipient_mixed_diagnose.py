#!/usr/bin/env python3
"""Bounded first-divergence replay of the retained mixed-recipient failure.

Prepare freezes inputs; run compiles only a trace-enabled native probe and
replays the frozen decoded reference. No export, fitting, timing campaign or
quality admission is performed.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / "results/native_expert_scaling/recipient_mixed_baseline_20261015"
SITE = ROOT / "results/native_expert_scaling/chatbot_source_runtime/site"
sys.path.insert(0, str(SITE))
PROTOCOL = ROOT / "docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_DIAGNOSTIC_PROTOCOL_20261015.md"
OLD_REVISION = "4b78ac715a6498995c25d4519943a77c8d588258"
OLD_PROBE_SHA = "9dfdc44494987db9116608d70261af9e6790f8b5856a0214260077dbf8090d12"
OLD_ARCHIVE_SHA = "8bb77b499d76031a128cf8044889a7a772953e96dbe4c65b670e518ae95d00cb"
LIMIT_SECONDS, JOB_SECONDS = 1800.0, 900.0
RSS_LIMIT, OUTPUT_LIMIT = 8 << 30, 2 << 30
TRACE_HEAD = struct.Struct("<4I")
ORGANS = {1: "q", 2: "k", 3: "v", 4: "o", 5: "gate", 6: "up", 7: "down", 8: "head"}
EXTRAS = {90: "pre_qkv_norm", 91: "pre_ffn_norm", 92: "active", 93: "attention",
          98: "final_norm"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def current_bytes(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def old_failure():
    old_binding_path = ORIGINAL / "binding.json"
    old = load(old_binding_path)
    need(old["schema"] == "RECIPIENT_MIXED_BASELINE_V1" and
         old["git_head"] == OLD_REVISION and
         not (ORIGINAL / "result.json").exists(), "original failure identity")
    fault = load(ORIGINAL / "audit.json.failure.json")
    diag = fault["diagnostics"]
    need(fault["status"] == "FAIL" and diag["field_count"] == 291 and
         diag["input_sha256_before_after"]["archive"] == OLD_ARCHIVE_SHA and
         diag["input_sha256_before_after"]["probe"] == OLD_PROBE_SHA and
         sha(ORIGINAL / "probe.stdout") == OLD_PROBE_SHA and
         sha(ORIGINAL / "recipient.mixed.bin") == OLD_ARCHIVE_SHA,
         "original numerical failure evidence")
    return old, fault


def old_c_blob_sha():
    raw = subprocess.check_output(
        ["git", "show", OLD_REVISION + ":benchmarks/native_expert_scaling/recipient_qwen_entry.c"],
        cwd=ROOT)
    return sha_bytes(raw)


def prepare(directory):
    old, fault = old_failure()
    old_c = HERE / "recipient_qwen_entry.c"
    expected_old = next(x for x in old["inputs"] if Path(x["path"]) == old_c)
    need(old_c_blob_sha() == expected_old["sha256"], "original C Git blob does not match binding")
    for item in old["inputs"]:
        path = Path(item["path"])
        need(path.is_file(), f"missing old input: {path}")
        if path != old_c:
            need(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                 f"changed original input: {path}")
    current_c = sha(old_c)
    need(current_c != expected_old["sha256"] and
         b"RC_DIAGNOSTIC" in old_c.read_bytes(), "missing isolated C trace change")
    old_files = [current_bytes(p) for p in sorted(ORIGINAL.rglob("*")) if p.is_file()]
    need(any(Path(x["path"]).name == "audit.json.failure.json" for x in old_files),
         "original failure receipt absent")
    new_files = [current_bytes(p) for p in (
        Path(__file__).resolve(), old_c, HERE / "recipient_mixed_audit.py",
        ROOT / "benchmarks/phase60/engine.c", HERE / "meth388_switch_thread_binding.h",
        PROTOCOL, Path(old["compiler"]), Path(old["openmp_dll"]), Path(old["python"]))]
    need(len({x["path"] for x in new_files}) == len(new_files), "duplicate new input")
    directory.mkdir(parents=True, exist_ok=False)
    binding = {"schema": "RECIPIENT_MIXED_DIAGNOSTIC_V1", "original_dir": str(ORIGINAL),
               "original_binding_sha256": sha(ORIGINAL / "binding.json"),
               "original_failure_sha256": sha(ORIGINAL / "audit.json.failure.json"),
               "original_probe_sha256": OLD_PROBE_SHA, "original_archive_sha256": OLD_ARCHIVE_SHA,
               "original_C_git_revision": OLD_REVISION,
               "original_C_sha256": expected_old["sha256"],
               "diagnostic_C_sha256": current_c,
               "original_inputs": old["inputs"], "original_files": old_files,
               "diagnostic_inputs": new_files, "compiler": old["compiler"],
               "openmp_dll": old["openmp_dll"], "python": old["python"],
               "reference_logit_sha256": fault["diagnostics"]["reference_logits_sha256"],
               "limits": {"job_seconds": JOB_SECONDS, "total_seconds": LIMIT_SECONDS,
                          "host_bytes": RSS_LIMIT, "output_bytes": OUTPUT_LIMIT},
               "scope": "Trace-only replay of retained source-derived packet and failed probe"}
    write(directory / "binding.json", binding)
    print(json.dumps({"prepared": str(directory),
                      "binding_sha256": sha(directory / "binding.json"),
                      "original_files": len(old_files), "new_inputs": len(new_files)}), flush=True)


def verify_binding(binding):
    need(binding["schema"] == "RECIPIENT_MIXED_DIAGNOSTIC_V1" and
         binding["original_dir"] == str(ORIGINAL), "diagnostic binding schema")
    need(sha(ORIGINAL / "binding.json") == binding["original_binding_sha256"] and
         sha(ORIGINAL / "audit.json.failure.json") == binding["original_failure_sha256"],
         "original binding/failure changed")
    need(old_c_blob_sha() == binding["original_C_sha256"] and
         sha(HERE / "recipient_qwen_entry.c") == binding["diagnostic_C_sha256"],
         "old/new C identity")
    for item in binding["original_inputs"]:
        path = Path(item["path"])
        if path == HERE / "recipient_qwen_entry.c":
            continue
        need(path.is_file() and path.stat().st_size == item["bytes"] and
             sha(path) == item["sha256"], f"changed original input: {path}")
    for collection in (binding["original_files"], binding["diagnostic_inputs"]):
        for item in collection:
            path = Path(item["path"])
            need(path.is_file() and path.stat().st_size == item["bytes"] and
                 sha(path) == item["sha256"], f"changed retained file: {path}")
    need(sha(ORIGINAL / "probe.stdout") == OLD_PROBE_SHA and
         sha(ORIGINAL / "recipient.mixed.bin") == OLD_ARCHIVE_SHA,
         "original probe/archive changed")


def job(directory, binding, name, argv, started, request=None, trace=None):
    import psutil
    need(time.monotonic() - started <= LIMIT_SECONDS, "diagnostic family time cap")
    stdout = directory / f"{name}.stdout"
    stderr = directory / f"{name}.stderr"
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(("OMP_", "KMP_", "GOMP_", "SILICON_")):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_WAIT_POLICY="ACTIVE", KMP_BLOCKTIME="infinite", KMP_AFFINITY="none",
               OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1", OPENBLAS_NUM_THREADS="6",
               MKL_NUM_THREADS="6", OMP_NUM_THREADS="6")
    env["PATH"] = str(Path(binding["compiler"]).parent) + os.pathsep + env.get("PATH", "")
    if trace is not None:
        need(not trace.exists(), "trace output already exists")
        env["SILICON_RECIPIENT_TRACE"] = str(trace)
    t = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        stdin = Path(request).open("rb") if request else subprocess.DEVNULL
        try:
            child = subprocess.Popen(list(map(str, argv)), stdin=stdin, stdout=out, stderr=err,
                                     cwd=ROOT, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        finally:
            if request:
                stdin.close()
        proc = psutil.Process(child.pid)
        created = proc.create_time()
        write(directory / f"{name}.owned.json", {"pid": child.pid, "create_time": created,
              "argv": list(map(str, argv)), "request": str(request) if request else None})
        peak = 0
        try:
            while child.poll() is None:
                try:
                    peak = max(peak, proc.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                need(peak <= RSS_LIMIT, f"{name}: RSS cap")
                need(time.monotonic() - t <= JOB_SECONDS, f"{name}: time cap")
                need(time.monotonic() - started <= LIMIT_SECONDS, "family time cap")
                time.sleep(0.1)
        except BaseException:
            if child.poll() is None and proc.create_time() == created:
                child.kill()
                child.wait()
            raise
    receipt = {"name": name, "argv": list(map(str, argv)), "exit_code": child.returncode,
               "seconds": time.monotonic() - t, "peak_rss_observed": peak,
               "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr),
               "request_sha256": sha(request) if request else None,
               "trace_sha256": sha(trace) if trace and trace.exists() else None}
    write(directory / f"{name}.terminal.json", receipt)
    need(child.returncode == 0, f"{name}: exit {child.returncode}")
    return receipt


def auditor_module():
    spec = importlib.util.spec_from_file_location(
        "recipient_mixed_audit_frozen", HERE / "recipient_mixed_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace_reference(directory):
    # Child-only numerical path. The parent controller does not import torch.
    A = auditor_module()
    import numpy as np
    import psutil
    import torch
    from safetensors import safe_open

    started = time.monotonic()
    process = psutil.Process()
    binding = load(directory / "binding.json")
    original = Path(binding["original_dir"])
    old = load(original / "binding.json")
    source_path = Path(old["source_directory"]) / "model.safetensors"
    catalog = A.Catalog(original / "manifest.json", original / "recipient_catalog.h",
                        original / "recipient.mixed.bin", OLD_ARCHIVE_SHA, source_path,
                        source_path.parent / "config.json", A.roles(), lambda: None)
    ids = np.fromfile(old["parity_ids"], dtype="<i4")
    need(len(ids) == 32, "reference ID count")
    reference_path = directory / "reference.trace"
    peak = 0

    def guard():
        nonlocal peak
        peak = max(peak, process.memory_info().rss)
        need(time.monotonic() - started <= JOB_SECONDS, "reference time cap")
        need(peak <= RSS_LIMIT, "reference RSS cap")

    class TracedReference(A.Reference):
        def __init__(self, cat, source, check, stream):
            super().__init__(cat, source, check)
            self.stream = stream
            self.position = -1
            self.last_input = None

        def record(self, layer, tag, values):
            if self.position >= 8:
                return
            values = np.asarray(values, dtype="<f4").reshape(-1)
            self.stream.write(TRACE_HEAD.pack(self.position, layer, tag, values.size))
            self.stream.write(values.tobytes())

        def quant(self, x, group):
            self.last_input = np.asarray(x, dtype=np.float32).copy()
            return A.Reference.quant(x, group)

        def project(self, name, q, scales):
            if name == "lm_head.weight":
                organ, layer = 8, 24
            else:
                match = re.fullmatch(
                    r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.([a-z]+)_proj\.weight", name)
                need(match is not None, f"trace projection name: {name}")
                layer = int(match.group(1))
                organ = {"q": 1, "k": 2, "v": 3, "o": 4,
                         "gate": 5, "up": 6, "down": 7}[match.group(2)]
                if layer == 0 and organ == 1:
                    self.position += 1
            if self.position < 8:
                extra = {1: 90, 4: 93, 5: 91, 7: 92}.get(organ)
                if extra is not None:
                    need(self.last_input is not None, "missing prequant input")
                    self.record(layer, extra, self.last_input)
                self.record(layer, organ * 10, q)
                self.record(layer, organ * 10 + 1, scales)
            value = super().project(name, q, scales)
            if self.position < 8 and organ != 8:
                self.record(layer, organ * 10 + 2, value)
            return value

        def head(self, norm):
            if self.position < 8:
                self.record(24, 98, norm)
            logits, top = super().head(norm)
            if self.position < 8:
                self.record(24, 82, logits)
            return logits, top

    with safe_open(str(source_path), framework="pt", device="cpu") as source:
        with reference_path.open("xb") as stream:
            model = TracedReference(catalog, source, guard, stream)
            _, logits, top8 = model.run(ids)
            stream.flush()
            os.fsync(stream.fileno())
    hashes = {str(position): sha_bytes(logits[position].astype("<f4").tobytes())
              for position in A.POSITIONS}
    need(hashes == binding["reference_logit_sha256"],
         "traced reference trajectory differs from failed audit")
    write(directory / "reference.meta.json",
          {"schema": "RECIPIENT_MIXED_REFERENCE_TRACE_V1", "logit_sha256": hashes,
           "top8": top8, "trace_sha256": sha(reference_path),
           "trace_bytes": reference_path.stat().st_size,
           "peak_sampled_rss_bytes": peak, "seconds": time.monotonic() - started})


def records(path):
    with path.open("rb") as f:
        while True:
            head = f.read(TRACE_HEAD.size)
            if not head:
                return
            need(len(head) == TRACE_HEAD.size, f"truncated trace header: {path}")
            position, layer, tag, count = TRACE_HEAD.unpack(head)
            need(position < 8 and layer <= 24 and 1 <= count <= 151936,
                 f"trace record geometry: {path}")
            raw = f.read(count * 4)
            need(len(raw) == count * 4, f"truncated trace values: {path}")
            yield (position, layer, tag), raw


def key_name(key):
    position, layer, tag = key
    if tag in EXTRAS:
        name = EXTRAS[tag]
    else:
        organ = ORGANS.get(tag // 10, "unknown")
        stage = {0: "activation_codes", 1: "activation_scales", 2: "projection_output"}.get(tag % 10, "unknown")
        name = organ + "." + stage
    return f"position={position},layer={layer},tag={tag},{name}"


def compare_traces(native_path, reference_path):
    import numpy as np

    native_all = list(records(native_path))
    reference = list(records(reference_path))
    need(len(native_all) > len(reference) and len(reference) >= 4000,
         "trace record census")
    native = [(key, raw) for key, raw in native_all
              if key[2] < 90 or (key[2] in EXTRAS and
                                  (key[2] != 98 or key[0] in (0, 7)))]
    need([key for key, _ in native] == [key for key, _ in reference],
         "native/reference trace tag/order mismatch")
    native_lookup = dict(native)
    reference_lookup = dict(reference)
    need(len(native_lookup) == len(native) and len(reference_lookup) == len(reference),
         "duplicate comparable trace tag")
    extras_count = {}
    for key, _ in native_all:
        extras_count[str(key[2])] = extras_count.get(str(key[2]), 0) + 1
    first_code = None
    first_scale = None
    first_numeric_1e3 = None
    first_bit_difference = None
    records_out = []
    for (key, raw_native), (_, raw_ref) in zip(native, reference):
        a = np.frombuffer(raw_native, dtype="<f4")
        b = np.frombuffer(raw_ref, dtype="<f4")
        need(len(a) == len(b) and np.isfinite(a).all() and np.isfinite(b).all(),
             f"trace finite/length: {key_name(key)}")
        mismatch = np.flatnonzero(a != b)
        if len(mismatch) and first_bit_difference is None:
            first_bit_difference = key_name(key)
        row = {"position": key[0], "layer": key[1], "tag": key[2],
               "stage": key_name(key), "count": len(a),
               "unequal_f32": int(len(mismatch)),
               "first_unequal_index": int(mismatch[0]) if len(mismatch) else None}
        if key[2] < 90 and key[2] % 10 == 0:
            need(np.array_equal(a, np.rint(a)) and np.array_equal(b, np.rint(b)),
                 f"noninteger activation codes: {key_name(key)}")
            row["code_mismatches"] = int(len(mismatch))
            if len(mismatch) and first_code is None:
                first_code = {**row, "native_code": float(a[mismatch[0]]),
                              "reference_code": float(b[mismatch[0]])}
                organ = key[2] // 10
                pretag = {1: 90, 2: 90, 3: 90, 4: 93,
                          5: 91, 6: 91, 7: 92, 8: 98}[organ]
                prekey = (key[0], key[1], pretag)
                scalekey = (key[0], key[1], key[2] + 1)
                if prekey in native_lookup and scalekey in native_lookup:
                    offset = int(mismatch[0])
                    group = offset // 32 if organ in (5, 6, 7) else 0
                    xn = np.frombuffer(native_lookup[prekey], dtype="<f4")[offset]
                    xr = np.frombuffer(reference_lookup[prekey], dtype="<f4")[offset]
                    sn = np.frombuffer(native_lookup[scalekey], dtype="<f4")[group]
                    sr = np.frombuffer(reference_lookup[scalekey], dtype="<f4")[group]
                    def half_margin(x, s):
                        ratio = float(x) / float(s)
                        return abs(ratio - (math.floor(ratio) + 0.5))
                    first_code["prequant"] = {
                        "native_input": float(xn), "reference_input": float(xr),
                        "native_scale": float(sn), "reference_scale": float(sr),
                        "native_half_step_margin": half_margin(xn, sn),
                        "reference_half_step_margin": half_margin(xr, sr)}
        else:
            da, db = a.astype(np.float64), b.astype(np.float64)
            denom = max(float(np.linalg.norm(db)), 1e-12)
            relative = float(np.linalg.norm(da - db) / denom)
            row["relative_l2"] = relative
            row["maximum_absolute_difference"] = float(np.max(np.abs(da - db)))
            if key[2] < 90 and key[2] % 10 == 1 and len(mismatch) and first_scale is None:
                first_scale = row
            if relative > 1e-3 and first_numeric_1e3 is None:
                first_numeric_1e3 = row
        records_out.append(row)
    by_position = {}
    for pos in range(8):
        selected = [row for row in records_out if row["position"] == pos]
        by_position[str(pos)] = {
            "records": len(selected),
            "activation_code_mismatches": sum(row.get("code_mismatches", 0) for row in selected),
            "worst_relative_l2": max((row.get("relative_l2", 0.0) for row in selected),
                                      default=0.0),
            "first_code_stage": next((row["stage"] for row in selected
                                      if row.get("code_mismatches", 0)), None)}
    return {"native_records": len(native_all), "compared_records": len(records_out),
            "native_tags": extras_count, "first_f32_difference": first_bit_difference,
            "first_activation_code_difference": first_code,
            "first_activation_scale_difference": first_scale,
            "first_relative_l2_gt_1e-3": first_numeric_1e3,
            "by_position": by_position, "records": records_out}


def run(directory):
    import psutil

    binding_path = directory / "binding.json"
    binding = load(binding_path)
    need(not (directory / "started.json").exists() and
         not (directory / "result.json").exists(), "diagnostic already started")
    verify_binding(binding)
    write(directory / "started.json", {"wall_time": time.time(),
                                        "binding_sha256": sha(binding_path)})
    started = time.monotonic()
    commands = []
    original = Path(binding["original_dir"])
    archive = original / "recipient.mixed.bin"
    header = original / "recipient_catalog.h"
    exe = directory / "recipient.diagnostic.exe"
    trace = directory / "native.trace"
    source = ROOT / "benchmarks/phase60/engine.c"
    compiler = binding["compiler"]
    try:
        compile_argv = [compiler, "-O3", "-std=c11", "-march=x86-64-v3",
                        "-ffp-contract=off", "-fopenmp", "-Wl,--no-insert-timestamp",
                        "-DSILICON_QWEN_RECIPIENT_ENTRY", "-DRC_DIAGNOSTIC",
                        '-DRECIPIENT_CATALOG="' + str(header).replace('\\', '/') + '"',
                        source, "-o", exe, "-lbcrypt", "-lm"]
        commands.append(job(directory, binding, "compile", compile_argv, started))
        request = original / "probe.request"
        commands.append(job(directory, binding, "probe",
                            [exe, archive, "probe", "2176", "6"], started,
                            request=request, trace=trace))
        need(sha(directory / "probe.stdout") == binding["original_probe_sha256"],
             "instrumented probe trajectory differs from retained failure")
        reference_argv = [binding["python"], "-I", "-S", "-B", "-X", "utf8",
                          Path(__file__).resolve(), "reference", "--directory", directory]
        commands.append(job(directory, binding, "reference", reference_argv, started))
        metadata = load(directory / "reference.meta.json")
        need(metadata["logit_sha256"] == binding["reference_logit_sha256"] and
             sha(directory / "reference.trace") == metadata["trace_sha256"],
             "reference trajectory differs from retained failure")
        comparison = compare_traces(trace, directory / "reference.trace")
        need(time.monotonic() - started <= LIMIT_SECONDS and
             psutil.Process().memory_info().rss <= RSS_LIMIT, "controller resource cap")
        output_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        need(output_bytes <= OUTPUT_LIMIT, "diagnostic output cap")
        verify_binding(binding)
        result = {"schema": "RECIPIENT_MIXED_DIAGNOSTIC_V1", "status": "TRACE_COMPARISON_COMPLETE",
                  "binding_sha256": sha(binding_path), "original_probe_sha256": OLD_PROBE_SHA,
                  "instrumented_probe_sha256": sha(directory / "probe.stdout"),
                  "original_reference_logit_sha256": binding["reference_logit_sha256"],
                  "instrumented_reference_logit_sha256": metadata["logit_sha256"],
                  "archive_sha256": sha(archive), "diagnostic_executable_sha256": sha(exe),
                  "native_trace_sha256": sha(trace),
                  "reference_trace_sha256": sha(directory / "reference.trace"),
                  "commands": commands, "trace_comparison": comparison,
                  "seconds": time.monotonic() - started, "output_bytes": output_bytes,
                  "quality_admitted": False, "cost_admitted": False, "goal_complete": False}
        write(directory / "result.json", result)
        print(json.dumps({"status": result["status"],
                          "first_code": comparison["first_activation_code_difference"],
                          "first_numeric": comparison["first_relative_l2_gt_1e-3"]}), flush=True)
    except BaseException as error:
        write(directory / "failure.json", {"error_type": type(error).__name__,
              "error": str(error), "commands": commands,
              "seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "reference"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == "prepare":
        prepare(directory)
    elif args.mode == "run":
        run(directory)
    else:
        trace_reference(directory)


if __name__ == "__main__":
    main()
