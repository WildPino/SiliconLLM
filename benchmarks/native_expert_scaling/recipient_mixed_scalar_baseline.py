#!/usr/bin/env python3
"""One sealed scalar-reference continuation of the failed mixed recipient.

Reuse the original archive and executable. A byte-identical probe and the
unchanged numerical gates must pass before any retained cost case is run.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE = ROOT / "results/native_expert_scaling/chatbot_source_runtime/site"
sys.path.insert(0, str(SITE))
import psutil

ORIGINAL = ROOT / "results/native_expert_scaling/recipient_mixed_baseline_20261015"
DIAG = ROOT / "results/native_expert_scaling/recipient_mixed_diagnostic_20261015"
PROTOCOL = ROOT / "docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_SCALAR_REPAIR_PROTOCOL_20261015.md"
OLD_REVISION = "4b78ac715a6498995c25d4519943a77c8d588258"
OLD_ARCHIVE_SHA = "8bb77b499d76031a128cf8044889a7a772953e96dbe4c65b670e518ae95d00cb"
OLD_PROBE_SHA = "9dfdc44494987db9116608d70261af9e6790f8b5856a0214260077dbf8090d12"
JOB_LIMIT, TOTAL_LIMIT = 900.0, 2400.0
RSS_LIMIT, OUTPUT_LIMIT = 8 << 30, 2 << 30


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def original_c_sha():
    raw = subprocess.check_output(
        ["git", "show", OLD_REVISION + ":benchmarks/native_expert_scaling/recipient_qwen_entry.c"],
        cwd=ROOT)
    return hashlib.sha256(raw).hexdigest()


def original_evidence():
    old = load(ORIGINAL / "binding.json")
    need(old["schema"] == "RECIPIENT_MIXED_BASELINE_V1" and
         old["git_head"] == OLD_REVISION and
         not (ORIGINAL / "result.json").exists(), "original failed attempt identity")
    failed = load(ORIGINAL / "audit.json.failure.json")
    need(failed["status"] == "FAIL" and
         failed["diagnostics"]["field_count"] == 291 and
         failed["diagnostics"]["gates"]["all_selected_argmax_equal"] is False,
         "original numerical failure absent")
    need(sha(ORIGINAL / "recipient.mixed.bin") == OLD_ARCHIVE_SHA and
         sha(ORIGINAL / "probe.stdout") == OLD_PROBE_SHA, "original archive/probe seal")
    old_c = HERE / "recipient_qwen_entry.c"
    c_entry = next((x for x in old["inputs"] if Path(x["path"]) == old_c), None)
    need(c_entry is not None and original_c_sha() == c_entry["sha256"],
         "original release C Git blob")
    for x in old["inputs"]:
        path = Path(x["path"])
        if path == old_c:
            continue  # Current C contains later trace guards; old Git blob is pinned above.
        need(path.is_file() and path.stat().st_size == x["bytes"] and
             sha(path) == x["sha256"], f"original input changed: {path}")
    for name in ("export", "compile", "probe"):
        terminal = load(ORIGINAL / (name + ".terminal.json"))
        need(terminal["exit_code"] == 0 and
             sha(ORIGINAL / (name + ".stdout")) == terminal["stdout_sha256"] and
             sha(ORIGINAL / (name + ".stderr")) == terminal["stderr_sha256"],
             f"original command receipt: {name}")
    return old, c_entry


def prepare(directory):
    old, old_c = original_evidence()
    need(PROTOCOL.is_file(), "scalar continuation protocol not saved")
    diagnostic = load(DIAG / "result.json")
    need(diagnostic["status"] == "TRACE_COMPARISON_COMPLETE" and
         diagnostic["original_probe_sha256"] == OLD_PROBE_SHA and
         diagnostic["instrumented_probe_sha256"] == OLD_PROBE_SHA and
         diagnostic["original_reference_logit_sha256"] ==
         diagnostic["instrumented_reference_logit_sha256"] ==
         load(ORIGINAL / "audit.json.failure.json")["diagnostics"]["reference_logits_sha256"],
         "diagnostic trajectory was not reproduced")
    check = DIAG / "stored_silu_check.json"
    stored = load(check)
    need(stored["gate_and_up_native_reference_identical"] is True and
         stored["elements"] == 4864 and
         stored["double_math_exp_rounded_f32_vs_native_mismatches"] == 0 and
         stored["native_vs_reference_mismatches"] == 814,
         "stored SiLU check does not support this reference repair")
    all_sites_path = DIAG / "stored_silu_all_sites.json"
    all_sites = load(all_sites_path)
    need(all_sites["schema"] == "RECIPIENT_MIXED_STORED_SILU_ALL_SITES_V1" and
         all_sites["status"] == "SCALAR_EXP_ALL_NATIVE_SITES_MATCH" and
         (all_sites["sites"], all_sites["elements"], all_sites["bitwise_mismatches"]) ==
         (192, 933888, 0) and
         all_sites["input_sha256_before_after"]["trace"] ==
         diagnostic["native_trace_sha256"] and
         all_sites["input_sha256_before_after"]["result"] == sha(DIAG / "result.json") and
         all_sites["input_sha256_before_after"]["reducer"] ==
         sha(HERE / "recipient_mixed_silu_check.py"),
         "stored all-site SiLU reduction did not pass")
    need([(c["name"], c["tokens"], c["new_tokens"]) for c in old["cases"]] ==
         [("prompt_0", 39, 64), ("prompt_1", 54, 64), ("prompt_2", 38, 64),
          ("context_512", 512, 64), ("context_2048", 2048, 64)],
         "original five cost cases changed")
    need(old["gates"] == {"worst_forward_seconds": .018,
                          "maximum_case_rate_spread": 1.10,
                          "probe_top1_exact": True, "probe_relative_l2": .001},
         "original cost/numerical gates changed")
    original_files = [record(p) for p in sorted(ORIGINAL.rglob("*")) if p.is_file()]
    diagnostic_files = [record(p) for p in sorted(DIAG.rglob("*")) if p.is_file()]
    new_inputs = [record(p) for p in (
        Path(__file__).resolve(), HERE / "recipient_mixed_scalar_audit.py",
        HERE / "recipient_mixed_audit.py", HERE / "recipient_mixed_baseline.py",
        HERE / "recipient_mixed_silu_check.py", HERE / "recipient_qwen_entry.c",
        PROTOCOL, check, all_sites_path,
        Path(old["python"]), Path(old["openmp_dll"]))]
    need(len({x["path"] for x in new_inputs}) == len(new_inputs), "duplicate continuation input")
    directory.mkdir(parents=True, exist_ok=False)
    binding = {"schema": "RECIPIENT_MIXED_SCALAR_CONTINUATION_V1",
               "original_dir": str(ORIGINAL), "diagnostic_dir": str(DIAG),
               "original_binding_sha256": sha(ORIGINAL / "binding.json"),
               "original_audit_failure_sha256": sha(ORIGINAL / "audit.json.failure.json"),
               "original_archive_sha256": OLD_ARCHIVE_SHA,
               "original_probe_sha256": OLD_PROBE_SHA,
               "original_C_git_revision": OLD_REVISION,
               "original_C_sha256": old_c["sha256"],
               "current_trace_C_sha256": sha(HERE / "recipient_qwen_entry.c"),
               "original_compile_terminal_sha256": sha(ORIGINAL / "compile.terminal.json"),
               "original_executable_sha256": sha(ORIGINAL / "recipient.mixed.exe"),
               "original_inputs": old["inputs"], "original_files": original_files,
               "diagnostic_files": diagnostic_files, "continuation_inputs": new_inputs,
               "stored_silu_check_sha256": sha(check),
               "stored_silu_all_sites_sha256": sha(all_sites_path),
               "cases": old["cases"], "probe_request": old["probe_request"],
               "parity_ids": old["parity_ids"], "source_directory": old["source_directory"],
               "source_sha256": old["source_sha256"], "python": old["python"],
               "compiler": old["compiler"], "openmp_dll": old["openmp_dll"],
               "limits": {"job_seconds": JOB_LIMIT, "total_seconds": TOTAL_LIMIT,
                          "host_bytes": RSS_LIMIT, "output_bytes": OUTPUT_LIMIT},
               "gates": old["gates"], "quality_admitted": False,
               "donor_transfer_admitted": False}
    write(directory / "binding.json", binding)
    print(json.dumps({"prepared": str(directory),
                      "binding_sha256": sha(directory / "binding.json"),
                      "original_files": len(original_files),
                      "diagnostic_files": len(diagnostic_files)}), flush=True)


def verify(binding):
    need(binding["schema"] == "RECIPIENT_MIXED_SCALAR_CONTINUATION_V1" and
         binding["original_dir"] == str(ORIGINAL) and
         binding["diagnostic_dir"] == str(DIAG), "continuation binding")
    old = load(ORIGINAL / "binding.json")
    need(binding["cases"] == old["cases"] and
         binding["probe_request"] == old["probe_request"] and
         binding["parity_ids"] == old["parity_ids"] and
         binding["gates"] == old["gates"] ==
         {"worst_forward_seconds": .018, "maximum_case_rate_spread": 1.10,
          "probe_top1_exact": True, "probe_relative_l2": .001},
         "case/request/gate binding changed")
    need(sha(ORIGINAL / "binding.json") == binding["original_binding_sha256"] and
         sha(ORIGINAL / "audit.json.failure.json") == binding["original_audit_failure_sha256"] and
         original_c_sha() == binding["original_C_sha256"] and
         sha(HERE / "recipient_qwen_entry.c") == binding["current_trace_C_sha256"],
         "original/current source identity")
    for item in binding["original_inputs"]:
        path = Path(item["path"])
        if path == HERE / "recipient_qwen_entry.c":
            continue
        need(path.is_file() and path.stat().st_size == item["bytes"] and
             sha(path) == item["sha256"], f"original input changed: {path}")
    for collection in (binding["original_files"], binding["diagnostic_files"],
                       binding["continuation_inputs"]):
        for item in collection:
            path = Path(item["path"])
            need(path.is_file() and path.stat().st_size == item["bytes"] and
                 sha(path) == item["sha256"], f"sealed input changed: {path}")
    need(sha(ORIGINAL / "recipient.mixed.bin") == binding["original_archive_sha256"] and
         sha(ORIGINAL / "probe.stdout") == binding["original_probe_sha256"] and
         sha(ORIGINAL / "recipient.mixed.exe") == binding["original_executable_sha256"] and
         sha(ORIGINAL / "compile.terminal.json") == binding["original_compile_terminal_sha256"],
         "retained release artifact/source receipt changed")


def environment():
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(("OMP_", "KMP_", "GOMP_", "SILICON_")):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_WAIT_POLICY="ACTIVE", KMP_BLOCKTIME="infinite", KMP_AFFINITY="none",
               OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1", OPENBLAS_NUM_THREADS="6",
               MKL_NUM_THREADS="6", OMP_NUM_THREADS="6")
    return env


def parse_generation(raw, log, context, maximum):
    # Reuse the frozen original reduction; this continuation changes no cost parser.
    spec = importlib.util.spec_from_file_location(
        "recipient_mixed_baseline_frozen", HERE / "recipient_mixed_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_generation(raw, log, context, maximum)


def run(directory):
    binding_path = directory / "binding.json"
    binding = load(binding_path)
    need(not (directory / "started.json").exists(), "continuation already started")
    verify(binding)
    write(directory / "started.json", {"time": time.time(),
                                        "binding_sha256": sha(binding_path)})
    started = time.monotonic()
    commands = []
    env = environment()
    env["PATH"] = str(Path(binding["compiler"]).parent) + os.pathsep + env.get("PATH", "")
    exe = ORIGINAL / "recipient.mixed.exe"
    archive = ORIGINAL / "recipient.mixed.bin"
    header = ORIGINAL / "recipient_catalog.h"
    manifest = ORIGINAL / "manifest.json"
    source = Path(binding["source_directory"]) / "model.safetensors"

    def job(name, argv, request=None):
        need(time.monotonic() - started < TOTAL_LIMIT, "continuation total time cap")
        t = time.monotonic()
        stdout, stderr = directory / f"{name}.stdout", directory / f"{name}.stderr"
        with stdout.open("xb") as out, stderr.open("xb") as err:
            stdin = Path(request).open("rb") if request else subprocess.DEVNULL
            try:
                child = subprocess.Popen(list(map(str, argv)), stdin=stdin, stdout=out,
                                         stderr=err, cwd=ROOT, env=env,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            finally:
                if request:
                    stdin.close()
            proc = psutil.Process(child.pid)
            created = proc.create_time()
            write(directory / f"{name}.owned.json", {"pid": child.pid,
                  "create_time": created, "argv": list(map(str, argv)),
                  "request": str(request) if request else None})
            peak = 0
            try:
                while child.poll() is None:
                    try:
                        peak = max(peak, proc.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    need(peak <= RSS_LIMIT, f"{name}: RSS cap")
                    need(time.monotonic() - t <= JOB_LIMIT, f"{name}: time cap")
                    need(time.monotonic() - started <= TOTAL_LIMIT, "total time cap")
                    time.sleep(.1)
            except BaseException:
                if child.poll() is None and proc.create_time() == created:
                    child.kill()
                    child.wait()
                raise
        receipt = {"name": name, "argv": list(map(str, argv)), "exit_code": child.returncode,
                   "seconds": time.monotonic() - t, "peak_rss_observed": peak,
                   "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr),
                   "request_sha256": sha(request) if request else None}
        write(directory / f"{name}.terminal.json", receipt)
        commands.append(receipt)
        print(json.dumps({"stage": name, "exit_code": child.returncode,
                          "seconds": receipt["seconds"]}), flush=True)
        need(child.returncode == 0, f"{name}: exit {child.returncode}")
        return stdout, stderr

    try:
        probe, _ = job("probe", [exe, archive, "probe", "2176", "6"],
                       binding["probe_request"])
        need(sha(probe) == OLD_PROBE_SHA, "replayed native probe differs from original failure")
        py = [binding["python"], "-I", "-S", "-B", "-X", "utf8"]
        job("audit", py + [HERE / "recipient_mixed_scalar_audit.py", "--source", source,
            "--archive", archive, "--archive-sha", OLD_ARCHIVE_SHA,
            "--catalog", header, "--manifest", manifest,
            "--ids", binding["parity_ids"], "--request", binding["probe_request"],
            "--probe", probe, "--out", directory / "audit.json"])
        audit = load(directory / "audit.json")
        need(audit["status"] == "PASS" and audit["field_count"] == 291 and
             audit["source_tensor_count"] == 290 and audit["ids_count"] == 32 and
             audit["selected_positions"] == [0, 7, 15, 31] and
             all(audit["gates"].values()) and
             audit["input_sha256_before_after"]["archive"] == OLD_ARCHIVE_SHA and
             audit["input_sha256_before_after"]["probe"] == OLD_PROBE_SHA,
             "scalar reference failed unchanged numerical gates")
        results = []
        for repetition in range(3):
            for case in binding["cases"]:
                out, err = job(f'generate_r{repetition}_{case["name"]}',
                               [exe, archive, "stream", "2176", "6"], case["request"])
                parsed = parse_generation(out.read_bytes(), err.read_text(encoding="utf-8"),
                                          case["tokens"], case["new_tokens"])
                results.append({"case": case["name"], "repetition": repetition, **parsed})
        verify(binding)
        by_case = {}
        for case in binding["cases"]:
            rows = [row for row in results if row["case"] == case["name"]]
            need(len(rows) == 3 and all(row["emitted_ids"] == rows[0]["emitted_ids"]
                                        for row in rows), f"repeat IDs: {case['name']}")
            rates = [row["native_decode_forwards"] / row["decode_s"] for row in rows]
            by_case[case["name"]] = {"rates": rates, "median": statistics.median(rates),
                                     "worst": min(rates), "worst_ms": 1000 / min(rates),
                                     "spread": max(rates) / min(rates)}
        passed = all(row["worst_ms"] <= 18 and row["spread"] <= 1.10
                     for row in by_case.values())
        output_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        need(output_bytes <= OUTPUT_LIMIT and time.monotonic() - started <= TOTAL_LIMIT,
             "continuation output/time cap")
        result = {"schema": "RECIPIENT_MIXED_SCALAR_CONTINUATION_V1",
                  "status": "MIXED_COST_ENVELOPE_PASS" if passed else "MIXED_COST_ENVELOPE_FAIL",
                  "binding_sha256": sha(binding_path), "original_archive_path": str(archive),
                  "artifact_sha256": sha(archive), "artifact_bytes": archive.stat().st_size,
                  "original_executable_path": str(exe), "executable_sha256": sha(exe),
                  "catalog_sha256": sha(header), "manifest_sha256": sha(manifest),
                  "original_failure_sha256": binding["original_audit_failure_sha256"],
                  "original_probe_sha256": OLD_PROBE_SHA,
                  "replayed_probe_sha256": sha(probe), "audit_sha256": sha(directory / "audit.json"),
                  "commands": commands, "cases": results, "cost_summary": by_case,
                  "seconds": time.monotonic() - started, "output_bytes": output_bytes,
                  "active_matrix_payload_bytes_per_forward": 357484032,
                  "physical_dram_measured": False, "quality_admitted": False,
                  "donor_transfer_admitted": False, "goal_complete": False}
        write(directory / "result.json", result)
        print(json.dumps({"status": result["status"], "cost_summary": by_case}), flush=True)
    except BaseException as error:
        write(directory / "failure.json", {"error_type": type(error).__name__,
              "error": str(error), "commands": commands,
              "seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else run)(args.directory.resolve())


if __name__ == "__main__":
    main()
