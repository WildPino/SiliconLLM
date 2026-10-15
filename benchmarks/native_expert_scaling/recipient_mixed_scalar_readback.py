#!/usr/bin/env python3
"""Independent readback of one retained mixed-recipient scalar continuation.

Reads sealed files and process receipts only. It imports no producer parser,
executes no model, and does not change the continuation's admission gates.
CLI: --directory RETAINED_RESULT_DIR --out EXCLUSIVE_REPORT_JSON
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "results/native_expert_scaling/chatbot_source_runtime/site"
sys.path.insert(0, str(SITE))
import psutil


SCHEMA = "RECIPIENT_MIXED_SCALAR_CONTINUATION_V1"
OLD_REVISION = "4b78ac715a6498995c25d4519943a77c8d588258"
OLD_ARCHIVE_SHA = "8bb77b499d76031a128cf8044889a7a772953e96dbe4c65b670e518ae95d00cb"
OLD_PROBE_SHA = "9dfdc44494987db9116608d70261af9e6790f8b5856a0214260077dbf8090d12"
EOS = (151645, 151643)
MASKS = [1, 4, 16, 64, 256, 1024]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def near(actual, reported, label):
    need(math.isfinite(actual) and math.isfinite(reported) and
         math.isclose(actual, reported, rel_tol=1e-11, abs_tol=1e-8), label)


def check_file_record(record, label):
    path = Path(record["path"])
    need(path.is_file() and path.stat().st_size == record["bytes"] and
         sha(path) == record["sha256"], f"changed {label}: {path}")


def old_c_git_sha():
    raw = subprocess.check_output(
        ["git", "show", OLD_REVISION + ":benchmarks/native_expert_scaling/recipient_qwen_entry.c"],
        cwd=ROOT)
    return hashlib.sha256(raw).hexdigest()


def check_owned(directory, name, terminal, expected_argv, expected_request):
    owned = load(directory / f"{name}.owned.json")
    need(owned["argv"] == expected_argv and terminal["argv"] == expected_argv and
         owned["request"] == expected_request, f"owned command: {name}")
    pid, created = owned["pid"], owned["create_time"]
    need(isinstance(pid, int) and pid > 0 and isinstance(created, (float, int)) and
         created > 0, f"owned process identity: {name}")
    try:
        current = psutil.Process(pid)
        same_process = abs(current.create_time() - created) < 0.001
    except psutil.NoSuchProcess:
        same_process = False
    need(not same_process, f"owned process still live: {name}")


def check_command(directory, name, command, expected_argv, request):
    terminal = load(directory / f"{name}.terminal.json")
    need(command == terminal and terminal["name"] == name and
         terminal["exit_code"] == 0 and terminal["seconds"] > 0 and
         0 <= terminal["peak_rss_observed"] <= 8 << 30 and
         terminal["seconds"] <= 900, f"terminal receipt: {name}")
    stdout = directory / f"{name}.stdout"
    stderr = directory / f"{name}.stderr"
    need(sha(stdout) == terminal["stdout_sha256"] and
         sha(stderr) == terminal["stderr_sha256"], f"command log SHA: {name}")
    expected_request = str(request) if request is not None else None
    need(terminal["request_sha256"] == (sha(request) if request is not None else None),
         f"request SHA: {name}")
    check_owned(directory, name, terminal, expected_argv, expected_request)
    return stdout, stderr


def decode_stream(stdout, stderr, prompt_tokens, maximum):
    raw = stdout.read_bytes()
    need(len(raw) >= 28 and raw[:8] == b"QWCO0001", "QWCO frame")
    count, stop, elapsed = struct.unpack_from("<IId", raw, 8)
    need(1 <= count <= maximum and stop in (0, 1) and
         len(raw) == 24 + 4 * count and math.isfinite(elapsed) and elapsed > 0,
         "QWCO extent/elapsed")
    ids = list(struct.unpack_from("<" + "i" * count, raw, 24))
    need(all(0 <= token < 151936 for token in ids) and
         all(token not in EOS for token in ids[:-1]) and
         stop == int(ids[-1] in EOS) and (stop or count == maximum),
         "QWCO generation/stop")
    events = [json.loads(line) for line in stderr.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    need([event.get("phase") for event in events] ==
         ["binding_start", "prefill"] + ["decode"] * count + ["binding_end"],
         "stderr event sequence")
    for binding in (events[0], events[-1]):
        need(binding.get("worker_masks") == MASKS and binding.get("pid") > 0,
             "worker affinity masks")
    need(events[0]["pid"] == events[-1]["pid"], "worker process continuity")
    prefill = events[1]
    need(prefill["context"] == prompt_tokens and prefill["threads"] == 6 and
         math.isfinite(prefill["seconds"]) and prefill["seconds"] > 0,
         "prefill event")
    steps = events[2:-1]
    for index, event in enumerate(steps):
        need((event["index"], event["generated_id"], event["context"], event["threads"]) ==
             (index, ids[index], prompt_tokens + index, 6) and
             event["native_forward"] is (ids[index] not in EOS) and
             math.isfinite(event["seconds"]) and event["seconds"] >= 0,
             f"decode event {index}")
    charged = sum(event["native_forward"] for event in steps)
    decode_seconds = sum(event["seconds"] for event in steps)
    need(charged > 0 and decode_seconds > 0 and
         elapsed + 1e-5 >= prefill["seconds"] + decode_seconds,
         "charged forward/elapsed")
    return {"emitted_ids": ids, "native_decode_forwards": charged,
            "decode_s": decode_seconds, "prefill_s": prefill["seconds"],
            "native_elapsed_s": elapsed, "stop": stop,
            "decode_steps": steps, "binding": [events[0], events[-1]]}


def check_case_row(actual, reported, name, repetition):
    need(reported["case"] == name and reported["repetition"] == repetition and
         reported["emitted_ids"] == actual["emitted_ids"] and
         reported["native_decode_forwards"] == actual["native_decode_forwards"] and
         reported["stop"] == actual["stop"] and
         reported["decode_steps"] == actual["decode_steps"] and
         reported["binding"] == actual["binding"], f"case record: {name}/{repetition}")
    for field in ("decode_s", "prefill_s", "native_elapsed_s"):
        near(actual[field], reported[field], f"case {field}: {name}/{repetition}")


def run(directory):
    binding_path = directory / "binding.json"
    result_path = directory / "result.json"
    need(binding_path.is_file() and result_path.is_file() and
         not (directory / "failure.json").exists() and
         not (directory / "audit.json.failure.json").exists(),
         "complete continuation only")
    before = {"binding": sha(binding_path), "result": sha(result_path),
              "readback": sha(Path(__file__))}
    binding, result = load(binding_path), load(result_path)
    need(binding["schema"] == result["schema"] == SCHEMA and
         result["binding_sha256"] == before["binding"] and
         binding["original_C_git_revision"] == OLD_REVISION and
         binding["original_archive_sha256"] == OLD_ARCHIVE_SHA and
         binding["original_probe_sha256"] == OLD_PROBE_SHA,
         "continuation binding/result identity")
    need(binding["gates"] == {"worst_forward_seconds": .018,
                              "maximum_case_rate_spread": 1.10,
                              "probe_top1_exact": True,
                              "probe_relative_l2": .001} and
         binding["quality_admitted"] is False and
         binding["donor_transfer_admitted"] is False,
         "frozen gates and scope")
    old_c = ROOT / "benchmarks/native_expert_scaling/recipient_qwen_entry.c"
    need(old_c_git_sha() == binding["original_C_sha256"] and
         sha(old_c) == binding["current_trace_C_sha256"], "old/current C identity")
    sealed_records = []
    for item in binding["original_inputs"]:
        if Path(item["path"]) != old_c:
            check_file_record(item, "original input")
            sealed_records.append(item)
        else:
            need(item["sha256"] == binding["original_C_sha256"], "old C input pin")
    for key in ("original_files", "diagnostic_files", "continuation_inputs"):
        for item in binding[key]:
            check_file_record(item, key)
            sealed_records.append(item)
    original = Path(binding["original_dir"])
    archive = original / "recipient.mixed.bin"
    executable = original / "recipient.mixed.exe"
    catalog = original / "recipient_catalog.h"
    manifest = original / "manifest.json"
    old_probe = original / "probe.stdout"
    need(sha(original / "binding.json") == binding["original_binding_sha256"] and
         sha(original / "audit.json.failure.json") == binding["original_audit_failure_sha256"] and
         sha(original / "compile.terminal.json") == binding["original_compile_terminal_sha256"] and
         sha(archive) == result["artifact_sha256"] == OLD_ARCHIVE_SHA and
         sha(executable) == result["executable_sha256"] == binding["original_executable_sha256"] and
         sha(catalog) == result["catalog_sha256"] and
         sha(manifest) == result["manifest_sha256"] and
         sha(old_probe) == OLD_PROBE_SHA and
         result["original_probe_sha256"] == result["replayed_probe_sha256"] == OLD_PROBE_SHA and
         result["original_archive_path"] == str(archive) and
         result["original_executable_path"] == str(executable) and
         result["artifact_bytes"] == archive.stat().st_size,
         "retained packet/executable/probe identity")
    need(sha(Path(binding["diagnostic_dir"]) / "stored_silu_all_sites.json") ==
         binding["stored_silu_all_sites_sha256"], "all-site precursor")
    cases = binding["cases"]
    names = [case["name"] for case in cases]
    need(names == ["prompt_0", "prompt_1", "prompt_2", "context_512", "context_2048"] and
         len(cases) == 5 and len(result["cases"]) == 15,
         "case census")
    expected_names = (["probe", "audit"] +
                      [f"generate_r{rep}_{name}" for rep in range(3) for name in names])
    need(len(result["commands"]) == 17 and
         [row["name"] for row in result["commands"]] == expected_names,
         "command count/order")
    probe, _ = check_command(directory, "probe", result["commands"][0],
                             [str(executable), str(archive), "probe", "2176", "6"],
                             binding["probe_request"])
    need(sha(probe) == OLD_PROBE_SHA and probe.read_bytes() == old_probe.read_bytes(),
         "probe byte identity")
    source = Path(binding["source_directory"]) / "model.safetensors"
    audit_argv = [str(binding["python"]), "-I", "-S", "-B", "-X", "utf8",
                  str(ROOT / "benchmarks/native_expert_scaling/recipient_mixed_scalar_audit.py"),
                  "--source", str(source), "--archive", str(archive),
                  "--archive-sha", OLD_ARCHIVE_SHA, "--catalog", str(catalog),
                  "--manifest", str(manifest), "--ids", str(binding["parity_ids"]),
                  "--request", str(binding["probe_request"]), "--probe", str(probe),
                  "--out", str(directory / "audit.json")]
    check_command(directory, "audit", result["commands"][1], audit_argv, None)
    audit_path = directory / "audit.json"
    audit = load(audit_path)
    need(sha(audit_path) == result["audit_sha256"] and audit["status"] == "PASS" and
         audit["field_count"] == 291 and audit["source_tensor_count"] == 290 and
         audit["ids_count"] == 32 and audit["selected_positions"] == [0, 7, 15, 31] and
         set(audit["gates"]) == {"all_selected_argmax_equal",
                                 "all_selected_full_logits_relative_l2_le_1e-3",
                                 "all_selected_final_norm_relative_l2_le_1e-3"} and
         all(audit["gates"].values()) and
         audit["input_sha256_before_after"]["archive"] == OLD_ARCHIVE_SHA and
         audit["input_sha256_before_after"]["probe"] == OLD_PROBE_SHA,
         "scalar parity audit")
    for name, path in {"source": source, "archive": archive, "catalog": catalog,
                       "manifest": manifest, "ids": Path(binding["parity_ids"]),
                       "request": Path(binding["probe_request"]), "probe": probe,
                       "config": source.parent / "config.json"}.items():
        need(audit["input_sha256_before_after"][name] == sha(path),
             f"audit input pin: {name}")
    parsed = []
    for repetition in range(3):
        for case in cases:
            name = f"generate_r{repetition}_{case['name']}"
            index = 2 + repetition * 5 + names.index(case["name"])
            stdout, stderr = check_command(
                directory, name, result["commands"][index],
                [str(executable), str(archive), "stream", "2176", "6"], case["request"])
            own = decode_stream(stdout, stderr, case["tokens"], case["new_tokens"])
            need(own["binding"][0]["pid"] == load(directory / f"{name}.owned.json")["pid"],
                 f"stderr/owned PID: {name}")
            check_case_row(own, result["cases"][index - 2], case["name"], repetition)
            parsed.append({"case": case["name"], "repetition": repetition, **own})
    summaries = {}
    for case in cases:
        rows = [row for row in parsed if row["case"] == case["name"]]
        need(len(rows) == 3 and all(row["emitted_ids"] == rows[0]["emitted_ids"]
                                    for row in rows), f"repeat IDs: {case['name']}")
        rates = [row["native_decode_forwards"] / row["decode_s"] for row in rows]
        summaries[case["name"]] = {"rates": rates,
                                   "median": statistics.median(rates),
                                   "worst": min(rates),
                                   "worst_ms": 1000 / min(rates),
                                   "spread": max(rates) / min(rates)}
    need(set(result["cost_summary"]) == set(summaries), "cost summary key set")
    for name, actual in summaries.items():
        saved = result["cost_summary"][name]
        need(set(saved) == set(actual), f"cost summary fields: {name}")
        for a, b in zip(actual["rates"], saved["rates"]):
            near(a, b, f"rate: {name}")
        for field in ("median", "worst", "worst_ms", "spread"):
            near(actual[field], saved[field], f"{field}: {name}")
    passed = all(row["worst_ms"] <= 18.0 and row["spread"] <= 1.10
                 for row in summaries.values())
    need(result["status"] == ("MIXED_COST_ENVELOPE_PASS" if passed else
                              "MIXED_COST_ENVELOPE_FAIL") and
         result["active_matrix_payload_bytes_per_forward"] == 357484032 and
         result["physical_dram_measured"] is False and
         result["quality_admitted"] is False and
         result["donor_transfer_admitted"] is False and
         result["goal_complete"] is False,
         "cost decision/scope")
    retained_bytes = sum(path.stat().st_size for path in directory.rglob("*")
                         if path.is_file() and path != result_path)
    need(result["output_bytes"] == retained_bytes <= 2 << 30 and
         0 < result["seconds"] <= 2400,
         "retained output/time limits")
    for item in sealed_records:
        check_file_record(item, "post-readback sealed input")
    need(old_c_git_sha() == binding["original_C_sha256"] and
         sha(old_c) == binding["current_trace_C_sha256"],
         "post-readback old/current C identity")
    after = {"binding": sha(binding_path), "result": sha(result_path),
             "readback": sha(Path(__file__))}
    need(after == before, "binding/result/readback changed during verification")
    return {"schema": "RECIPIENT_MIXED_SCALAR_READBACK_V1",
            "status": "READBACK_PASS", "cost_status": result["status"],
            "input_sha256_before_after": before,
            "commands": 17, "cases": 15, "audit_gates": audit["gates"],
            "cost_summary": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    need(args.directory.is_dir() and args.out.parent.is_dir() and
         not args.out.exists(), "input/output directory or exclusive output")
    report = run(args.directory.resolve())
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
