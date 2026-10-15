"""Frozen synthetic Granite child256 operator-bundle cost screen.

Preparing this file performs no compilation or benchmark. Explicit --check-only
is required before the separate --run command can begin timing processes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "results/native_expert_scaling/chatbot_source_runtime/site"
sys.path.insert(0, str(SITE))
import psutil


HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/GRANITE_CHILD256_COST_PROTOCOL_20261015.md"
PROFILE = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925/meth374_switch_physical_workers_contract_result.json"
FILES = {
    "controller": Path(__file__).resolve(),
    "fixture": HERE / "granite_child256_cost_cpu.c",
    "protocol": PROTOCOL,
    "q4_header": ROOT / "benchmarks/phase60/strat01_q4k_q8k.h",
    "f32_header": ROOT / "benchmarks/phase60/strat01_f32_dot_reference_generic.h",
    "worker_helper": HERE / "meth388_switch_thread_binding.h",
    "physical_profile": PROFILE,
}
FLAGS = ("-O3", "-std=c11", "-march=x86-64-v3", "-fno-fast-math",
         "-ffp-contract=off", "-fopenmp")
LIMIT_S = 900
LIMIT_BYTES = 6 << 30
LIMIT_OUTPUT = 512 << 20


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert all(profile["gates"].values()), "physical_profile_not_admitted"
    cc = Path(profile["compile"]["argv"][0]).resolve()
    dll = cc.parent / "libomp.dll"
    assert sha(cc) == profile["compile"]["compiler_sha256"]
    assert sha(dll) == profile["compile"]["runtime_sha256"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "schema": "granite_child256_cost_freeze_v1",
        "git_head": head,
        "inputs_sha256": {name: sha(path) for name, path in FILES.items()},
        "compiler": str(cc), "compiler_sha256": sha(cc),
        "openmp_dll": str(dll), "openmp_dll_sha256": sha(dll),
        "compile_flags": list(FLAGS),
        "psutil_version": psutil.__version__,
        "psutil_package_sha256": sha(Path(psutil.__file__).resolve()),
        "psutil_native_sha256": sha(Path(psutil._psutil_windows.__file__).resolve()),
        "synthetic_only": True,
        "no_attention_rope_kv_generation": True,
    }


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check_binding(path):
    supplied = json.loads(path.read_text(encoding="utf-8"))
    if supplied != frozen():
        raise RuntimeError("freeze_binding_mismatch")
    return supplied


def parse_lines(path):
    events = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        events.append(json.loads(raw))
    return events


def placement(event):
    assert event["worker_physical_cores"] == 6
    rows = event["worker_affinity"]
    assert [r["slot"] for r in rows] == [0, 1, 2]
    assert [r["actual_mask"] for r in rows] == [1, 4, 16]
    assert len({r["windows_thread_id"] for r in rows}) == 3
    assert all(r["group"] == 0 for r in rows)
    events = event["worker_binding_events"]
    assert {r["slot"] for r in events} == {0, 1, 2}
    assert all(r["group"] == 0 and r["actual_mask"] == 1 << (2 * r["slot"]) for r in events)


def size_of(directory):
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def invoke(binary, mode, profile, label, output, clock_start):
    stdout = output / f"{label}.stdout.jsonl"
    stderr = output / f"{label}.stderr.log"
    cmd = [str(binary), str(mode), str(profile)]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OMP_", "KMP_", "GOMP_", "SILICON_WORKER_BINDING_"))}
    env.update(json.loads(PROFILE.read_text(encoding="utf-8"))["runtime_environment"])
    env["OMP_NUM_THREADS"] = "3"
    start = time.monotonic()
    peak = 0
    failure = None
    with stdout.open("xb") as so, stderr.open("xb") as se:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=so, stderr=se)
        try:
            while child.poll() is None:
                usage = psutil.Process().memory_info().rss
                try:
                    usage += psutil.Process(child.pid).memory_info().rss
                except psutil.NoSuchProcess:
                    pass
                peak = max(peak, usage)
                if peak > LIMIT_BYTES or time.monotonic() - start > 120 or time.monotonic() - clock_start > LIMIT_S or size_of(output) > LIMIT_OUTPUT:
                    raise RuntimeError("time_memory_or_output_cap")
                time.sleep(0.1)
            code = child.wait()
        except BaseException as exc:
            failure = f"{type(exc).__name__}:{exc}"
            if child.poll() is None:
                child.kill()
                child.wait()
            code = child.returncode
    receipt = {"argv": cmd, "exit_code": code, "elapsed_s": time.monotonic() - start,
               "peak_parent_child_rss_bytes": peak, "stdout": str(stdout), "stderr": str(stderr),
               "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr)}
    if failure:
        receipt["failure"] = failure
    exclusive_json(output / f"{label}.receipt.json", receipt)
    if failure:
        raise RuntimeError(failure)
    if code:
        raise RuntimeError(f"fixture_exit_{code}:{stderr.read_text(encoding='utf-8')[-2048:]}")
    return receipt, parse_lines(stdout)


def expected_sample(row, mode, profile):
    assert row["event"] == "sample" and row["mode"] == mode and row["profile"] == profile
    assert row["rep"] in (0, 1, 2) and row["fixture"] in range(8)
    assert 0 < row["ms"] < 120000
    assert len(row["sha256"]) == 64
    expected = {
        "i8_coefficients": 125832192,
        "i8_scale_bytes": 491532,
        "q4_coefficients": 150994944 if mode else 301989888,
        "q4_payload_bytes": 84934656 if mode else 169869312,
        "router_weight_bytes": 3145728,
        "selector_weight_bytes": 1572864,
        "q8_quantized_values": 73728 if mode else 122880,
    }
    assert all(row[k] == v for k, v in expected.items()), ("cost_ledger", row)


def check_events(events):
    assert [r["event"] for r in events] == ["ready", "control", "check_output", "control", "check_output"]
    ready = events[0]
    placement(ready)
    assert ready["schema"] == "granite_child256_cost_v1" and ready["synthetic_only"]
    assert (ready["layers"], ready["dimension"], ready["experts"], ready["topk"],
            ready["parent_width"], ready["child_width"], ready["vocab"]) == (24, 1024, 32, 8, 512, 256, 49155)
    assert ready["stored_i8_coefficients"] == 125832192
    assert ready["stored_i8_scale_bytes"] == 491532
    assert ready["stored_q4_parent_bytes"] == ready["stored_q4_child_bytes"] == 679477248
    for mode in (0, 1):
        control, output = events[1 + 2 * mode:3 + 2 * mode]
        assert control["mode"] == output["mode"] == mode
        assert control["q4_full_rows"] == (589824 if mode == 0 else 294912)
        assert control["q4_relative_l2"] <= 1e-5
        assert control["i8_sampled_rows"] == 832 and control["router_layers"] == control["selector_layers"] == 24
        assert len(output["sha256"]) == 64


def timing_events(events, mode, profile):
    assert len(events) == 26
    assert events[0]["event"] == "ready" and events[-1]["event"] == "finished"
    placement(events[0]); placement(events[-1])
    assert events[0]["stored_q4_parent_bytes"] == events[0]["stored_q4_child_bytes"] == 679477248
    assert events[-1]["peak_rss_bytes"] <= LIMIT_BYTES
    samples = events[1:-1]
    assert [(r["rep"], r["fixture"]) for r in samples] == [(r, f) for r in range(3) for f in range(8)]
    for row in samples:
        expected_sample(row, mode, profile)
    return samples


def decision(samples):
    groups = {}
    hashes = {}
    for item in samples:
        mode, profile, process = item["mode"], item["profile"], item["process"]
        for row in item["samples"]:
            key = (mode, profile, row["fixture"])
            groups.setdefault(key, {}).setdefault(process, []).append(row["ms"])
            hkey = (mode, row["fixture"])
            hashes.setdefault(hkey, set()).add(row["sha256"])
    assert all(len(v) == 1 for v in hashes.values()), "unstable_full_output_sha256"
    gates = {}
    for key, processes in groups.items():
        assert len(processes) == 3 and all(len(v) == 3 for v in processes.values())
        medians = [statistics.median(v) for v in processes.values()]
        all_values = [v for values in processes.values() for v in values]
        gates[str(key)] = {"process_medians_ms": medians,
                           "max_min_ratio_all_9": max(all_values) / min(all_values),
                           "all_process_medians_le_14ms": all(v <= 14 for v in medians),
                           "all_9_max_min_le_1_10": max(all_values) / min(all_values) <= 1.10}
    assert len(gates) == 32
    return {"per_mode_profile_fixture": gates,
            "all_14ms": all(v["all_process_medians_le_14ms"] for v in gates.values()),
            "all_stable": all(v["all_9_max_min_le_1_10"] for v in gates.values()),
            "hashes_stable": True}


def execute(args):
    binding = check_binding(args.binding)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "granite_child256_cost_result_v1", "binding_sha256": sha(args.binding),
              "mode": "check_only" if args.check_only else "timing", "synthetic_only": True,
              "not_model_inference": True, "runs": []}
    try:
        dll = out / "libomp.dll"
        shutil.copyfile(binding["openmp_dll"], dll)
        assert sha(dll) == binding["openmp_dll_sha256"]
        binary = out / "granite_child256_cost_cpu.exe"
        cmd = [binding["compiler"], *FLAGS, str(FILES["fixture"]), "-o", str(binary),
               "-lm", "-lpsapi", "-lbcrypt"]
        cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
        result["compile"] = {"argv": cmd, "exit_code": cp.returncode,
                             "stdout": cp.stdout[-8192:], "stderr": cp.stderr[-8192:]}
        assert cp.returncode == 0, "compile_failed"
        result["binary_sha256"] = sha(binary)
        if args.check_only:
            receipt, events = invoke(binary, 2, 0, "numeric_control", out, started)
            result["runs"].append(receipt)
            check_events(events)
            result["controls"] = events[1:]
            check_binding(args.binding)
            result["decision"] = "numeric_pass"
        else:
            checked = json.loads(args.check_receipt.read_text(encoding="utf-8"))
            assert checked["binding_sha256"] == result["binding_sha256"]
            assert checked["binary_sha256"] == result["binary_sha256"]
            assert checked["decision"] == "numeric_pass", "numeric_check_required"
            assert len(checked["runs"]) == 1
            prior_run = checked["runs"][0]
            prior_stdout = Path(prior_run["stdout"]).resolve()
            assert prior_stdout == (args.check_receipt.resolve().parent / "numeric_control.stdout.jsonl")
            assert sha(prior_stdout) == prior_run["stdout_sha256"]
            assert prior_run["exit_code"] == 0
            check_events(parse_lines(prior_stdout))
            schedule = []
            for profile in (0, 1):
                for process, modes in enumerate(((0, 1), (1, 0), (0, 1))):
                    for mode in modes:
                        schedule.append((mode, profile, process))
            result["schedule"] = schedule
            samples = []
            for mode, profile, process in schedule:
                label = f"p{profile}_r{process}_m{mode}"
                receipt, events = invoke(binary, mode, profile, label, out, started)
                result["runs"].append(receipt)
                part = timing_events(events, mode, profile)
                samples.append({"mode": mode, "profile": profile, "process": process, "samples": part})
            result["gates"] = decision(samples)
            check_binding(args.binding)
            result["decision"] = "bundle_cost_pass" if result["gates"]["all_14ms"] and result["gates"]["all_stable"] else "bundle_cost_fail"
    except Exception as exc:
        result["decision"] = "control_or_execution_fail"
        result["error"] = f"{type(exc).__name__}:{exc}"
    result["retained_process_receipts"] = [str(p) for p in sorted(out.glob("*.receipt.json"))]
    result["elapsed_s"] = time.monotonic() - started
    result["output_bytes"] = size_of(out)
    exclusive_json(out / "result.json", result)
    print(json.dumps({"result": str(out / "result.json"), "decision": result["decision"]}))
    return 0 if result["decision"] in ("numeric_pass", "bundle_cost_pass", "bundle_cost_fail") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", type=Path)
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--check-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        if args.binding or args.check_receipt or args.output:
            parser.error("--freeze accepts no other path")
        exclusive_json(args.freeze, frozen())
        print(str(args.freeze.resolve()))
        return 0
    if not args.binding or not args.output or (args.run and not args.check_receipt):
        parser.error("--check-only/--run require --binding and --output; --run also needs --check-receipt")
    return execute(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"cost_controller_failure:{type(exc).__name__}:{exc}", file=sys.stderr)
        sys.exit(2)
