"""Independent, bounded CPU audit of first-attention trace and metric receipts.

The default audit reads saved traces and does no attention projection work.
--replay independently evaluates the F32 source and rank-256 candidate on CPU.
No producer module is imported and no GPU API is called.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "6"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results/native_expert_scaling/chatbot_source_runtime/site"))
import psutil
import torch
import torch.nn.functional as F
torch.set_num_threads(6)
torch.set_num_interop_threads(1)

DOC = ROOT / "docs/research/PIPELINE_REASSESSMENT_20261015"
CONFIG = DOC / "large_family_metadata/config120.json"
REFERENCE = DOC / "large_family_metadata/reference_model.py"
RANK_CODE = ROOT / "benchmarks/native_expert_scaling/first_attention_rank.py"
RANK_PROTOCOL = DOC / "FIRST_ATTENTION_RANK_PROTOCOL_20261015.md"
AUDIT_PROTOCOL = DOC / "FIRST_ATTENTION_CPU_AUDIT_PROTOCOL_20261015.md"
RANK = 256
MAX_SECONDS = 3600
MAX_RSS = 6 << 30
MAX_OUTPUT_BYTES = 2 << 20
SHAPES = {
    "model.layers.0.input_layernorm.weight": (2880,),
    "model.layers.0.self_attn.q_proj.weight": (4096, 2880),
    "model.layers.0.self_attn.q_proj.bias": (4096,),
    "model.layers.0.self_attn.k_proj.weight": (512, 2880),
    "model.layers.0.self_attn.k_proj.bias": (512,),
    "model.layers.0.self_attn.v_proj.weight": (512, 2880),
    "model.layers.0.self_attn.v_proj.bias": (512,),
    "model.layers.0.self_attn.o_proj.weight": (2880, 4096),
    "model.layers.0.self_attn.o_proj.bias": (2880,),
    "model.layers.0.self_attn.sinks": (64,),
}
LAYOUT = "query,kv_head,query_per_kv_head,allowed_key_oldest_to_newest,sink_last"
GATES = ("pooled_branch_l2", "sequence_l2_p95", "tv_mean_including_sink",
         "tv_p95_including_sink")
LIMITS = (.01, .05, .01, .05)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def bounded(started):
    need(time.monotonic() - started < MAX_SECONDS, "audit_deadline")
    need(psutil.Process().memory_info().rss < MAX_RSS, "audit_rss_cap")


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(4 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def located(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def pinned(desc):
    p = located(desc["path"])
    need(p.stat().st_size == desc["bytes"] and file_sha(p) == desc["sha256"],
         "file_pin:" + str(p))
    return p


def raw_bf16(desc, shape):
    p = pinned(desc)
    need(desc["dtype"] == "BF16" and tuple(desc["shape"]) == tuple(shape) and
         desc["bytes"] == math.prod(shape) * 2, "raw_tensor_shape_dtype")
    value = torch.frombuffer(p.read_bytes(), dtype=torch.uint16).clone().view(torch.bfloat16)
    value = value.reshape(shape)
    need(bool(torch.isfinite(value.float()).all()), "nonfinite_source_tensor")
    return value


def check_inputs(manifest_path, binding_path, result):
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    b = json.loads(binding_path.read_text(encoding="utf-8"))
    need(m["schema"] == "FIRST_ATTENTION_SOURCE_V1" and
         m["source_model"] == "openai/gpt-oss-120b" and
         m["source_revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
         "source_revision")
    need(b["schema"] == "FIRST_ATTENTION_RANK_BINDING_V1" and
         b["manifest_path"] == str(manifest_path) and
         b["rank"] == RANK and b["steps"] == 512, "rank_binding_contract")
    need(result["binding_sha256"] == file_sha(binding_path) and
         result["manifest_sha256"] == file_sha(manifest_path) and
         result["post_binding_match"] is True, "run_input_seal")
    paths = {"code": RANK_CODE, "protocol": RANK_PROTOCOL,
             "config": CONFIG, "reference": REFERENCE,
             "manifest": manifest_path, "torch": Path(torch.__file__).resolve(),
             "torch_native": Path(torch._C.__file__).resolve(),
             "torch_cuda": Path(torch.__file__).resolve().parent / "cuda" / "__init__.py",
             "psutil": Path(psutil.__file__).resolve(),
             "psutil_native": Path(psutil._psutil_windows.__file__).resolve()}
    for filename in ("torch_cpu.dll", "torch_cuda.dll", "torch_python.dll",
                     "c10.dll", "c10_cuda.dll"):
        paths[filename] = Path(torch.__file__).resolve().parent / "lib" / filename
    need(set(b["sha256"]) == set(paths), "binding_runtime_file_set")
    for key, path in paths.items():
        need(b["sha256"][key] == file_sha(path), "binding_byte_hash:" + key)
    need(b["torch_version"] == torch.__version__ and
         b["cuda_version"] == torch.version.cuda and
         b["psutil_version"] == psutil.__version__, "binding_runtime_versions")
    for key in ("code", "protocol", "config", "reference"):
        payload = paths[key].read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") +
                                b"\0" + payload).hexdigest()
        need(b["git_blobs"][key] == git_blob, "binding_git_blob:" + key)
    need(located(m["config"]["path"]) == CONFIG.resolve() and
         located(m["reference"]["path"]) == REFERENCE.resolve(), "official_paths")
    for k in ("config", "reference", "index"):
        pinned(m[k])
    need(set(m["tensors"]) == set(SHAPES), "tensor_set")
    for name, shape in SHAPES.items():
        d = m["tensors"][name]
        need(d["name"] == name and d["source_shard"] and d["source_offset"] >= 0,
             "tensor_provenance")
        need(tuple(d["shape"]) == shape and d["dtype"] == "BF16" and
             d["bytes"] == 2 * math.prod(shape), "tensor_descriptor")
        pinned(d)
    ids = m["embedding_rows"]["token_ids"]
    need(ids == sorted(set(ids)), "embedding_ids_order")
    raw_bf16(m["embedding_rows"], (len(ids), 2880))
    for d in m["input_pins"] + m["tokenizer_files"]:
        pinned(d)
    need(file_sha(manifest_path.parent / "selection_plan.json") == m["selection_plan_sha256"] and
         file_sha(manifest_path.parent / "eligibility_plan.json") == m["eligibility_plan_sha256"],
         "selection_plan_pin")
    cases = m["cases"]
    need(len(cases) == 36 and len({c["id"] for c in cases}) == 36, "cases_unique_36")
    fit = [c for c in cases if c["split"] == "FIT"]
    dev = [c for c in cases if c["split"] == "DEV"]
    need(len(fit) == 24 and len(dev) == 12, "fit_dev_count")
    fd = {c["domain"] for c in fit}; dd = {c["domain"] for c in dev}
    need(len(fd) == 8 and len(dd) == 4 and not fd & dd and
         dd == {"smol-magpie-ultra", "metamathqa-50k", "smol-constraints",
                "self-oss-instruct"}, "domain_holdout")
    for group in (fit, dev):
        need(all(sum(c["domain"] == domain for c in group) == 3
                 for domain in {c["domain"] for c in group}), "three_per_domain")
    for key in ("group_key", "first_user_key"):
        need(not {c[key] for c in fit} & {c[key] for c in dev}, "split_disjoint_" + key)
    observed = min(len(c["input_ids"]) for c in cases)
    for q in range(observed):
        if len({c["input_ids"][q] for c in cases}) != 1:
            observed = q
            break
    need(m["common_prefix_length"] == observed and observed < 256, "query_mask_common")
    ids_set = set(ids)
    for c in cases:
        used = c["input_ids"]; full = c["canonical_input_ids"]
        need(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["id"]) is not None,
             "safe_case_id")
        need(used == full[:256] and len(full) == c["original_length"] and
             observed < len(used) <= 256 and set(used) <= ids_set, "case_ids")
        for key, values in (("used_ids_sha256", used), ("canonical_ids_sha256", full)):
            digest = hashlib.sha256(struct.pack("<" + "I" * len(values), *values)).hexdigest()
            need(c[key] == digest, "case_ids_sha:" + key)
    return m, b


def source_hash(m):
    t = {name: raw_bf16(m["tensors"][name], shape) for name, shape in SHAPES.items()}
    p = "model.layers.0.self_attn."
    values = {
        "norm.scale": t["model.layers.0.input_layernorm.weight"].float(),
        "qkv.weight": torch.cat([t[p + k + "_proj.weight"] for k in ("q", "k", "v")]),
        "qkv.bias": torch.cat([t[p + k + "_proj.bias"] for k in ("q", "k", "v")]),
        "out.weight": t[p + "o_proj.weight"], "out.bias": t[p + "o_proj.bias"],
        "sinks": t[p + "sinks"],
    }
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode() + b"\0")
        h.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        h.update(str(value.dtype).encode("ascii") + b"\0")
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest(), values


def load_trace(path, case, dtype):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    n = len(case["input_ids"])
    need(obj["input_ids"] == case["input_ids"] and obj["split"] == case["split"] and
         obj["domain"] == case["domain"], "trace_case_identity")
    branch = obj["branch"]
    need(isinstance(branch, torch.Tensor) and branch.shape == (n, 2880) and
         branch.dtype == dtype and bool(torch.isfinite(branch.float()).all()), "trace_branch")
    packed = obj["prob_packed"]
    need(packed["layout"] == LAYOUT and packed["length"] == n and
         packed["window"] == 128 and packed["heads"] == 64, "packed_header")
    values = packed["values"]
    count = 64 * sum(min(q + 1, 128) + 1 for q in range(n))
    need(values.shape == (count,) and values.dtype == dtype and
         bool(torch.isfinite(values.float()).all()), "packed_values")
    rows = []
    offset = 0
    for q in range(n):
        width = min(q + 1, 128) + 1
        row = values[offset:offset + 64 * width].reshape(64, width).float()
        need(bool((row >= 0).all()) and
             float((row.sum(-1) - 1).abs().max()) <= (.016 if dtype == torch.bfloat16 else .0002),
             "packed_probability_simplex")
        rows.append(row)
        offset += 64 * width
    need(offset == count, "packed_exhaustive")
    return branch.float(), rows


def percentile_higher(numbers):
    ordered = sorted(numbers)
    return ordered[math.ceil(.95 * len(ordered)) - 1]


def compare_trace(candidate, teacher, start_query, bias):
    cb, cr = candidate; tb, tr = teacher
    need(len(cr) == len(tr) == cb.shape[0] == tb.shape[0], "trace_length_match")
    delta = (cb - tb).double()
    reference = tb.double()
    err2 = float(delta[start_query:].square().sum())
    norm2 = float(reference[start_query:].square().sum())
    allerr2 = float(delta.square().sum())
    allnorm2 = float(reference.square().sum())
    biasnorm2 = float((reference[start_query:] - bias.double()).square().sum())
    tv = []; all_tv = []
    for q, (a, b) in enumerate(zip(cr, tr)):
        # Source BF16 values remain untouched on disk; only metric copies normalize.
        a = a.double() / a.double().sum(-1, keepdim=True)
        b = b.double() / b.double().sum(-1, keepdim=True)
        row_tv = (a - b).abs().sum(-1).mul(.5).tolist()
        all_tv.extend(row_tv)
        if q >= start_query:
            tv.extend(row_tv)
    return {"error2": err2, "norm2": norm2,
            "seq_l2": math.sqrt(err2 / max(norm2, 1e-30)),
            "tv": tv, "all_error2": allerr2, "all_norm2": allnorm2,
            "all_seq_l2": math.sqrt(allerr2 / max(allnorm2, 1e-30)),
            "bias_removed_error2": err2, "bias_removed_norm2": biasnorm2,
            "all_tv": all_tv}


def aggregate(rows):
    def pooled(a, b):
        return math.sqrt(sum(r[a] for r in rows) / max(sum(r[b] for r in rows), 1e-30))
    tv = [v for r in rows for v in r["tv"]]
    all_tv = [v for r in rows for v in r["all_tv"]]
    return {"pooled_branch_l2": pooled("error2", "norm2"),
            "sequence_l2_p95": percentile_higher([r["seq_l2"] for r in rows]),
            "tv_mean_including_sink": sum(tv) / len(tv),
            "tv_p95_including_sink": percentile_higher(tv),
            "worst_sequence_l2": max(r["seq_l2"] for r in rows),
            "worst_tv_row": max(tv),
            "all_token_pooled_branch_l2_diagnostic": pooled("all_error2", "all_norm2"),
            "bias_removed_pooled_branch_l2_diagnostic": pooled("bias_removed_error2", "bias_removed_norm2"),
            "all_token_tv_mean_diagnostic": sum(all_tv) / len(all_tv),
            "all_token_tv_p95_diagnostic": percentile_higher(all_tv)}


def close_metrics(actual, recorded, label):
    for key, val in actual.items():
        # Double accumulation vs producer F32 reductions may differ slightly.
        other = recorded[key]
        need(math.isfinite(other) and abs(val - other) <= 2e-5 + 2e-4 * abs(val),
             "metric_disagreement:" + label + ":" + key)


def check_outputs(output, result):
    need(result["schema"] == "FIRST_ATTENTION_RANK_RESULT_V1" and
         result["status"] in ("INCONCLUSIVE_PRECISION_FLOOR", "RANK256_F32_REJECT",
                              "RANK256_F32_GATE_PASS_PHYSICAL_Q4_PENDING"), "result_status")
    listed = {Path(d["path"]).resolve(): d for d in result["owned_outputs"]}
    need(len(listed) == len(result["owned_outputs"]), "owned_output_unique")
    for path, d in listed.items():
        need(path.parent == output and path.is_file() and path.stat().st_size == d["bytes"]
             and file_sha(path) == d["sha256"], "owned_output_sha:" + path.name)
    extras = {p.resolve() for p in output.iterdir() if p.is_file()} - set(listed)
    need(extras <= {output / "result.json", output / "watchdog_done"}, "unexpected_output_files")
    owned = json.loads((output / "owned_process.json").read_text())
    need(owned["pid"] == result["pid"] and
         owned["pid_create_time"] == result["pid_create_time"] and
         owned["binding_sha256"] == result["binding_sha256"], "owned_process_custody")
    need((output / "watchdog_done").read_text().strip() == "complete" and
         not (output / "watchdog_failure.json").exists(), "watchdog_completion")
    need(result["source_parameter_sha256_before"] ==
         result["source_parameter_sha256_after"], "source_before_after")
    need(result["gpu_released_before_posthash"] is True and
         result["cpu_rss_bytes"] < 8 << 30 and
         result["gpu_peak_allocated_bytes"] < 8 << 30 and
         result["gpu_peak_reserved_bytes"] < 9 << 30 and
         result["wall_seconds"] < 9000, "producer_resource_receipt")


def check_factors(output, result, m):
    expected = {"q_left": (4096, RANK), "q_right": (RANK, 2880),
                "o_left": (2880, RANK), "o_right": (RANK, 4096)}
    final = torch.load(output / "rank256_f32_factors.pt", map_location="cpu", weights_only=True)
    initial = torch.load(output / "rank256_initial_factors.pt", map_location="cpu", weights_only=True)
    for stage in (initial, final):
        need(stage["rank"] == RANK and set(stage["factors"]) == set(expected), "factor_set")
        for name, shape in expected.items():
            value = stage["factors"][name]
            need(value.shape == shape and value.dtype == torch.float32 and
                 bool(torch.isfinite(value).all()), "factor_shape_finite:" + name)
    need(sum(math.prod(s) for s in expected.values()) == final["coefficients"] == 3571712,
         "factor_count")
    need(file_sha(output / "rank256_f32_factors.pt") == result["factor_sha256"], "factor_file_sha")
    ledger = (output / "optimizer_ledger.jsonl").read_text().splitlines()
    need(len(ledger) == 512 and file_sha(output / "optimizer_ledger.jsonl") ==
         result["optimizer_ledger_sha256"], "optimizer_ledger_complete")
    fit_ids = sorted(c["id"] for c in m["cases"] if c["split"] == "FIT")
    for i, line in enumerate(ledger):
        row = json.loads(line)
        need(row["step"] == row["adam_step"] == i + 1 and
             row["case_id"] == fit_ids[i % 24] and row["all_finite"] is True and
             all(math.isfinite(row[k]) for k in ("branch_loss", "tv", "total", "gradient_l2")) and
             abs(row["total"] - row["branch_loss"] - row["tv"]) <= 1e-4,
             "optimizer_step_ledger")
    h = hashlib.sha256()
    for name in ("q_left", "q_right", "o_left", "o_right"):
        h.update(final["factors"][name].numpy().tobytes(order="C"))
    need(h.hexdigest() == json.loads(ledger[-1])["parameters_sha256"],
         "final_factor_ledger_hash")
    opt = torch.load(output / "rank256_optimizer_final.pt", map_location="cpu", weights_only=True)
    need(opt["steps"] == 512 and len(opt["optimizer"]["state"]) == 4 and
         set(opt["last_gradients"]) == set(expected), "optimizer_final_state")
    for name, shape in expected.items():
        g = opt["last_gradients"][name]
        need(g.shape == shape and bool(torch.isfinite(g).all()), "last_gradient_finite")
    return final["factors"]


def audit(manifest_path, binding_path, output, report_path, replay):
    started = time.monotonic()
    manifest_path = manifest_path.resolve(); binding_path = binding_path.resolve()
    output = output.resolve(); report_path = report_path.resolve()
    need(not report_path.is_relative_to(output) and not report_path.exists(),
         "independent_report_path")
    audit_code = Path(__file__).resolve()
    initial_pins = {"audit_code": file_sha(audit_code),
                    "audit_protocol": file_sha(AUDIT_PROTOCOL),
                    "manifest": file_sha(manifest_path),
                    "binding": file_sha(binding_path),
                    "producer_result": file_sha(output / "result.json")}
    result = json.loads((output / "result.json").read_text())
    m, _ = check_inputs(manifest_path, binding_path, result)
    check_outputs(output, result)
    source_digest, source_values = source_hash(m)
    need(source_digest == result["source_parameter_sha256_before"], "raw_source_parameter_hash")
    common = m["common_prefix_length"]
    bias = source_values["out.bias"].float()
    precision = {"FIT": [], "DEV": []}
    candidate = {"FIT": [], "DEV": []}
    for case in sorted(m["cases"], key=lambda c: c["id"]):
        cid = case["id"]
        teacher = load_trace(output / ("source_" + cid + ".pt"), case, torch.bfloat16)
        control = load_trace(output / ("f32_source_" + cid + ".pt"), case, torch.float32)
        precision[case["split"]].append((cid, compare_trace(control, teacher, common, bias)))
        if result["status"] != "INCONCLUSIVE_PRECISION_FLOOR":
            rank_trace = load_trace(output / ("candidate_" + cid + ".pt"), case, torch.float32)
            candidate[case["split"]].append((cid, compare_trace(rank_trace, teacher, common, bias)))
        bounded(started)
    recomputed = {"precision": {}, "rank256": {}}
    for split in ("FIT", "DEV"):
        actual = aggregate([r for _, r in precision[split]])
        recomputed["precision"][split] = actual
        close_metrics(actual, result["precision"][split], "precision_" + split)
        recorded_cases = {r["id"]: r for r in result["cases"]} if result["status"] == "INCONCLUSIVE_PRECISION_FLOOR" else None
        if recorded_cases:
            for cid, values in precision[split]:
                close_metrics({k: v for k, v in values.items() if k not in ("tv", "all_tv")},
                              recorded_cases[cid], "precision_case_" + cid)
        if candidate[split]:
            rank_actual = aggregate([r for _, r in candidate[split]])
            recomputed["rank256"][split] = rank_actual
            close_metrics(rank_actual, result["rank256"][split], "rank_" + split)
            recorded_cases = {r["id"]: r for r in result["per_case"][split]}
            for cid, values in candidate[split]:
                close_metrics({k: v for k, v in values.items() if k not in ("tv", "all_tv")},
                              recorded_cases[cid], "rank_case_" + cid)
    precision_pass = all(all(recomputed["precision"][s][key] <= limit
                             for key, limit in zip(GATES, LIMITS)) for s in ("FIT", "DEV"))
    recorded_precision_pass = all(all(result["precision"][s][key] <= limit
                                      for key, limit in zip(GATES, LIMITS))
                                  for s in ("FIT", "DEV"))
    need(precision_pass == recorded_precision_pass, "precision_gate_boundary_disagreement")
    need(precision_pass == (result["status"] != "INCONCLUSIVE_PRECISION_FLOOR"),
         "precision_floor_status")
    if candidate["FIT"]:
        final = check_factors(output, result, m)
        rank_pass = all(all(recomputed["rank256"][s][key] <= limit
                            for key, limit in zip(GATES, LIMITS)) for s in ("FIT", "DEV"))
        recorded_rank_pass = all(all(result["rank256"][s][key] <= limit
                                     for key, limit in zip(GATES, LIMITS))
                                 for s in ("FIT", "DEV"))
        need(rank_pass == recorded_rank_pass, "rank_gate_boundary_disagreement")
        need(rank_pass == (result["status"] == "RANK256_F32_GATE_PASS_PHYSICAL_Q4_PENDING"),
             "rank_status")
    else:
        final = None
        need(not any(output.glob("candidate_*.pt")), "candidate_after_precision_floor")
    if replay:
        need(final is not None, "replay_requires_candidate")
        # Full graph replay is a separate optional stage and deliberately not
        # required for the first precision-floor audit.
        replay_all(m, source_values, final, output, started)
    check_inputs(manifest_path, binding_path, result)
    check_outputs(output, result)
    final_pins = {"audit_code": file_sha(audit_code),
                  "audit_protocol": file_sha(AUDIT_PROTOCOL),
                  "manifest": file_sha(manifest_path),
                  "binding": file_sha(binding_path),
                  "producer_result": file_sha(output / "result.json")}
    need(final_pins == initial_pins, "audit_pin_changed_during_run")
    memory = psutil.Process().memory_info()
    audit_result = {"schema": "FIRST_ATTENTION_CPU_AUDIT_V1", "status": "PASS",
                    "producer_status": result["status"], "source_parameter_sha256": source_digest,
                    "pins_sha256": final_pins, "recomputed_metrics": recomputed,
                    "decisions": {"precision_pass": precision_pass,
                                  "rank256_pass": rank_pass if final is not None else None},
                    "settings": {"cpu_threads": torch.get_num_threads(),
                                 "interop_threads": torch.get_num_interop_threads(),
                                 "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]},
                    "peak_wset_bytes": getattr(memory, "peak_wset", None),
                    "rss_bytes_at_report": memory.rss,
                    "replay": replay, "seconds": time.monotonic() - started,
                    "remaining_witnesses": ["GPU training execution and intermediate optimizer state",
                                           "source probability capture is manual graph checked by official context",
                                           "whole-model behavior and physical Q4 remain untested"]}
    encoded = json.dumps(audit_result, indent=2, sort_keys=True) + "\n"
    need(len(encoded) <= MAX_OUTPUT_BYTES, "audit_report_cap")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(encoded)


def yarn_rotate(q, k, n):
    half = 32
    base = 150000.0
    freq = base ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    low = half * math.log(4096 / (32 * 2 * math.pi)) / math.log(base)
    high = half * math.log(4096 / (1 * 2 * math.pi)) / math.log(base)
    need(0 < low < high < half - 1, "yarn_ranges")
    ramp = (torch.arange(half, dtype=torch.float32) - low) / (high - low)
    mask = 1 - ramp.clamp(0, 1)
    inv = (1 / (32 * freq)) * (1 - mask) + (1 / freq) * mask
    phase = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    concentration = 1 + .1 * math.log(32)
    cosine = (phase.cos() * concentration)[:, None, None, :]
    sine = (phase.sin() * concentration)[:, None, None, :]

    def rotate(value):
        shape = value.shape
        value = value.reshape(n, -1, 1, 64)
        first, second = value[..., :32], value[..., 32:]
        return torch.cat((first * cosine - second * sine,
                          second * cosine + first * sine), dim=-1).reshape(shape)
    return rotate(q), rotate(k)


def cpu_attention(x, weights, factors):
    """Separate F32 equations; `factors=None` evaluates uncompressed Q/O."""
    n = x.shape[0]
    norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    norm = norm * weights["norm.scale"]
    qkv_weight = weights["qkv.weight"]
    qkv_bias = weights["qkv.bias"]
    if factors is None:
        qkv = F.linear(norm, qkv_weight, qkv_bias)
        q = qkv[:, :4096]
        k = qkv[:, 4096:4608]
        v = qkv[:, 4608:5120]
    else:
        q = F.linear(F.linear(norm, factors["q_right"]),
                     factors["q_left"], qkv_bias[:4096])
        k = F.linear(norm, qkv_weight[4096:4608], qkv_bias[4096:4608])
        v = F.linear(norm, qkv_weight[4608:5120], qkv_bias[4608:5120])
    q, k = yarn_rotate(q.reshape(n, 8, 8, 64), k.reshape(n, 8, 64), n)
    v = v.reshape(n, 8, 64)
    key = k[:, :, None, :].expand(n, 8, 8, 64)
    value = v[:, :, None, :].expand(n, 8, 8, 64)
    logits = torch.einsum("qhmd,khmd->hmqk", q, key)
    logits *= .125
    key_ids = torch.arange(n)[None, :]
    query_ids = torch.arange(n)[:, None]
    legal = (key_ids <= query_ids) & (key_ids >= query_ids - 127)
    logits.masked_fill_(~legal[None, None, :, :], -float("inf"))
    sinks = weights["sinks"].reshape(8, 8, 1, 1).expand(8, 8, n, 1)
    prob = torch.softmax(torch.cat((logits, sinks), dim=-1), dim=-1)
    context = torch.einsum("hmqk,khmd->qhmd", prob[..., :-1], value).reshape(n, 4096)
    if factors is None:
        branch = F.linear(context, weights["out.weight"], weights["out.bias"])
    else:
        branch = F.linear(F.linear(context, factors["o_right"]),
                          factors["o_left"], weights["out.bias"])
    return branch, prob


def replay_all(m, source_values, factors, output, started):
    weights = {k: v.float() for k, v in source_values.items()}
    embedding = raw_bf16(m["embedding_rows"],
                         (len(m["embedding_rows"]["token_ids"]), 2880)).float()
    lookup = {token: i for i, token in enumerate(m["embedding_rows"]["token_ids"])}
    with torch.no_grad():
        for case in sorted(m["cases"], key=lambda c: c["id"]):
            bounded(started)
            x = embedding[[lookup[token] for token in case["input_ids"]]]
            for prefix, factor_set in (("f32_source_", None), ("candidate_", factors)):
                trace = load_trace(output / (prefix + case["id"] + ".pt"), case,
                                   torch.float32)
                branch, prob = cpu_attention(x, weights, factor_set)
                saved_branch, saved_rows = trace
                difference = torch.linalg.vector_norm(branch - saved_branch)
                relative = float(difference / torch.linalg.vector_norm(saved_branch).clamp_min(1e-12))
                need(relative <= 3e-4, "cpu_replay_branch:" + prefix + case["id"])
                max_abs = 0.0
                for q, saved in enumerate(saved_rows):
                    first = max(0, q - 127)
                    rebuilt = torch.cat((prob[:, :, q, first:q + 1], prob[:, :, q, -1:]),
                                        dim=-1).reshape(64, -1)
                    max_abs = max(max_abs, float((rebuilt - saved).abs().max()))
                need(max_abs <= 5e-4, "cpu_replay_probability:" + prefix + case["id"])
            bounded(started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    audit(args.manifest, args.binding, args.output, args.report, args.replay)
