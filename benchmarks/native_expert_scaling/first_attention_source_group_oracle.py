"""Sealed CPU-only one-of-16 source attention contribution oracle."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
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

PROTOCOL = ROOT / "docs/research/PIPELINE_REASSESSMENT_20261015/FIRST_ATTENTION_SOURCE_GROUP_ORACLE_PROTOCOL_20261015.md"
MAX_WALL = 600
MAX_RSS = 3 << 30
MAX_OUTPUT = 512 << 20
LAYOUT = "query,kv_head,query_per_kv_head,allowed_key_oldest_to_newest,sink_last"
NEEDED = {
    "model.layers.0.input_layernorm.weight": (2880,),
    "model.layers.0.self_attn.v_proj.weight": (512, 2880),
    "model.layers.0.self_attn.v_proj.bias": (512,),
    "model.layers.0.self_attn.o_proj.weight": (2880, 4096),
    "model.layers.0.self_attn.o_proj.bias": (2880,),
}


def need(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_of(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve()


def check_desc(d, shape):
    p = path_of(d["path"])
    need(d["dtype"] == "BF16" and tuple(d["shape"]) == tuple(shape) and
         d["bytes"] == p.stat().st_size == math.prod(shape) * 2 and
         d["sha256"] == sha(p), "source_descriptor:" + str(p))
    return p


def bf16(d, shape):
    p = check_desc(d, shape)
    t = torch.frombuffer(p.read_bytes(), dtype=torch.uint16).clone().view(torch.bfloat16)
    t = t.reshape(shape)
    need(bool(torch.isfinite(t.float()).all()), "source_values_finite")
    return t.float()


def pins(args):
    manifest = args.manifest.resolve(); producer = args.producer.resolve()
    numerical = args.numerical_audit.resolve()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    r = json.loads((producer / "result.json").read_text(encoding="utf-8"))
    a = json.loads(numerical.read_text(encoding="utf-8"))
    need(m["schema"] == "FIRST_ATTENTION_SOURCE_V1" and
         m["source_model"] == "openai/gpt-oss-120b" and
         m["source_revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
         "source_identity")
    need(r["status"] == "FAILED_GPU_RELEASE" and r["post_binding_match"] is True and
         r["gpu_released_before_posthash"] is False and
         r["source_trace_count"] == 36, "producer_cleanup_status")
    need(a["status"] == "PASS_NUMERICAL_REJECTION_PRODUCER_FAILED" and
         a["producer_status"] == "FAILED_GPU_RELEASE" and
         a["decisions"]["independent_numerical_status"] == "RANK256_F32_REJECT" and
         a["decisions"]["precision_pass"] is True and
         a["decisions"]["rank256_pass"] is False and
         a["pins_sha256"]["producer_result"] == sha(producer / "result.json") and
         a["pins_sha256"]["manifest"] == sha(manifest), "numerical_audit_custody")
    cases = m["cases"]
    need(len(cases) == 36 and len({c["id"] for c in cases}) == 36 and
         sum(c["split"] == "FIT" for c in cases) == 24 and
         sum(c["split"] == "DEV" for c in cases) == 12 and
         m["common_prefix_length"] == 62, "fixed_36_cases_query_mask")
    rows = m["embedding_rows"]
    check_desc(rows, (len(rows["token_ids"]), 2880))
    files = {"code": Path(__file__).resolve(), "protocol": PROTOCOL.resolve(),
             "manifest": manifest, "producer_result": producer / "result.json",
             "numerical_audit": numerical, "embedding_rows": path_of(rows["path"]),
             "source_config": path_of(m["config"]["path"]),
             "source_reference": path_of(m["reference"]["path"]),
             "torch": Path(torch.__file__).resolve(),
             "torch_native": Path(torch._C.__file__).resolve(),
             "psutil": Path(psutil.__file__).resolve(),
             "psutil_native": Path(psutil._psutil_windows.__file__).resolve()}
    for name in ("torch_cpu.dll", "torch_python.dll", "c10.dll"):
        files[name] = Path(torch.__file__).resolve().parent / "lib" / name
    trace_keys = set()
    for name, shape in NEEDED.items():
        d = m["tensors"][name]
        files["tensor:" + name] = check_desc(d, shape)
    owner = {Path(d["path"]).resolve(): d["sha256"] for d in r["owned_outputs"]}
    for case in cases:
        need(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case["id"]) is not None,
             "safe_case_id")
        need(len(case["input_ids"]) > 62 and len(case["input_ids"]) <= 256,
             "case_token_length")
        for stem in ("source_", "f32_source_"):
            p = (producer / (stem + case["id"] + ".pt")).resolve()
            need(p.parent == producer and owner.get(p) is not None, "owned_trace:" + case["id"])
            key = stem + case["id"]
            files[key] = p
            trace_keys.add(key)
    frozen = {key: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
              for key, p in sorted(files.items())}
    for key, value in frozen.items():
        if key in trace_keys:
            need(owner[Path(value["path"])] == value["sha256"], "producer_owned_trace_sha")
    return frozen, m, r


def save_json_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def check_bound(start, output):
    need(time.monotonic() - start < MAX_WALL, "wall_10min")
    memory = psutil.Process().memory_info()
    need(memory.rss < MAX_RSS and getattr(memory, "peak_wset", memory.rss) < MAX_RSS,
         "rss_or_peak_3gib")
    need(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) < MAX_OUTPUT,
         "output_512mib")


def trace(path, case, dtype, with_prob):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    n = len(case["input_ids"])
    branch = obj["branch"]
    need(obj["input_ids"] == case["input_ids"] and obj["split"] == case["split"] and
         obj["domain"] == case["domain"] and branch.shape == (n, 2880) and
         branch.dtype == dtype and bool(torch.isfinite(branch.float()).all()),
         "trace_identity_shape")
    if not with_prob:
        return branch.float(), None
    packed = obj["prob_packed"]
    need(packed["layout"] == LAYOUT and packed["window"] == 128 and
         packed["heads"] == 64 and packed["length"] == n and
         packed["values"].dtype == dtype, "probability_layout")
    values = packed["values"].float()
    need(values.numel() == 64 * sum(min(q + 1, 128) + 1 for q in range(n)) and
         bool(torch.isfinite(values).all()) and bool((values >= 0).all()),
         "probability_values")
    return branch.float(), values


def context_from_prob(values, v):
    n = v.shape[0]
    context = torch.empty((n, 4096), dtype=torch.float32)
    offset = 0
    for q in range(n):
        first = max(0, q - 127)
        count = q - first + 1
        row = values[offset:offset + 64 * (count + 1)].reshape(8, 8, count + 1)
        need(float((row.sum(-1) - 1).abs().max()) <= .0002,
             "source_f32_probability_simplex")
        # Last category is the sink; it contributes no V vector.
        ctx = torch.einsum("hmc,chd->hmd", row[:, :, :count], v[first:q + 1])
        context[q] = ctx.reshape(4096)
        offset += 64 * (count + 1)
    need(offset == values.numel(), "probability_exact_consumption")
    return context


def case_oracle(case, m, rows, lookup, w, output, common):
    n = len(case["input_ids"])
    x = rows[[lookup[i] for i in case["input_ids"]]]
    truth, _ = trace(output["producer"] / ("source_" + case["id"] + ".pt"),
                     case, torch.bfloat16, False)
    saved_f32, probs = trace(output["producer"] / ("f32_source_" + case["id"] + ".pt"),
                             case, torch.float32, True)
    with torch.no_grad():
        norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
        norm *= w["norm"]
        v = F.linear(norm, w["v_weight"], w["v_bias"]).reshape(n, 8, 64)
        context = context_from_prob(probs, v)
        contributions = torch.stack([
            F.linear(context[:, 256*g:256*(g+1)].double(),
                     w["o_weight"][:, 256*g:256*(g+1)].double())
            for g in range(16)], dim=1)
        bias = w["o_bias"].double()
        target = truth.double() - bias
        dot = (contributions * target[:, None, :]).sum(-1)
        norm2 = contributions.square().sum(-1)
        a = torch.where(norm2 > 0, dot / norm2.clamp_min(1e-300),
                        torch.zeros_like(dot))
        target2 = target.square().sum(-1, keepdim=True)
        formula_residual2 = (target2 - 2*a*dot + a.square()*norm2).clamp_min(0)
        residual2 = (a[:, :, None] * contributions - target[:, None, :]).square().sum(-1)
        need(bool(((residual2 - formula_residual2).abs() <=
                   1e-11 * target2.clamp_min(1e-30)).all()),
             "direct_vs_analytic_residuals")
        need(bool(torch.isfinite(a).all()) and bool(torch.isfinite(residual2).all()),
             "finite_unbounded_real_oracle")
        chosen = residual2.argmin(dim=1)  # torch.argmin keeps the first tied ID.
        i = torch.arange(n)
        chosen_a = a[i, chosen]
        oracle = bias + chosen_a[:, None] * contributions[i, chosen]
        all_group = bias + contributions.sum(dim=1)
        clipped_a = a.clamp(0, 2)
        clipped_residual2 = (target2 - 2*clipped_a*dot + clipped_a.square()*norm2).clamp_min(0)
        clipped_id = clipped_residual2.argmin(dim=1)
        clipped = bias + clipped_a[i, clipped_id][:, None] * contributions[i, clipped_id]
        full_relative = float(torch.linalg.vector_norm(all_group - saved_f32.double()) /
                              torch.linalg.vector_norm(saved_f32.double()).clamp_min(1e-30))
        need(full_relative <= 3e-4, "all_groups_vs_saved_f32:" + case["id"])
        record = {"case_id": case["id"], "split": case["split"], "domain": case["domain"],
                  "input_ids": case["input_ids"], "common_prefix_length": common,
                  "selected_group_id": chosen.to(torch.int16), "mass_f64": chosen_a,
                  "analytic_residual2_f64": residual2,
                  "quadratic_formula_residual2_f64": formula_residual2,
                  "oracle_branch_f64": oracle, "all_group_branch_f64": all_group,
                  "control_relative_l2_vs_saved_f32": full_relative}
        torch.save(record, output["directory"] / ("case_" + case["id"] + ".pt"))
        return {"case_id": case["id"], "split": case["split"],
                "domain": case["domain"], "all_group": scores(all_group, truth, common),
                "oracle": scores(oracle, truth, common),
                "clipped_diagnostic": scores(clipped, truth, common),
                "control_relative_l2_vs_saved_f32": full_relative,
                "negative_mass_count": int((chosen_a < 0).sum()),
                "abs_mass_gt_2_count": int((chosen_a.abs() > 2).sum()),
                "mass_abs_mean": float(chosen_a.abs().mean()),
                "mass_abs_max": float(chosen_a.abs().max()),
                "variable_negative_mass_count": int((chosen_a[common:] < 0).sum()),
                "variable_abs_mass_gt_2_count": int((chosen_a[common:].abs() > 2).sum()),
                "variable_mass_abs_mean": float(chosen_a[common:].abs().mean()),
                "variable_mass_abs_max": float(chosen_a[common:].abs().max()),
                "selected_group_counts": [int((chosen == g).sum()) for g in range(16)],
                "variable_selected_group_counts":
                    [int((chosen[common:] == g).sum()) for g in range(16)],
                "tokens": n, "variable_tokens": n - common}


def scores(branch, truth, common):
    difference = branch.double() - truth.double()
    reference = truth.double()
    e2 = float(difference[common:].square().sum())
    n2 = float(reference[common:].square().sum())
    return {"error2": e2, "norm2": n2,
            "sequence_l2": math.sqrt(e2 / max(n2, 1e-30)),
            "all_token_error2": float(difference.square().sum()),
            "all_token_norm2": float(reference.square().sum())}


def aggregate(rows, key):
    selected = [r[key] for r in rows]
    seq = sorted(r["sequence_l2"] for r in selected)
    e = sum(r["error2"] for r in selected)
    n = sum(r["norm2"] for r in selected)
    ae = sum(r["all_token_error2"] for r in selected)
    an = sum(r["all_token_norm2"] for r in selected)
    return {"pooled_branch_l2": math.sqrt(e / max(n, 1e-30)),
            "sequence_l2_p95": seq[math.ceil(.95 * len(seq)) - 1],
            "worst_sequence_l2": seq[-1],
            "all_token_pooled_l2_diagnostic": math.sqrt(ae / max(an, 1e-30))}


def run(args, binding):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    proc = psutil.Process()
    save_json_exclusive(output / "owned_process.json",
                        {"pid": proc.pid, "pid_create_time": proc.create_time(),
                         "start_wall_time": time.time(), "binding_sha256": sha(args.binding)})
    receipt = {"schema": "FIRST_ATTENTION_SOURCE_GROUP_ORACLE_V1",
               "status": "STARTED", "binding_sha256": sha(args.binding),
               "producer_status": "FAILED_GPU_RELEASE",
               "numerical_audit_sha256": sha(args.numerical_audit)}
    try:
        current, m, r = pins(args)
        need(current == binding["files"] and
             binding["torch_version"] == torch.__version__ and
             binding["psutil_version"] == psutil.__version__, "frozen_binding_changed")
        rows_desc = m["embedding_rows"]
        rows = bf16(rows_desc, (len(rows_desc["token_ids"]), 2880))
        lookup = {t: i for i, t in enumerate(rows_desc["token_ids"])}
        t = m["tensors"]
        w = {"norm": bf16(t["model.layers.0.input_layernorm.weight"], (2880,)),
             "v_weight": bf16(t["model.layers.0.self_attn.v_proj.weight"], (512, 2880)),
             "v_bias": bf16(t["model.layers.0.self_attn.v_proj.bias"], (512,)),
             "o_weight": bf16(t["model.layers.0.self_attn.o_proj.weight"], (2880, 4096)),
             "o_bias": bf16(t["model.layers.0.self_attn.o_proj.bias"], (2880,))}
        cases = sorted(m["cases"], key=lambda c: c["id"])
        records = []
        for case in cases:
            check_bound(start, output)
            records.append(case_oracle(case, m, rows, lookup, w,
                                       {"producer": args.producer.resolve(),
                                        "directory": output}, m["common_prefix_length"]))
            check_bound(start, output)
        metrics = {split: {kind: aggregate([r for r in records if r["split"] == split], kind)
                           for kind in ("all_group", "oracle", "clipped_diagnostic")}
                   for split in ("FIT", "DEV")}
        control_pass = all(metrics[s]["all_group"]["pooled_branch_l2"] <= .01 and
                           metrics[s]["all_group"]["sequence_l2_p95"] <= .05
                           for s in ("FIT", "DEV"))
        oracle_pass = all(metrics[s]["oracle"]["pooled_branch_l2"] <= .01 and
                          metrics[s]["oracle"]["sequence_l2_p95"] <= .05
                          for s in ("FIT", "DEV"))
        receipt.update({"status": ("INCONCLUSIVE_CONTROL" if not control_pass else
                                   "ONE_GROUP_ORACLE_LOWER_BOUND_PASS" if oracle_pass else
                                   "ONE_GROUP_ORACLE_LOWER_BOUND_FAIL"),
                        "metrics": metrics, "cases": records,
                        "control_pass": control_pass, "oracle_pass": oracle_pass,
                        "query_mask_start": m["common_prefix_length"],
                        "fit_cases": 24, "dev_cases": 12,
                        "selection_uses_teacher_on_dev": True,
                        "mass_definition": "unbounded_signed_real_f64",
                        "negative_mass_count": sum(x["negative_mass_count"] for x in records),
                        "abs_mass_gt_2_count": sum(x["abs_mass_gt_2_count"] for x in records),
                        "variable_negative_mass_count":
                            sum(x["variable_negative_mass_count"] for x in records),
                        "variable_abs_mass_gt_2_count":
                            sum(x["variable_abs_mass_gt_2_count"] for x in records),
                        "admitted": False,
                        "owned_pid": proc.pid, "owned_pid_create_time": proc.create_time()})
        need(pins(args)[0] == binding["files"], "post_binding_changed")
    except Exception as exc:
        receipt["status"] = "FAILED"
        receipt["failure"] = f"{type(exc).__name__}:{exc}"
    receipt["wall_seconds"] = time.monotonic() - start
    memory = proc.memory_info()
    receipt["rss_bytes"] = memory.rss
    receipt["peak_wset_bytes"] = getattr(memory, "peak_wset", None)
    receipt["cpu_threads"] = torch.get_num_threads()
    receipt["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    receipt["owned_outputs"] = [{"path": str(p), "bytes": p.stat().st_size,
                                  "sha256": sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
    save_json_exclusive(output / "result.json", receipt)
    save_json_exclusive(output / "terminal.json",
                        {"status": receipt["status"], "result_sha256": sha(output / "result.json"),
                         "pid": proc.pid, "pid_create_time": proc.create_time()})
    need(receipt["status"] != "FAILED", "oracle_run_failed")
    print(json.dumps({"status": receipt["status"], "result": str(output / "result.json")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--producer", type=Path, required=True)
    ap.add_argument("--numerical-audit", type=Path, required=True)
    ap.add_argument("--freeze", type=Path)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--binding", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    need((args.freeze is not None) != args.run, "choose_freeze_or_run")
    if args.freeze is not None:
        need(args.binding is None and args.output is None, "freeze_options")
        files, _, _ = pins(args)
        save_json_exclusive(args.freeze.resolve(),
                            {"schema": "FIRST_ATTENTION_SOURCE_GROUP_ORACLE_BINDING_V1",
                             "files": files, "torch_version": torch.__version__,
                             "psutil_version": psutil.__version__,
                             "cpu_threads": torch.get_num_threads(),
                             "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
    else:
        need(args.binding is not None and args.output is not None, "run_options")
        binding = json.loads(args.binding.read_text(encoding="utf-8"))
        need(binding["schema"] == "FIRST_ATTENTION_SOURCE_GROUP_ORACLE_BINDING_V1" and
             binding["cpu_threads"] == 6 and binding["cuda_visible_devices"] == "",
             "binding_profile")
        run(args, binding)


if __name__ == "__main__":
    main()
