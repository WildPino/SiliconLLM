"""Actual-weight first-attention Q/O rank-256 falsifier; no acquisition code.

Only layer zero of pinned GPT-OSS-120B is evaluated. The source reference is
loaded by selected AST nodes; no model checkpoint or expert/head is imported.
"""
import argparse
import ast
import gc
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import sys
import time
import subprocess

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "results/native_expert_scaling/chatbot_source_runtime/site"
sys.path.insert(0, str(SITE))
import psutil
IS_WATCHDOG = sys.argv[1:2] == ["--watchdog"]
if not IS_WATCHDOG:
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)

HERE = Path(__file__).resolve().parent
DOC = ROOT / "docs/research/PIPELINE_REASSESSMENT_20261015"
CONFIG = DOC / "large_family_metadata/config120.json"
REFERENCE = DOC / "large_family_metadata/reference_model.py"
PROTOCOL = DOC / "FIRST_ATTENTION_RANK_PROTOCOL_20261015.md"
MODEL_REV = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
REF_SHA = "319ba3c6e4d827eb3115fe4838942af448376b513c02cea4b6b471bf6c21a346"
CONFIG_SHA = "933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566"
RANK = 256
MAX_STEPS = 512
LR = 1e-3
RIDGE = 1e-4
MAX_GPU_ALLOC = 8 << 30
MAX_GPU_RESERVE = 9 << 30
MAX_CPU_RSS = 8 << 30
MAX_OUTPUT = 2 << 30
MAX_GPU_SECONDS = 7200
MAX_TOTAL_SECONDS = 9000
ACTIVE_WATCHDOG = None
KEY_SHAPES = {
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


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def descriptor(d, shape, dtype="BF16"):
    path = local_path(d["path"])
    require(tuple(d["shape"]) == tuple(shape) and d["dtype"] == dtype, "tensor_shape_dtype")
    expected = math.prod(shape) * (2 if dtype == "BF16" else 4)
    require(d["bytes"] == expected == path.stat().st_size, "tensor_exact_byte_length")
    require(sha(path) == d["sha256"], "tensor_sha256")
    return path


def file_pin(d):
    path = local_path(d["path"])
    require(path.stat().st_size == d["bytes"] and sha(path) == d["sha256"], "file_pin")
    return path


def ids_sha(ids):
    return hashlib.sha256(struct.pack("<" + "I" * len(ids), *ids)).hexdigest()


def validate_manifest(path):
    m = json.loads(path.read_text(encoding="utf-8"))
    require(m["schema"] == "FIRST_ATTENTION_SOURCE_V1", "manifest_schema")
    require(m["source_model"] == "openai/gpt-oss-120b" and m["source_revision"] == MODEL_REV,
            "source_revision")
    require(sha(CONFIG) == CONFIG_SHA and sha(REFERENCE) == REF_SHA, "official_pins")
    for key, fixed, seal in (("config", CONFIG, CONFIG_SHA), ("reference", REFERENCE, REF_SHA)):
        d = m[key]
        require(local_path(d["path"]) == fixed.resolve() and d["bytes"] == fixed.stat().st_size
                and d["sha256"] == seal, key + "_descriptor")
        require(file_pin(d) == fixed.resolve(), key + "_source_path")
    file_pin(m["index"])
    require(set(m["tensors"]) == set(KEY_SHAPES), "exact_first_attention_tensor_keys")
    for name, shape in KEY_SHAPES.items():
        d = m["tensors"][name]
        require(d["name"] == name and d["source_shard"] and d["source_offset"] >= 0,
                "source_tensor_provenance")
        descriptor(d, shape)
    er = m["embedding_rows"]
    token_ids = er["token_ids"]
    require(token_ids == sorted(set(token_ids)) and all(0 <= i < 201088 for i in token_ids),
            "embedding_token_ids")
    descriptor(er, (len(token_ids), 2880))
    cases = m["cases"]
    require(len(cases) == 36 and len({c["id"] for c in cases}) == 36, "case_count_unique")
    require(sum(c["split"] == "FIT" for c in cases) == 24 and
            sum(c["split"] == "DEV" for c in cases) == 12, "fit_dev_count")
    fit_domains = {c["domain"] for c in cases if c["split"] == "FIT"}
    dev_domains = {c["domain"] for c in cases if c["split"] == "DEV"}
    require(len(fit_domains) == 8 and len(dev_domains) == 4 and
            not fit_domains & dev_domains, "whole_source_domain_holdout")
    fit_groups = {c["group_key"] for c in cases if c["split"] == "FIT"}
    dev_groups = {c["group_key"] for c in cases if c["split"] == "DEV"}
    fit_users = {c["first_user_key"] for c in cases if c["split"] == "FIT"}
    dev_users = {c["first_user_key"] for c in cases if c["split"] == "DEV"}
    require(not fit_groups & dev_groups and not fit_users & dev_users, "split_group_disjoint")
    for domain in fit_domains:
        require(sum(c["domain"] == domain and c["split"] == "FIT" for c in cases) == 3,
                "three_fit_per_domain")
    for domain in dev_domains:
        require(sum(c["domain"] == domain and c["split"] == "DEV" for c in cases) == 3,
                "three_dev_per_domain")
    require(dev_domains == {"smol-magpie-ultra", "metamathqa-50k",
                            "smol-constraints", "self-oss-instruct"}, "fixed_dev_domains")
    universe = set(token_ids)
    common = m["common_prefix_length"]
    require(isinstance(common, int) and 0 <= common < 256, "common_prefix_length")
    for c in cases:
        ids = c["input_ids"]
        canonical = c["canonical_input_ids"]
        require(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["id"]) is not None, "safe_case_id")
        require(c["split"] in ("FIT", "DEV") and common < len(ids) <= 256 and
                len(canonical) == c["original_length"] and
                ids == canonical[:256] and set(ids) <= universe, "case_ids")
        require(c["canonical_ids_sha256"] == ids_sha(canonical), "canonical_ids_hash")
        require(c["used_ids_sha256"] == ids_sha(ids), "capped_ids_hash")
        require(len(c["canonical_ids_sha256"]) == 64 and len(c["prefix_sha256"]) == 64
                and isinstance(c["messages"], list), "case_source_pins")
    prefixes = [c["input_ids"] for c in cases]
    observed_common = 0
    while observed_common < min(len(ids) for ids in prefixes) and all(
            ids[observed_common] == prefixes[0][observed_common] for ids in prefixes):
        observed_common += 1
    require(common == observed_common, "actual_common_prefix_length")
    require(m["input_pins"] and m["tokenizer_files"] and len(m["selection_plan_sha256"]) == 64,
            "corpus_selection_pins")
    for d in [*m["input_pins"], *m["tokenizer_files"]]:
        file_pin(d)
    require(sha(path.parent / "selection_plan.json") == m["selection_plan_sha256"] and
            sha(path.parent / "eligibility_plan.json") == m["eligibility_plan_sha256"],
            "selection_and_eligibility_pins")
    return m


def load_raw_bf16(d, shape):
    path = descriptor(d, shape)
    # frombuffer is read-only; clone before moving to GPU.
    value = torch.frombuffer(path.read_bytes(), dtype=torch.uint16).clone().view(torch.bfloat16).reshape(shape)
    require(bool(torch.isfinite(value.float()).all()), "source_tensor_finite")
    return value


def official_nodes():
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"))
    names = {"ModelConfig", "RMSNorm", "RotaryEmbedding", "_apply_rotary_emb", "sdpa", "AttentionBlock"}
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    require({n.name for n in selected} == names, "official_ast_nodes")
    scope = {"torch": torch, "math": math, "dataclass": dataclass}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(REFERENCE), "exec"), scope)
    return scope


def source_block(scope, tensors, device):
    config = scope["ModelConfig"]()
    hf = json.loads(CONFIG.read_text(encoding="utf-8"))
    exact_fields = {
        "num_hidden_layers": "num_hidden_layers", "experts_per_token": "experts_per_token",
        "vocab_size": "vocab_size", "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size", "swiglu_limit": "swiglu_limit",
        "head_dim": "head_dim", "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads", "sliding_window": "sliding_window",
        "initial_context_length": "initial_context_length", "rope_theta": "rope_theta",
    }
    for reference_name, hf_name in exact_fields.items():
        require(getattr(config, reference_name) == hf[hf_name],
                "reference_config_" + reference_name)
    require(config.num_experts == hf["num_local_experts"] and
            config.rope_scaling_factor == hf["rope_scaling"]["factor"] and
            config.rope_ntk_alpha == hf["rope_scaling"]["beta_slow"] and
            config.rope_ntk_beta == hf["rope_scaling"]["beta_fast"] and
            hf["rope_scaling"]["original_max_position_embeddings"] ==
            config.initial_context_length and hf["rope_scaling"]["rope_type"] == "yarn" and
            hf["layer_types"][0] == "sliding_attention", "reference_rope_layer_config")
    block = scope["AttentionBlock"](config, 0, device=device).eval()
    prefix = "model.layers.0.self_attn."
    with torch.no_grad():
        block.norm.scale.copy_(tensors["model.layers.0.input_layernorm.weight"].float())
        block.qkv.weight.copy_(torch.cat([tensors[prefix + k + "_proj.weight"]
                                          for k in ("q", "k", "v")], dim=0))
        block.qkv.bias.copy_(torch.cat([tensors[prefix + k + "_proj.bias"]
                                        for k in ("q", "k", "v")], dim=0))
        block.out.weight.copy_(tensors[prefix + "o_proj.weight"])
        block.out.bias.copy_(tensors[prefix + "o_proj.bias"])
        block.sinks.copy_(tensors[prefix + "sinks"])
    for p in block.parameters():
        p.requires_grad_(False)
    return block


def probabilities(q, k, v, sinks, scale, window):
    n, heads, multiple, dim = q.shape
    kk = k[:, :, None, :].expand(-1, -1, multiple, -1)
    vv = v[:, :, None, :].expand(-1, -1, multiple, -1)
    s = sinks.reshape(heads, multiple, 1, 1).expand(-1, -1, n, -1)
    mask = torch.triu(q.new_full((n, n), -float("inf")), diagonal=1)
    if window:
        mask += torch.tril(mask.new_full((n, n), -float("inf")), diagonal=-window)
    logits = torch.einsum("qhmd,khmd->hmqk", q, kk)
    logits *= scale
    logits += mask[None, None, :, :]
    probs = torch.softmax(torch.cat([logits, s], dim=-1), dim=-1)
    context = torch.einsum("hmqk,khmd->qhmd", probs[..., :-1], vv).reshape(n, -1)
    return context, probs


def forward_attention(x, block, qfactor=None, ofactor=None, f32=False):
    dtype = torch.float32 if f32 else torch.bfloat16
    x = x.to(dtype)
    if f32:
        normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + block.norm.eps)
        normed = normed * block.norm.scale.float()
        qkv_weight = block.qkv.weight.float()
        qkv_bias = block.qkv.bias.float()
    else:
        normed = block.norm(x)
        qkv_weight, qkv_bias = block.qkv.weight, block.qkv.bias
    if qfactor is None:
        qkv = F.linear(normed, qkv_weight, qkv_bias)
        q = qkv[:, :4096].contiguous()
    else:
        q = F.linear(F.linear(normed.float(), qfactor[1]), qfactor[0], qkv_bias[:4096].float())
        kv = F.linear(normed.float(), qkv_weight[4096:].float(), qkv_bias[4096:].float())
        qkv = torch.cat([q, kv], dim=-1)
    k, v = qkv[:, 4096:4608].contiguous(), qkv[:, 4608:5120].contiguous()
    q = q.view(-1, 8, 8, 64); k = k.view(-1, 8, 64); v = v.view(-1, 8, 64)
    q, k = block.rope(q, k)
    context, probs = probabilities(q, k, v, block.sinks.float() if f32 else block.sinks,
                                   block.sm_scale, 128)
    if ofactor is None:
        branch = F.linear(context, block.out.weight.float() if f32 else block.out.weight,
                          block.out.bias.float() if f32 else block.out.bias)
    else:
        branch = F.linear(F.linear(context.float(), ofactor[1]), ofactor[0],
                          block.out.bias.float())
    return branch.float(), probs.float(), context.float(), normed.float()


def relative_l2(a, b):
    return float(torch.linalg.vector_norm(a.float() - b.float()) /
                 torch.linalg.vector_norm(b.float()).clamp_min(1e-12))


def tv_rows(a, b):
    a = a.float(); b = b.float()
    a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    b = b / b.sum(-1, keepdim=True).clamp_min(1e-12)
    return 0.5 * (a - b).abs().sum(-1).flatten()


def metrics(rows):
    e2 = sum(r["error2"] for r in rows)
    n2 = sum(r["norm2"] for r in rows)
    seq = sorted(r["seq_l2"] for r in rows)
    tv = torch.cat([r["tv"] for r in rows]).sort().values
    all_e2 = sum(r["all_error2"] for r in rows)
    all_n2 = sum(r["all_norm2"] for r in rows)
    all_tv = torch.cat([r["all_tv"] for r in rows]).sort().values
    bias_e2 = sum(r["bias_removed_error2"] for r in rows)
    bias_n2 = sum(r["bias_removed_norm2"] for r in rows)
    def p95_list(x):
        return x[min(len(x) - 1, math.ceil(.95 * len(x)) - 1)]
    return {"pooled_branch_l2": math.sqrt(e2 / max(n2, 1e-30)),
            "sequence_l2_p95": p95_list(seq),
            "tv_mean_including_sink": float(tv.mean()),
            "tv_p95_including_sink": float(p95_list(tv)),
            "worst_sequence_l2": max(seq), "worst_tv_row": float(tv[-1]),
            "all_token_pooled_branch_l2_diagnostic": math.sqrt(all_e2 / max(all_n2, 1e-30)),
            "bias_removed_pooled_branch_l2_diagnostic": math.sqrt(bias_e2 / max(bias_n2, 1e-30)),
            "all_token_tv_mean_diagnostic": float(all_tv.mean()),
            "all_token_tv_p95_diagnostic": float(p95_list(all_tv))}


def measure_case(candidate, target, case, common, bias):
    branch, prob = candidate
    truth, truth_prob = target
    all_error = (branch.float() - truth.float()).flatten()
    all_ref = truth.float().flatten()
    error = (branch[common:].float() - truth[common:].float()).flatten()
    ref = truth[common:].float().flatten()
    bias_removed = (truth[common:].float() - bias.float()).flatten()
    return {"id": case["id"], "domain": case["domain"], "split": case["split"],
            "error2": float(error.square().sum()), "norm2": float(ref.square().sum()),
            "seq_l2": relative_l2(branch[common:], truth[common:]),
            "tv": tv_rows(prob[..., common:, :], truth_prob[..., common:, :]).cpu(),
            "all_error2": float(all_error.square().sum()),
            "all_norm2": float(all_ref.square().sum()),
            "bias_removed_error2": float(error.square().sum()),
            "bias_removed_norm2": float(bias_removed.square().sum()),
            "all_seq_l2": relative_l2(branch, truth),
            "all_tv": tv_rows(prob, truth_prob).cpu()}


def weighted_svd(weight, inputs, rank):
    x = inputs.float()
    cov = x.T @ x / x.shape[0]
    damping = RIDGE * torch.diagonal(cov).mean().clamp_min(1e-12)
    chol = torch.linalg.cholesky(cov + damping * torch.eye(cov.shape[0], device=x.device))
    u, sigma, vh = torch.linalg.svd(weight.float() @ chol, full_matrices=False)
    left = u[:, :rank] * sigma[:rank].sqrt()[None, :]
    right = sigma[:rank].sqrt()[:, None] * torch.linalg.solve_triangular(
        chol.T, vh[:rank].T, upper=True).T
    return torch.nn.Parameter(left.contiguous()), torch.nn.Parameter(right.contiguous()), float(damping)


def o_initializer(context, branches, rank):
    x = context.float(); y = branches.float()
    cov = x.T @ x / x.shape[0]
    damping = RIDGE * torch.diagonal(cov).mean().clamp_min(1e-12)
    stable = cov + damping * torch.eye(cov.shape[0], device=x.device)
    ls = torch.linalg.solve(stable, x.T @ y / x.shape[0]).T
    left, right, _ = weighted_svd(ls, x, rank)
    return left, right, float(damping)


def packed_prob(prob):
    """Query-major, then 64 heads, then allowed keys oldest-to-newest plus sink."""
    p = prob.detach().cpu()
    n = p.shape[-2]
    require(tuple(p.shape) == (8, 8, n, n + 1), "probability_shape")
    flat = p.reshape(64, n, n + 1)
    pieces = []
    for q in range(n):
        first = max(0, q - 127)
        pieces.append(torch.cat((flat[:, q, first:q + 1], flat[:, q, -1:]),
                                dim=-1).reshape(-1))
    return {"values": torch.cat(pieces).contiguous(), "length": n,
            "layout": "query,kv_head,query_per_kv_head,allowed_key_oldest_to_newest,sink_last",
            "window": 128, "heads": 64}


def factor_state(params):
    return {name: p.detach().cpu().clone() for name, p in
            zip(("q_left", "q_right", "o_left", "o_right"), params)}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def parameter_hash(params):
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().float().cpu().numpy().tobytes(order="C"))
    return h.hexdigest()


def source_parameter_hash(block):
    h = hashlib.sha256()
    for name, param in sorted(block.named_parameters()):
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(tuple(param.shape)).encode("ascii") + b"\0")
        h.update(str(param.dtype).encode("ascii") + b"\0")
        h.update(param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def source_parameter_identity(block):
    return {name: {"data_ptr": param.data_ptr(), "version": param._version,
                   "requires_grad": param.requires_grad,
                   "shape": list(param.shape), "dtype": str(param.dtype)}
            for name, param in block.named_parameters()}


def runtime_seal(manifest_path):
    paths = {"code": Path(__file__).resolve(), "protocol": PROTOCOL, "config": CONFIG,
             "reference": REFERENCE, "manifest": manifest_path.resolve(),
             "torch": Path(torch.__file__).resolve(),
             "torch_native": Path(torch._C.__file__).resolve(),
             "torch_cuda": Path(torch.cuda.__file__).resolve(),
             "psutil": Path(psutil.__file__).resolve(),
             "psutil_native": Path(psutil._psutil_windows.__file__).resolve()}
    for filename in ("torch_cpu.dll", "torch_cuda.dll", "torch_python.dll",
                     "c10.dll", "c10_cuda.dll"):
        library = Path(torch.__file__).resolve().parent / "lib" / filename
        require(library.is_file(), "missing_runtime_library_" + filename)
        paths[filename] = library
    subprocess = __import__("subprocess")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    blobs = {name: subprocess.check_output(["git", "hash-object", str(path)], cwd=ROOT,
                                           text=True).strip() for name, path in paths.items()
             if name in ("code", "protocol", "config", "reference")}
    return {"schema": "FIRST_ATTENTION_RANK_BINDING_V1", "git_head": head,
            "sha256": {name: sha(path) for name, path in paths.items()},
            "git_blobs": blobs,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "psutil_version": psutil.__version__, "manifest_path": str(manifest_path.resolve()),
            "rank": RANK, "steps": MAX_STEPS, "learning_rate": LR, "ridge": RIDGE}


def same_frozen_inputs(actual, frozen):
    # An unrelated later commit may move HEAD; pinned blob and byte hashes do not.
    return {k: v for k, v in actual.items() if k != "git_head"} == {
        k: v for k, v in frozen.items() if k != "git_head"}


def atomic_json(path, value):
    require(not path.exists(), "exclusive_json_exists")
    pending = path.with_name(path.name + ".publishing")
    with pending.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    # Windows rename fails if the final exclusive destination already exists.
    os.rename(pending, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--freeze", type=Path)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--binding", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    require(args.freeze is not None or args.run, "freeze_or_run_required")
    require(not (args.freeze is not None and args.run), "exclusive_command")
    manifest_path = args.manifest.resolve()
    validate_manifest(manifest_path)
    seal = runtime_seal(manifest_path)
    if args.freeze is not None:
        require(not args.binding and not args.output, "freeze_paths")
        args.freeze.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.freeze, seal)
        return
    require(args.binding and args.output and
            same_frozen_inputs(seal, json.loads(args.binding.read_text())),
            "binding_mismatch")
    run(args, seal)


def run(args, seal):
    global ACTIVE_WATCHDOG
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    pid = os.getpid(); created = psutil.Process(pid).create_time()
    receipt = {"schema": "FIRST_ATTENTION_RANK_RESULT_V1", "binding_sha256": sha(args.binding),
               "manifest_sha256": seal["sha256"]["manifest"], "pid": pid,
               "pid_create_time": created, "status": "STARTED", "owned_outputs": []}
    atomic_json(output / "owned_process.json",
                {"pid": pid, "pid_create_time": created, "start_wall_time": time.time(),
                 "wall_deadline_seconds": MAX_TOTAL_SECONDS, "cpu_rss_cap": MAX_CPU_RSS,
                 "output_cap": MAX_OUTPUT, "binding_sha256": receipt["binding_sha256"]})
    watchdog_log = (output / "watchdog.log").open("xb")
    watcher = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-X", "utf8",
                                str(Path(__file__).resolve()), "--watchdog",
                                str(pid), str(created), str(output), str(time.time())],
                               cwd=ROOT, stdout=watchdog_log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    ACTIVE_WATCHDOG = watcher
    atomic_json(output / "watchdog_process.json",
                {"pid": watcher.pid, "pid_create_time": psutil.Process(watcher.pid).create_time()})
    try:
        ready_deadline = time.monotonic() + 30
        while not (output / "watchdog_ready.json").exists():
            require(watcher.poll() is None and time.monotonic() < ready_deadline,
                    "watchdog_failed_to_start")
            time.sleep(.05)
        result = experiment(validate_manifest(args.manifest.resolve()), output, receipt, start)
        receipt.update(result)
    except Exception as exc:
        receipt["status"] = "FAILED"
        receipt["failure"] = f"{type(exc).__name__}:{exc}"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        receipt["gpu_released_before_posthash"] = torch.cuda.memory_allocated() == 0
        if not receipt["gpu_released_before_posthash"]:
            receipt["status"] = "FAILED_GPU_RELEASE"
    receipt["wall_seconds"] = time.monotonic() - start
    receipt["cpu_rss_bytes"] = psutil.Process(pid).memory_info().rss
    receipt["gpu_allocated_bytes"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    receipt["gpu_reserved_bytes"] = torch.cuda.memory_reserved() if torch.cuda.is_available() else None
    receipt["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    receipt["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    try:
        validate_manifest(args.manifest.resolve())
        receipt["post_binding_match"] = same_frozen_inputs(runtime_seal(args.manifest.resolve()), seal)
    except Exception as exc:
        receipt["post_binding_match"] = False
        receipt["post_binding_failure"] = f"{type(exc).__name__}:{exc}"
    if not receipt["post_binding_match"]:
        receipt["status"] = "FAILED_POST_BINDING"
    receipt["owned_outputs"] = [{"path": str(p), "bytes": p.stat().st_size,
                                 "sha256": sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
    atomic_json(output / "result.json", receipt)
    (output / "watchdog_done").write_text("complete\n", encoding="ascii")
    watcher.wait(timeout=10)
    watchdog_log.close()
    require(watcher.returncode == 0, "watchdog_nonzero_exit")
    print(json.dumps({"status": receipt["status"], "result": str(output / "result.json")}))
    require(receipt["status"] in ("RANK256_F32_GATE_PASS_PHYSICAL_Q4_PENDING",
                                   "RANK256_F32_REJECT", "INCONCLUSIVE_PRECISION_FLOOR"),
            "experiment_failed")


def watchdog(pid, created, output, start_wall):
    target = psutil.Process(pid)
    require(abs(target.create_time() - created) < .01, "watchdog_target_identity")
    atomic_json(output / "watchdog_ready.json",
                {"pid": os.getpid(), "target_pid": pid, "target_created": created})
    while True:
        if (output / "watchdog_done").exists():
            return
        try:
            if abs(target.create_time() - created) > .01 or not target.is_running():
                return
            rss = target.memory_info().rss + psutil.Process().memory_info().rss
            bytes_out = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            gpu_marker = output / "gpu_phase.json"
            gpu_expired = (gpu_marker.exists() and time.time() -
                           json.loads(gpu_marker.read_text())["start_wall_time"] > MAX_GPU_SECONDS)
            reason = ("gpu_deadline" if gpu_expired else
                      "wall_deadline" if time.time() - start_wall > MAX_TOTAL_SECONDS else
                      "cpu_rss_cap" if rss > MAX_CPU_RSS else
                      "output_cap" if bytes_out > MAX_OUTPUT else None)
            if reason:
                atomic_json(output / "watchdog_failure.json",
                            {"reason": reason, "target_pid": pid, "target_created": created,
                             "watchdog_pid": os.getpid(), "rss_bytes": rss,
                             "output_bytes": bytes_out, "wall_seconds": time.time() - start_wall})
                target.kill()
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(.5)


def guard(start, gpu_start, output):
    require(ACTIVE_WATCHDOG is not None and ACTIVE_WATCHDOG.poll() is None,
            "watchdog_not_live")
    require(time.monotonic() - start < MAX_TOTAL_SECONDS and
            time.monotonic() - gpu_start < MAX_GPU_SECONDS, "deadline")
    require(psutil.Process().memory_info().rss < MAX_CPU_RSS, "cpu_rss_cap")
    require(torch.cuda.memory_allocated() < MAX_GPU_ALLOC and
            torch.cuda.memory_reserved() < MAX_GPU_RESERVE and
            torch.cuda.max_memory_allocated() < MAX_GPU_ALLOC and
            torch.cuda.max_memory_reserved() < MAX_GPU_RESERVE, "gpu_memory_cap")
    require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < MAX_OUTPUT,
            "output_cap")


def experiment(m, output, receipt, start):
    atomic_json(output / "gpu_phase.json", {"start_wall_time": time.time()})
    require(torch.cuda.is_available(), "cuda_required")
    free, total = torch.cuda.mem_get_info()
    require(free >= 4096 << 20, "gpu_free_preflight_4gib")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(MAX_GPU_ALLOC / total)
    torch.manual_seed(20261015); random.seed(20261015)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_start = time.monotonic()
    stage_start = gpu_start
    receipt["stage_seconds"] = {}
    device = torch.device("cuda")
    weights = {name: load_raw_bf16(m["tensors"][name], shape).to(device)
               for name, shape in KEY_SHAPES.items()}
    scope = official_nodes(); block = source_block(scope, weights, device)
    source_before = source_parameter_hash(block)
    source_identity_before = source_parameter_identity(block)
    receipt["source_parameter_sha256_before"] = source_before
    receipt["source_parameter_identity_before"] = source_identity_before
    receipt["stage_seconds"]["source_weight_import"] = time.monotonic() - stage_start
    stage_start = time.monotonic()
    er = m["embedding_rows"]
    rows = load_raw_bf16(er, (len(er["token_ids"]), 2880))
    lookup = {token: j for j, token in enumerate(er["token_ids"])}
    cases = sorted(m["cases"], key=lambda c: c["id"])
    fit = [c for c in cases if c["split"] == "FIT"]
    dev = [c for c in cases if c["split"] == "DEV"]
    targets = {}; precision = []
    for case in cases:
        guard(start, gpu_start, output)
        x = rows[[lookup[i] for i in case["input_ids"]]].to(device)
        with torch.no_grad():
            captured = []
            handle = block.out.register_forward_hook(
                lambda _module, inputs, branch: captured.append(
                    (inputs[0].detach().clone(), branch.detach().clone())))
            try:
                official = block(x)
            finally:
                handle.remove()
            require(len(captured) == 1, "official_out_capture_count")
            official_context, official_branch = captured[0]
            bfbranch, bfprob, bfcontext, _ = forward_attention(x, block)
            require(torch.equal(bfcontext.to(torch.bfloat16), official_context),
                    "manual_vs_official_pre_o_context")
            require(torch.equal(bfbranch.to(torch.bfloat16), official_branch),
                    "manual_vs_official_pre_residual_branch")
            require(torch.equal(x + official_branch, official),
                    "official_residual_identity")
            fbranch, fprob, _, _ = forward_attention(x, block, f32=True)
        teacher_branch = official_branch.float()
        precision.append(measure_case((fbranch, fprob), (teacher_branch, bfprob), case,
                                      m["common_prefix_length"], block.out.bias))
        targets[case["id"]] = (teacher_branch.cpu(), bfprob.cpu())
        torch.save({"branch": official_branch.cpu(),
                    "prob_packed": packed_prob(bfprob.to(torch.bfloat16)),
                    "input_ids": case["input_ids"], "split": case["split"],
                    "domain": case["domain"]}, output / ("source_" + case["id"] + ".pt"))
        torch.save({"branch": fbranch.cpu(), "prob_packed": packed_prob(fprob),
                    "input_ids": case["input_ids"], "split": case["split"],
                    "domain": case["domain"]}, output / ("f32_source_" + case["id"] + ".pt"))
        del x, official, official_context, official_branch, teacher_branch, captured
        del bfbranch, bfprob, bfcontext, fbranch, fprob
    precision_gate = {split: metrics([r for r in precision if r["split"] == split])
                      for split in ("FIT", "DEV")}
    receipt["stage_seconds"]["teacher_and_precision_control"] = time.monotonic() - stage_start
    stage_start = time.monotonic()
    # A source precision floor failure is inconclusive for the rank recipe.
    if any(precision_gate[split]["pooled_branch_l2"] > .01 or
           precision_gate[split]["sequence_l2_p95"] > .05 or
           precision_gate[split]["tv_mean_including_sink"] > .01 or
           precision_gate[split]["tv_p95_including_sink"] > .05
           for split in ("FIT", "DEV")):
        source_after = source_parameter_hash(block)
        require(source_after == source_before, "source_block_mutated")
        require(source_parameter_identity(block) == source_identity_before,
                "source_identity_or_version_mutated")
        return {"status": "INCONCLUSIVE_PRECISION_FLOOR", "precision": precision_gate,
                "source_parameter_sha256_after": source_after,
                "cases": [{k: v for k, v in r.items() if k not in ("tv", "all_tv")}
                          for r in precision]}
    with torch.no_grad():
        normalized = []; source_branch = []
        for case in fit:
            x = rows[[lookup[i] for i in case["input_ids"]]].to(device)
            _, _, _, normed = forward_attention(x, block, f32=True)
            normalized.append(normed[m["common_prefix_length"]:]);
            source_branch.append(targets[case["id"]][0][m["common_prefix_length"]:].to(device))
        qleft, qright, q_damping = weighted_svd(block.qkv.weight[:4096].float(),
                                                  torch.cat(normalized), RANK)
        contexts = []
        for case in fit:
            x = rows[[lookup[i] for i in case["input_ids"]]].to(device)
            _, _, context, _ = forward_attention(x, block, (qleft, qright), f32=True)
            contexts.append(context[m["common_prefix_length"]:])
        oleft, oright, o_damping = o_initializer(torch.cat(contexts),
            torch.cat(source_branch) - block.out.bias.float(), RANK)
        del normalized, source_branch, contexts
    guard(start, gpu_start, output)
    receipt["stage_seconds"]["fit_initialization"] = time.monotonic() - stage_start
    params = (qleft, qright, oleft, oright)
    require(sum(p.numel() for p in params) == 3571712, "factor_coefficient_count")
    torch.save({"factors": factor_state(params), "rank": RANK,
                "q_ridge": q_damping, "o_ridge": o_damping},
               output / "rank256_initial_factors.pt")
    optimizer = torch.optim.Adam(params, lr=LR)
    ledger_path = output / "optimizer_ledger.jsonl"
    ledger_stream = ledger_path.open("x", encoding="utf-8")
    optimization_start = time.monotonic()
    for step in range(MAX_STEPS):
        guard(start, gpu_start, output)
        case = fit[step % len(fit)]
        x = rows[[lookup[i] for i in case["input_ids"]]].to(device)
        truth, prob = (t.to(device) for t in targets[case["id"]])
        branch, candidate_prob, _, _ = forward_attention(x, block,
                                        (qleft, qright), (oleft, oright), f32=True)
        common = m["common_prefix_length"]
        branch_loss = (branch[common:] - truth[common:]).square().mean() / truth[common:].square().mean().clamp_min(1e-12)
        tv = tv_rows(candidate_prob[..., common:, :], prob[..., common:, :]).mean()
        loss = branch_loss + tv
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in params))
        require(math.isfinite(grad_norm), "finite_gradient")
        optimizer.step()
        p_hash = parameter_hash(params)
        row = {"step": step + 1, "case_id": case["id"], "branch_loss": float(branch_loss.detach()),
               "tv": float(tv.detach()), "total": float(loss.detach()),
               "gradient_l2": grad_norm, "parameters_sha256": p_hash,
               "adam_step": step + 1, "all_finite": all(torch.isfinite(p).all().item() for p in params),
               "gpu_allocated": torch.cuda.memory_allocated(),
               "gpu_reserved": torch.cuda.memory_reserved()}
        ledger_stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        ledger_stream.flush(); os.fsync(ledger_stream.fileno())
        require(row["all_finite"], "finite_factors")
        if step == 1:
            estimate = (time.monotonic() - optimization_start) / 2 * MAX_STEPS
            atomic_json(output / "first_two_step_resource_probe.json",
                        {"gpu_allocated_bytes": torch.cuda.memory_allocated(),
                         "gpu_reserved_bytes": torch.cuda.memory_reserved(),
                         "cpu_rss_bytes": psutil.Process().memory_info().rss,
                         "seconds_since_gpu_start": time.monotonic() - gpu_start,
                         "estimated_512_update_seconds": estimate})
            require(torch.cuda.memory_allocated() < MAX_GPU_ALLOC and
                    torch.cuda.memory_reserved() < MAX_GPU_RESERVE,
                    "first_two_step_gpu_probe")
            require(estimate < 3600, "REVIEW_REQUIRED_ESTIMATED_OVER_1H")
        del x, truth, prob, branch, candidate_prob, loss
    ledger_stream.close()
    receipt["stage_seconds"]["optimization_512_updates"] = time.monotonic() - optimization_start
    stage_start = time.monotonic()
    factor_path = output / "rank256_f32_factors.pt"
    torch.save({"factors": factor_state(params),
                "rank": RANK, "coefficients": 3571712}, factor_path)
    torch.save({"optimizer": cpu_tree(optimizer.state_dict()),
                "last_gradients": {name: p.grad.detach().cpu().clone() for name, p in
                                   zip(("q_left", "q_right", "o_left", "o_right"), params)},
                "steps": MAX_STEPS}, output / "rank256_optimizer_final.pt")
    evaluated = {"FIT": [], "DEV": []}
    with torch.no_grad():
        for case in cases:
            guard(start, gpu_start, output)
            x = rows[[lookup[i] for i in case["input_ids"]]].to(device)
            predicted = forward_attention(x, block, (qleft, qright),
                                          (oleft, oright), f32=True)
            truth = tuple(t.to(device) for t in targets[case["id"]])
            evaluated[case["split"]].append(measure_case(predicted[:2], truth, case,
                                                         m["common_prefix_length"],
                                                         block.out.bias))
            torch.save({"branch": predicted[0].cpu(), "prob_packed": packed_prob(predicted[1]),
                        "input_ids": case["input_ids"], "split": case["split"],
                        "domain": case["domain"]}, output / ("candidate_" + case["id"] + ".pt"))
    report = {split: metrics(values) for split, values in evaluated.items()}
    receipt["stage_seconds"]["export_and_final_evaluation"] = time.monotonic() - stage_start
    passed = all(report[split]["pooled_branch_l2"] <= .01 and
                 report[split]["sequence_l2_p95"] <= .05 and
                 report[split]["tv_mean_including_sink"] <= .01 and
                 report[split]["tv_p95_including_sink"] <= .05
                 for split in ("FIT", "DEV"))
    source_after = source_parameter_hash(block)
    require(source_after == source_before, "source_block_mutated")
    require(source_parameter_identity(block) == source_identity_before,
            "source_identity_or_version_mutated")
    return {"status": "RANK256_F32_GATE_PASS_PHYSICAL_Q4_PENDING" if passed else "RANK256_F32_REJECT",
            "source_parameter_sha256_after": source_after,
            "precision": precision_gate, "rank256": report,
            "q_ridge": q_damping, "o_ridge": o_damping,
            "factor_sha256": sha(factor_path), "optimizer_ledger_sha256": sha(ledger_path),
            "source_trace_count": len(targets), "gpu_initial_free_bytes": free,
            "gpu_total_bytes": total,
            "per_case": {split: [{k: v for k, v in r.items() if k not in ("tv", "all_tv")}
                                 for r in values]
                         for split, values in evaluated.items()}}


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--watchdog":
        watchdog(int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5]))
    else:
        main()
