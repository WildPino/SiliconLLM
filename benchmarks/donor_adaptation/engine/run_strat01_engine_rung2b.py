#!/usr/bin/env python3
"""Build, execute once, and adjudicate STRAT-01 dense block-0 Rung 2B."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ENGINE = ROOT / "benchmarks" / "phase60" / "engine.c"
RUNG2A_HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2a.h"
RUNG2B_HEADER = ROOT / "benchmarks" / "phase60" / "strat01_gguf_rung2b.h"
REFERENCE_SOURCE = HERE / "strat01_engine_rung2a_reference.cpp"
REFERENCE_BUILDER = HERE / "build_strat01_engine_rung2b_reference.py"
PROTOCOL = ROOT / "docs" / "research" / "donor_adaptation" / "probes" / "STRAT_01_GIGACHAT31_ENGINE_RUNG2B_PROTOCOL_20260921.md"
DEFAULT_MODEL = ROOT / "benchmarks" / "donor_adaptation" / "density" / "results" / "strat01_gigachat_q4_97045b2" / "GigaChat3.1-10B-A1.8B-q4_K_M.gguf"
DEFAULT_OUTPUT = HERE / "results" / "strat01_gigachat_engine_rung2b_20260921"
PINNED_LLAMA = Path(r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-bind-5b335f4")
LLAMA_COMMIT = "5b335f413e4f73b0809c4fe39af894efbcc6a0d2"
EXPECTED_BYTES = 6_474_702_976
EXPECTED_SHA256 = "68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb"
TOKENS = [1, 72, 14, 14129, 14, 2135, 1512, 2015]
POSITIONS = list(range(8))
ARMS = ("prefill8", "cached7p1")
GENERAL_LIMITS = (2e-3, 1e-2)
TERMINAL_LIMITS = (1e-3, 5e-3)
CONTINUITY_LIMITS = (2e-6, 1e-5)
COMPILE_FLAGS = ["-std=c11", "-O3", "-mavx2", "-mfma"]
EXPECTED_C_CONFIG = (
    "reference=llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2;cpu_threads=1;"
    "n_ctx=8;n_batch=8;n_ubatch=8;flash_attn=false;offload_kqv=false;"
    "type_k=f16;type_v=f16-no-allocation-mla;cache=layer-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "ffn=block0-rmsnorm-q4kq8k-gate-up-silu-q6kq8k-down-residual;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-token-major;adjudication=external-reference-only"
)
EXPECTED_REFERENCE_CONFIG = {
    "n_gpu_layers": 0, "n_threads": 1, "n_threads_batch": 1,
    "requested_n_ctx": 8, "resolved_n_ctx": 256, "n_batch": 8, "n_ubatch": 8,
    "flash_attn": "disabled", "offload_kqv": False, "op_offload": False,
    "type_k": "F16", "type_v": "F16",
}
SHAPES = {
    "ffn_norm-0": [1536, 8], "ffn_up-0": [8960, 8],
    "ffn_gate-0": [8960, 8], "ffn_swiglu-0": [8960, 8],
    "ffn_out-0": [1536, 8], "l_out-0": [1536, 8],
}
EXPECTED_OPS = {
    "ffn_norm-0": "MUL", "ffn_up-0": "MUL_MAT", "ffn_gate-0": "MUL_MAT",
    "ffn_swiglu-0": "GLU", "ffn_out-0": "MUL_MAT", "l_out-0": "ADD",
}
NON_CLAIMS = [
    "routed/shared MoE or later-layer parity", "full logits, tokenizer, sampling, or generation parity",
    "language-model quality through the C engine", "RAM fit or accepted-token throughput",
    "any speed or SPEED_LEDGER claim",
]


class RunnerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_value(args: list[str], cwd: Path = ROOT) -> str | None:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_command(command: list[str], *, output: Path, label: str, timeout: int) -> dict[str, Any]:
    started_utc, started = utc_now(), time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)
        record: dict[str, Any] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        record = {"returncode": None, "stdout": str(getattr(exc, "stdout", "") or ""), "stderr": str(getattr(exc, "stderr", "") or ""), "spawn_or_timeout_error": str(exc)}
    (output / f"{label}.stdout.log").write_text(record["stdout"], encoding="utf-8")
    (output / f"{label}.stderr.log").write_text(record["stderr"], encoding="utf-8")
    record.update({"command": command, "started_utc": started_utc, "seconds": time.perf_counter()-started})
    return record


def require_ok(record: dict[str, Any], label: str) -> None:
    if record.get("returncode") != 0:
        detail = record.get("spawn_or_timeout_error") or record.get("stderr") or record.get("stdout") or ""
        raise RunnerError(f"{label} failed (rc={record.get('returncode')}): {str(detail)[-1600:]}")


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label} is missing or malformed: {exc}") from exc


def contained_file(root: Path, reported: str, expected_bytes: int, label: str) -> Path:
    candidate = Path(reported)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RunnerError(f"{label} path is absent or escapes its raw directory") from exc
    if not resolved.is_file() or resolved.stat().st_size != expected_bytes:
        raise RunnerError(f"{label} payload byte count mismatch")
    return resolved


def load_payload(path: Path, count: int, digest: str, label: str) -> np.ndarray:
    if sha256_file(path) != digest:
        raise RunnerError(f"{label} payload SHA-256 mismatch")
    values = np.fromfile(path, dtype=np.dtype("<f4"))
    if values.size != count or not bool(np.isfinite(values).all()):
        raise RunnerError(f"{label} payload count/finiteness mismatch")
    return values


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    c, r = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if c.shape != r.shape or not c.size or not bool(np.isfinite(c).all() and np.isfinite(r).all()):
        raise RunnerError("metric input mismatch")
    delta = c-r
    return {
        "nrmse": math.sqrt(float(np.dot(delta,delta))/max(float(np.dot(r,r)),1e-30)),
        "normalized_max": float(np.max(np.abs(delta)))/max(float(np.max(np.abs(r))),1e-6),
    }


def judged(candidate: np.ndarray, reference: np.ndarray, limits: tuple[float,float]) -> dict[str, Any]:
    result: dict[str, Any] = metrics(candidate,reference)
    result.update({"nrmse_limit":limits[0],"normalized_max_limit":limits[1]})
    result["pass"] = result["nrmse"] <= limits[0] and result["normalized_max"] <= limits[1]
    return result


def token7(values: np.ndarray, shape: list[int]) -> np.ndarray:
    per_token = math.prod(shape[:-1])
    if values.size != per_token*8:
        raise RunnerError("token-7 extraction shape mismatch")
    return values[7*per_token:]


def source_inventory() -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(), "engine": ENGINE, "rung2a_header": RUNG2A_HEADER,
        "rung2b_header": RUNG2B_HEADER, "reference_source": REFERENCE_SOURCE,
        "reference_builder": REFERENCE_BUILDER, "protocol": PROTOCOL,
    }
    missing = [name for name,path in paths.items() if not path.is_file()]
    if missing:
        raise RunnerError("missing source(s): "+", ".join(missing))
    return {name:{"path":str(path),"sha256":sha256_file(path)} for name,path in paths.items()}


def validate_c(c_root: Path, sources: dict[str, Any], model: Path, expected_config: str = EXPECTED_C_CONFIG) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    report = read_json(c_root/"strat01_rung2b.json","C report")
    required = {"command","c_state","self_certifies_pass","input_path","byte_size","sha256","reference_revision","CONFIG","compiler_family","compiler_embedded_version","compiler_resolved_path_and_full_version","engine_source_sha256","rung2b_source_sha256","token_ids","positions","arms","timing_or_rate_claim"}
    if not isinstance(report,dict) or set(report)!=required:
        raise RunnerError("C report schema mismatch")
    if report["command"]!="--strat01-gguf-rung2b" or report["c_state"]!="ENGINE_RUNG2B_OUTPUT_READY_PENDING_REFERENCE" or report["self_certifies_pass"] is not False:
        raise RunnerError("C report state mismatch")
    if Path(report["input_path"]).resolve(strict=True)!=model.resolve(strict=True) or report["byte_size"]!=EXPECTED_BYTES or report["sha256"]!=EXPECTED_SHA256:
        raise RunnerError("C artifact identity mismatch")
    if report["reference_revision"]!=f"llama.cpp {LLAMA_COMMIT}" or report["CONFIG"]!=expected_config:
        raise RunnerError("C reference/config mismatch")
    if report["engine_source_sha256"]!=sources["engine"]["sha256"] or report["rung2b_source_sha256"]!=sources["rung2b_header"]["sha256"]:
        raise RunnerError("C source hash mismatch")
    if report["compiler_family"]!="clang" or report["compiler_resolved_path_and_full_version"]!="EXTERNAL_RUNNER_REQUIRED" or report["token_ids"]!=TOKENS or report["positions"]!=POSITIONS or report["timing_or_rate_claim"] is not None:
        raise RunnerError("C compiler/input/non-speed contract mismatch")
    if report["arms"]!=[{"name":"prefill8","manifest":"prefill8_manifest.json"},{"name":"cached7p1","manifest":"cached7p1_manifest.json"}]:
        raise RunnerError("C paired-arm contract mismatch")
    tensors: dict[str,np.ndarray]={};manifests={}
    for arm in ARMS:
        manifest=read_json(c_root/f"{arm}_manifest.json",f"C {arm} manifest")
        if not isinstance(manifest,dict) or set(manifest)!={"arm","payload_encoding","shape_order","tensors"} or manifest["arm"]!=arm or manifest["payload_encoding"]!="IEEE-754 binary32 little-endian":
            raise RunnerError(f"C {arm} manifest schema mismatch")
        by_name={item.get("name"):item for item in manifest["tensors"] if isinstance(item,dict)}
        if set(by_name)!=set(SHAPES) or len(by_name)!=len(manifest["tensors"]):
            raise RunnerError(f"C {arm} checkpoint set mismatch")
        for name,shape in SHAPES.items():
            item=by_name[name];count=math.prod(shape)
            if item.get("logical_shape")!=shape or item.get("op")!=EXPECTED_OPS[name] or item.get("ordinal")!=0 or item.get("type")!="F32" or item.get("payload_order")!="token,feature" or item.get("byte_count")!=4*count:
                raise RunnerError(f"C {arm}/{name} identity mismatch")
            path=contained_file(c_root,item["path"],4*count,f"C {arm}/{name}")
            tensors[f"{arm}/{name}"]=load_payload(path,count,item["sha256"],f"C {arm}/{name}")
        manifests[arm]=manifest
    return tensors,{"report":report,"manifests":manifests}


def validate_reference(ref_root: Path, model: Path) -> tuple[dict[str,np.ndarray],dict[str,Any]]:
    root=read_json(ref_root/"manifest.json","reference root manifest")
    required={"schema","state","llama_cpp_commit","model","config","fixed_tokens","fixed_positions","arms"}
    if not isinstance(root,dict) or set(root)!=required or root["schema"]!="strat01_engine_rung2b_reference_manifest_v1" or root["state"]!="REFERENCE_TRACE_READY_PENDING_C_ENGINE" or root["llama_cpp_commit"]!=LLAMA_COMMIT:
        raise RunnerError("reference root schema/state mismatch")
    model_record=root["model"]
    if Path(model_record.get("path","")).resolve(strict=True)!=model.resolve(strict=True) or model_record.get("bytes")!=EXPECTED_BYTES or model_record.get("sha256")!=EXPECTED_SHA256:
        raise RunnerError("reference artifact identity mismatch")
    if root["config"]!=EXPECTED_REFERENCE_CONFIG or root["fixed_tokens"]!=TOKENS or root["fixed_positions"]!=POSITIONS or root["arms"]!=[{"arm":a,"manifest":f"{a}/manifest.json"} for a in ARMS]:
        raise RunnerError("reference configuration/arm mismatch")
    tensors: dict[str,np.ndarray]={};manifests={}
    for arm in ARMS:
        manifest=read_json(ref_root/arm/"manifest.json",f"reference {arm} manifest")
        if not isinstance(manifest,dict) or set(manifest)!={"schema","arm","logical_cache_contract","callback_records","logical_selection","payloads"} or manifest["schema"]!=root["schema"] or manifest["arm"]!=arm:
            raise RunnerError(f"reference {arm} manifest schema mismatch")
        selection=manifest["logical_selection"]
        payloads={(item.get("logical"),item.get("kind")):item for item in manifest["payloads"] if isinstance(item,dict)}
        for name,shape in SHAPES.items():
            selected=selection.get(name)
            if not isinstance(selected,dict) or selected.get("logical_shape")!=shape:
                raise RunnerError(f"reference {arm}/{name} selection missing")
            sources=[selected.get("source")] if arm=="prefill8" else [selected.get("prefix_source"),selected.get("final_source")]
            if selected.get("composition")!=("single_prefill8_callback" if arm=="prefill8" else "prefix7_then_final1"):
                raise RunnerError(f"reference {arm}/{name} composition mismatch")
            expected_tokens=[8] if arm=="prefill8" else [7,1]
            for source,tokens in zip(sources,expected_tokens):
                if not isinstance(source,dict) or source.get("name")!=name or source.get("op")!=EXPECTED_OPS[name] or source.get("ordinal")!=0 or source.get("shape")!=shape[:-1]+[tokens] or str(source.get("type","")).upper() not in {"F32","F16"}:
                    raise RunnerError(f"reference {arm}/{name} callback identity mismatch")
            item=payloads.get((name,"full"));count=math.prod(shape)
            if not isinstance(item,dict) or item.get("byte_count")!=4*count:
                raise RunnerError(f"reference {arm}/{name} full payload missing")
            path=contained_file(ref_root,item["path"],4*count,f"reference {arm}/{name}")
            tensors[f"{arm}/{name}"]=load_payload(path,count,item["sha256"],f"reference {arm}/{name}")
        manifests[arm]=manifest
    return tensors,{"root":root,"manifests":manifests}


def adjudicate(c: dict[str,np.ndarray],r: dict[str,np.ndarray]) -> dict[str,Any]:
    checkpoint_results=[];failures=[]
    for arm in ARMS:
        for name in SHAPES:
            limits=TERMINAL_LIMITS if name=="l_out-0" else GENERAL_LIMITS
            result=judged(c[f"{arm}/{name}"],r[f"{arm}/{name}"],limits)
            result.update({"arm":arm,"checkpoint":name});checkpoint_results.append(result)
            if not result["pass"]:failures.append(f"{arm}/{name}")
    continuity=[]
    for implementation,values in (("c_engine",c),("pinned_reference",r)):
        result=judged(token7(values["prefill8/l_out-0"],SHAPES["l_out-0"]),token7(values["cached7p1/l_out-0"],SHAPES["l_out-0"]),CONTINUITY_LIMITS)
        result["implementation"]=implementation;continuity.append(result)
        if not result["pass"]:failures.append(f"{implementation}/l_out_continuity")
    ref_gate=r["prefill8/ffn_gate-0"].astype(np.float64)
    ref_up=r["prefill8/ffn_up-0"].astype(np.float64)
    ref_swiglu=r["prefill8/ffn_swiglu-0"]
    controls={
        "gate_up_swap":judged((ref_up/(1.0+np.exp(-ref_up))*ref_gate).astype(np.float32),ref_swiglu,GENERAL_LIMITS),
        "omit_silu":judged((ref_gate*ref_up).astype(np.float32),ref_swiglu,GENERAL_LIMITS),
        "mutated_q6_down_input":judged(np.zeros_like(r["prefill8/ffn_out-0"]),r["prefill8/ffn_out-0"],GENERAL_LIMITS),
        "omit_final_residual":judged(r["prefill8/ffn_out-0"],r["prefill8/l_out-0"],TERMINAL_LIMITS),
    }
    controls_pass=all(not item["pass"] for item in controls.values())
    if not controls_pass:failures.append("causal_negative_controls")
    return {
        "status":"PASS_ENGINE_RUNG2B" if not failures else "FAIL_ENGINE_RUNG2B",
        "checkpoint_results":checkpoint_results,"continuity_results":continuity,
        "negative_controls":controls,"negative_controls_pass":controls_pass,"failures":failures,
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--apparatus-only",action="store_true")
    args=parser.parse_args();model=args.model.resolve();output=args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"refusing non-empty or non-directory output path: {output}")
    output.mkdir(parents=True,exist_ok=True)
    started_utc,started=utc_now(),time.perf_counter();status="VOID_ENGINE_RUNG2B";errors=[];commands={};sources={};c_meta={};ref_meta={}
    adjudication: dict[str,Any]={"status":"NOT_RUN","checkpoint_results":[],"continuity_results":[],"negative_controls":{},"failures":[]}
    compiler=shutil.which("clang");binary=None;reference_binary=None
    artifact={"path":str(model),"expected_bytes":EXPECTED_BYTES,"expected_sha256":EXPECTED_SHA256,"bytes":None,"sha256":None}
    try:
        sources=source_inventory()
        if not compiler:raise RunnerError("clang is unavailable")
        if git_value(["git","rev-parse","HEAD"],PINNED_LLAMA)!=LLAMA_COMMIT or git_value(["git","status","--porcelain"],PINNED_LLAMA):
            raise RunnerError("llama.cpp checkout is not the clean pinned revision")
        if not args.apparatus_only and (not model.is_file() or model.stat().st_size!=EXPECTED_BYTES):
            raise RunnerError("accepted artifact is absent or has wrong size")
        commands["clang_version"]=run_command([compiler,"--version"],output=output,label="clang_version",timeout=30);require_ok(commands["clang_version"],"clang --version")
        binary=output/"engine_rung2b.exe"
        commands["compile_c"]=run_command([compiler,*COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],output=output,label="compile_c",timeout=600);require_ok(commands["compile_c"],"C build")
        commands["c_selftest"]=run_command([str(binary),"--strat01-gguf-rung2b-selftest"],output=output,label="c_selftest",timeout=300);require_ok(commands["c_selftest"],"C Rung-2B selftest")
        commands["legacy_selftest"]=run_command([str(binary),"--kselftest"],output=output,label="legacy_selftest",timeout=300);require_ok(commands["legacy_selftest"],"legacy selftest")
        commands["python_tests"]=run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_engine_rung2b"],output=output,label="python_tests",timeout=300);require_ok(commands["python_tests"],"Rung-2B Python tests")
        build_dir=output/"reference_build"
        commands["build_reference"]=run_command([sys.executable,"-B",str(REFERENCE_BUILDER),"--build-dir",str(build_dir),"--config","Release"],output=output,label="build_reference",timeout=3600);require_ok(commands["build_reference"],"reference build")
        lines=[line.strip() for line in commands["build_reference"]["stdout"].splitlines() if line.strip()]
        if not lines:raise RunnerError("reference builder reported no binary")
        reference_binary=Path(lines[-1]).resolve(strict=True);reference_binary.relative_to(build_dir.resolve(strict=True))
        commands["reference_selftest"]=run_command([str(reference_binary),"--self-test"],output=output,label="reference_selftest",timeout=300);require_ok(commands["reference_selftest"],"reference selftest")
        if args.apparatus_only:
            status="APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            digest=sha256_file(model);artifact.update({"bytes":model.stat().st_size,"sha256":digest})
            if digest!=EXPECTED_SHA256:raise RunnerError("accepted artifact SHA-256 mismatch")
            ref_root=output/"pinned_reference"
            commands["accepted_artifact_pinned_reference"]=run_command([str(reference_binary),"--model",str(model),"--out-dir",str(ref_root),"--all"],output=output,label="accepted_artifact_pinned_reference",timeout=21600);require_ok(commands["accepted_artifact_pinned_reference"],"accepted-artifact reference")
            ref_tensors,ref_meta=validate_reference(ref_root,model)
            c_root=output/"c_engine";c_root.mkdir()
            commands["accepted_artifact_c_engine"]=run_command([str(binary),"--strat01-gguf-rung2b",str(model),"--out-dir",str(c_root)],output=output,label="accepted_artifact_c_engine",timeout=21600);require_ok(commands["accepted_artifact_c_engine"],"accepted-artifact C engine")
            sources=source_inventory();c_tensors,c_meta=validate_c(c_root,sources,model)
            adjudication=adjudicate(c_tensors,ref_tensors);status=adjudication["status"]
    except RunnerError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance={
        "started_utc":started_utc,"finished_utc":utc_now(),"seconds":time.perf_counter()-started,
        "git_head":git_value(["git","rev-parse","HEAD"]),"git_status_porcelain":git_value(["git","status","--porcelain"]),
        "source_hashes":sources,"artifact":artifact,
        "environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd(),"clang_path":compiler},
        "binaries":{"c_engine":{"path":str(binary) if binary else None,"sha256":sha256_file(binary) if binary and binary.is_file() else None},"pinned_reference":{"path":str(reference_binary) if reference_binary else None,"sha256":sha256_file(reference_binary) if reference_binary and reference_binary.is_file() else None}},
        "commands":commands,
    }
    record={"schema":"strat01_gigachat_engine_rung2b_adjudication_v1","status":status,"scope":"block-0 dense SwiGLU parity from ffn_inp-0 through l_out-0","errors":errors,"adjudication":adjudication,"c_metadata":c_meta,"reference_metadata":ref_meta,"non_claims":NON_CLAIMS,"provenance":provenance}
    manifest={"schema":"strat01_gigachat_engine_rung2b_run_manifest_v1","status":status,"errors":errors,"provenance":provenance}
    write_json(output/"adjudication.json",record);write_json(output/"run_manifest.json",manifest)
    print(json.dumps({"status":status,"output":str(output),"errors":errors},indent=2))
    return 0 if status in {"PASS_ENGINE_RUNG2B","APPARATUS_READY_NO_DONOR_EXECUTION"} else 2


if __name__=="__main__":
    raise SystemExit(main())
