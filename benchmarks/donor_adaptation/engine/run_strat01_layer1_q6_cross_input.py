#!/usr/bin/env python3
"""Execute the frozen layer-1 routed/shared Q6 cross-input diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_engine_rung2c as r2c
from benchmarks.donor_adaptation.engine import run_strat01_q4k_q8k_avx2_parity as q4base

HERE = Path(__file__).resolve().parent
ENGINE = r2c.ENGINE
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_q6_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_PROTOCOL_20260923.md"
TESTS = HERE / "test_strat01_layer1_q6_cross_input.py"
MODEL = r2c.MODEL
C_ROOT = HERE / "results/strat01_gigachat_engine_reference_generic_propagation_20260923/c_engine"
REFERENCE_ROOT = HERE / "results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference"
PREDECESSOR = HERE / "results/strat01_gigachat_engine_reference_generic_propagation_20260923/adjudication.json"
PREDECESSOR_SHA = "79a19806fcf80df3e4d8a84335eaaecfc7beb9aef543f021dd1da01f6558b9c3"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_q6_cross_input_20260923"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_q6_cross_input_apparatus_20260923"

INPUTS = {
    "topk": (C_ROOT / "prefill8_ffn_moe_topk-1.i32", 128, "557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95", "i32"),
    "current_routed": (C_ROOT / "prefill8_ffn_moe_swiglu-1.f32", 163840, "ba62cdfd1689cd299dd476e3ac8a31d1b5e374a4028596c8472ef2b7345cf8fa", "f32"),
    "reference_routed": (REFERENCE_ROOT / "prefill8/ffn_moe_swiglu-1.full.f32le", 163840, "5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8", "f32"),
    "current_shared": (C_ROOT / "prefill8_ffn_swiglu-1.f32", 40960, "30ec7ceafc08c3d30b73c21919a7d31619f69b9cd8045421e00c2e38af81a6b8", "f32"),
    "reference_shared": (REFERENCE_ROOT / "prefill8/ffn_swiglu-1.full.f32le", 40960, "fa0d1b4fe5da3f561ac30b8b8cb95520f359232528ea66c1d60e9244642c6003", "f32"),
    "current_routed_down": (C_ROOT / "prefill8_ffn_moe_down-1.f32", 196608, "2a2d153787683a520a5c6e3632406a33de2b7b22d0b47ad33521eb0d57f6ffe8", "f32"),
    "reference_routed_down": (REFERENCE_ROOT / "prefill8/ffn_moe_down-1.full.f32le", 196608, "d19ce37968e1c2c6b04dffbc56ebba7ed25d704db2447e45031786b4710bb0af", "f32"),
    "current_shared_down": (C_ROOT / "prefill8_ffn_shexp-1.f32", 49152, "dd05bd9e815a4a2e7e2957004b157b8ae48f12a60019409963f17e39583051dd", "f32"),
    "reference_shared_down": (REFERENCE_ROOT / "prefill8/ffn_shexp-1.full.f32le", 49152, "7bc8b2104cacb7742e57df0c34a1af64fadbff8cc61006ed1aedc034a030ddae", "f32"),
}
TEST_MODULES = tuple("benchmarks.donor_adaptation.engine." + path.stem for path in sorted(HERE.glob("test_strat01_*.py")))


class DiagnosticError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return r2c.base.sha256_file(path)


def source_inventory() -> dict[str, dict[str, str]]:
    paths = {"runner": Path(__file__).resolve(), "tests": TESTS, "protocol": PROTOCOL, "engine": ENGINE,
             "header": HEADER, "q4_header": ROOT / "benchmarks/phase60/strat01_q4k_q8k.h",
             "rung2b_header": r2c.RUNG2B_HEADER, "rung2c_header": r2c.RUNG2C_HEADER}
    if any(not path.is_file() for path in paths.values()):
        raise DiagnosticError("layer-1 Q6 source inventory incomplete")
    return {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}


def clean_sources_at_head(sources: dict[str, dict[str, str]]) -> None:
    relative = [Path(item["path"]).resolve().relative_to(ROOT.resolve()) for item in sources.values()]
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *map(str, relative)], cwd=ROOT, check=False).returncode:
        raise DiagnosticError("layer-1 Q6 implementation/protocol differs from HEAD")
    for path in relative:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True, check=False).returncode:
            raise DiagnosticError(f"untracked layer-1 Q6 source: {path}")


def validate_inputs() -> dict[str, Path]:
    if sha(PREDECESSOR) != PREDECESSOR_SHA:
        raise DiagnosticError("propagation predecessor hash mismatch")
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if predecessor.get("status") != "FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION" or predecessor.get("errors") != [] or predecessor.get("donor_graph_executions") != 2 or predecessor.get("reference_graph_executions") != 0:
        raise DiagnosticError("propagation predecessor state mismatch")
    result: dict[str, Path] = {}
    for name, (path, size, digest, _) in INPUTS.items():
        if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
            raise DiagnosticError(f"input identity mismatch: {name}")
        result[name] = path.resolve(strict=True)
    if INPUTS["topk"][2] != "557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95":
        raise DiagnosticError("top-k identity mismatch")
    return result


def load_f32(path: Path, count: int) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    if value.size != count or not bool(np.isfinite(value).all()):
        raise DiagnosticError(f"invalid F32 payload: {path}")
    return value


def load_output(root: Path, item: Any, count: int, label: str) -> np.ndarray:
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"} or item["bytes"] != count * 4:
        raise DiagnosticError(f"output schema mismatch: {label}")
    path = r2c.base.contained_file(root, item["path"], count * 4, label)
    if sha(path) != item["sha256"]:
        raise DiagnosticError(f"output digest mismatch: {label}")
    return load_f32(path, count)


def judged(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    return r2c.base.judged(candidate, reference, r2c.base.GENERAL_LIMITS)


def detailed(candidate: np.ndarray, reference: np.ndarray, routed: bool, ids: np.ndarray) -> dict[str, Any]:
    result = judged(candidate, reference)
    if routed:
        c, r = candidate.reshape(8, 4, 1536), reference.reshape(8, 4, 1536)
        result["per_token_slot"] = [dict(judged(c[t, s], r[t, s]), token=t, slot=s, expert_id=int(ids[t, s])) for t in range(8) for s in range(4)]
    else:
        c, r = candidate.reshape(8, 1536), reference.reshape(8, 1536)
        result["per_token"] = [dict(judged(c[t], r[t]), token=t) for t in range(8)]
    return result


def validate_report(root: Path, model: Path, sources: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report = json.loads((root / "strat01_layer1_q6_cross_input.json").read_text(encoding="utf-8"))
    required = {"command","state","self_certifies_pass","model","topk","inputs","tensors","outputs","controls","engine_source_sha256","diagnostic_source_sha256","compiler_family","donor_graph_executions","reference_graph_executions","timing_or_rate_claim"}
    if set(report) != required or report["command"] != "--strat01-layer1-q6-cross-input" or report["state"] != "OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False:
        raise DiagnosticError("report schema/state mismatch")
    if report["model"] != {"path": str(model), "bytes": r2c.base.EXPECTED_BYTES, "sha256": r2c.base.EXPECTED_SHA256}:
        raise DiagnosticError("reported model identity mismatch")
    ids = np.asarray(report["topk"].get("ids"), dtype=np.int64)
    if report["topk"].get("path") != str(paths["topk"]) or report["topk"].get("bytes") != 128 or report["topk"].get("sha256") != INPUTS["topk"][2] or ids.shape != (32,) or bool(((ids < 0) | (ids >= 64)).any()):
        raise DiagnosticError("reported top-k mismatch")
    expected_inputs = {name: {"path": str(paths[key]), "bytes": INPUTS[key][1], "sha256": INPUTS[key][2]} for name, key in (("current_routed_swiglu","current_routed"),("reference_routed_swiglu","reference_routed"),("current_shared_swiglu","current_shared"),("reference_shared_swiglu","reference_shared"))}
    if report["inputs"] != expected_inputs:
        raise DiagnosticError("reported input identity mismatch")
    expected_tensors = {"routed": ("blk.1.ffn_down_exps.weight", 14, 3, [1280,1536,64]), "shared": ("blk.1.ffn_down_shexp.weight", 14, 2, [1280,1536])}
    for name, expected in expected_tensors.items():
        item = report["tensors"].get(name)
        if not isinstance(item, dict) or item.get("name") != expected[0] or item.get("type") != expected[1] or item.get("rank") != expected[2] or item.get("dims") != expected[3] or item.get("span",0) <= 0:
            raise DiagnosticError(f"tensor descriptor mismatch: {name}")
    counts = {"reference_routed_current_q6":49152,"current_routed_current_q6":49152,"reference_shared_current_q6":12288,"current_shared_current_q6":12288,"control_negated_reference_routed":49152,"control_negated_reference_shared":12288,"control_mutated_expert_id":49152}
    if set(report["outputs"]) != set(counts):
        raise DiagnosticError("output set mismatch")
    values = {name: load_output(root, report["outputs"][name], count, name) for name, count in counts.items()}
    if report["controls"].get("q8_deterministic") is not True or report["controls"].get("mutated_expert_id") == int(ids[0]):
        raise DiagnosticError("C causal control state mismatch")
    if report["engine_source_sha256"] != sources["engine"]["sha256"] or report["diagnostic_source_sha256"] != sources["header"]["sha256"] or report["compiler_family"] != "clang" or report["donor_graph_executions"] != 0 or report["reference_graph_executions"] != 0 or report["timing_or_rate_claim"] is not None:
        raise DiagnosticError("source/execution contract mismatch")
    return values, {"report": report, "ids": ids.reshape(8,4)}


def adjudicate(values: dict[str, np.ndarray], meta: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    ids = meta["ids"]
    targets = {
        "routed": load_f32(paths["reference_routed_down"], 49152),
        "shared": load_f32(paths["reference_shared_down"], 12288),
    }
    current_targets = {
        "routed": load_f32(paths["current_routed_down"], 49152),
        "shared": load_f32(paths["current_shared_down"], 12288),
    }
    judgments = {
        "reference_routed_current_q6": detailed(values["reference_routed_current_q6"], targets["routed"], True, ids),
        "current_routed_current_q6": detailed(values["current_routed_current_q6"], targets["routed"], True, ids),
        "reference_shared_current_q6": detailed(values["reference_shared_current_q6"], targets["shared"], False, ids),
        "current_shared_current_q6": detailed(values["current_shared_current_q6"], targets["shared"], False, ids),
    }
    replay = {
        "current_routed_byte_exact": bool(np.array_equal(values["current_routed_current_q6"], current_targets["routed"])),
        "current_shared_byte_exact": bool(np.array_equal(values["current_shared_current_q6"], current_targets["shared"])),
    }
    controls = {
        "negated_reference_routed_rejects": not judged(values["control_negated_reference_routed"], targets["routed"])["pass"],
        "negated_reference_shared_rejects": not judged(values["control_negated_reference_shared"], targets["shared"])["pass"],
        "mutated_expert_id_rejects": not judged(values["control_mutated_expert_id"], targets["routed"])["pass"],
    }
    mutation_refused: dict[str, bool] = {}
    for name, path in paths.items():
        raw = bytearray(path.read_bytes()); raw[len(raw)//2] ^= 1
        mutation_refused[name] = hashlib.sha256(raw).hexdigest() != INPUTS[name][2]
    if not all(replay.values()) or not all(controls.values()) or not all(mutation_refused.values()):
        raise DiagnosticError("replay or causal control failure")
    routed_pass = judgments["reference_routed_current_q6"]["pass"]
    shared_pass = judgments["reference_shared_current_q6"]["pass"]
    status = ("LAYER1_SWIGLU_RESIDUAL_SUFFICIENT" if routed_pass and shared_pass else
              "LAYER1_Q6_OPERATOR_RESIDUALS" if not routed_pass and not shared_pass else
              "ROUTED_Q6_OPERATOR_RESIDUAL" if not routed_pass else "SHARED_Q6_OPERATOR_RESIDUAL")
    return {"status": status, "judgments": judgments, "replay_controls": replay, "causal_controls": controls, "mutated_inputs_refused": mutation_refused}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--model",type=Path,default=MODEL);parser.add_argument("--output-dir",type=Path);parser.add_argument("--apparatus-only",action="store_true");args=parser.parse_args()
    model=args.model.resolve();output=(args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True,exist_ok=True);started=datetime.now(timezone.utc).isoformat();clock=time.perf_counter();status="VOID_LAYER1_Q6_CROSS_INPUT";errors=[];commands={};sources={};report={};result={"status":"NOT_RUN"};compiler=shutil.which("clang");binary=None
    try:
        sources=source_inventory();paths=validate_inputs()
        if not compiler: raise DiagnosticError("clang unavailable")
        binary=output/"engine_layer1_q6_cross_input.exe";commands["compile"]=q4base.run_command([compiler,*r2c.base.COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],output,"compile",600);q4base.require_ok(commands["compile"],"compile")
        commands["python_tests"]=q4base.run_command([sys.executable,"-B","-m","unittest","-v",*TEST_MODULES],output,"all_strat01_unittests",1800);q4base.require_ok(commands["python_tests"],"all STRAT-01 tests")
        for index,option in enumerate((*q4base.SELFTESTS,"--strat01-layer1-q6-cross-input-selftest")):
            label=f"selftest_{index:02d}";commands[label]=q4base.run_command([str(binary),option],output,label,300);q4base.require_ok(commands[label],option)
        if args.apparatus_only: status="APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean_sources_at_head(sources)
            if not model.is_file() or model.stat().st_size!=r2c.base.EXPECTED_BYTES or sha(model)!=r2c.base.EXPECTED_SHA256: raise DiagnosticError("model identity mismatch")
            root=output/"diagnostic";root.mkdir();command=[str(binary),"--strat01-layer1-q6-cross-input",str(model),"--topk",str(paths["topk"]),"--c-routed",str(paths["current_routed"]),"--reference-routed",str(paths["reference_routed"]),"--c-shared",str(paths["current_shared"]),"--reference-shared",str(paths["reference_shared"]),"--out-dir",str(root)]
            commands["diagnostic"]=q4base.run_command(command,output,"diagnostic",21600);q4base.require_ok(commands["diagnostic"],"diagnostic");sources=source_inventory();values,meta=validate_report(root,model,sources,paths);report=meta["report"];result=adjudicate(values,meta,paths);status=result["status"]
    except (DiagnosticError,r2c.RunnerError,r2c.base.RunnerError,q4base.ParityError) as exc: errors.append(str(exc))
    except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
    provenance={"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-clock,"git_head_observed" if args.apparatus_only else "git_head":r2c.base.git_value(["git","rev-parse","HEAD"]),"source_hashes":sources,"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd(),"clang_path":compiler},"binary":{"path":str(binary) if binary else None,"sha256":sha(binary) if binary and binary.is_file() else None},"commands":commands}
    record={"schema":"strat01_layer1_q6_cross_input_v1","status":status,"errors":errors,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":result,"c_report":report,"non_claims":["production repair","graph rerun","later layers","quality","RAM","rate"],"provenance":provenance};r2c.base.write_json(output/"adjudication.json",record);print(json.dumps({"status":status,"output":str(output),"errors":errors},indent=2));return 0 if status in {"APPARATUS_READY_NO_DONOR_EXECUTION","LAYER1_SWIGLU_RESIDUAL_SUFFICIENT","LAYER1_Q6_OPERATOR_RESIDUALS","ROUTED_Q6_OPERATOR_RESIDUAL","SHARED_Q6_OPERATOR_RESIDUAL"} else 2


if __name__ == "__main__": raise SystemExit(main())
