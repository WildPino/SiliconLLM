#!/usr/bin/env python3
"""Run the frozen layer-1 selected routed-up projection input split."""
from __future__ import annotations

import argparse, copy, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from benchmarks.donor_adaptation.engine import run_strat01_layer1_routed_swiglu_component_cross_input as sw
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE = Path(__file__).resolve().parent
ENGINE, DEFAULT_MODEL = sw.ENGINE, sw.DEFAULT_MODEL
HEADER = ROOT / "benchmarks/phase60/strat01_gguf_layer1_routed_up_projection_cross_input.h"
PROTOCOL = ROOT / "docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_UP_PROJECTION_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS = HERE / "test_strat01_layer1_routed_up_projection_cross_input.py"
DEFAULT_OUTPUT = HERE / "results/strat01_gigachat_engine_layer1_routed_up_projection_cross_input_20260924"
DEFAULT_APPARATUS = HERE / "results/strat01_gigachat_engine_layer1_routed_up_projection_cross_input_apparatus_20260924"
PREDECESSOR = sw.DEFAULT_OUTPUT / "adjudication.json"
PREDECESSOR_SHA = "04b46ac959b096bf8d2df8f1735882e37afbafacdc43c1c4eadff7560093a705"
PREDECESSOR_C = sw.DEFAULT_OUTPUT / "diagnostic/sse2_reference_gate_c_up.downstream.f32le"
PREDECESSOR_C_SHA = "81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
REF_ROOT, C_ROOT = sw.REF_ROOT, sw.C_ROOT
TOPK, TOPK_SHA = sw.TOPK, sw.TOPK_SHA
ARM_NAMES = ("captured_reference_up", "captured_production_c_up", "current_up_on_reference_norm", "current_up_on_c_norm", "control_reference_norm_token7_negated")
ARM_ORIGINS = ("capture_reference", "capture_c", "reference_norm", "c_norm", "reference_norm_control")
VALID_STATUSES = {"LAYER1_ROUTED_UPSTREAM_NORM_RESIDUAL_SUFFICIENT", "LAYER1_ROUTED_UP_PROJECTION_RESIDUAL_SUFFICIENT"}
INPUTS = {
    "reference_norm": (REF_ROOT / "prefill8/ffn_norm-1.full.f32le", "3f4826dd184c9442fa34807c9649aeae88343c5894c7aac531e5b5a209806dde", 49152),
    "c_norm": (C_ROOT / "prefill8_ffn_norm-1.f32", "6d2080dcb6eb22c6e9b8f38f8d2beef04cf56aec02161775b65f911dc595f8d7", 49152),
    "reference_up": sw.INPUTS["reference_up"], "c_up": sw.INPUTS["c_up"], "reference_gate": sw.INPUTS["reference_gate"],
    "ref_q": sw.INPUTS["ref_q"], "ref_k": sw.INPUTS["ref_k"], "ref_ffn_inp": sw.INPUTS["ref_ffn_inp"],
    "ref_shared_out": sw.INPUTS["ref_shared_out"], "ref_weights": sw.INPUTS["ref_weights"], "ref_target": sw.INPUTS["ref_target"],
}
TWINS = {
    "reference_norm": REF_ROOT / "cached7p1/ffn_norm-1.full.f32le", "c_norm": C_ROOT / "cached7p1_ffn_norm-1.f32",
    "reference_up": sw.TWINS["reference_up"], "c_up": sw.TWINS["c_up"], "reference_gate": sw.TWINS["reference_gate"],
    "ref_q": sw.TWINS["ref_q"], "ref_k": sw.TWINS["ref_k"], "ref_ffn_inp": sw.TWINS["ref_ffn_inp"],
    "ref_shared_out": sw.TWINS["ref_shared_out"], "ref_weights": sw.TWINS["ref_weights"], "ref_target": sw.TWINS["ref_target"],
}


class RunnerError(RuntimeError): pass


def sources():
    paths = {"runner":Path(__file__).resolve(),"tests":TESTS,"protocol":PROTOCOL,"engine":ENGINE,"header":HEADER,"swiglu_header":sw.HEADER,"swiglu_runner":Path(sw.__file__).resolve(),"shared_sse2":sw.SHARED}
    if any(not p.is_file() for p in paths.values()): raise RunnerError("missing routed-up source")
    return {n:{"path":str(p),"sha256":base.sha256_file(p)} for n,p in paths.items()}


def clean(source_map):
    rel=[Path(v["path"]).resolve().relative_to(ROOT.resolve()) for v in source_map.values()]
    if subprocess.run(["git","diff","--quiet","HEAD","--",*map(str,rel)],cwd=ROOT,check=False).returncode: raise RunnerError("routed-up sources differ from HEAD")
    for p in rel:
        if subprocess.run(["git","ls-files","--error-unmatch",str(p)],cwd=ROOT,capture_output=True,check=False).returncode: raise RunnerError(f"untracked routed-up source: {p}")


def evidence():
    found={}
    for name,(path,digest,size) in INPUTS.items():
        if not path.is_file() or path.stat().st_size!=size or base.sha256_file(path)!=digest: raise RunnerError(f"routed-up input mismatch: {name}")
        twin=TWINS[name]
        if not twin.is_file() or twin.stat().st_size!=size or base.sha256_file(twin)!=digest: raise RunnerError(f"routed-up twin mismatch: {name}")
        found[name]=path.resolve(strict=True)
    for path,digest,size,label in ((TOPK,TOPK_SHA,128,"top-k"),(PREDECESSOR,PREDECESSOR_SHA,None,"predecessor"),(PREDECESSOR_C,PREDECESSOR_C_SHA,196608,"predecessor C output")):
        if not path.is_file() or (size is not None and path.stat().st_size!=size) or base.sha256_file(path)!=digest: raise RunnerError(label+" mismatch")
    found["topk"]=TOPK.resolve(strict=True);found["prior_c"]=PREDECESSOR_C.resolve(strict=True);return found


def arm_manifest(outputs):
    if not isinstance(outputs,dict) or tuple(outputs)!=ARM_NAMES: raise RunnerError("routed-up arm labels/order mismatch")
    sizes={"up":163840,"swiglu":163840,"down":196608,"moe_out":49152,"downstream":196608}
    for i,(name,item) in enumerate(outputs.items()):
        if set(item)!={"origin",*sizes} or item["origin"]!=ARM_ORIGINS[i]: raise RunnerError("routed-up arm metadata mismatch")
        for key,size in sizes.items():
            if set(item[key])!={"path","bytes","sha256"} or item[key]["bytes"]!=size: raise RunnerError("routed-up payload schema mismatch")


def validate_report(root,model,source_map):
    report=json.loads((root/"strat01_layer1_routed_up_projection_cross_input.json").read_text(encoding="utf-8"));required={"command","state","self_certifies_pass","model","inputs","outputs","engine_source_sha256","diagnostic_source_sha256","compiler_family","up_arms_completed","q6_arms_completed","donor_graph_executions","reference_graph_executions","timing_or_rate_claim"}
    if set(report)!=required or report["command"]!="--strat01-layer1-routed-up-projection-cross-input" or report["state"]!="OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or report["self_certifies_pass"] is not False: raise RunnerError("routed-up report schema mismatch")
    expected={name:{"path":str(INPUTS[name][0].resolve()),"bytes":INPUTS[name][2],"sha256":INPUTS[name][1]} for name in tuple(INPUTS)[:-1]};expected["topk"]={"path":str(TOPK.resolve()),"bytes":128,"sha256":TOPK_SHA}
    if report["inputs"]!=expected or report["model"]!={"path":str(model),"bytes":base.EXPECTED_MODEL_BYTES,"sha256":base.EXPECTED_MODEL_SHA}: raise RunnerError("routed-up identity mismatch")
    if report["up_arms_completed"]!=3 or report["q6_arms_completed"]!=5 or report["donor_graph_executions"] or report["reference_graph_executions"] or report["timing_or_rate_claim"] is not None: raise RunnerError("routed-up execution-count mismatch")
    if report["engine_source_sha256"]!=source_map["engine"]["sha256"] or report["diagnostic_source_sha256"]!=source_map["header"]["sha256"] or report["compiler_family"]!="clang": raise RunnerError("routed-up source provenance mismatch")
    arm_manifest(report["outputs"]);values={}
    for name,item in report["outputs"].items(): values[name]={key:base.load_f32(base.contained(root,item[key]["path"],item[key]["bytes"],item[key]["sha256"],name+" "+key),item[key]["bytes"]//4,name+" "+key) for key in ("up","swiglu","down","moe_out","downstream")}
    return values,report


def descriptive(candidate,reference,shape):
    result=base.metrics(candidate,reference);a,b=np.asarray(candidate).reshape(shape),np.asarray(reference).reshape(shape);result["per_token"]=[dict(base.metrics(a[i].ravel(),b[i].ravel()),token=i) for i in range(shape[0])];return result


def adjudicate(values,report,frozen):
    target=base.load_f32(frozen["ref_target"],49152,"reference target");prior_c=base.load_f32(frozen["prior_c"],49152,"predecessor C output")
    if values[ARM_NAMES[0]]["downstream"].tobytes()!=target.tobytes() or values[ARM_NAMES[1]]["downstream"].tobytes()!=prior_c.tobytes(): raise RunnerError("routed-up anchor replay mismatch")
    if values[ARM_NAMES[3]]["up"].tobytes()!=values[ARM_NAMES[1]]["up"].tobytes(): raise RunnerError("routed-up C-input replay mismatch")
    judgments={name:base.judged(values[name]["downstream"],target) for name in ARM_NAMES}
    if not judgments[ARM_NAMES[0]]["pass"] or judgments[ARM_NAMES[1]]["pass"] or judgments[ARM_NAMES[4]]["pass"]: raise RunnerError("routed-up precondition contradiction")
    status="LAYER1_ROUTED_UPSTREAM_NORM_RESIDUAL_SUFFICIENT" if judgments[ARM_NAMES[2]]["pass"] else "LAYER1_ROUTED_UP_PROJECTION_RESIDUAL_SUFFICIENT"
    mutations={}
    for name,(path,digest,size) in INPUTS.items(): data=bytearray(path.read_bytes());data[len(data)//2]^=1;mutations[name]=not base.identity_matches(bytes(data),size,digest)
    data=bytearray(TOPK.read_bytes());data[len(data)//2]^=1;mutations["topk"]=not base.identity_matches(bytes(data),128,TOPK_SHA)
    if not all(mutations.values()): raise RunnerError("routed-up mutation control failed")
    swapped=copy.deepcopy(report["outputs"]);items=list(swapped.items());items[2],items[3]=items[3],items[2]
    try: arm_manifest(dict(items));label_swap=False
    except RunnerError: label_swap=True
    if not label_swap: raise RunnerError("routed-up label swap accepted")
    ref=values[ARM_NAMES[0]]
    return {"status":status,"downstream":judgments,"up_metrics":{n:descriptive(values[n]["up"],ref["up"],(8,4,1280)) for n in ARM_NAMES[:4]},"swiglu_metrics":{n:descriptive(values[n]["swiglu"],ref["swiglu"],(8,4,1280)) for n in ARM_NAMES[:4]},"down_metrics":{n:descriptive(values[n]["down"],ref["down"],(8,4,1536)) for n in ARM_NAMES[:4]},"routed_output_metrics":{n:descriptive(values[n]["moe_out"],ref["moe_out"],(8,1536)) for n in ARM_NAMES[:4]},"controls":{"reference_anchor_byte_exact":True,"production_c_replay_byte_exact":True,"computed_c_up_replay_byte_exact":True,"schedule_twins_byte_exact":True,"mutated_inputs_refused":mutations,"label_swap_rejected":label_swap}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",type=Path,default=DEFAULT_MODEL);parser.add_argument("--output-dir",type=Path);parser.add_argument("--apparatus-only",action="store_true");args=parser.parse_args();model=args.model.resolve();out=(args.output_dir or (DEFAULT_APPARATUS if args.apparatus_only else DEFAULT_OUTPUT)).resolve()
    if out.exists(): raise SystemExit(f"output already exists: {out}")
    out.mkdir(parents=True);started=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();status="VOID_LAYER1_ROUTED_UP_PROJECTION";errors=[];commands={};report={};result={"status":"NOT_RUN"};source_map={};compiler=shutil.which("clang");binary=None;invocations=0;up_completed=q6_completed=0;artifact={"path":str(model),"expected_bytes":base.EXPECTED_MODEL_BYTES,"expected_sha256":base.EXPECTED_MODEL_SHA,"bytes":None,"sha256":None,"opened":False}
    try:
        source_map=sources()
        if not compiler: raise RunnerError("clang unavailable")
        commands["clang_version"]=base.run_command([compiler,"--version"],out,"clang_version",30);base.require_ok(commands["clang_version"],"clang");binary=out/"engine_layer1_routed_up_projection.exe";commands["compile"]=base.run_command([compiler,*base.COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],out,"compile",600);base.require_ok(commands["compile"],"compile")
        for label,flag in (("up_selftest","--strat01-layer1-routed-up-projection-cross-input-selftest"),("swiglu_selftest","--strat01-layer1-routed-swiglu-component-cross-input-selftest"),("q6_selftest","--strat01-layer1-q6-cross-input-selftest"),("propagation_selftest","--strat01-layer1-routed-q6-residual-propagation-selftest"),("legacy_selftest","--kselftest")):
            commands[label]=base.run_command([str(binary),flag],out,label,300);base.require_ok(commands[label],label)
        commands["python_tests"]=base.run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_layer1_routed_up_projection_cross_input"],out,"python_tests",300);base.require_ok(commands["python_tests"],"tests")
        if args.apparatus_only: status="APPARATUS_READY_NO_DONOR_EXECUTION"
        else:
            clean(source_map)
            if not model.is_file() or model.stat().st_size!=base.EXPECTED_MODEL_BYTES or base.sha256_file(model)!=base.EXPECTED_MODEL_SHA: raise RunnerError("artifact mismatch")
            artifact.update({"bytes":model.stat().st_size,"sha256":base.EXPECTED_MODEL_SHA,"opened":True});frozen=evidence();root=out/"diagnostic";root.mkdir();command=[str(binary),"--strat01-layer1-routed-up-projection-cross-input",str(model),"--topk",str(frozen["topk"]),"--ref-norm",str(frozen["reference_norm"]),"--c-norm",str(frozen["c_norm"]),"--ref-up",str(frozen["reference_up"]),"--c-up",str(frozen["c_up"]),"--ref-gate",str(frozen["reference_gate"]),"--ref-q",str(frozen["ref_q"]),"--ref-k",str(frozen["ref_k"]),"--ref-ffn-inp",str(frozen["ref_ffn_inp"]),"--ref-shared-out",str(frozen["ref_shared_out"]),"--ref-weights",str(frozen["ref_weights"]),"--out-dir",str(root)];invocations=1;commands["diagnostic"]=base.run_command(command,out,"diagnostic",21600)
            if commands["diagnostic"]["returncode"]:
                failure=root/"strat01_layer1_routed_up_projection_cross_input.json"
                if failure.is_file():
                    try: data=json.loads(failure.read_text(encoding="utf-8"));up_completed=int(data.get("up_arms_completed",0));q6_completed=int(data.get("q6_arms_completed",0))
                    except (OSError,ValueError,json.JSONDecodeError): pass
            base.require_ok(commands["diagnostic"],"diagnostic");up_completed,q6_completed=3,5;source_map=sources();values,report=validate_report(root,model,source_map);result=adjudicate(values,report,frozen);status=result["status"]
    except (RunnerError,base.RunnerError) as exc: errors.append(str(exc))
    except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
    record={"schema":"strat01_layer1_routed_up_projection_cross_input_v1","status":status,"errors":errors,"diagnostic_invocations":invocations,"up_arms_completed":up_completed,"q6_arms_completed":q6_completed,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":result,"c_report":report,"provenance":{"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-tick,"git_head_observed" if args.apparatus_only else "git_head":base.git_value(["git","rev-parse","HEAD"]),"source_hashes":source_map,"predecessor":{"path":str(PREDECESSOR),"sha256":PREDECESSOR_SHA},"artifact":artifact,"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd()},"binary":{"path":str(binary) if binary else None,"sha256":base.sha256_file(binary) if binary and binary.is_file() else None},"commands":commands},"non_claims":["graph execution","production repair","later layers","quality/RAM/rate"]};base.write_json(out/"adjudication.json",record);print(json.dumps({"status":status,"output":str(out),"errors":errors},indent=2));return 0 if status=="APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2


if __name__=="__main__": raise SystemExit(main())
