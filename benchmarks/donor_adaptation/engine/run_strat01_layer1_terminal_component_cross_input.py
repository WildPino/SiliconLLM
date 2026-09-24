#!/usr/bin/env python3
"""Execute the frozen layer-1 terminal-component cross-input diagnostic."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmarks.donor_adaptation.engine import run_strat01_layer2_attn_rmsnorm_cross_input as an
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE=Path(__file__).resolve().parent; ENGINE=an.ENGINE; DEFAULT_MODEL=an.DEFAULT_MODEL
HEADER=ROOT/"benchmarks/phase60/strat01_gguf_layer1_terminal_component_cross_input.h"
PROTOCOL=ROOT/"docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS=HERE/"test_strat01_layer1_terminal_component_cross_input.py"
DEFAULT_OUTPUT=HERE/"results/strat01_gigachat_engine_layer1_terminal_component_cross_input_20260924"
DEFAULT_APPARATUS=HERE/"results/strat01_gigachat_engine_layer1_terminal_component_cross_input_apparatus_20260924"
PREDECESSOR=an.DEFAULT_OUTPUT/"adjudication.json"; PREDECESSOR_SHA="a42540afb96a70dfb411092d3f416cbbf840c83479302ac2eb1fd8bc9ad5d430"
PREDECESSOR_OUTPUT=an.DEFAULT_OUTPUT/"diagnostic/captured_c_norm.f32le"; PREDECESSOR_OUTPUT_SHA="81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
INTEGRATION=HERE/"results/strat01_gigachat_engine_post_f16_swiglu_production_integration_20260924/adjudication.json"; INTEGRATION_SHA="d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94"
REF_ROOT=HERE/"results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference"
C_ROOT=HERE/"results/strat01_gigachat_engine_post_f16_swiglu_production_integration_20260924/c_engine"
ARM_NAMES=("captured_ref_l_out","captured_c_l_out","computed_ref_inp_ref_out","computed_c_inp_c_out","cross_ref_inp_c_out","cross_c_inp_ref_out","control_ref_out_token7_negated")
ARM_META={
 ARM_NAMES[0]:("captured","reference","reference",False),ARM_NAMES[1]:("captured","c","c",False),
 ARM_NAMES[2]:("computed","reference","reference",False),ARM_NAMES[3]:("computed","c","c",False),
 ARM_NAMES[4]:("computed","reference","c",False),ARM_NAMES[5]:("computed","c","reference",False),
 ARM_NAMES[6]:("computed","reference","reference",True),
}
VALID_STATUSES={"LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT","LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT","LAYER1_TERMINAL_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT","LAYER1_TERMINAL_COMPONENT_RESIDUALS_JOINTLY_SUFFICIENT"}
PINNED={
 "ref_q":an.PINNED["ref_q"],"ref_k":an.PINNED["ref_k"],"ref_target":an.PINNED["ref_target"],
 "ref_ffn_inp":(REF_ROOT/"prefill8/ffn_inp-1.full.f32le","99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8",49152),
 "ref_ffn_out":(REF_ROOT/"prefill8/ffn_out-1.full.f32le","e8cdbbf3154447c90fa2dbddff5a454bd37200091fa2c5aaf7021104e76c104e",49152),
 "ref_l_out":(REF_ROOT/"prefill8/l_out-1.full.f32le","40d5a0f07fbb77c1ef73ca24f81cb35ea0df32457faa8d04d6c5cd33cd9f1d5f",49152),
 "c_ffn_inp":(C_ROOT/"prefill8_ffn_inp-1.f32","c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2",49152),
 "c_ffn_out":(C_ROOT/"prefill8_ffn_out-1.f32","2932f1d3b23fc439ebdbee791a031a93e0681724f62935967e324b19a09196f5",49152),
 "c_l_out":(C_ROOT/"prefill8_l_out-1.f32","9af8cec3f42781e9f4cac6af1ea13f6a63b751a5323201a8a18f91b3f7b7bbcb",49152),
}
TWINS={
 "ref_q":an.TWINS["ref_q"],"ref_k":an.TWINS["ref_k"],"ref_target":an.TWINS["ref_target"],
 "ref_ffn_inp":REF_ROOT/"cached7p1/ffn_inp-1.full.f32le","ref_ffn_out":REF_ROOT/"cached7p1/ffn_out-1.full.f32le","ref_l_out":REF_ROOT/"cached7p1/l_out-1.full.f32le",
 "c_ffn_inp":C_ROOT/"cached7p1_ffn_inp-1.f32","c_ffn_out":C_ROOT/"cached7p1_ffn_out-1.f32","c_l_out":C_ROOT/"cached7p1_l_out-1.f32",
}
EXPECTED_SUM_SHA={ARM_NAMES[2]:PINNED["ref_l_out"][1],ARM_NAMES[3]:PINNED["c_l_out"][1],ARM_NAMES[4]:"36626b5d7d7159caa92be092b83b64f176dc01997bb7291e6f2038d24c413bfa",ARM_NAMES[5]:"a0edcb0717a4d717dbd8a9a14f03441b7dd865a99bd25b52b07ef50dac03b42f"}
FROZEN={"nrmse":0.002642086799405445,"normalized_max":0.004687597394884091}

class RunnerError(RuntimeError): pass

def sources():
 p={"runner":Path(__file__).resolve(),"tests":TESTS,"protocol":PROTOCOL,"engine":ENGINE,"header":HEADER,"attn_norm_header":an.CROSS_HEADER,"attn_norm_runner":Path(an.__file__).resolve(),"kva_header":an.KVA_HEADER,"rms_header":an.RMS_HEADER,"rung2a":an.RUNG2A,"rung2c":an.RUNG2C,"base_runner":Path(base.__file__).resolve()}
 if any(not x.is_file() for x in p.values()): raise RunnerError("missing terminal-component source")
 return {n:{"path":str(x),"sha256":base.sha256_file(x)} for n,x in p.items()}

def clean(s):
 rel=[Path(x["path"]).resolve().relative_to(ROOT.resolve()) for x in s.values()]
 if subprocess.run(["git","diff","--quiet","HEAD","--",*map(str,rel)],cwd=ROOT,check=False).returncode: raise RunnerError("terminal-component sources differ from HEAD")
 for p in rel:
  if subprocess.run(["git","ls-files","--error-unmatch",str(p)],cwd=ROOT,capture_output=True,check=False).returncode: raise RunnerError(f"untracked terminal-component source: {p}")

def frozen_evidence():
 out={}
 for n,(p,h,z) in PINNED.items():
  if not p.is_file() or p.stat().st_size!=z or base.sha256_file(p)!=h: raise RunnerError(f"terminal-component input mismatch: {n}")
  twin=TWINS[n]
  if not twin.is_file() or twin.stat().st_size!=z or base.sha256_file(twin)!=h: raise RunnerError(f"terminal-component twin mismatch: {n}")
  out[n]=p.resolve(strict=True)
 for p,h,label in ((PREDECESSOR,PREDECESSOR_SHA,"attention RMSNorm predecessor"),(INTEGRATION,INTEGRATION_SHA,"layer-1 integration predecessor"),(PREDECESSOR_OUTPUT,PREDECESSOR_OUTPUT_SHA,"predecessor output")):
  if not p.is_file() or base.sha256_file(p)!=h: raise RunnerError(label+" mismatch")
 out["prior"]=PREDECESSOR_OUTPUT.resolve(strict=True); return out

def arm_manifest(outputs):
 if not isinstance(outputs,dict) or tuple(outputs)!=ARM_NAMES: raise RunnerError("terminal-component arm labels/order mismatch")
 for name,item in outputs.items():
  if set(item)!={"kind","ffn_inp_origin","ffn_out_origin","output_control","l_out","norm","projection","prefix","downstream"}: raise RunnerError("terminal-component arm schema mismatch")
  if (item["kind"],item["ffn_inp_origin"],item["ffn_out_origin"],item["output_control"])!=ARM_META[name]: raise RunnerError("terminal-component arm metadata mismatch")
  for key,size in (("l_out",49152),("norm",49152),("projection",18432),("prefix",16384),("downstream",196608)):
   if set(item[key])!={"path","bytes","sha256"} or item[key]["bytes"]!=size: raise RunnerError("terminal-component payload schema mismatch")

def validate_report(root,model,s):
 r=json.loads((root/"strat01_layer1_terminal_component_cross_input.json").read_text(encoding="utf-8")); required={"command","state","self_certifies_pass","model","inputs","matrices","outputs","engine_source_sha256","diagnostic_source_sha256","compiler_family","donor_graph_executions","reference_graph_executions","timing_or_rate_claim"}
 if set(r)!=required or r["command"]!="--strat01-layer1-terminal-component-cross-input" or r["state"]!="OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or r["self_certifies_pass"] is not False: raise RunnerError("terminal-component report schema mismatch")
 if r["model"]!={"path":str(model),"bytes":base.EXPECTED_MODEL_BYTES,"sha256":base.EXPECTED_MODEL_SHA}: raise RunnerError("terminal-component model mismatch")
 order=("ref_q","ref_k","ref_ffn_inp","ref_ffn_out","ref_l_out","c_ffn_inp","c_ffn_out","c_l_out"); expected={n:{"path":str(PINNED[n][0].resolve()),"bytes":PINNED[n][2],"sha256":PINNED[n][1]} for n in order}
 if r["inputs"]!=expected: raise RunnerError("terminal-component inputs mismatch")
 matrices={"attn_norm":{"name":"blk.2.attn_norm.weight","type":"F32","shape":[1536],"offset":578811136,"file_offset":584914048,"span":6144},"projection":{"name":"blk.2.attn_kv_a_mqa.weight","type":"Q4_K","shape":[1536,576]},"kv_norm":{"name":"blk.2.attn_kv_a_norm.weight","type":"F32","shape":[512]},"v_b":{"name":"blk.2.attn_v_b.weight","type":"Q4_K","shape":[512,192,32]}}
 if r["matrices"]!=matrices: raise RunnerError("terminal-component descriptors mismatch")
 if r["engine_source_sha256"]!=s["engine"]["sha256"] or r["diagnostic_source_sha256"]!=s["header"]["sha256"] or r["compiler_family"]!="clang" or r["donor_graph_executions"]!=0 or r["reference_graph_executions"]!=0 or r["timing_or_rate_claim"] is not None: raise RunnerError("terminal-component provenance mismatch")
 arm_manifest(r["outputs"]); values={}
 for n,item in r["outputs"].items(): values[n]={k:base.load_f32(base.contained(root,item[k]["path"],item[k]["bytes"],item[k]["sha256"],n+" "+k),item[k]["bytes"]//4,n+" "+k) for k in ("l_out","norm","projection","prefix","downstream")}
 return values,r

def classify(j):
 if not j[ARM_NAMES[0]]["pass"] or j[ARM_NAMES[1]]["pass"] or j[ARM_NAMES[3]]["pass"]: raise RunnerError("terminal-component anchor contradiction")
 out_fails=not j[ARM_NAMES[4]]["pass"]; inp_fails=not j[ARM_NAMES[5]]["pass"]
 if out_fails and inp_fails:return "LAYER1_TERMINAL_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT"
 if out_fails:return "LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT"
 if inp_fails:return "LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT"
 return "LAYER1_TERMINAL_COMPONENT_RESIDUALS_JOINTLY_SUFFICIENT"

def descriptive(candidate,reference,shape):
 x=base.metrics(candidate,reference); a=np.asarray(candidate).reshape(shape); b=np.asarray(reference).reshape(shape); x["per_token"]=[dict(base.metrics(a[i],b[i]),token=i) for i in range(shape[0])]; return x

def adjudicate(values,report,f):
 ref=base.load_f32(f["ref_target"],49152,"reference target"); prior=base.load_f32(f["prior"],49152,"predecessor output"); ref_sum=base.load_f32(f["ref_l_out"],12288,"reference l_out"); c_sum=base.load_f32(f["c_l_out"],12288,"C l_out")
 if values[ARM_NAMES[0]]["l_out"].tobytes()!=ref_sum.tobytes() or values[ARM_NAMES[1]]["l_out"].tobytes()!=c_sum.tobytes(): raise RunnerError("terminal-component captured anchor mismatch")
 for key in ("l_out","norm","projection","prefix","downstream"):
  if values[ARM_NAMES[2]][key].tobytes()!=values[ARM_NAMES[0]][key].tobytes(): raise RunnerError("terminal-component reference replay mismatch")
  if values[ARM_NAMES[3]][key].tobytes()!=values[ARM_NAMES[1]][key].tobytes(): raise RunnerError("terminal-component C replay mismatch")
 for name,digest in EXPECTED_SUM_SHA.items():
  if hashlib.sha256(values[name]["l_out"].astype("<f4",copy=False).tobytes()).hexdigest()!=digest: raise RunnerError("terminal-component cross-sum mismatch: "+name)
 if values[ARM_NAMES[0]]["downstream"].tobytes()!=ref.tobytes() or values[ARM_NAMES[1]]["downstream"].tobytes()!=prior.tobytes(): raise RunnerError("terminal-component downstream anchor mismatch")
 judgments={n:base.judged(values[n]["downstream"],ref) for n in ARM_NAMES[:6]}
 for metric,expected in FROZEN.items():
  if abs(judgments[ARM_NAMES[1]][metric]-expected)>1e-12: raise RunnerError("terminal-component frozen metric mismatch")
 control=base.judged(values[ARM_NAMES[6]]["downstream"],ref)
 if control["pass"]: raise RunnerError("terminal-component planted control failed")
 mutations={}
 for n,(_,digest,size) in PINNED.items(): data=bytearray(f[n].read_bytes()); data[len(data)//2]^=1; mutations[n]=not base.identity_matches(bytes(data),size,digest)
 if not all(mutations.values()): raise RunnerError("terminal-component mutation failed")
 swapped=copy.deepcopy(report["outputs"]); items=list(swapped.items()); items[4],items[5]=items[5],items[4]
 try: arm_manifest(dict(items)); label=False
 except RunnerError: label=True
 if not label: raise RunnerError("terminal-component label swap failed")
 component_metrics={"c_ffn_inp_vs_ref":descriptive(base.load_f32(f["c_ffn_inp"],12288,"C ffn input"),base.load_f32(f["ref_ffn_inp"],12288,"reference ffn input"),(8,1536)),"c_ffn_out_vs_ref":descriptive(base.load_f32(f["c_ffn_out"],12288,"C ffn output"),base.load_f32(f["ref_ffn_out"],12288,"reference ffn output"),(8,1536))}
 return {"status":classify(judgments),"arms_vs_reference":judgments,"terminal_sum_metrics":{n:descriptive(values[n]["l_out"],ref_sum,(8,1536)) for n in ARM_NAMES[:6]},"component_metrics":component_metrics,"control_vs_reference":control,"controls":{"reference_replay_all_stages_byte_exact":True,"c_replay_all_stages_byte_exact":True,"captured_anchors_byte_exact":True,"schedule_twins_byte_exact":True,"mutated_inputs_refused":mutations,"label_swap_rejected":label}}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,default=DEFAULT_MODEL); p.add_argument("--output-dir",type=Path); p.add_argument("--apparatus-only",action="store_true"); a=p.parse_args(); model=a.model.resolve(); out=(a.output_dir or (DEFAULT_APPARATUS if a.apparatus_only else DEFAULT_OUTPUT)).resolve()
 if out.exists(): raise SystemExit(f"output already exists: {out}")
 out.mkdir(parents=True); started=datetime.now(timezone.utc).isoformat(); tick=time.perf_counter(); status="VOID_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT"; errors=[]; commands={}; report={}; adj={"status":"NOT_RUN"}; source_map={}; compiler=shutil.which("clang"); binary=None; invocations=0; artifact={"path":str(model),"expected_bytes":base.EXPECTED_MODEL_BYTES,"expected_sha256":base.EXPECTED_MODEL_SHA,"bytes":None,"sha256":None,"opened":False}
 try:
  source_map=sources()
  if not compiler: raise RunnerError("clang unavailable")
  commands["clang_version"]=base.run_command([compiler,"--version"],out,"clang_version",30); base.require_ok(commands["clang_version"],"clang"); binary=out/"engine_layer1_terminal_component.exe"; commands["compile"]=base.run_command([compiler,*base.COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],out,"compile",600); base.require_ok(commands["compile"],"compile")
  tests=(("terminal_component_selftest","--strat01-layer1-terminal-component-cross-input-selftest"),("attn_norm_selftest","--strat01-layer2-attn-rmsnorm-cross-input-selftest"),("kva_selftest","--strat01-layer2-kv-a-projection-cross-input-selftest"),("rms_selftest","--strat01-layer2-kv-rmsnorm-cross-input-selftest"),("partition_selftest","--strat01-layer2-kv-partition-cross-input-selftest"),("legacy_selftest","--kselftest"))
  for label,flag in tests: commands[label]=base.run_command([str(binary),flag],out,label,300); base.require_ok(commands[label],label)
  commands["python_tests"]=base.run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_layer1_terminal_component_cross_input"],out,"python_tests",300); base.require_ok(commands["python_tests"],"tests")
  if a.apparatus_only: status="APPARATUS_READY_NO_DONOR_EXECUTION"
  else:
   clean(source_map)
   if not model.is_file() or model.stat().st_size!=base.EXPECTED_MODEL_BYTES or base.sha256_file(model)!=base.EXPECTED_MODEL_SHA: raise RunnerError("artifact mismatch")
   artifact.update({"bytes":model.stat().st_size,"sha256":base.EXPECTED_MODEL_SHA,"opened":True}); f=frozen_evidence(); root=out/"diagnostic"; root.mkdir(); cmd=[str(binary),"--strat01-layer1-terminal-component-cross-input",str(model),"--ref-q",str(f["ref_q"]),"--ref-k",str(f["ref_k"]),"--ref-ffn-inp",str(f["ref_ffn_inp"]),"--ref-ffn-out",str(f["ref_ffn_out"]),"--ref-l-out",str(f["ref_l_out"]),"--c-ffn-inp",str(f["c_ffn_inp"]),"--c-ffn-out",str(f["c_ffn_out"]),"--c-l-out",str(f["c_l_out"]),"--out-dir",str(root)]; invocations=1; commands["diagnostic"]=base.run_command(cmd,out,"diagnostic",21600); base.require_ok(commands["diagnostic"],"diagnostic"); source_map=sources(); values,report=validate_report(root,model,source_map); adj=adjudicate(values,report,f); status=adj["status"]
 except (RunnerError,base.RunnerError) as exc: errors.append(str(exc))
 except Exception as exc: errors.append(f"unexpected {type(exc).__name__}: {exc}")
 record={"schema":"strat01_layer1_terminal_component_cross_input_v1","status":status,"errors":errors,"diagnostic_invocations":invocations,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":adj,"c_report":report,"provenance":{"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-tick,"git_head_observed" if a.apparatus_only else "git_head":base.git_value(["git","rev-parse","HEAD"]),"source_hashes":source_map,"predecessors":{"attention_rmsnorm":{"path":str(PREDECESSOR),"sha256":PREDECESSOR_SHA},"layer1_integration":{"path":str(INTEGRATION),"sha256":INTEGRATION_SHA}},"artifact":artifact,"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd()},"binary":{"path":str(binary) if binary else None,"sha256":base.sha256_file(binary) if binary and binary.is_file() else None},"commands":commands},"non_claims":["repair","producer rerun","uncaptured operator","later layers","quality/RAM/rate"]}; base.write_json(out/"adjudication.json",record); print(json.dumps({"status":status,"output":str(out),"errors":errors},indent=2)); return 0 if status=="APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2

if __name__=="__main__": raise SystemExit(main())
