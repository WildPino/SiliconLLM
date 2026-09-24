#!/usr/bin/env python3
"""Execute the frozen layer-2 KV RMSNorm cross-input diagnostic."""
from __future__ import annotations
import argparse, copy, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_partition_cross_input as partition
from benchmarks.donor_adaptation.engine import run_strat01_layer2_attention_cross_input as attention
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE=Path(__file__).resolve().parent; ENGINE=partition.ENGINE; RUNG2A=partition.RUNG2A; RUNG2C=partition.RUNG2C
INHERITED_HEADER=partition.INHERITED_HEADER; ATTENTION_HEADER=partition.PREDECESSOR_HEADER; PARTITION_HEADER=partition.CROSS_HEADER
CROSS_HEADER=ROOT/"benchmarks/phase60/strat01_gguf_layer2_kv_rmsnorm_cross_input.h"
PROTOCOL=ROOT/"docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS=HERE/"test_strat01_layer2_kv_rmsnorm_cross_input.py"; RAW=attention.RAW; DEFAULT_MODEL=partition.DEFAULT_MODEL
DEFAULT_OUTPUT=HERE/"results/strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_20260924"
DEFAULT_APPARATUS=HERE/"results/strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_apparatus_20260924"
PREDECESSOR_DIR=partition.DEFAULT_OUTPUT; PREDECESSOR_ADJUDICATION_SHA="c6f59af8cf83f413cee14ae38ed451270531badc1a71095875a5f39680eebe7f"
PREDECESSOR_OUTPUT_SHA="81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
FROZEN_PREDECESSOR_METRICS={"nrmse":0.002642086799405445,"normalized_max":0.004687597394884091}
ARM_NAMES=("captured_ref_norm","captured_c_norm","computed_ref_input","computed_c_input","control_ref_input_token7_negated","control_norm_weight_negated")
ARM_META={"captured_ref_norm":("captured","reference",False,False),"captured_c_norm":("captured","c",False,False),"computed_ref_input":("computed","reference",False,False),"computed_c_input":("computed","c",False,False),"control_ref_input_token7_negated":("computed","reference",True,False),"control_norm_weight_negated":("computed","reference",False,True)}
VALID_STATUSES={"LAYER2_KV_RMSNORM_FAILS_EXACT_REFERENCE_INPUT","LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT"}
PINNED_FILES={
 "ref_q":(RAW/"pinned_reference/prefill8/Qcur-2.full.f32le","46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea",589824),
 "ref_k":(RAW/"pinned_reference/prefill8/Kcur-2.full.f32le","b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b",18432),
 "ref_pre":(RAW/"pinned_reference/prefill8/kv_cmpr_pe-2.full.f32le","81c439e40a9dfd06221a86b7cdc9b6019ea8742e6cf11371d1d2c50546346c61",18432),
 "ref_norm":(RAW/"pinned_reference/prefill8/kv_cmpr-2.full.f32le","0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91",16384),
 "ref_target":(RAW/"pinned_reference/prefill8/kqv_out-2.full.f32le","d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e",196608),
 "c_pre":(RAW/"c_engine/prefill8_kv_cmpr_pe-2.f32","aae2f54a6391b84c156c3b61f3206be1170afb7e09995cc891f1fd9e7ebfc444",18432),
 "c_norm":(RAW/"c_engine/prefill8_kv_cmpr-2.f32","e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9",16384),
}
SCHEDULE_TWINS={"ref_q":RAW/"pinned_reference/cached7p1/Qcur-2.full.f32le","ref_k":RAW/"pinned_reference/cached7p1/Kcur-2.full.f32le","ref_pre":RAW/"pinned_reference/cached7p1/kv_cmpr_pe-2.full.f32le","ref_norm":RAW/"pinned_reference/cached7p1/kv_cmpr-2.full.f32le","ref_target":RAW/"pinned_reference/cached7p1/kqv_out-2.full.f32le","c_pre":RAW/"c_engine/cached7p1_kv_cmpr_pe-2.f32","c_norm":RAW/"c_engine/cached7p1_kv_cmpr-2.f32"}
EVIDENCE_FILES={"canonical_recovery":(attention.RECOVERY,attention.RECOVERY_SHA),"partition_predecessor":(PREDECESSOR_DIR/"adjudication.json",PREDECESSOR_ADJUDICATION_SHA),"reference_manifest":attention.EVIDENCE_FILES["reference_prefill_manifest"],"c_manifest":attention.EVIDENCE_FILES["c_prefill_manifest"]}
class RunnerError(RuntimeError): pass

def source_inventory():
 p={"runner":Path(__file__).resolve(),"tests":TESTS,"protocol":PROTOCOL,"engine":ENGINE,"rung2a":RUNG2A,"rung2c":RUNG2C,"inherited_cross_header":INHERITED_HEADER,"attention_header":ATTENTION_HEADER,"partition_header":PARTITION_HEADER,"rmsnorm_header":CROSS_HEADER}
 m=[n for n,x in p.items() if not x.is_file()]
 if m: raise RunnerError("missing layer-2 KV RMSNorm source(s): "+", ".join(m))
 return {n:{"path":str(x),"sha256":base.sha256_file(x)} for n,x in p.items()}

def clean_sources_at_head(s):
 rel=[Path(x["path"]).resolve().relative_to(ROOT.resolve()) for x in s.values()]
 if subprocess.run(["git","diff","--quiet","HEAD","--",*map(str,rel)],cwd=ROOT,check=False).returncode: raise RunnerError("layer-2 KV RMSNorm sources differ from HEAD")
 for p in rel:
  if subprocess.run(["git","ls-files","--error-unmatch",str(p)],cwd=ROOT,capture_output=True,check=False).returncode: raise RunnerError(f"untracked layer-2 KV RMSNorm source: {p}")

def validate_frozen_evidence():
 r={}
 for n,(p,h,z) in PINNED_FILES.items():
  if not p.is_file() or p.stat().st_size!=z or base.sha256_file(p)!=h: raise RunnerError(f"pinned layer-2 {n} identity mismatch")
  t=SCHEDULE_TWINS[n]
  if not t.is_file() or t.stat().st_size!=z or base.sha256_file(t)!=h: raise RunnerError(f"cached schedule twin mismatch for {n}")
  r[n]=p.resolve(strict=True)
 for n,(p,h) in EVIDENCE_FILES.items():
  if not p.is_file() or base.sha256_file(p)!=h: raise RunnerError(f"layer-2 evidence {n} identity mismatch")
  r[n]=p.resolve(strict=True)
 prior=PREDECESSOR_DIR/"diagnostic/c_latent__ref_positional.f32le"
 if not prior.is_file() or prior.stat().st_size!=196608 or base.sha256_file(prior)!=PREDECESSOR_OUTPUT_SHA: raise RunnerError("partition predecessor output identity mismatch")
 r["predecessor_output"]=prior.resolve(strict=True); return r

def validate_arm_manifest(o):
 if not isinstance(o,dict) or tuple(o)!=ARM_NAMES: raise RunnerError("layer-2 KV RMSNorm arm labels/order mismatch")
 req={"kind","source","input_control","weight_control","prefix","downstream"}
 for n,x in o.items():
  if not isinstance(x,dict) or set(x)!=req: raise RunnerError(f"arm schema mismatch: {n}")
  k,s,ic,wc=ARM_META[n]
  if (x["kind"],x["source"],x["input_control"],x["weight_control"])!=(k,s,ic,wc): raise RunnerError(f"arm metadata mismatch: {n}")
  if set(x["prefix"])!={"path","bytes","sha256"} or x["prefix"]["bytes"]!=16384 or set(x["downstream"])!={"path","bytes","sha256"} or x["downstream"]["bytes"]!=196608: raise RunnerError(f"arm payload schema mismatch: {n}")

def validate_report(root,model,sources):
 try: r=json.loads((root/"strat01_layer2_kv_rmsnorm_cross_input.json").read_text(encoding="utf-8"))
 except (OSError,json.JSONDecodeError) as e: raise RunnerError(f"KV RMSNorm report missing/malformed: {e}") from e
 req={"command","state","self_certifies_pass","model","inputs","split","matrices","outputs","engine_source_sha256","diagnostic_source_sha256","compiler_family","donor_graph_executions","timing_or_rate_claim"}
 if set(r)!=req or r["command"]!="--strat01-layer2-kv-rmsnorm-cross-input" or r["state"]!="OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or r["self_certifies_pass"] is not False: raise RunnerError("KV RMSNorm report schema/state mismatch")
 if r["model"]!={"path":str(model),"bytes":base.EXPECTED_MODEL_BYTES,"sha256":base.EXPECTED_MODEL_SHA}: raise RunnerError("KV RMSNorm model identity mismatch")
 ei={n:{"path":str(PINNED_FILES[n][0].resolve()),"bytes":PINNED_FILES[n][2],"sha256":PINNED_FILES[n][1]} for n in ("ref_q","ref_k","ref_pre","ref_norm","c_pre","c_norm")}
 if r["inputs"]!=ei or r["split"]!=512: raise RunnerError("KV RMSNorm input/split mismatch")
 if r["matrices"]!={"norm":{"name":"blk.2.attn_kv_a_norm.weight","type":"F32","shape":[512],"offset":578809088,"file_offset":584912000,"span":2048},"v_b":{"name":"blk.2.attn_v_b.weight","type":"Q4_K","shape":[512,192,32],"offset":589434112,"file_offset":595537024,"span":1769472}}: raise RunnerError("KV RMSNorm descriptor mismatch")
 if r["engine_source_sha256"]!=sources["engine"]["sha256"] or r["diagnostic_source_sha256"]!=sources["rmsnorm_header"]["sha256"] or r["compiler_family"]!="clang" or r["donor_graph_executions"]!=0 or r["timing_or_rate_claim"] is not None: raise RunnerError("KV RMSNorm provenance/execution mismatch")
 validate_arm_manifest(r["outputs"]); prefixes={}; outputs={}
 for n,x in r["outputs"].items():
  prefixes[n]=base.load_f32(base.contained(root,x["prefix"]["path"],16384,x["prefix"]["sha256"],n+" prefix"),4096,n+" prefix")
  outputs[n]=base.load_f32(base.contained(root,x["downstream"]["path"],196608,x["downstream"]["sha256"],n),49152,n)
 return prefixes,outputs,r

def classify(j):
 if not j["captured_ref_norm"]["pass"] or j["captured_c_norm"]["pass"] or j["computed_c_input"]["pass"]: raise RunnerError("frozen KV RMSNorm anchor contradiction")
 return "LAYER2_KV_RMSNORM_FAILS_EXACT_REFERENCE_INPUT" if not j["computed_ref_input"]["pass"] else "LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT"

def adjudicate(prefixes,outputs,report,f):
 ref=base.load_f32(f["ref_target"],49152,"reference target"); prior=base.load_f32(f["predecessor_output"],49152,"partition predecessor")
 if prefixes["captured_ref_norm"].tobytes()!=base.load_f32(f["ref_norm"],4096,"reference norm").tobytes() or prefixes["captured_c_norm"].tobytes()!=base.load_f32(f["c_norm"],4096,"C norm").tobytes(): raise RunnerError("captured normalized-prefix replay mismatch")
 if prefixes["computed_c_input"].tobytes()!=prefixes["captured_c_norm"].tobytes() or outputs["computed_c_input"].tobytes()!=outputs["captured_c_norm"].tobytes(): raise RunnerError("production C RMSNorm replay mismatch")
 if outputs["captured_ref_norm"].tobytes()!=ref.tobytes() or outputs["captured_c_norm"].tobytes()!=prior.tobytes(): raise RunnerError("downstream anchor replay mismatch")
 judged={n:base.judged(outputs[n],ref) for n in ARM_NAMES[:4]}
 for m,e in FROZEN_PREDECESSOR_METRICS.items():
  if abs(judged["captured_c_norm"][m]-e)>1e-12: raise RunnerError(f"frozen partition metric mismatch: {m}")
 controls={n:base.judged(outputs[n],ref) for n in ARM_NAMES[4:]}
 if any(x["pass"] for x in controls.values()): raise RunnerError("KV RMSNorm planted control did not reject")
 muts={}
 for n,(_,h,z) in PINNED_FILES.items():
  b=bytearray(f[n].read_bytes());b[len(b)//2]^=1;muts[n]=not base.identity_matches(bytes(b),z,h)
 if not all(muts.values()): raise RunnerError("KV RMSNorm mutation control failed")
 swapped=copy.deepcopy(report["outputs"]); items=list(swapped.items());items[2],items[3]=items[3],items[2]
 try: validate_arm_manifest(dict(items)); label=False
 except RunnerError: label=True
 if not label: raise RunnerError("KV RMSNorm label swap not rejected")
 return {"status":classify(judged),"arms_vs_reference":judged,"prefixes_vs_reference_norm":{n:base.judged(prefixes[n],prefixes["captured_ref_norm"]) for n in ARM_NAMES[:4]},"controls_vs_reference":controls,"controls":{"captured_reference_byte_exact":True,"captured_predecessor_byte_exact":True,"computed_c_prefix_and_downstream_byte_exact":True,"frozen_metrics_reproduced_within_1e-12":True,"schedule_twins_byte_exact":True,"mutated_inputs_refused":muts,"scientific_label_swap_rejected":label}}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",type=Path,default=DEFAULT_MODEL);p.add_argument("--output-dir",type=Path);p.add_argument("--apparatus-only",action="store_true");a=p.parse_args();model=a.model.resolve();out=(a.output_dir or (DEFAULT_APPARATUS if a.apparatus_only else DEFAULT_OUTPUT)).resolve()
 if out.exists(): raise SystemExit(f"output already exists: {out}")
 out.mkdir(parents=True);started_utc=datetime.now(timezone.utc).isoformat();started=time.perf_counter();status="VOID_LAYER2_KV_RMSNORM_CROSS_INPUT";errors=[];commands={};report={};adj={"status":"NOT_RUN"};sources={};compiler=shutil.which("clang");binary=None;invocations=0;artifact={"path":str(model),"expected_bytes":base.EXPECTED_MODEL_BYTES,"expected_sha256":base.EXPECTED_MODEL_SHA,"bytes":None,"sha256":None,"opened":False}
 try:
  sources=source_inventory()
  if not compiler: raise RunnerError("clang unavailable")
  commands["clang_version"]=base.run_command([compiler,"--version"],out,"clang_version",30);base.require_ok(commands["clang_version"],"clang version");binary=out/"engine_layer2_kv_rmsnorm_cross_input.exe";commands["compile"]=base.run_command([compiler,*base.COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],out,"compile",600);base.require_ok(commands["compile"],"KV RMSNorm build")
  for label,flag in (("rmsnorm_selftest","--strat01-layer2-kv-rmsnorm-cross-input-selftest"),("partition_selftest","--strat01-layer2-kv-partition-cross-input-selftest"),("attention_selftest","--strat01-layer2-attention-cross-input-selftest"),("legacy_selftest","--kselftest")):
   commands[label]=base.run_command([str(binary),flag],out,label,300);base.require_ok(commands[label],label)
  commands["python_tests"]=base.run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_layer2_kv_rmsnorm_cross_input"],out,"python_tests",300);base.require_ok(commands["python_tests"],"KV RMSNorm Python tests")
  if a.apparatus_only: status="APPARATUS_READY_NO_DONOR_EXECUTION"
  else:
   clean_sources_at_head(sources)
   if not model.is_file() or model.stat().st_size!=base.EXPECTED_MODEL_BYTES or base.sha256_file(model)!=base.EXPECTED_MODEL_SHA: raise RunnerError("accepted artifact identity mismatch")
   artifact.update({"bytes":model.stat().st_size,"sha256":base.EXPECTED_MODEL_SHA,"opened":True});f=validate_frozen_evidence();root=out/"diagnostic";root.mkdir();cmd=[str(binary),"--strat01-layer2-kv-rmsnorm-cross-input",str(model),"--ref-q",str(f["ref_q"]),"--ref-k",str(f["ref_k"]),"--ref-pre",str(f["ref_pre"]),"--ref-norm",str(f["ref_norm"]),"--c-pre",str(f["c_pre"]),"--c-norm",str(f["c_norm"]),"--out-dir",str(root)];invocations=1;commands["boundary_diagnostic"]=base.run_command(cmd,out,"boundary_diagnostic",21600);base.require_ok(commands["boundary_diagnostic"],"KV RMSNorm diagnostic");sources=source_inventory();prefixes,outputs,report=validate_report(root,model,sources);adj=adjudicate(prefixes,outputs,report,f);status=adj["status"]
 except (RunnerError,base.RunnerError) as e: errors.append(str(e))
 except Exception as e: errors.append(f"unexpected {type(e).__name__}: {e}")
 prov={"started_utc":started_utc,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-started,"git_head_observed" if a.apparatus_only else "git_head":base.git_value(["git","rev-parse","HEAD"]),"source_hashes":sources,"evidence_hashes":{n:{"path":str(p),"sha256":h} for n,(p,h) in EVIDENCE_FILES.items()},"artifact":artifact,"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd(),"clang_path":compiler},"binary":{"path":str(binary) if binary else None,"sha256":base.sha256_file(binary) if binary and binary.is_file() else None},"commands":commands}
 record={"schema":"strat01_layer2_kv_rmsnorm_cross_input_adjudication_v1","status":status,"errors":errors,"diagnostic_invocations":invocations,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":adj,"c_report":report,"provenance":prov,"non_claims":["repaired layer 2","executed/adjudicated KV-A projection","output projection/FFN/later layers","tokenizer/logits/generation","quality/RAM/rate"]};base.write_json(out/"adjudication.json",record);print(json.dumps({"status":status,"output":str(out),"errors":errors},indent=2));return 0 if status=="APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2
if __name__=="__main__": raise SystemExit(main())
