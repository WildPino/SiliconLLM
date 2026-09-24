#!/usr/bin/env python3
"""Execute the frozen layer-2 KV-A projection cross-input diagnostic."""
from __future__ import annotations
import argparse,copy,json,os,platform,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_rmsnorm_cross_input as rms
from benchmarks.donor_adaptation.engine import run_strat01_layer2_attention_cross_input as attention
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base
HERE=Path(__file__).resolve().parent;ENGINE=rms.ENGINE;RUNG2A=rms.RUNG2A;RUNG2C=rms.RUNG2C
CROSS_HEADER=ROOT/"benchmarks/phase60/strat01_gguf_layer2_kv_a_projection_cross_input.h";RMS_HEADER=rms.CROSS_HEADER
PROTOCOL=ROOT/"docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_A_PROJECTION_CROSS_INPUT_PROTOCOL_20260924.md";TESTS=HERE/"test_strat01_layer2_kv_a_projection_cross_input.py";RAW=attention.RAW;DEFAULT_MODEL=rms.DEFAULT_MODEL
DEFAULT_OUTPUT=HERE/"results/strat01_gigachat_engine_layer2_kv_a_projection_cross_input_20260924";DEFAULT_APPARATUS=HERE/"results/strat01_gigachat_engine_layer2_kv_a_projection_cross_input_apparatus_20260924"
RECOVERY=HERE/"results/strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_offline_recovery1_20260924/adjudication.json";RECOVERY_SHA="e4f4fb67e73e7089fee3f7900c768894b9c9bf0399b5c978169c06f7b07ca976";PREDECESSOR_OUTPUT_SHA="81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254";FROZEN={"nrmse":0.002642086799405445,"normalized_max":0.004687597394884091}
ARM_NAMES=("captured_ref_projection","captured_c_projection","computed_ref_input","computed_c_input","control_ref_input_token7_negated","control_ref_projection_prefix_negated")
ARM_META={"captured_ref_projection":("captured","reference",False,False),"captured_c_projection":("captured","c",False,False),"computed_ref_input":("computed","reference",False,False),"computed_c_input":("computed","c",False,False),"control_ref_input_token7_negated":("computed","reference",True,False),"control_ref_projection_prefix_negated":("computed","reference",False,True)}
VALID_STATUSES={"LAYER2_KV_A_PROJECTION_FAILS_EXACT_REFERENCE_INPUT","LAYER2_ATTN_NORM_INPUT_RESIDUAL_SUFFICIENT"}
PINNED={"ref_q":(RAW/"pinned_reference/prefill8/Qcur-2.full.f32le","46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea",589824),"ref_k":(RAW/"pinned_reference/prefill8/Kcur-2.full.f32le","b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b",18432),"ref_attn":(RAW/"pinned_reference/prefill8/attn_norm-2.full.f32le","7bc7ba62b9e5b779cf079ab212f9b328e247c979423992f42aff786560115f02",49152),"ref_projection":(RAW/"pinned_reference/prefill8/kv_cmpr_pe-2.full.f32le","81c439e40a9dfd06221a86b7cdc9b6019ea8742e6cf11371d1d2c50546346c61",18432),"ref_target":(RAW/"pinned_reference/prefill8/kqv_out-2.full.f32le","d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e",196608),"c_attn":(RAW/"c_engine/prefill8_attn_norm-2.f32","b55cdc958cb9e48ff014c740dc4bfe6b1ec81ef839ee95618018e426673e112d",49152),"c_projection":(RAW/"c_engine/prefill8_kv_cmpr_pe-2.f32","aae2f54a6391b84c156c3b61f3206be1170afb7e09995cc891f1fd9e7ebfc444",18432)}
TWINS={"ref_q":RAW/"pinned_reference/cached7p1/Qcur-2.full.f32le","ref_k":RAW/"pinned_reference/cached7p1/Kcur-2.full.f32le","ref_attn":RAW/"pinned_reference/cached7p1/attn_norm-2.full.f32le","ref_projection":RAW/"pinned_reference/cached7p1/kv_cmpr_pe-2.full.f32le","ref_target":RAW/"pinned_reference/cached7p1/kqv_out-2.full.f32le","c_attn":RAW/"c_engine/cached7p1_attn_norm-2.f32","c_projection":RAW/"c_engine/cached7p1_kv_cmpr_pe-2.f32"}
class RunnerError(RuntimeError):pass
def sources():
 p={"runner":Path(__file__).resolve(),"tests":TESTS,"protocol":PROTOCOL,"engine":ENGINE,"rung2a":RUNG2A,"rung2c":RUNG2C,"rms_header":RMS_HEADER,"projection_header":CROSS_HEADER}
 if any(not x.is_file() for x in p.values()):raise RunnerError("missing KV-A source")
 return {n:{"path":str(x),"sha256":base.sha256_file(x)} for n,x in p.items()}
def clean(s):
 rel=[Path(x["path"]).resolve().relative_to(ROOT.resolve()) for x in s.values()]
 if subprocess.run(["git","diff","--quiet","HEAD","--",*map(str,rel)],cwd=ROOT,check=False).returncode:raise RunnerError("KV-A sources differ from HEAD")
 for p in rel:
  if subprocess.run(["git","ls-files","--error-unmatch",str(p)],cwd=ROOT,capture_output=True,check=False).returncode:raise RunnerError(f"untracked KV-A source: {p}")
def frozen_evidence():
 r={}
 for n,(p,h,z) in PINNED.items():
  if not p.is_file() or p.stat().st_size!=z or base.sha256_file(p)!=h:raise RunnerError(f"KV-A input mismatch: {n}")
  t=TWINS[n]
  if not t.is_file() or t.stat().st_size!=z or base.sha256_file(t)!=h:raise RunnerError(f"KV-A twin mismatch: {n}")
  r[n]=p.resolve(strict=True)
 if not RECOVERY.is_file() or base.sha256_file(RECOVERY)!=RECOVERY_SHA:raise RunnerError("RMSNorm recovery mismatch")
 prior=rms.RAW/"diagnostic/captured_c_norm.f32le"
 if not prior.is_file() or base.sha256_file(prior)!=PREDECESSOR_OUTPUT_SHA:raise RunnerError("RMSNorm predecessor output mismatch")
 r["prior"]=prior.resolve(strict=True);return r
def arm_manifest(o):
 if not isinstance(o,dict) or tuple(o)!=ARM_NAMES:raise RunnerError("KV-A arm labels/order mismatch")
 for n,x in o.items():
  if set(x)!={"kind","source","input_control","projection_control","projection","prefix","downstream"}:raise RunnerError("KV-A arm schema mismatch")
  if (x["kind"],x["source"],x["input_control"],x["projection_control"])!=ARM_META[n]:raise RunnerError("KV-A arm metadata mismatch")
  for k,z in (("projection",18432),("prefix",16384),("downstream",196608)):
   if set(x[k])!={"path","bytes","sha256"} or x[k]["bytes"]!=z:raise RunnerError("KV-A payload schema mismatch")
def validate_report(root,model,s):
 r=json.loads((root/"strat01_layer2_kv_a_projection_cross_input.json").read_text(encoding="utf-8"));req={"command","state","self_certifies_pass","model","inputs","matrices","outputs","engine_source_sha256","diagnostic_source_sha256","compiler_family","donor_graph_executions","timing_or_rate_claim"}
 if set(r)!=req or r["command"]!="--strat01-layer2-kv-a-projection-cross-input" or r["state"]!="OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or r["self_certifies_pass"] is not False:raise RunnerError("KV-A report schema mismatch")
 if r["model"]!={"path":str(model),"bytes":base.EXPECTED_MODEL_BYTES,"sha256":base.EXPECTED_MODEL_SHA}:raise RunnerError("KV-A model mismatch")
 ei={n:{"path":str(PINNED[n][0].resolve()),"bytes":PINNED[n][2],"sha256":PINNED[n][1]} for n in ("ref_q","ref_k","ref_attn","ref_projection","c_attn","c_projection")}
 if r["inputs"]!=ei:raise RunnerError("KV-A inputs mismatch")
 if r["matrices"]!={"projection":{"name":"blk.2.attn_kv_a_mqa.weight","type":"Q4_K","shape":[1536,576],"offset":578311424,"file_offset":584414336,"span":497664},"norm":{"name":"blk.2.attn_kv_a_norm.weight","type":"F32","shape":[512]},"v_b":{"name":"blk.2.attn_v_b.weight","type":"Q4_K","shape":[512,192,32]}}:raise RunnerError("KV-A descriptors mismatch")
 if r["engine_source_sha256"]!=s["engine"]["sha256"] or r["diagnostic_source_sha256"]!=s["projection_header"]["sha256"] or r["compiler_family"]!="clang" or r["donor_graph_executions"]!=0 or r["timing_or_rate_claim"] is not None:raise RunnerError("KV-A provenance mismatch")
 arm_manifest(r["outputs"]);vals={}
 for n,x in r["outputs"].items():vals[n]={k:base.load_f32(base.contained(root,x[k]["path"],x[k]["bytes"],x[k]["sha256"],n+" "+k),x[k]["bytes"]//4,n+" "+k) for k in ("projection","prefix","downstream")}
 return vals,r
def classify(j):
 if not j["captured_ref_projection"]["pass"] or j["captured_c_projection"]["pass"] or j["computed_c_input"]["pass"]:raise RunnerError("KV-A anchor contradiction")
 return "LAYER2_KV_A_PROJECTION_FAILS_EXACT_REFERENCE_INPUT" if not j["computed_ref_input"]["pass"] else "LAYER2_ATTN_NORM_INPUT_RESIDUAL_SUFFICIENT"
def projection_judged(candidate,reference):
 x=base.metrics(candidate,reference);x.update({"nrmse_limit":base.LIMITS[0],"normalized_max_limit":base.LIMITS[1]});x["pass"]=x["nrmse"]<=base.LIMITS[0] and x["normalized_max"]<=base.LIMITS[1];c=np.asarray(candidate).reshape(8,576);q=np.asarray(reference).reshape(8,576);x["per_token"]=[dict(base.metrics(c[i],q[i]),token=i) for i in range(8)];return x
def adjudicate(v,r,f):
 ref=base.load_f32(f["ref_target"],49152,"reference target");prior=base.load_f32(f["prior"],49152,"prior")
 if v["captured_ref_projection"]["projection"].tobytes()!=base.load_f32(f["ref_projection"],4608,"ref projection").tobytes() or v["captured_c_projection"]["projection"].tobytes()!=base.load_f32(f["c_projection"],4608,"C projection").tobytes():raise RunnerError("captured projection mismatch")
 for k in ("projection","prefix","downstream"):
  if v["computed_c_input"][k].tobytes()!=v["captured_c_projection"][k].tobytes():raise RunnerError("computed C projection replay mismatch")
 if v["captured_ref_projection"]["downstream"].tobytes()!=ref.tobytes() or v["captured_c_projection"]["downstream"].tobytes()!=prior.tobytes():raise RunnerError("KV-A downstream replay mismatch")
 j={n:base.judged(v[n]["downstream"],ref) for n in ARM_NAMES[:4]}
 for m,e in FROZEN.items():
  if abs(j["captured_c_projection"][m]-e)>1e-12:raise RunnerError("KV-A frozen metric mismatch")
 c={n:base.judged(v[n]["downstream"],ref) for n in ARM_NAMES[4:]}
 if any(x["pass"] for x in c.values()):raise RunnerError("KV-A planted control failed")
 muts={}
 for n,(_,h,z) in PINNED.items():b=bytearray(f[n].read_bytes());b[len(b)//2]^=1;muts[n]=not base.identity_matches(bytes(b),z,h)
 if not all(muts.values()):raise RunnerError("KV-A mutation failed")
 sw=copy.deepcopy(r["outputs"]);it=list(sw.items());it[2],it[3]=it[3],it[2]
 try:arm_manifest(dict(it));label=False
 except RunnerError:label=True
 if not label:raise RunnerError("KV-A label swap failed")
 return {"status":classify(j),"arms_vs_reference":j,"projection_metrics":{n:projection_judged(v[n]["projection"],v["captured_ref_projection"]["projection"]) for n in ARM_NAMES[:4]},"controls_vs_reference":c,"controls":{"computed_c_all_stages_byte_exact":True,"captured_anchors_byte_exact":True,"schedule_twins_byte_exact":True,"mutated_inputs_refused":muts,"label_swap_rejected":label}}
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",type=Path,default=DEFAULT_MODEL);p.add_argument("--output-dir",type=Path);p.add_argument("--apparatus-only",action="store_true");a=p.parse_args();model=a.model.resolve();out=(a.output_dir or (DEFAULT_APPARATUS if a.apparatus_only else DEFAULT_OUTPUT)).resolve()
 if out.exists():raise SystemExit(f"output already exists: {out}")
 out.mkdir(parents=True);start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();status="VOID_LAYER2_KV_A_PROJECTION_CROSS_INPUT";errors=[];commands={};report={};adj={"status":"NOT_RUN"};s={};compiler=shutil.which("clang");binary=None;inv=0;artifact={"path":str(model),"expected_bytes":base.EXPECTED_MODEL_BYTES,"expected_sha256":base.EXPECTED_MODEL_SHA,"bytes":None,"sha256":None,"opened":False}
 try:
  s=sources()
  if not compiler:raise RunnerError("clang unavailable")
  commands["clang_version"]=base.run_command([compiler,"--version"],out,"clang_version",30);base.require_ok(commands["clang_version"],"clang");binary=out/"engine_layer2_kv_a_projection_cross_input.exe";commands["compile"]=base.run_command([compiler,*base.COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],out,"compile",600);base.require_ok(commands["compile"],"compile")
  for label,flag in (("kva_selftest","--strat01-layer2-kv-a-projection-cross-input-selftest"),("rms_selftest","--strat01-layer2-kv-rmsnorm-cross-input-selftest"),("partition_selftest","--strat01-layer2-kv-partition-cross-input-selftest"),("legacy_selftest","--kselftest")):commands[label]=base.run_command([str(binary),flag],out,label,300);base.require_ok(commands[label],label)
  commands["python_tests"]=base.run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_layer2_kv_a_projection_cross_input"],out,"python_tests",300);base.require_ok(commands["python_tests"],"tests")
  if a.apparatus_only:status="APPARATUS_READY_NO_DONOR_EXECUTION"
  else:
   clean(s)
   if not model.is_file() or model.stat().st_size!=base.EXPECTED_MODEL_BYTES or base.sha256_file(model)!=base.EXPECTED_MODEL_SHA:raise RunnerError("artifact mismatch")
   artifact.update({"bytes":model.stat().st_size,"sha256":base.EXPECTED_MODEL_SHA,"opened":True});f=frozen_evidence();root=out/"diagnostic";root.mkdir();cmd=[str(binary),"--strat01-layer2-kv-a-projection-cross-input",str(model),"--ref-q",str(f["ref_q"]),"--ref-k",str(f["ref_k"]),"--ref-attn",str(f["ref_attn"]),"--ref-projection",str(f["ref_projection"]),"--c-attn",str(f["c_attn"]),"--c-projection",str(f["c_projection"]),"--out-dir",str(root)];inv=1;commands["diagnostic"]=base.run_command(cmd,out,"diagnostic",21600);base.require_ok(commands["diagnostic"],"diagnostic");s=sources();v,report=validate_report(root,model,s);adj=adjudicate(v,report,f);status=adj["status"]
 except (RunnerError,base.RunnerError) as e:errors.append(str(e))
 except Exception as e:errors.append(f"unexpected {type(e).__name__}: {e}")
 record={"schema":"strat01_layer2_kv_a_projection_cross_input_v1","status":status,"errors":errors,"diagnostic_invocations":inv,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":adj,"c_report":report,"provenance":{"started_utc":start,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-tick,"git_head_observed" if a.apparatus_only else "git_head":base.git_value(["git","rev-parse","HEAD"]),"source_hashes":s,"artifact":artifact,"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd()},"binary":{"path":str(binary) if binary else None,"sha256":base.sha256_file(binary) if binary and binary.is_file() else None},"commands":commands},"non_claims":["repair","later operators","quality/RAM/rate"]};base.write_json(out/"adjudication.json",record);print(json.dumps({"status":status,"output":str(out),"errors":errors},indent=2));return 0 if status=="APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2
if __name__=="__main__":raise SystemExit(main())
