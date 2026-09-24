#!/usr/bin/env python3
"""Execute the frozen layer-2 attention RMSNorm cross-input diagnostic."""
from __future__ import annotations
import argparse, copy, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_a_projection_cross_input as kva
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_rmsnorm_cross_input as rms
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE=Path(__file__).resolve().parent; ENGINE=kva.ENGINE; RUNG2A=kva.RUNG2A; RUNG2C=kva.RUNG2C
KVA_HEADER=kva.CROSS_HEADER; RMS_HEADER=kva.RMS_HEADER
CROSS_HEADER=ROOT/"benchmarks/phase60/strat01_gguf_layer2_attn_rmsnorm_cross_input.h"
PROTOCOL=ROOT/"docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTN_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md"
TESTS=HERE/"test_strat01_layer2_attn_rmsnorm_cross_input.py"; RAW=kva.RAW; DEFAULT_MODEL=kva.DEFAULT_MODEL
DEFAULT_OUTPUT=HERE/"results/strat01_gigachat_engine_layer2_attn_rmsnorm_cross_input_20260924"
DEFAULT_APPARATUS=HERE/"results/strat01_gigachat_engine_layer2_attn_rmsnorm_cross_input_apparatus_repair1_20260924"
PREDECESSOR=kva.DEFAULT_OUTPUT/"adjudication.json"; PREDECESSOR_SHA="d16af478a412e22457f11e4d0946e8c7d9fec1c842dfee9657fb213ec8cc06d1"
PREDECESSOR_OUTPUT=kva.DEFAULT_OUTPUT/"diagnostic/captured_c_projection.f32le"; PREDECESSOR_OUTPUT_SHA="81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254"
FROZEN={"nrmse":0.002642086799405445,"normalized_max":0.004687597394884091}
ARM_NAMES=("captured_ref_norm","captured_c_norm","computed_ref_input","computed_c_input","control_ref_input_token7_negated","control_norm_weight_negated")
ARM_META={"captured_ref_norm":("captured","reference",False,False),"captured_c_norm":("captured","c",False,False),"computed_ref_input":("computed","reference",False,False),"computed_c_input":("computed","c",False,False),"control_ref_input_token7_negated":("computed","reference",True,False),"control_norm_weight_negated":("computed","reference",False,True)}
VALID_STATUSES={"LAYER2_ATTN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT","LAYER1_TERMINAL_RESIDUAL_SUFFICIENT"}
PINNED={
 "ref_q":(RAW/"pinned_reference/prefill8/Qcur-2.full.f32le","46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea",589824),
 "ref_k":(RAW/"pinned_reference/prefill8/Kcur-2.full.f32le","b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b",18432),
 "ref_input":(RAW/"pinned_reference/prefill8/l_out-1.full.f32le","40d5a0f07fbb77c1ef73ca24f81cb35ea0df32457faa8d04d6c5cd33cd9f1d5f",49152),
 "ref_norm":(RAW/"pinned_reference/prefill8/attn_norm-2.full.f32le","7bc7ba62b9e5b779cf079ab212f9b328e247c979423992f42aff786560115f02",49152),
 "ref_target":(RAW/"pinned_reference/prefill8/kqv_out-2.full.f32le","d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e",196608),
 "c_input":(RAW/"c_engine/prefill8_l_out-1.f32","9af8cec3f42781e9f4cac6af1ea13f6a63b751a5323201a8a18f91b3f7b7bbcb",49152),
 "c_norm":(RAW/"c_engine/prefill8_attn_norm-2.f32","b55cdc958cb9e48ff014c740dc4bfe6b1ec81ef839ee95618018e426673e112d",49152),
}
TWINS={"ref_q":RAW/"pinned_reference/cached7p1/Qcur-2.full.f32le","ref_k":RAW/"pinned_reference/cached7p1/Kcur-2.full.f32le","ref_input":RAW/"pinned_reference/cached7p1/l_out-1.full.f32le","ref_norm":RAW/"pinned_reference/cached7p1/attn_norm-2.full.f32le","ref_target":RAW/"pinned_reference/cached7p1/kqv_out-2.full.f32le","c_input":RAW/"c_engine/cached7p1_l_out-1.f32","c_norm":RAW/"c_engine/cached7p1_attn_norm-2.f32"}

class RunnerError(RuntimeError): pass

def sources():
 p={"runner":Path(__file__).resolve(),"tests":TESTS,"protocol":PROTOCOL,"engine":ENGINE,"rung2a":RUNG2A,"rung2c":RUNG2C,"rms_header":RMS_HEADER,"kva_header":KVA_HEADER,"attn_norm_header":CROSS_HEADER,"kva_runner":Path(kva.__file__).resolve(),"rms_runner":Path(rms.__file__).resolve(),"base_runner":Path(base.__file__).resolve()}
 if any(not x.is_file() for x in p.values()): raise RunnerError("missing attention RMSNorm source")
 return {n:{"path":str(x),"sha256":base.sha256_file(x)} for n,x in p.items()}

def clean(s):
 rel=[Path(x["path"]).resolve().relative_to(ROOT.resolve()) for x in s.values()]
 if subprocess.run(["git","diff","--quiet","HEAD","--",*map(str,rel)],cwd=ROOT,check=False).returncode: raise RunnerError("attention RMSNorm sources differ from HEAD")
 for p in rel:
  if subprocess.run(["git","ls-files","--error-unmatch",str(p)],cwd=ROOT,capture_output=True,check=False).returncode: raise RunnerError(f"untracked attention RMSNorm source: {p}")

def frozen_evidence():
 r={}
 for n,(p,h,z) in PINNED.items():
  if not p.is_file() or p.stat().st_size!=z or base.sha256_file(p)!=h: raise RunnerError(f"attention RMSNorm input mismatch: {n}")
  t=TWINS[n]
  if not t.is_file() or t.stat().st_size!=z or base.sha256_file(t)!=h: raise RunnerError(f"attention RMSNorm twin mismatch: {n}")
  r[n]=p.resolve(strict=True)
 if not PREDECESSOR.is_file() or base.sha256_file(PREDECESSOR)!=PREDECESSOR_SHA: raise RunnerError("KV-A predecessor mismatch")
 if not PREDECESSOR_OUTPUT.is_file() or base.sha256_file(PREDECESSOR_OUTPUT)!=PREDECESSOR_OUTPUT_SHA: raise RunnerError("KV-A predecessor output mismatch")
 r["prior"]=PREDECESSOR_OUTPUT.resolve(strict=True); return r

def arm_manifest(o):
 if not isinstance(o,dict) or tuple(o)!=ARM_NAMES: raise RunnerError("attention RMSNorm arm labels/order mismatch")
 for n,x in o.items():
  if set(x)!={"kind","source","input_control","weight_control","norm","projection","prefix","downstream"}: raise RunnerError("attention RMSNorm arm schema mismatch")
  if (x["kind"],x["source"],x["input_control"],x["weight_control"])!=ARM_META[n]: raise RunnerError("attention RMSNorm arm metadata mismatch")
  for k,z in (("norm",49152),("projection",18432),("prefix",16384),("downstream",196608)):
   if set(x[k])!={"path","bytes","sha256"} or x[k]["bytes"]!=z: raise RunnerError("attention RMSNorm payload schema mismatch")

def validate_report(root,model,s):
 r=json.loads((root/"strat01_layer2_attn_rmsnorm_cross_input.json").read_text(encoding="utf-8")); req={"command","state","self_certifies_pass","model","inputs","matrices","outputs","engine_source_sha256","diagnostic_source_sha256","compiler_family","donor_graph_executions","timing_or_rate_claim"}
 if set(r)!=req or r["command"]!="--strat01-layer2-attn-rmsnorm-cross-input" or r["state"]!="OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION" or r["self_certifies_pass"] is not False: raise RunnerError("attention RMSNorm report schema mismatch")
 if r["model"]!={"path":str(model),"bytes":base.EXPECTED_MODEL_BYTES,"sha256":base.EXPECTED_MODEL_SHA}: raise RunnerError("attention RMSNorm model mismatch")
 ei={n:{"path":str(PINNED[n][0].resolve()),"bytes":PINNED[n][2],"sha256":PINNED[n][1]} for n in ("ref_q","ref_k","ref_input","ref_norm","c_input","c_norm")}
 if r["inputs"]!=ei: raise RunnerError("attention RMSNorm inputs mismatch")
 if r["matrices"]!={"attn_norm":{"name":"blk.2.attn_norm.weight","type":"F32","shape":[1536],"offset":578811136,"file_offset":584914048,"span":6144},"projection":{"name":"blk.2.attn_kv_a_mqa.weight","type":"Q4_K","shape":[1536,576]},"kv_norm":{"name":"blk.2.attn_kv_a_norm.weight","type":"F32","shape":[512]},"v_b":{"name":"blk.2.attn_v_b.weight","type":"Q4_K","shape":[512,192,32]}}: raise RunnerError("attention RMSNorm descriptors mismatch")
 if r["engine_source_sha256"]!=s["engine"]["sha256"] or r["diagnostic_source_sha256"]!=s["attn_norm_header"]["sha256"] or r["compiler_family"]!="clang" or r["donor_graph_executions"]!=0 or r["timing_or_rate_claim"] is not None: raise RunnerError("attention RMSNorm provenance mismatch")
 arm_manifest(r["outputs"]); vals={}
 for n,x in r["outputs"].items(): vals[n]={k:base.load_f32(base.contained(root,x[k]["path"],x[k]["bytes"],x[k]["sha256"],n+" "+k),x[k]["bytes"]//4,n+" "+k) for k in ("norm","projection","prefix","downstream")}
 return vals,r

def classify(j):
 if not j["captured_ref_norm"]["pass"] or j["captured_c_norm"]["pass"] or j["computed_c_input"]["pass"]: raise RunnerError("attention RMSNorm anchor contradiction")
 return "LAYER2_ATTN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT" if not j["computed_ref_input"]["pass"] else "LAYER1_TERMINAL_RESIDUAL_SUFFICIENT"

def descriptive(candidate,reference,shape):
 x=base.metrics(candidate,reference); c=np.asarray(candidate).reshape(shape); q=np.asarray(reference).reshape(shape); x["per_token"]=[dict(base.metrics(c[i],q[i]),token=i) for i in range(shape[0])]; return x

def adjudicate(v,r,f):
 ref=base.load_f32(f["ref_target"],49152,"reference target"); prior=base.load_f32(f["prior"],49152,"prior")
 if v["captured_ref_norm"]["norm"].tobytes()!=base.load_f32(f["ref_norm"],12288,"ref norm").tobytes() or v["captured_c_norm"]["norm"].tobytes()!=base.load_f32(f["c_norm"],12288,"C norm").tobytes(): raise RunnerError("captured attention norm mismatch")
 for k in ("norm","projection","prefix","downstream"):
  if v["computed_c_input"][k].tobytes()!=v["captured_c_norm"][k].tobytes(): raise RunnerError("computed C attention RMSNorm replay mismatch")
 if v["captured_ref_norm"]["downstream"].tobytes()!=ref.tobytes() or v["captured_c_norm"]["downstream"].tobytes()!=prior.tobytes(): raise RunnerError("attention RMSNorm downstream replay mismatch")
 j={n:base.judged(v[n]["downstream"],ref) for n in ARM_NAMES[:4]}
 for m,e in FROZEN.items():
  if abs(j["captured_c_norm"][m]-e)>1e-12: raise RunnerError("attention RMSNorm frozen metric mismatch")
 c={n:base.judged(v[n]["downstream"],ref) for n in ARM_NAMES[4:]}
 if any(x["pass"] for x in c.values()): raise RunnerError("attention RMSNorm planted control failed")
 muts={}
 for n,(_,h,z) in PINNED.items(): b=bytearray(f[n].read_bytes()); b[len(b)//2]^=1; muts[n]=not base.identity_matches(bytes(b),z,h)
 if not all(muts.values()): raise RunnerError("attention RMSNorm mutation failed")
 sw=copy.deepcopy(r["outputs"]); it=list(sw.items()); it[2],it[3]=it[3],it[2]
 try: arm_manifest(dict(it)); label=False
 except RunnerError: label=True
 if not label: raise RunnerError("attention RMSNorm label swap failed")
 return {"status":classify(j),"arms_vs_reference":j,"norm_metrics":{n:descriptive(v[n]["norm"],v["captured_ref_norm"]["norm"],(8,1536)) for n in ARM_NAMES[:4]},"controls_vs_reference":c,"controls":{"computed_c_all_stages_byte_exact":True,"captured_anchors_byte_exact":True,"schedule_twins_byte_exact":True,"mutated_inputs_refused":muts,"label_swap_rejected":label}}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,default=DEFAULT_MODEL); p.add_argument("--output-dir",type=Path); p.add_argument("--apparatus-only",action="store_true"); a=p.parse_args(); model=a.model.resolve(); out=(a.output_dir or (DEFAULT_APPARATUS if a.apparatus_only else DEFAULT_OUTPUT)).resolve()
 if out.exists(): raise SystemExit(f"output already exists: {out}")
 out.mkdir(parents=True); start=datetime.now(timezone.utc).isoformat(); tick=time.perf_counter(); status="VOID_LAYER2_ATTN_RMSNORM_CROSS_INPUT"; errors=[]; commands={}; report={}; adj={"status":"NOT_RUN"}; s={}; compiler=shutil.which("clang"); binary=None; inv=0; artifact={"path":str(model),"expected_bytes":base.EXPECTED_MODEL_BYTES,"expected_sha256":base.EXPECTED_MODEL_SHA,"bytes":None,"sha256":None,"opened":False}
 try:
  s=sources()
  if not compiler: raise RunnerError("clang unavailable")
  commands["clang_version"]=base.run_command([compiler,"--version"],out,"clang_version",30); base.require_ok(commands["clang_version"],"clang"); binary=out/"engine_layer2_attn_rmsnorm_cross_input.exe"; commands["compile"]=base.run_command([compiler,*base.COMPILE_FLAGS,str(ENGINE),"-o",str(binary),"-lm"],out,"compile",600); base.require_ok(commands["compile"],"compile")
  for label,flag in (("attn_norm_selftest","--strat01-layer2-attn-rmsnorm-cross-input-selftest"),("kva_selftest","--strat01-layer2-kv-a-projection-cross-input-selftest"),("rms_selftest","--strat01-layer2-kv-rmsnorm-cross-input-selftest"),("partition_selftest","--strat01-layer2-kv-partition-cross-input-selftest"),("legacy_selftest","--kselftest")): commands[label]=base.run_command([str(binary),flag],out,label,300); base.require_ok(commands[label],label)
  commands["python_tests"]=base.run_command([sys.executable,"-B","-m","unittest","-v","benchmarks.donor_adaptation.engine.test_strat01_layer2_attn_rmsnorm_cross_input"],out,"python_tests",300); base.require_ok(commands["python_tests"],"tests")
  if a.apparatus_only: status="APPARATUS_READY_NO_DONOR_EXECUTION"
  else:
   clean(s)
   if not model.is_file() or model.stat().st_size!=base.EXPECTED_MODEL_BYTES or base.sha256_file(model)!=base.EXPECTED_MODEL_SHA: raise RunnerError("artifact mismatch")
   artifact.update({"bytes":model.stat().st_size,"sha256":base.EXPECTED_MODEL_SHA,"opened":True}); f=frozen_evidence(); root=out/"diagnostic"; root.mkdir(); cmd=[str(binary),"--strat01-layer2-attn-rmsnorm-cross-input",str(model),"--ref-q",str(f["ref_q"]),"--ref-k",str(f["ref_k"]),"--ref-input",str(f["ref_input"]),"--ref-norm",str(f["ref_norm"]),"--c-input",str(f["c_input"]),"--c-norm",str(f["c_norm"]),"--out-dir",str(root)]; inv=1; commands["diagnostic"]=base.run_command(cmd,out,"diagnostic",21600); base.require_ok(commands["diagnostic"],"diagnostic"); s=sources(); v,report=validate_report(root,model,s); adj=adjudicate(v,report,f); status=adj["status"]
 except (RunnerError,base.RunnerError) as e: errors.append(str(e))
 except Exception as e: errors.append(f"unexpected {type(e).__name__}: {e}")
 record={"schema":"strat01_layer2_attn_rmsnorm_cross_input_v1","status":status,"errors":errors,"diagnostic_invocations":inv,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":adj,"c_report":report,"provenance":{"started_utc":start,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-tick,"git_head_observed" if a.apparatus_only else "git_head":base.git_value(["git","rev-parse","HEAD"]),"source_hashes":s,"artifact":artifact,"environment":{"platform":platform.platform(),"python":sys.version,"numpy":np.__version__,"cwd":os.getcwd()},"binary":{"path":str(binary) if binary else None,"sha256":base.sha256_file(binary) if binary and binary.is_file() else None},"commands":commands},"non_claims":["repair","earlier layer-1 internals","later operators","quality/RAM/rate"]}; base.write_json(out/"adjudication.json",record); print(json.dumps({"status":status,"output":str(out),"errors":errors},indent=2)); return 0 if status=="APPARATUS_READY_NO_DONOR_EXECUTION" or status in VALID_STATUSES else 2

if __name__=="__main__": raise SystemExit(main())
