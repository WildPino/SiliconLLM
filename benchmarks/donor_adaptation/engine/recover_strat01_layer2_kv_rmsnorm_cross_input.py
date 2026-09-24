#!/usr/bin/env python3
"""Recover the completed layer-2 KV RMSNorm run without producer execution."""
from __future__ import annotations
import json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmarks.donor_adaptation.engine import run_strat01_layer2_kv_rmsnorm_cross_input as runner
from benchmarks.donor_adaptation.engine import run_strat01_rung2c_cross_input as base

HERE=Path(__file__).resolve().parent
RAW=runner.DEFAULT_OUTPUT
DEFAULT_OUTPUT=HERE/"results/strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_offline_recovery1_20260924"
PROTOCOL=ROOT/"docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_OFFLINE_RECOVERY_PROTOCOL_20260924.md"
TESTS=HERE/"test_recover_strat01_layer2_kv_rmsnorm_cross_input.py"
RAW_SHA="2c628c3a9503cf882c2e6ce42a6c8211a5767b6269dc0694b148623da383aaaf"
RAW_HEAD="a1ee4f61f8b10f763108f4e37830e642f7b859f6"
RAW_BINARY_SHA="8aa61ddb65aae9f4a1a8ee7c8009cee8609d56b956c8faf009ec7f5085ca8235"
RAW_ERROR="unexpected ValueError: cannot reshape array of size 4096 into shape (8,6144)"
STATUS_MAP={"LAYER2_KV_RMSNORM_FAILS_EXACT_REFERENCE_INPUT":"LAYER2_KV_RMSNORM_FAILS_EXACT_REFERENCE_INPUT_RECOVERED","LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT":"LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT_RECOVERED"}
class RecoveryError(RuntimeError):pass

def source_inventory():
 p={"recovery":Path(__file__).resolve(),"tests":TESTS,"protocol":PROTOCOL,"scientific_runner":Path(runner.__file__).resolve(),"engine":runner.ENGINE,"rmsnorm_header":runner.CROSS_HEADER}
 m=[n for n,x in p.items() if not x.is_file()]
 if m:raise RecoveryError("missing recovery source(s): "+", ".join(m))
 return {n:{"path":str(x),"sha256":base.sha256_file(x)} for n,x in p.items()}

def clean_sources_at_head(s):
 rel=[Path(x["path"]).resolve().relative_to(ROOT.resolve()) for x in s.values()]
 if subprocess.run(["git","diff","--quiet","HEAD","--",*map(str,rel)],cwd=ROOT,check=False).returncode:raise RecoveryError("offline recovery sources differ from HEAD")
 for p in rel:
  if subprocess.run(["git","ls-files","--error-unmatch",str(p)],cwd=ROOT,capture_output=True,check=False).returncode:raise RecoveryError(f"untracked recovery source: {p}")

def validate_raw()->dict[str,Any]:
 p=RAW/"adjudication.json"
 if not p.is_file() or base.sha256_file(p)!=RAW_SHA:raise RecoveryError("raw adjudication identity mismatch")
 try:r=json.loads(p.read_text(encoding="utf-8"))
 except (OSError,json.JSONDecodeError) as e:raise RecoveryError(f"raw adjudication malformed: {e}") from e
 if r.get("schema")!="strat01_layer2_kv_rmsnorm_cross_input_adjudication_v1" or r.get("status")!="VOID_LAYER2_KV_RMSNORM_CROSS_INPUT" or r.get("errors")!=[RAW_ERROR]:raise RecoveryError("raw VOID identity/status/error mismatch")
 if r.get("diagnostic_invocations")!=1 or r.get("donor_graph_executions")!=0 or r.get("reference_graph_executions")!=0:raise RecoveryError("raw execution accounting mismatch")
 pvn=r.get("provenance",{})
 if pvn.get("git_head")!=RAW_HEAD or pvn.get("binary",{}).get("sha256")!=RAW_BINARY_SHA:raise RecoveryError("raw commit/binary mismatch")
 if r.get("c_report",{}).get("state")!="OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION":raise RecoveryError("raw C report incomplete")
 return r

def recover(raw,sources):
 frozen=runner.validate_frozen_evidence();prefixes,outputs,report=runner.validate_report(RAW/"diagnostic",runner.DEFAULT_MODEL.resolve(),sources);adj=runner.adjudicate(prefixes,outputs,report,frozen)
 if adj["status"] not in STATUS_MAP:raise RecoveryError("unexpected recovered scientific status")
 adj["raw_scientific_status"]=adj["status"];adj["status"]=STATUS_MAP[adj["status"]];return adj,report

def main():
 out=DEFAULT_OUTPUT.resolve()
 if out.exists():raise SystemExit(f"output already exists: {out}")
 out.mkdir(parents=True);started=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();status="VOID_LAYER2_KV_RMSNORM_OFFLINE_RECOVERY";errors=[];adj={"status":"NOT_RUN"};report={};sources={}
 try:sources=source_inventory();clean_sources_at_head(sources);raw=validate_raw();adj,report=recover(raw,sources);status=adj["status"]
 except (RecoveryError,runner.RunnerError,base.RunnerError) as e:errors.append(str(e))
 except Exception as e:errors.append(f"unexpected {type(e).__name__}: {e}")
 record={"schema":"strat01_layer2_kv_rmsnorm_offline_recovery_v1","status":status,"errors":errors,"recovered_from":{"path":str((RAW/"adjudication.json").resolve()),"sha256":RAW_SHA,"raw_status":"VOID_LAYER2_KV_RMSNORM_CROSS_INPUT","raw_error":RAW_ERROR},"new_diagnostic_invocations":0,"model_opened":False,"donor_graph_executions":0,"reference_graph_executions":0,"adjudication":adj,"c_report":report,"provenance":{"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.perf_counter()-tick,"git_head":base.git_value(["git","rev-parse","HEAD"]),"source_hashes":sources,"environment":{"platform":platform.platform(),"python":sys.version,"cwd":os.getcwd()}},"non_claims":["new producer evidence","repaired layer 2","quality/RAM/rate"]}
 base.write_json(out/"adjudication.json",record);print(json.dumps({"status":status,"output":str(out),"errors":errors},indent=2));return 0 if status in STATUS_MAP.values() else 2
if __name__=="__main__":raise SystemExit(main())
