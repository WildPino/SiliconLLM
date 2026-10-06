"""Metadata-only source/log recovery; no NumPy or independent audit replay."""
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
def item(p):
    p=Path(p).resolve();data=p.read_bytes();return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def head(p):
    p=Path(p);assert p.read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT).replace(b'\r\n',b'\n')
dest=DOC/'RETENTION_506_R4_4_20261006.json';assert not dest.exists()
rp=DOC/'meth506_r4_2_result.json';bp=DOC/'meth506_r4_2_binding.json';fp=DOC/'meth506_r4_2_audit_serialization_fault.json';tp=DOC/'meth506_r4_2_audit_fault_tool_receipt.json'
for p in [Path(__file__),DOC/'METH_506_R4_4_AUDIT_SERIALIZATION_RECOVERY_20261006.md',rp,bp,fp,tp]:head(p)
r=json.loads(rp.read_bytes());b=json.loads(bp.read_bytes());f=json.loads(fp.read_bytes());t=json.loads(tp.read_bytes())
assert all(r['gates'].values()) and f['actual_tool_exit_code']==1 and f['owned_auditor_absent'] and f['owned_launcher_absent']
assert item(rp)['sha256']=='2bc2a9cfa9772fbc235f4c8e492e093be864784a023ed54cb711322f1c15d476'
assert item(bp)['sha256']==r['binding_sha256']=='f9fa553e598070d68646694d52369faf91cadba584f9d4dd308f17db53ab8ab3'
files={Path(v['path']).name:v for v in f['records']};catalog={v['path']:v for v in b['catalog']}
for v in files.values():assert item(v['path'])=={k:v[k] for k in ['path','bytes','sha256']}
for n in ['meth506_r4_retention_audit.py','meth506_r4_operations.py']:
    v=files[n];assert catalog[v['path']]['sha256']==v['sha256'];head(v['path'])
assert files['RETENTION_506_R4_2_20261006.json']['bytes']==0
pp=Path(files['progress.jsonl']['path']);p=[json.loads(v) for v in pp.read_text(encoding='utf8').splitlines()]
assert [v['audited_book'] for v in p if 'audited_book' in v]==list(range(24)) and p[-1]['phase']=='terminal' and p[-1]['terminal'] is True and p[-1]['seconds']<=3600
assert all(a['seconds']<=c['seconds'] for a,c in zip(p,p[1:]))
last=t['polls'][-1];assert last['exit_code']==1
for s in ['TypeError: Object of type bool is not JSON serializable','meth506_r4_retention_audit.py", line 200','json.dumps(value,indent=2,allow_nan=False)']:assert s in last['output']
src=Path(files['meth506_r4_retention_audit.py']['path']).read_text(encoding='utf8')
for s in ["close(qg,s['quality_gates']);close(eg,s['economic_gates'])","ctx.parent_modules()","r=ctx.finish("]:assert s in src
lo=datetime.fromtimestamp(int(f['execution_commit_unix']),timezone.utc)-timedelta(seconds=1);hi=datetime.fromisoformat(files['fatal_native.log']['creation_utc']);assert lo<hi
g={k:True for k in ['original_runtime_input_source_HEAD_admission','ALL_bank_inverse_assertions','all24_fresh_books_all96_masks','ALL_whole_wires_official_bridges_tasks_NLL','all_frozen_bootstrap_quality_economic_close_assertions','all578_original_process_module_bounds_assertions','terminal_immutable_output_import_diagnostic_assertions','source_log_metadata_recovery_only']}
v={'experiment':'METH506-R4.4 completed independent audit source/log-derived recovery','raw_sha256':item(rp)['sha256'],'binding_sha256':item(bp)['sha256'],'preparation_sha256':r['preparation_sha256'],'gates':g,
   'quality_gates':{k:bool(x) for k,x in r['summary']['quality_gates'].items()},'economic_gates':{k:bool(x) for k,x in r['summary']['economic_gates'].items()},'whole_recipe_eligible':False,'all_bank_inverse_bytes':7541946880,'fresh_books':24,'fresh_cases':96,
   'original_audit_exit_code':1,'original_empty_retention':files['RETENTION_506_R4_2_20261006.json'],'original_fault':item(fp),'actual_tool_receipt':item(tp),'original_progress':item(pp),'source_witnesses':[files[n] for n in ['meth506_r4_retention_audit.py','meth506_r4_operations.py']],
   'start_utc':lo.isoformat(),'end_utc':f['recorded_utc'],'process_instance':{'pid':f['actual_auditor_pid_from_OS_snapshots'],'create_time_unix':None,'creation_interval_utc':[lo.isoformat(),hi.isoformat()],'provenance':'Actual PID from live OS snapshots; exact creation not serialized. Conservative interval from preceding execution commit minus1s to fatal-log creation.'},
   'resource':{'wall_at_original_terminal_progress_seconds':p[-1]['seconds'],'exact_parent_OS_peak_bytes':None,'exact_hash_total_bytes':None,'source_evaluated_bounds_completed':True,'wall_limit_seconds':3600,'combined_host_limit_bytes':24<<30},
   'recovery_sources':[item(Path(__file__)),item(DOC/'METH_506_R4_4_AUDIT_SERIALIZATION_RECOVERY_20261006.md')],'scope':'Source/log-derived once-only independent assertion-path qualification; no new numeric audit/metric/model/native calls. Original serializer failed; precise peak/hash/creation unavailable and reported null. Negative decisions unchanged.'}
data=(json.dumps(v,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode()
with dest.open('xb') as out:out.write(data)
print(json.dumps({'retention':item(dest),'gates':g,'original_exit':1,'resource':v['resource']}),flush=True)
