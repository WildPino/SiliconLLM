"""Metadata admission of original first fault and distinct repaired retention."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';DEST=DOC/'ADMISSION_485_R1_20261006.json'
assert not DEST.exists() and sys.flags.optimize==0
def item(p):
    p=Path(p);h=hashlib.sha256(p.read_bytes());return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
rawpath=DOC/'meth485_shared_integer_backend_result.failure.json';retpath=DOC/'RETENTION_485_R1_20261006.json';oldpath=DOC/'RETENTION_485_20261006.failure.json'
raw=json.loads(rawpath.read_bytes());ret=json.loads(retpath.read_bytes());old=json.loads(oldpath.read_bytes());assert all(ret['gates'].values()) and ret['raw']['sha256']==item(rawpath)['sha256'] and ret['main_completed'] is False
files=[]
for label,value,path,exitcode,winname in [('main',raw,rawpath,1,'meth485_windows_terminal.json'),('audit',old,oldpath,1,'meth485_audit_windows_terminal.json'),('audit_r1',ret,retpath,0,'meth485_audit_r1_windows_terminal.json')]:
    rp=DOC/f'meth485_{label}_sole_first_registration.json';reg=json.loads(rp.read_bytes());assert reg['attempt']==1 and reg['actual_exit_code']==exitcode and reg['process_instance']==value['process_instance'] and reg['record_sha256']==item(path)['sha256']
    wp=ROOT/'results/native_expert_scaling'/winname;win=json.loads(wp.read_bytes());assert win['query_available'] and not win['matching_scientific_events'] and win['instances'][0]['pid']==value['process_instance']['pid'] and win['instances'][0]['create_time_unix']==value['process_instance']['create_time_unix'];files.extend([item(rp),item(wp)])
r={'experiment':'METH485 original first ABI fault plus independent numbered retention admission','raw':item(rawpath),'original_audit_fault':item(oldpath),'retention_r1':item(retpath),'records':files,'gates':{'ALL_R1_retention_gates':True,'THREE_actual_sole_executions':True,'THREE_available_zero_matching_Windows_faults':True,'original_main_and_audit_faults_unchanged':True},'decision':'Original main failed symbol resolution before explicit CUDA initialization/GEMM/model. Algebra/speed/quality not experimentally qualified. Next candidate requires new frozen namespace and corrected bound Windows export, never a replay.'}
with DEST.open('xb') as f:f.write((json.dumps(r,indent=2)+'\n').replace('\n','\r\n').encode())
print(json.dumps({'admission':str(DEST),'sha256':item(DEST)['sha256'],'gates':r['gates']}))
