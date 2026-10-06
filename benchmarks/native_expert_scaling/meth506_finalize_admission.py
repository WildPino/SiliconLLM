"""Metadata-only admission of fixed 506 observations; never runs science."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import meth506_operations as O

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(4<<20):h.update(data)
    return h.hexdigest()
def item(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':digest(p)}
dest=O.DOC/'ADMISSION_506_20261006.json';assert not dest.exists()
binding=json.loads(O.BIND.read_bytes());prep=O.PREPRAW;audit=O.DOC/'RETENTION_506_20261006.json'
raw=json.loads(O.RAW.read_bytes());retained=json.loads(audit.read_bytes());prepared=json.loads(prep.read_bytes())
assert all(raw['gates'].values()) and all(retained['gates'].values()) and all(prepared['gates'].values())
assert retained['raw_sha256']==digest(O.RAW) and raw['preparation_sha256']==retained['preparation_sha256']==digest(prep)
assert raw['binding_sha256']==retained['binding_sha256']==prepared['binding_sha256']==digest(O.BIND)
windows=[]
for stage in ['prepare','main','audit']:
    p=O.ROOT/f'results/native_expert_scaling/meth506_r3_{stage}_windows_terminal.json';v=json.loads(p.read_bytes());assert v['query_available'] and not v['matching_scientific_events'];windows.append(item(p))
assert retained['quality_gates']==raw['summary']['quality_gates'] and retained['economic_gates']==raw['summary']['economic_gates']
assert all(c['returncode']==0 for c in raw['commands']) and not prepared['commands']
for rel,sha in binding['preserved'].items():assert digest(O.ROOT/rel)==sha
assert digest(O.ROOT/'benchmarks/phase60/engine.c')==binding['engine_sha256']
outputs=sum(p.stat().st_size for folder in [O.ROOT/'results/native_expert_scaling/meth506_artifact',O.ROOT/'results/native_expert_scaling/meth506_r2_artifact',O.PREP,O.OUT,O.AUDIT] for p in folder.iterdir() if p.is_file())+sum(p.stat().st_size for p in O.DOC.glob('meth506*.json'))
assert outputs<=24<<30
for p in [prepared,raw,retained]:
    resource=p['resource'];assert resource['wall_seconds']<=3600 and resource['parent_peak_bytes']+resource['native_peak_bytes']<=24<<30
r={'experiment':'METH506 retained whole feasibility admission','binding':item(O.BIND),'preparation':item(prep),'raw':item(O.RAW),'retention':item(audit),'windows':windows,'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=O.ROOT,text=True).strip(),'argv':sys.argv,
   'gates':{'all_frozen_source_only_preparation_and_main_apparatus':True,'independent_ALL_bank_payload_whole_wires_fresh_donor_quality_and_rates':True,'all_processes_successful_actual_modules_and_windows_terminal':True,'foreign3_engine_and_bounded_retained_outputs':True},
   'quality_gates':retained['quality_gates'],'economic_gates':retained['economic_gates'],'whole_recipe_eligible':retained['whole_recipe_eligible'],'total_new_output_bytes_before_this_receipt':outputs,'main_completed':True,
   'scope':'Exploratory source128 complete exact columns, unchanged505 kernel with original full WI. Neither generic runtime nor this admission completes compact-core/useful-n/LUT/physical-DRAM/other-family/scale goal.'}
O.write(dest,r);print(json.dumps({'admission':item(dest),'gates':r['gates'],'quality_gates':r['quality_gates'],'economic_gates':r['economic_gates'],'whole_recipe_eligible':r['whole_recipe_eligible'],'outputs_bytes':outputs}),flush=True)
