"""518 fresh used retained norm/index/role binding; no full payload or projection."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth518_operations import ROOT,DOC,BIND,Meter,write


def main():
    ctx=None
    try:
        ctx=Meter('binding',60,128<<20);catalog={}
        def item(path,expected=None):
            p=Path(path).resolve();sha=ctx.digest(p)
            if expected:assert sha==expected,str(p)
            v={'path':str(p),'bytes':p.stat().st_size,'sha256':sha};catalog[str(p)]=v;return v
        aref=item(DOC/'ADMISSION_517_20261007.json','a985cd64f968032b8e5f4b483cfd172595ff47561a8f611f7c7e4d0899445d8a')
        prior=json.loads(Path(aref['path']).read_bytes())
        assert all(prior['gates'].values()) and prior['main_completed'] and prior['independent_audit_completed']
        for k in ['binding','raw','audit','retention','windows']:item(prior[k]['path'],prior[k]['sha256'])
        old=json.loads(Path(prior['binding']['path']).read_bytes())
        raw=json.loads(Path(prior['raw']['path']).read_bytes())
        assert not any(v['certified_cells'] or v['candidates'] for v in raw['parents'])
        data={k:item(old['data'][k]['path'],old['data'][k]['sha256']) for k in ['uid','occurrences']}
        for k in [f'row_norm2_e{e:03}.npy' for e in range(1,128)]:data[k]=item(old['data'][k]['path'],old['data'][k]['sha256'])
        for v in raw['output_inventory']:
            name=Path(v['path']).name
            if name in ['prefix_index.bin','work.npy']:data[name]=item(v['path'],v['sha256'])
        assert 'prefix_index.bin' in data and 'work.npy' in data
        runtime=[item(v['path'],v['sha256']) for v in old['runtime_files']]
        science=[]
        paths=sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth518*'))+[DOC/'METH_518_CERTIFICATE_ENERGY_OBSTRUCTION_PROTOCOL_20261007.md']
        for p in paths:
            rel=p.relative_to(ROOT).as_posix()
            committed=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
            assert p.read_bytes().replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n')
            science.append(item(p))
        for rel,sha in old['preserved'].items():item(ROOT/rel,sha)
        assert sys.version==old['python'] and str(Path(sys.executable).resolve())==old['executable']
        ctx.r['gates'].update(prior517_complete_admission_and_zero_realized_certificates=True,
                             actual_used_norm_roles_index_work_runtime_science_fresh_SHA=True)
        value={**ctx.r,'experiment':'METH518 exact energy capacity obstruction',
             'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'python':sys.version,'executable':str(Path(sys.executable).resolve()),'packages':old['packages'],
             'catalog':list(catalog.values()),'data':data,'scientific':science,'runtime_files':runtime,
             'preserved':old['preserved'],'prior517_admission':aref,
             'limits':{'binding':[60,128<<20],'main':[60,512<<20],'audit':[60,512<<20],'combined_output_bytes':16<<20},
             'resource_before_serialization':ctx.resources(),'ended_compute_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'scope':'Fresh used retained geometry bytes only; no full payload, capture, source projection or solver replay.'}
        write(BIND,value);sha=ctx.digest(BIND)
        write(ROOT/'results/native_expert_scaling/meth518_binding_resource.json',
              {'binding_sha256':sha,'process_instance':ctx.r['process_instance'],**ctx.resources(),
               'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        ctx.guard();ctx.timer.cancel()
        print(json.dumps({'binding_sha256':sha,'files':len(catalog),'resource':ctx.resources()}),flush=True)
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
