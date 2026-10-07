"""Bind reusable523 numerical prefix and original source before new atom masks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from meth528_operations import Meter,ROOT,DOC,BIND,write


def main():
    ctx=None
    try:
        ctx=Meter('binding',180,512<<20); catalog={}
        def item(path,expected=None):
            p=Path(path).resolve(); sha=ctx.digest(p)
            if expected: assert sha==expected,str(p)
            v=dict(path=str(p),bytes=p.stat().st_size,sha256=sha); catalog[str(p)]=v; return v
        ar=item(DOC/'ADMISSION_523_20261007.json','246b21ea1725af54199f1ccf58b45bc7b20ec19c86be4157a084f9735027b939')
        admission=json.loads(Path(ar['path']).read_bytes())
        assert not admission['main_completed'] and admission['independent_audit_completed']
        assert not admission['gates']['original_main_900s_envelope']
        assert all(v for k,v in admission['gates'].items() if k!='original_main_900s_envelope')
        refs={k:item(admission[k]['path'],admission[k]['sha256']) for k in ('binding','raw','audit','retention','windows')}
        old=json.loads(Path(refs['binding']['path']).read_bytes()); raw=json.loads(Path(refs['raw']['path']).read_bytes())
        assert len(raw['gates'])==7 and all(raw['gates'].values()) and raw['decision']=='SINGLE_REFERENCE_SOURCE_FOLD_HINGES_RECIPE_CLOSED'
        legacy_inventory=raw['output_inventory']
        for v in legacy_inventory: item(v['path'],v['sha256'])
        for name in ('meth523_main_result.failure.json','meth523_main_metadata_repair1_result.json'):
            item(DOC/name)
        data={k:item(v['path'],v['sha256']) for k,v in old['data'].items()}
        byname={Path(v['path']).stem:v for v in legacy_inventory}
        for key in ('signed_dots','predictions','hinges','signs','anchors','fold64'):
            v=byname[key]; data['old_'+key]=item(v['path'],v['sha256'])
        assert len(old['source_extents'])==508 and sum(v['bytes'] for v in old['source_extents'])==601211904
        for v in old['source_extents']: assert ctx.digest(v['path'],v['offset'],v['bytes'])==v['sha256']
        runtime=[item(v['path'],v['sha256']) for v in old['runtime_files']]
        for v in old['scientific']: item(v['path'],v['sha256'])
        wrapper=Path(shutil.which('clang.exe')).resolve(); toolroot=wrapper.parent.parent
        compiler=wrapper.parent/'clang-21.exe'; assert compiler.is_file()
        files=set(p for p in compiler.parent.iterdir() if p.is_file())
        for folder in (toolroot/'x86_64-w64-mingw32',toolroot/'include',toolroot/'lib/clang/21'):
            assert folder.is_dir(); files.update(p for p in folder.rglob('*') if p.is_file())
        toolchain=[item(p) for p in sorted(files)]
        for name in ('msvcrt.dll','kernel32.dll','KernelBase.dll','ntdll.dll','psapi.dll'):
            item(Path(os.environ['SystemRoot'])/'System32'/name)
        scientific=[]
        for p in sorted((ROOT/'benchmarks/native_expert_scaling').glob('meth528*'))+[DOC/'METH_528_VARIABLE_ATOM_PROTOCOL_20261007.md']:
            saved=subprocess.check_output(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
            assert saved.replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n'); scientific.append(item(p))
        for rel,sha in old['preserved'].items(): item(ROOT/rel,sha)
        assert sys.version==old['python'] and str(Path(sys.executable).resolve())==old['executable']
        ctx.r['gates'].update(original523_failed_resource_AND_admitted_complete7_gate_numerical_prefix_explicit=True,
            ALL_original_outputs_inputs_source_extents_runtime_and_compiler_SHA_bound=True,
            committed_new_atom_code_protocol_without_new_mask_width_or_candidate_values=True)
        write(BIND,{**ctx.r,'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'python':sys.version,'executable':old['executable'],'packages':old['packages'],'catalog':list(catalog.values()),
            'data':data,'legacy':refs,'legacy_admission':ar,'legacy_output_inventory':legacy_inventory,
            'upstream_original_main_completed':False,'payload':old['payload'],'payload_stat':old['payload_stat'],
            'parents':old['parents'],'source_extents':old['source_extents'],'runtime_files':runtime,
            'compiler':dict(path=str(compiler),sysroot=str(toolroot/'x86_64-w64-mingw32'),files=toolchain,target='x86_64-w64-windows-gnu'),
            'scientific':scientific,'preserved':old['preserved'],
            'limits':{'binding':[180,512<<20],'main':[240,512<<20],'audit':[300,512<<20],'combined_output_bytes':2<<30},
            'resource_before_serialization':ctx.resources()})
        sha=ctx.digest(BIND); write(ROOT/'results/native_expert_scaling/meth528_binding_resource.json',
            dict(binding_sha256=sha,process_instance=ctx.r['process_instance'],**ctx.resources()))
        ctx.guard();ctx.timer.cancel(); print(json.dumps(dict(binding_sha256=sha,resource=ctx.resources())))
    except BaseException:
        if ctx:ctx.fail()
        raise


if __name__=='__main__':main()
