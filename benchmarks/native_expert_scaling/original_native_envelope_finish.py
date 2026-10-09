"""Missing-only native completion after metadata fault; adopted fixture/base compile.

All old kernels/weights remain immutable. Exported wider core contains no new
knowledge. Native-to-native preservation does not requalify donor/GPU quality.
"""
import argparse
import json
import math
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925';sys.path.insert(0,str(B))
from original_falcon_whole_recovery import extent,check_inputs,memory_reader
from original_packed_capacity import SITE,sha,write,extract,FIELD,raw
sys.path.insert(0,str(SITE))


def read_fields(path):
    with Path(path).open('rb') as f:
        assert f.read(8)==b'E4BPv001';h=struct.unpack('<16I',f.read(64));size=struct.unpack('<Q',f.read(8))[0]
        assert h==(1,65537,256,96,8,6,512,16,4,128,5,1152,128,8,1,110)
        assert size==Path(path).stat().st_size==507505920
        fields=[];offset=80+110*104
        for _ in range(110):
            row=FIELD.unpack(f.read(104));name=row[0].split(b'\0')[0].decode();dtype,rank=row[1:3];shape=row[3:3+rank]
            assert dtype in (1,2) and row[-2]==offset and row[-1]==math.prod(shape)*(4 if dtype==1 else 1)
            fields.append(dict(name=name,dtype=dtype,shape=shape,offset=offset,bytes=row[-1]));offset+=row[-1]
        assert offset==size and len({r['name'] for r in fields})==110
    return h,fields


def bind(a):
    import numpy as np
    import psutil
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    parent=DOC/'original_native_envelope_binding_20261009.json'
    b=json.loads(parent.read_bytes());check_inputs(b)
    old_dir=ROOT/'results/native_expert_scaling/original_native_envelope_20261009'
    failure=DOC/'original_native_envelope_result_20261009.launcher_failure.json'
    f=json.loads(failure.read_bytes());fault=json.loads((old_dir/'first_fault.json').read_bytes())
    receipt=json.loads((old_dir/'compile_base.receipt.json').read_bytes())
    assert f['exit_code']==1 and fault['stage']=='compile_base' and 'multiple values' in fault['error']
    assert receipt['exit_code']==0 and len(fault['children'])==1 and not fault['speed'] and not fault['profile'] and fault['parity'] is None
    extra=[Path(__file__),parent,failure,DOC/'original_native_envelope_result_20261009.worker.log',
        DOC/'ORIGINAL_NATIVE_ENVELOPE_FINISH_PROTOCOL_20261009.md']+sorted(p for p in old_dir.iterdir() if p.is_file())
    b.update(worker_path=str(Path(__file__).resolve()),completion=dict(directory=str(old_dir.resolve()),
        first_failure=extent(failure),inherited_compile=receipt,prior_held_family_seconds=f['elapsed_seconds']),
        inputs=list({i['path']:i for i in b['inputs']+[extent(p) for p in extra]}.values()))
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(b['inputs']),inherited_compile=1,new_children=11)),flush=True)


def launch(a):
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes())
    assert b['schema']=='ORIGINAL_NATIVE_ENVELOPE_BINDING_V1' and Path(sys.executable).resolve()==Path(b['python']).resolve()
    proc=psutil.Process();proc.cpu_affinity([11]);own={proc.pid,*(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name','cmdline']):
        name=(p.info['name'] or '').lower();argv=p.info['cmdline'] or []
        if p.pid in own:continue
        if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():continue
        assert not name.startswith(('python','gcc','clang','engine','packed_original','native_base','native_wide')),('overlap',p.pid,name)
    log=a.out.with_suffix('.worker.log');terminal=a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out,log,terminal,a.out.with_suffix('.launcher_failure.json')))
    worker=None;peak=0;reader=memory_reader();seen={};record=dict(freeze=a.freeze,binding_sha256=a.binding_sha,launcher_pid=proc.pid)
    def guard():
        nonlocal peak
        peak=max(peak,reader(worker));assert time.monotonic()-start<=b['limits']['seconds'],'family deadline'
        assert peak+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family OS'
        assert log.stat().st_size<=4<<20,'log cap'
    try:
        check_inputs(b);env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        argv=[b['python'],'-I','-S','-B','-X','utf8',b['worker_path'],'--worker','--binding',str(a.binding.resolve()),
            '--binding-sha',a.binding_sha,'--freeze',a.freeze,'--directory',str(a.directory.resolve()),'--out',str(a.out.resolve())]
        record['command']=argv
        with log.open('xb') as stream:
            worker=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=8)
            record.update(worker_pid=worker.pid,worker_creation_time=psutil.Process(worker.pid).create_time());offset=0
            while worker.poll() is None:
                guard()
                try:
                    for child in psutil.Process(worker.pid).children(recursive=True):
                        try:name=child.name().lower();created=child.create_time();exe=Path(child.exe().removeprefix('\\\\?\\')).resolve()
                        except psutil.NoSuchProcess:continue
                        seen[(child.pid,created)]=child
                        assert name in b['allowed_child_names'],('child name',name)
                        if name.startswith('native_'):assert exe==a.directory.resolve()/name
                        elif name=='conhost.exe':assert exe==Path('C:/Windows/System32/conhost.exe').resolve()
                        else:assert exe.parent==Path(b['compiler']).parent
                except psutil.NoSuchProcess:pass
                with log.open('rb') as f:
                    f.seek(offset);chunk=f.read();end=chunk.rfind(b'\n')+1
                    if end:print(chunk[:end].decode('utf8',errors='replace'),end='',flush=True);offset+=end
                time.sleep(.1)
        guard();record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak,launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert worker.returncode==0,log.read_text(errors='replace')[-6000:]
        check_inputs(b);r=json.loads(a.out.read_bytes());assert r['schema']=='ORIGINAL_NATIVE_ENVELOPE_RESULT_V1'
        assert r['source_calls']==r['optimizer_updates']==r['GPU_calls']==0 and r['parity']['passed'] and len(r['children'])==11 and len(r['inherited_children'])==1
        assert peak+r['max_direct_child_OS_peak']+proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'family worker+direct child+launcher'
        outputs=[extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in outputs)<=b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start,result_sha256=sha(a.out),output_files=outputs,
            decision=r['decision'],resource_gates=True,scope='Direct child held by worker;observed nested linker allowed but its separate peak not held.')
        write(terminal,record);print(json.dumps(dict(terminal=str(terminal),worker_pid=worker.pid,exit_code=worker.returncode,seconds=record['elapsed_seconds'],decision=r['decision'])),flush=True)
    except BaseException as e:
        for (pid,created),child in reversed(list(seen.items())):
            try:
                if child.is_running() and child.create_time()==created:child.kill()
            except psutil.NoSuchProcess:pass
        if worker is not None:
            if worker.poll() is None:worker.kill();worker.wait()
            peak=max(peak,reader(worker));record.update(exit_code=worker.returncode,worker_OS_peak_through_exit=peak)
        record.update(fault=repr(e),elapsed_seconds=time.monotonic()-start,launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.launcher_failure.json'),record);raise


def export_wide(source,dest):
    import numpy as np
    header,fields=read_fields(source);specs=[];cursor=80+110*104
    for e in fields:
        shape=list(e['shape']);organ=e['name'].rsplit('.',1)[-1]
        if e['name'].startswith('layers.') and e['name'].split('.')[1]!='5':
            if organ=='in_proj':shape=[2048,256]
            elif organ=='conv_w':shape=[1024,4]
            elif organ in ('conv_b','dt_b','Dskip'):shape=[1024]
            elif organ=='A_log':shape=[1024,96]
            elif organ=='x_proj':shape=[240,1024]
            elif organ=='dt_proj':shape=[1024,48]
            elif organ=='out_proj':shape=[256,1024]
        size=math.prod(shape)*(4 if e['dtype']==1 else 1)
        specs.append(dict(**e,new_shape=shape,new_offset=cursor,new_bytes=size));cursor+=size
    h=list(header);h[6]=1024;h[7]=48
    table=b''.join(FIELD.pack(e['name'].encode(),e['dtype'],len(e['new_shape']),*(e['new_shape']+[0]*(4-len(e['new_shape']))),e['new_offset'],e['new_bytes']) for e in specs)
    unchanged=[];modified=[]
    with Path(source).open('rb') as f,dest.open('xb') as out:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        try:
            out.write(b'E4BPv001'+struct.pack('<16I',*h)+struct.pack('<Q',cursor)+table)
            for e in specs:
                assert out.tell()==e['new_offset'];organ=e['name'].rsplit('.',1)[-1]
                if list(e['shape'])==e['new_shape']:
                    for pos in range(e['offset'],e['offset']+e['bytes'],8<<20):out.write(mm[pos:min(pos+(8<<20),e['offset']+e['bytes'])])
                    unchanged.append(e['name']);continue
                src=np.ndarray(e['shape'],dtype='<f4',buffer=mm,offset=e['offset']);dst=np.zeros(e['new_shape'],dtype='<f4')
                if organ=='in_proj':dst[:512]=src[:512];dst[512:1024]=src[:512];dst[1024:1536]=src[512:];dst[1536:]=src[512:]
                elif organ in ('conv_w','conv_b','dt_b','Dskip','A_log'):dst[:512]=src;dst[512:]=src
                elif organ=='x_proj':dst[:16,:512]=src[:16];dst[48:,:512]=src[16:]
                elif organ=='dt_proj':dst[:512,:16]=src;dst[512:,:16]=src
                elif organ=='out_proj':dst[:,:512]=src
                else:raise AssertionError(organ)
                assert np.isfinite(dst).all();out.write(dst.tobytes());modified.append(e['name']);del src,dst
            assert out.tell()==cursor
        finally:mm.close()
    # Independent elementwise verification of every changed block and zero padding.
    with Path(source).open('rb') as old,dest.open('rb') as new:
        om=mmap.mmap(old.fileno(),0,access=mmap.ACCESS_READ);nm=mmap.mmap(new.fileno(),0,access=mmap.ACCESS_READ)
        try:
            for e in specs:
                if e['name'] in unchanged:
                    for k in range(0,e['bytes'],8<<20):assert om[e['offset']+k:e['offset']+min(k+(8<<20),e['bytes'])]==nm[e['new_offset']+k:e['new_offset']+min(k+(8<<20),e['bytes'])]
                    continue
                src=np.ndarray(e['shape'],dtype='<f4',buffer=om,offset=e['offset']);dst=np.ndarray(e['new_shape'],dtype='<f4',buffer=nm,offset=e['new_offset']);organ=e['name'].rsplit('.',1)[-1]
                if organ=='in_proj':assert np.array_equal(dst[:512],src[:512]) and np.array_equal(dst[512:1024],src[:512]) and np.array_equal(dst[1024:1536],src[512:]) and np.array_equal(dst[1536:],src[512:])
                elif organ in ('conv_w','conv_b','dt_b','Dskip','A_log'):assert np.array_equal(dst[:512],src) and np.array_equal(dst[512:],src)
                elif organ=='x_proj':assert np.array_equal(dst[:16,:512],src[:16]) and np.array_equal(dst[48:,:512],src[16:]) and not np.any(dst[:,512:]) and not np.any(dst[16:48])
                elif organ=='dt_proj':assert np.array_equal(dst[:512,:16],src) and np.array_equal(dst[512:,:16],src) and not np.any(dst[:,16:])
                else:assert organ=='out_proj' and np.array_equal(dst[:,:512],src) and not np.any(dst[:,512:])
                del src,dst
        finally:om.close();nm.close()
    return dict(artifact=extent(dest),modified_fields=modified,unchanged_fields=unchanged,all_fields_verified=True,new_knowledge=False,construction='zero read injection with duplicated recurrent channels')


def worker(a):
    import numpy as np
    import psutil
    start=time.monotonic();assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    assert (np.__version__,psutil.__version__)==('2.4.6','7.2.2')
    proc=psutil.Process();proc.cpu_affinity([10]);reader=memory_reader();max_child=0;children=[];stage='startup';parity=None;speed=[];profile=[]
    def guard():
        assert time.monotonic()-start<=b['limits']['seconds']-b['limits']['reserve_seconds'],'worker reserve'
        assert proc.memory_info().peak_wset+max_child<=b['limits']['OS_bytes'],'worker/direct child OS'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=b['limits']['output_bytes'],'namespace'
    def event(**kw):guard();print(json.dumps(dict(seconds=time.monotonic()-start,**kw)),flush=True)
    def run(argv,name):
        nonlocal max_child
        guard();t=time.monotonic();peak=0;log=a.directory/(name+'.log')
        with log.open('xb') as stream:
            child=subprocess.Popen([str(p) for p in argv],stdout=stream,stderr=subprocess.STDOUT,creationflags=8)
            created=psutil.Process(child.pid).create_time()
            try:
                while child.poll() is None:peak=max(peak,reader(child));max_child=max(max_child,peak);guard();time.sleep(.05)
                peak=max(peak,reader(child));max_child=max(max_child,peak)
            except BaseException:
                if child.poll() is None:child.kill();child.wait()
                peak=max(peak,reader(child));max_child=max(max_child,peak);raise
            finally:
                receipt=dict(command=[str(p) for p in argv],pid=child.pid,creation_time=created,exit_code=child.returncode,held_OS_peak=peak,seconds=time.monotonic()-t,log=extent(log))
                children.append(receipt);write(a.directory/(name+'.receipt.json'),receipt)
        assert child.returncode==0,(name,child.returncode,log.read_text(errors='replace'));event(stage='child_complete',name=name,held_OS_peak=peak,child_seconds=receipt['seconds']);return receipt
    try:
        stage='adopt_extraction';old_dir=Path(b['completion']['directory']);body=json.loads((old_dir/'original_bodies.json').read_bytes());raw(a.directory/'original_packed_bodies.h',(old_dir/'original_packed_bodies.h').read_bytes());write(a.directory/'adopted_original_bodies.json',body)
        wrapper=(B/'original_packed_capacity.c').read_text();assert wrapper.count('int main(')==1
        wrapper=wrapper[:wrapper.index('int main(')]
        for name,value in [('DN',512),('DTR',16)]:
            old=f'#define {name} {value}';assert wrapper.count(old)==1
            wrapper=wrapper.replace(old,f'#ifndef {name}\n{old}\n#endif')
        cpath=a.directory/'native_envelope.c';raw(cpath,(wrapper+(B/'original_native_envelope_main.c').read_text()).encode())
        stage='adopt_wide';wide=json.loads((old_dir/'wide_export.json').read_bytes());assert extent(wide['artifact']['path'])==wide['artifact'];write(a.directory/'adopted_wide_export.json',wide);event(stage=stage,bytes=wide['artifact']['bytes'])
        exe={};models={'base':Path(b['packed']['path']),'wide':Path(wide['artifact']['path'])}
        for arm in b['controls']:
            stage='compile_'+arm['name'];exe[arm['name']]=a.directory/f"native_{arm['name']}.exe"
            if arm['name']=='base':
                raw(exe['base'],(old_dir/'native_base.exe').read_bytes());assert sha(exe['base'])==sha(old_dir/'native_base.exe');continue
            run([b['compiler'],'-O3','-mavx2','-mfma','-march=znver2',f"-DDN={arm['DN']}",f"-DDTR={arm['DTR']}",cpath,'-I',a.directory,'-o',exe[arm['name']],'-lm'],stage)
        query=a.directory/'requests.u32'
        with query.open('xb') as f:
            f.write(struct.pack('<I',3))
            for rec in b['sequences']:f.write(struct.pack('<I',rec['tokens'])+np.asarray(rec['ids'],dtype='<u4').tobytes())
        native_parity={}
        for name in ('base','wide'):
            stage='parity_'+name;prefix=a.directory/stage
            run([exe[name],models[name],query,'parity',prefix],stage);native_parity[name]=json.loads(prefix.with_suffix('.native.json').read_bytes())
            assert prefix.with_suffix('.witness').read_bytes()==Path(b['expected_witness']['path']).read_bytes(),'independent integer witnesses'
        stage='parity_adjudication';total=sum(r['tokens'] for r in b['sequences']);metric=[];first_fault=None;offset=0
        base=np.memmap(a.directory/'parity_base.f32',dtype='<f4',mode='r',shape=(total,65537));wide_logits=np.memmap(a.directory/'parity_wide.f32',dtype='<f4',mode='r',shape=(total,65537))
        rb=(a.directory/'parity_base.routes').read_bytes();rw=(a.directory/'parity_wide.routes').read_bytes()
        assert len(rb)==len(rw)==total*6*64
        for rec in b['sequences']:
            worst=0.;sqdiff=0.;sqref=0.;ids_bad=0;max_mass=0.;max_defect=0.;argmax_bad=0
            for j in range(rec['tokens']):
                p=offset+j;x=np.asarray(base[p],dtype='f8');y=np.asarray(wide_logits[p],dtype='f8');ds=float(np.dot(x-y,x-y));rs=float(np.dot(x,x));er=math.sqrt(ds)/max(math.sqrt(rs),1e-12)
                assert np.isfinite(x).all() and np.isfinite(y).all();worst=max(worst,er);sqdiff+=ds;sqref+=rs;argmax_bad+=int(np.argmax(x)!=np.argmax(y))
                for l in range(6):
                    start_byte=(p*6+l)*64;bi=np.frombuffer(rb,dtype='<i4',count=8,offset=start_byte);wi=np.frombuffer(rw,dtype='<i4',count=8,offset=start_byte)
                    bm=np.frombuffer(rb,dtype='<f4',count=8,offset=start_byte+32);wm=np.frombuffer(rw,dtype='<f4',count=8,offset=start_byte+32)
                    bad=not np.array_equal(bi,wi);delta=float(np.max(np.abs(bm.astype('f8')-wm.astype('f8'))));defect=max(abs(float(bm.astype('f8').sum())-1),abs(float(wm.astype('f8').sum())-1))
                    ids_bad+=int(bad);max_mass=max(max_mass,delta);max_defect=max(max_defect,defect)
                    if first_fault is None and (bad or delta>b['criteria']['mass_delta'] or defect>b['criteria']['mass_defect'] or er>b['criteria']['full_head_relative_RMS']):first_fault=dict(id=rec['id'],position=j,site=l,relative_RMS=er,ID_mismatch=bool(bad),mass_delta=delta,mass_defect=defect)
            metric.append(dict(id=rec['id'],tokens=rec['tokens'],worst_row_relative_RMS=worst,history_relative_RMS=math.sqrt(sqdiff)/max(math.sqrt(sqref),1e-12),ID_mismatch_calls=ids_bad,mass_max_delta=max_mass,mass_max_defect=max_defect,argmax_mismatch_rows=argmax_bad));offset+=rec['tokens']
        del base,wide_logits
        parity=dict(passed=first_fault is None,first_fault=first_fault,cases=metric,integer_coordinates_per_arm=18432,native=native_parity)
        write(a.directory/'parity_adjudication.json',parity);assert parity['passed'],('native duplication parity',first_fault)
        event(stage='parity_qualified',cases=metric)
        for ordinal,order in enumerate(b['speed_orders']):
            for name in order:
                stage=f'speed_{ordinal}_{name}';prefix=a.directory/stage
                run([exe[name],models[name],query,'speed',prefix],stage)
                speed.append(dict(ordinal=ordinal,arm=name,result=json.loads(prefix.with_suffix('.native.json').read_bytes())))
        for name in b['profile_order']:
            stage='profile_'+name;prefix=a.directory/stage;run([exe[name],models[name],query,'profile',prefix],stage)
            profile.append(dict(arm=name,result=json.loads(prefix.with_suffix('.native.json').read_bytes())))
        summary={}
        for name in ('base','wide'):
            arm=[r for r in speed if r['arm']==name];cases=[]
            for i,seq in enumerate(b['sequences']):
                times=[r['result']['requests'][i]['batch1_seconds'] for r in arm];median=float(np.median(times))
                prof=next(r['result'] for r in profile if r['arm']==name)['requests'][i]
                cases.append(dict(id=seq['id'],tokens=seq['tokens'],seconds=times,median_seconds=median,median_raw_IDs_per_second=seq['tokens']/median,
                    profile_seconds=prof['batch1_seconds'],profile_minus_unprofiled_median=prof['batch1_seconds']-median,components=prof['components']))
            summary[name]=dict(cases=cases,median_aggregate_raw_IDs_per_second=total/float(np.median([sum(x['batch1_seconds'] for x in r['result']['requests']) for r in arm])),minimum_case_median_raw_IDs_per_second=min(c['median_raw_IDs_per_second'] for c in cases))
        ratio=summary['base']['median_aggregate_raw_IDs_per_second']/summary['wide']['median_aggregate_raw_IDs_per_second']
        affordable=summary['wide']['minimum_case_median_raw_IDs_per_second']>=b['criteria']['raw_envelope_IDs_per_second'] and ratio<=b['criteria']['max_raw_slowdown']
        result=dict(schema='ORIGINAL_NATIVE_ENVELOPE_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='WIDER_NATIVE_RAW_ENVELOPE_SUPPORTS_FINITE_LEARNED_CORE_PILOT' if affordable else 'WIDER_NATIVE_RAW_ENVELOPE_COST_GATE_FAIL',
            original_bodies=body,wide_export=wide,parity=parity,speed=speed,profile=profile,summary=summary,aggregate_raw_slowdown=ratio,
            raw_cost_gate=affordable,children=children,compiled_executables={k:extent(v) for k,v in exe.items()},
            source_calls=0,optimizer_updates=0,GPU_calls=0,quality_admission=False,speed_admission=False,useful_large_n_admission=False,
            physical_DRAM_bytes=None,logical_selected_pair_code_bytes_per_token=3*6*8*256*128//2,
            scope='Real poor-quality Adam25 bank/fullV;zero-read duplicate wider core,no new information. Complete batch1 forced histories+fullhead argmax/finiteness/mass consumer,postload. Not useful accepted generation;no DRAM counters. Profiling difference includes timing noise,not isolated QPC overhead. Nested linker separate peak unknown.',
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=max_child,elapsed_seconds=time.monotonic()-start)
        result.update(inherited_children=[b['completion']['inherited_compile']],adopted_wide_export=True,prior_held_family_seconds=b['completion']['prior_held_family_seconds']);assert len(children)==11;write(a.out,result);event(stage='complete',decision=result['decision'],summary=summary)
    except BaseException as e:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(e),parity=parity,speed=speed,profile=profile,children=children,
            elapsed_seconds=time.monotonic()-start,worker_OS_peak_snapshot=proc.memory_info().peak_wset,max_direct_child_OS_peak=max_child));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
