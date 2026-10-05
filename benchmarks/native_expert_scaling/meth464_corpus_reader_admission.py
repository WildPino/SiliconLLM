"""Retained-parent admission of isolated corpus transport after native463 crash."""
import os
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
PROTOCOL = DOC / 'METH_464_CORPUS_READER_ADMISSION_PROTOCOL_20261005.md'
CHILD = ROOT / 'benchmarks/native_expert_scaling/meth464_isolated_corpus_reader.py'
OUT = ROOT / 'results/native_expert_scaling/meth464_corpus_reader_admission'
RAW = DOC / 'meth464_corpus_reader_admission_result.json'
CORPUS = ROOT / 'data/external/pg19/data/train-00014-of-00023-54b567998cd5eb4b.parquet'
CORPUS_SHA = '15f3ecb5e314ce58bbf8eae927b51bf893b9fdabe4a04c8e246547206357a2d5'
PYTHON = ROOT / 'results/native_expert_scaling/meth324_switch_reference/venv/Scripts/python.exe'
DIRECT = {'meth463_external_termination_result.json':'b3a46001a14449d3a60926f03681fe657d383d99b8b90e2f45929d6e76ed576d',
          'meth463_prospective_bindings.json':'65309f005dfb17d3c02abcdf3382b4a947b6a2ae365ccbcc2616fb4c99cd6211',
          'meth382_switch_multi_span_manifest.json':'96b3ffd09b0bdd42d7c990006e8744e39b5e1f32b7ae40aaf3b3dca98bf4db78',
          'meth362_switch_multi_span_manifest.json':'c387fc7b831b6b7bb50634686ac39b8e8873f20fedc7dec9b8556f62c54de2cf'}


def committed(path):
    rel=Path(path).resolve().relative_to(ROOT).as_posix()
    assert Path(path).read_bytes()==subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel],cwd=ROOT),rel


def write_new(path,value):
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').replace('\n','\r\n').encode('utf-8')
    assert len(data)<=1<<20
    with Path(path).open('xb') as stream:stream.write(data)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert args.out.resolve()==RAW.resolve() and not RAW.exists() and not RAW.with_suffix('.failure.json').exists() and not OUT.exists()
    assert os.name=='nt' and sys.flags.optimize==0
    start=time.monotonic();stage='bindings';peak=0;aggregate_peak=0;hashed=0;child=None
    result={'experiment':'METH464-isolated-corpus-reader-admission','retained_record_sha256':{},'runtime_file_sha256':{},'gates':{},'native_or_model_commands':0}
    process=psutil.Process()
    def guard():
        nonlocal peak,aggregate_peak
        info=process.memory_info();peak=max(peak,info.rss,getattr(info,'peak_wset',0));rss=info.rss
        if child is not None and child.poll() is None:
            try:
                root=psutil.Process(child.pid);rss+=root.memory_info().rss+sum(p.memory_info().rss for p in root.children(recursive=True))
            except psutil.NoSuchProcess:pass
        aggregate_peak=max(aggregate_peak,rss)
        assert time.monotonic()-start<=300 and max(peak,aggregate_peak)<=4<<30,'main300s4GiB'
        assert not OUT.exists() or sum(p.stat().st_size for p in OUT.iterdir())<=2<<30,'output2GiB'
    def digest(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(4<<20):h.update(block);hashed+=len(block);guard()
        return h.hexdigest()
    def jobs():
        own={os.getpid(),*(p.pid for p in process.parents())};preserved=[]
        for p in psutil.process_iter(['name','cmdline']):
            if p.pid in own:continue
            name,argv=(p.info['name'] or '').lower(),p.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():preserved.append(p.pid);continue
            assert not(name.startswith('python') or name=='clang.exe' or(name.startswith('meth') and name.endswith('.exe'))),(p.pid,name)
        return preserved
    try:
        result['preserved_daemons_before']=jobs()
        for path in (Path(__file__),CHILD,PROTOCOL,*(DOC/name for name in DIRECT)):committed(path)
        result['controller_sha256']=digest(__file__);result['child_source_sha256']=digest(CHILD);result['protocol_sha256']=digest(PROTOCOL)
        result['head_at_execution']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for name,expected in DIRECT.items():assert digest(DOC/name)==expected;result['retained_record_sha256'][name]=expected
        bindings=json.loads((DOC/'meth463_prospective_bindings.json').read_text(encoding='utf-8'))
        assert digest(CORPUS)==CORPUS_SHA
        assert metadata.version('pyarrow')=='24.0.0'
        assert digest(ROOT/'.venv/Lib/site-packages/pyarrow/arrow.dll')=='b9a22a56c2e4c08e6b13924b65552531075f659116737e0a2242b2496c788b24'
        for path,row in bindings['runtime']['files'].items():assert digest(path)==row['sha256'];result['runtime_file_sha256'][path]=row['sha256']
        for path in (ROOT/'.venv/Lib/site-packages/pyarrow/arrow.dll',ROOT/'.venv/Lib/site-packages/pyarrow/_parquet.cp312-win_amd64.pyd',ROOT/'.venv/Lib/site-packages/pyarrow/lib.cp312-win_amd64.pyd'):
            result['runtime_file_sha256'][str(path)]=digest(path)
        for name in ('pyarrow','psutil'):
            row=bindings['runtime']['packages'][name];assert metadata.version(name)==row['version'] and digest(row['metadata_path'])==row['sha256']
            result['runtime_file_sha256'][row['metadata_path']]=row['sha256']
        result['corpus']={'path':str(CORPUS),'bytes':CORPUS.stat().st_size,'sha256':CORPUS_SHA}
        controls={}
        for name in ('meth382_switch_multi_span_manifest.json','meth362_switch_multi_span_manifest.json'):
            for book in json.loads((DOC/name).read_text(encoding='utf-8'))['items']:
                key=str(book['corpus_row']);assert key not in controls;controls[key]=book['whole_source_utf8_sha256']
        assert len(controls)==48
        for rel,row in bindings['preserved_unrelated_files'].items():assert digest(ROOT/rel)==row['sha256']
        result['engine_sha256']=digest(ROOT/'benchmarks/phase60/engine.c');committed(ROOT/'benchmarks/phase60/engine.c')
        OUT.mkdir();write_new(OUT/'known_original_controls.json',controls)
        result['gates']['source_corpus_runtime_first_fault_and_controls_bound']=True
        write_new(OUT/'parent_admission_checkpoint.json',result)
        stage='isolated_reader'
        argv=[str(PYTHON),'-u','-X','faulthandler',str(CHILD),'--corpus',str(CORPUS),'--out',str(OUT),'--controls',str(OUT/'known_original_controls.json')]
        result['reader_command']={'argv':argv.copy()}
        with (OUT/'reader.stdout.log').open('xb') as stdout,(OUT/'reader.stderr.log').open('xb') as stderr:
            child_start=time.monotonic();child=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,env=os.environ.copy())
            result['reader_command']['pid']=child.pid;write_new(OUT/'parent_child_started.json',result['reader_command'])
            while child.poll() is None:guard();assert time.monotonic()-child_start<=180,'reader180s';time.sleep(.25)
        result['reader_command'].update(returncode=child.returncode,seconds=time.monotonic()-child_start)
        assert child.returncode==0,('reader_terminal_failure',child.returncode)
        summary=json.loads((OUT/'reader.summary.json').read_text(encoding='utf-8'))
        assert summary['known_original_book_sha256']==controls and summary['rows']==1243
        stage='independent_UTF8_transport_validation'
        spool=OUT/'corpus_utf8.bin';index=struct.Struct('<QQQ32s');known={};offset=16+1243*index.size;total_characters=0
        with spool.open('rb') as stream:
            assert stream.read(16)==struct.pack('<8sII',b'M464TX01',1243,index.size)
            entries=[index.unpack(stream.read(index.size)) for _ in range(1243)]
            for row,(begin,length,characters,expected) in enumerate(entries):
                assert begin==offset and length>0 and characters>0
                data=stream.read(length);assert len(data)==length and hashlib.sha256(data).digest()==expected
                assert len(data.decode('utf-8'))==characters
                if str(row) in controls:assert expected.hex()==controls[str(row)];known[str(row)]=expected.hex()
                total_characters+=characters;offset+=length;guard()
            assert not stream.read(1) and offset==spool.stat().st_size==summary['spool_bytes'] and known==controls
        result['reader_summary']=summary
        result['transport']={'path':str(spool),'bytes':offset,'sha256':digest(spool),'rows':1243,'index_record_bytes':index.size,'index_bytes':16+1243*index.size,'total_characters':total_characters,'known_original_book_sha256':known}
        result['gates'].update(isolated_single_thread_reader_terminal_success=True,all1243_UTF8_rows_offsets_lengths_and_hashes_verified=True,all48_known_original_source_book_hashes_exact=True)
        result['retained_output_files']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir())]
        for rel,row in bindings['preserved_unrelated_files'].items():assert digest(ROOT/rel)==row['sha256']
        assert digest(ROOT/'benchmarks/phase60/engine.c')==result['engine_sha256']
        result['preserved_daemons_after']=jobs();result['gates']['engine_and_unrelated_work_preserved_no_model_selection']=True;guard()
        result['resource']={'main_seconds_through_validation':time.monotonic()-start,'parent_OS_peak_bytes':peak,'maximum_sampled_parent_descendant_sum_bytes':aggregate_peak,'bytes_hashed':hashed,'new_output_bytes':sum(p.stat().st_size for p in OUT.iterdir())}
        result['decision']='isolated_original_corpus_transport_admitted_freeze_new_source_manifest_without_Arrow_in_tokenizer_process'
        result['scope']='Corpus transport only; ALL rows are catalogue data, NOT consumed calibration/quality cases.48 known source controls. No source selection/new tokenization/model/fit/native forward/geometry/quality/rate/DRAM; no causal crash attribution.'
        write_new(RAW,result);guard()
        print(json.dumps({'sha256':digest(RAW),'gates':result['gates'],'transport_bytes':offset,'resource':result['resource']}),flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                root=psutil.Process(child.pid)
                for p in root.children(recursive=True):p.kill()
                root.kill()
            except psutil.NoSuchProcess:pass
            child.wait()
        if child is not None:result.setdefault('reader_command',{})['returncode']=child.returncode
        result.update(stage=stage,error=repr(error),seconds=time.monotonic()-start,parent_OS_peak_bytes=peak,maximum_sampled_parent_descendant_sum_bytes=aggregate_peak)
        if OUT.exists():result['partial_output_files']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size<8<<20 else None} for p in OUT.iterdir()]
        write_new(RAW.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
