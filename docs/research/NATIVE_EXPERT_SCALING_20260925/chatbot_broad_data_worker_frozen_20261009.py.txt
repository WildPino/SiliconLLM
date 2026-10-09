"""Pinned public dialogue acquisition, isolated selection and canonical ID adoption."""
import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from chatbot_falcon_usability import SITE,sha,write
sys.path.insert(0,str(SITE))
from chatbot_broad_data_reader import key
REPO='HuggingFaceTB/smoltalk'
REV='5feaf2fd3ffca7c237fc38d1861bc30365d48ffa'
SNAPSHOT=ROOT/'results/native_expert_scaling/chatbot_broad_data_sources_20261009'
FOREIGN={'benchmarks/donor_adaptation/configs/_manifest.json':'fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d',
 'benchmarks/donor_adaptation/density/build_document_holdout.py':'c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9',
 'docs/research/RESEARCH_INDEX.md':'99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273'}


def snapshot(a):
    SNAPSHOT.mkdir(exist_ok=False)
    urls=dict(api='https://huggingface.co/api/datasets/'+REPO+'/revision/'+REV+'?blobs=true',
              card='https://huggingface.co/datasets/'+REPO+'/resolve/'+REV+'/README.md')
    files=[]
    for name,url in urls.items():
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read(1<<20)
        assert len(data)<1<<20
        p=SNAPSHOT/('api.json' if name=='api' else 'README.md')
        with p.open('xb') as f:f.write(data)
        files.append(dict(url=url,path=str(p),bytes=len(data),sha256=sha(p)))
    api=json.loads((SNAPSHOT/'api.json').read_bytes());assert api['sha']==REV
    selected=[r for r in api['siblings'] if r['rfilename'] in ('data/all/train-00000-of-00009.parquet','data/all/test-00000-of-00001.parquet')]
    assert len(selected)==2 and sum(r['size'] for r in selected)==328261485
    write(a.out,dict(schema='BROAD_CHAT_SOURCE_SNAPSHOT_V1',dataset=REPO,revision=REV,files=files,selected_shards=selected,
                    scope='ONE of nine training shards and complete official test shard; not the complete training mixture.'))
    print(json.dumps(dict(snapshot=str(a.out),sha256=sha(a.out),download_bytes=328261485)),flush=True)


def bind(a):
    source=json.loads(a.snapshot.read_bytes());assert source['revision']==REV
    files=[a.snapshot,*[Path(r['path']) for r in source['files']],Path(__file__),B/'chatbot_broad_data_reader.py',
      B/'chatbot_hybrid_engine_chat.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
      DOC/'CHATBOT_BROAD_DATA_PROTOCOL_20261009.md',Path(sys.executable),
      B/'chatbot_hybrid_transfer_cases_v1.json',B/'chatbot_falcon_usability_cases_v1.json',
      DOC/'chatbot_hybrid_pilot_result_repair1_20261008.json']
    arrow=ROOT/'.venv/Lib/site-packages/pyarrow'
    files += [arrow/n for n in ('__init__.py','arrow.dll','lib.cp312-win_amd64.pyd','_parquet.cp312-win_amd64.pyd','parquet/core.py')]
    files += [ROOT/'.venv/Lib/site-packages/pyarrow-24.0.0.dist-info/METADATA',SITE/'psutil/__init__.py',
              SITE/'tokenizers/__init__.py',SITE/'tokenizers/tokenizers.pyd',SITE/'jinja2/__init__.py']
    tokenizer_source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    files += [tokenizer_source/n for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja')]
    protected=set()
    cases=json.loads((B/'chatbot_hybrid_transfer_cases_v1.json').read_bytes())
    for row in cases['cases']+cases['reserved_cases']:
        protected.add(key(row['prompt']))
        for m in row.get('history',[]):
            if m['role']=='user':protected.add(key(m['content']))
    for row in json.loads((B/'chatbot_falcon_usability_cases_v1.json').read_bytes())['cases']:protected.add(key(row['prompt']))
    for row in json.loads((DOC/'chatbot_hybrid_pilot_result_repair1_20261008.json').read_bytes())['supervision_records']:
        for m in row['messages']:
            if m['role']=='user':protected.add(key(m['content']))
    assert all(sha(ROOT/p)==s for p,s in FOREIGN.items());files += [ROOT/p for p in FOREIGN]
    b=dict(schema='HYBRID_BROAD_DATA_BINDING_V1',python=str(Path(sys.executable).resolve()),worker_path=str(Path(__file__).resolve()),
       source_snapshot=str(a.snapshot.resolve()),source_revision=REV,source_repo=REPO,download_bytes=328261485,
       reader_path=str((B/'chatbot_broad_data_reader.py').resolve()),tokenizer_source=str(tokenizer_source),
       protected_prompt_keys=sorted(protected),selected_count=624,source_count=13,quotas=dict(FIT=32,DEV=8,RESERVED=8),
       limits=dict(seconds=600,OS_bytes=6<<30,output_bytes=768<<20),allowed_worker_children=['python.exe'],
       runtime_binding_scope='Pinned public shard LFS bytes plus selected source/runtime/Arrow/Rust/template/code/protocol/Python extents; isolated Arrow reader without Torch/Rust. No full DLL tree or independent Parquet decoder.',
       inputs=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in dict.fromkeys(files)])
    write(a.out,b);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(b['inputs']),protected_keys=len(protected))),flush=True)


def worker(a):
    start=time.monotonic();assert sha(a.binding)==a.binding_sha
    b=json.loads(a.binding.read_bytes());assert b['schema']=='HYBRID_BROAD_DATA_BINDING_V1'
    assert Path(sys.executable).resolve()==Path(b['python']).resolve()
    import psutil
    from chatbot_falcon_usability_launch import Memory
    proc=psutil.Process();proc.cpu_affinity([0]);a.directory.mkdir(exist_ok=False)
    getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    getmem.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];getmem.restype=wintypes.BOOL
    def peak(p):
        m=Memory();m.cb=ctypes.sizeof(m);assert getmem(wintypes.HANDLE(int(p._handle)),ctypes.byref(m),m.cb)
        return m.PeakWorkingSetSize
    child=None;child_peak=0;phase='download';downloads=[]
    def guard():
        assert time.monotonic()-start<b['limits']['seconds']-30,'worker reserve'
        assert proc.memory_info().peak_wset+child_peak<=b['limits']['OS_bytes'],'worker/reader OS'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=b['limits']['output_bytes'],'output cap'
    try:
        source=json.loads(Path(b['source_snapshot']).read_bytes());reader_config=dict(protected_prompt_keys=b['protected_prompt_keys'])
        for shard in sorted(source['selected_shards'],key=lambda r:r['rfilename']):
            split='test' if '/test-' in shard['rfilename'] else 'train';path=a.directory/(split+'.parquet')
            url='https://huggingface.co/datasets/'+REPO+'/resolve/'+REV+'/'+shard['rfilename']+'?download=true'
            begin=time.monotonic();hasher=hashlib.sha256();count=0
            with urllib.request.urlopen(url,timeout=30) as response,path.open('xb') as output:
                for chunk in iter(lambda:response.read(1<<20),b''):
                    output.write(chunk);hasher.update(chunk);count+=len(chunk)
                    assert count<=shard['size'];guard()
                    if count%(16<<20)==0:print(json.dumps(dict(stage='download',split=split,bytes=count,total=shard['size'])),flush=True)
                output.flush();os.fsync(output.fileno())
            assert count==shard['size'] and hasher.hexdigest()==shard['lfs']['sha256']
            rec=dict(path=str(path.resolve()),bytes=count,sha256=hasher.hexdigest(),url=url,seconds=time.monotonic()-begin)
            downloads.append(rec);reader_config[split]=rec
            write(a.directory/(split+'.download.json'),rec)
        phase='reader';config=a.directory/'reader_config.json';write(config,reader_config);log=a.directory/'reader.log'
        command=[b['python'],'-I','-S','-B','-X','utf8',b['reader_path'],'--config',str(config.resolve()),'--directory',str(a.directory.resolve())]
        begin=time.monotonic()
        with log.open('xb') as output:
            child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,creationflags=0x08000000)
            try:
                while child.poll() is None:
                    child_peak=max(child_peak,peak(child));guard();time.sleep(.1)
            finally:
                if child.poll() is None:child.kill();child.wait()
                child_peak=max(child_peak,peak(child))
                reader_receipt=dict(command=command,pid=child.pid,exit_code=child.returncode,seconds=time.monotonic()-begin,
                                    OS_peak_through_exit=child_peak,log_sha256=sha(log))
                write(a.directory/'reader.process.json',reader_receipt)
        assert child.returncode==0,log.read_text(errors='replace')
        selected=json.loads((a.directory/'selected_raw.json').read_bytes());assert selected['selected']==624
        phase='tokenizer_adoption'
        from chatbot_hybrid_engine_chat import ChatTokenizer
        tokenizer=ChatTokenizer(b['tokenizer_source']);records=selected['records'];groups=set();prefixes=set();lengths={};summaries={}
        for row in records:
            assert row['group_key']==key(next(m['content'] for m in row['messages'] if m['role']=='user'))
            prefix=json.dumps(row['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
            assert hashlib.sha256(prefix).hexdigest()==row['prefix_sha256']
            assert row['group_key'] not in groups and row['prefix_sha256'] not in prefixes
            groups.add(row['group_key']);prefixes.add(row['prefix_sha256'])
            assert all(key(m['content']) not in set(b['protected_prompt_keys']) for m in row['messages'] if m['role']=='user')
            ids=tokenizer.encode(row['messages']);assert ids[0]==17 and all(0<=i<65537 for i in ids)
            row['input_ids']=ids;row['input_length']=len(ids);row['donor_reply_available']=False
            band='le256' if len(ids)<=256 else ('257_768' if len(ids)<=768 else ('769_8192' if len(ids)<=8192 else 'gt8192'))
            row['length_band']=band
            lengths.setdefault(row['split'],Counter())[band]+=1
            summaries.setdefault(row['source'],Counter())[row['split']]+=1
            guard()
        assert len(groups)==len(prefixes)==624 and len(summaries)==13
        assert all(dict(v)==dict(RESERVED=8,FIT=32,DEV=8) for v in summaries.values())
        corpus=a.directory/'corpus.json'
        write(corpus,dict(schema='BROAD_CHAT_DONOR_INPUTS_V1',source_repo=REPO,source_revision=REV,records=records,
          source_counts={k:dict(v) for k,v in summaries.items()},length_bands={k:dict(v) for k,v in lengths.items()},
          source_calls=0,target_labels=0,external_answers_are_donor_labels=False,scope='Selected last-user prefixes include upstream assistant history; not donor-own-history observations. All lengths retained without truncation.'))
        assert not any(v in sys.modules for v in ('torch','transformers','pyarrow'))
        report=dict(schema='HYBRID_BROAD_DATA_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
         process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),decision='BROAD_DATA_ADOPTION_PASS',selected=624,
         gates=dict(pinned_shard_LFS_bytes=True,isolated_reader_exit=True,selected_prefix_transport=True,all624_unique_groups_and_prefixes=True,
                    full_test_groups_excluded_from_selected_train=True,old_queries_excluded=True,source_template_ID_adoption=True,
                    source_quotas=True,no_length_truncation=True),downloads=downloads,reader=reader_receipt,
         upstream_rows=selected['upstream_rows'],upstream_source_rows=selected['upstream_source_rows'],counters=selected['counters'],
         source_counts={k:dict(v) for k,v in summaries.items()},length_bands={k:dict(v) for k,v in lengths.items()},
         min_input_ids=min(r['input_length'] for r in records),max_input_ids=max(r['input_length'] for r in records),
         prior_assistant_cases=sum(r['prior_assistant_turns']>0 for r in records),
         corpus=dict(path=str(corpus.resolve()),bytes=corpus.stat().st_size,sha256=sha(corpus)),
         worker_OS_peak_snapshot=proc.memory_info().peak_wset,reader_OS_peak_through_exit=child_peak,
         seconds=time.monotonic()-start,source_calls=0,model_calls=0,optimizer_updates=0,GPU_calls=0,reserved_queries=0,
         scope='One of nine training shards plus all official test rows. New donor inputs, zero donor labels/quality/admission. Exact selected-group disjointness, not semantic decontamination or an independent Parquet decoder.')
        guard();write(a.out,report)
        print(json.dumps(dict(stage='adoption_complete',selected=624,length_bands=report['length_bands'],corpus=report['corpus'])),flush=True)
    except BaseException as error:
        if child and child.poll() is None:child.kill();child.wait()
        write(a.directory/'first_failure.json',dict(phase=phase,fault=repr(error),seconds=time.monotonic()-start,completed_downloads=downloads))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path);p.add_argument('--acquire-metadata',action='store_true')
    p.add_argument('--bind',action='store_true');p.add_argument('--binding',type=Path);p.add_argument('--binding-sha')
    p.add_argument('--freeze');p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();snapshot(a) if a.acquire_metadata else (bind(a) if a.bind else worker(a))
