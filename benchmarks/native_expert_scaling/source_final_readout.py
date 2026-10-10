"""Paired BF16/F64 donor heads on retained full h24; no model or history call."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling'
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from original_packed_capacity import SITE,extent,sha,write
sys.path.insert(0,str(SITE))
from original_latent_readout_attribution import row_metrics,aggregate
V,D=65537,2048


def fields(path):
    with Path(path).open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
    result={}
    for name,shape in (('lm_head.weight',[V,D]),('model.final_layernorm.weight',[D])):
        item=header[name];assert item['dtype']=='BF16' and item['shape']==shape
        start,end=item['data_offsets'];assert end-start==math.prod(shape)*2
        assert 8+size+end<=Path(path).stat().st_size
        result[name]=dict(offset=8+size+start,bytes=end-start,shape=shape,dtype='BF16')
    return result


def alignment(arms,g):
    flags=[]
    for arm in arms:
        for split in ('FIT','DEV'):
            rows=[r for r in arm['records'] if r['split']==split];agg=aggregate(rows)
            gates=dict(mean_KL=agg['case_KL']<=g['mean_KL'],mean_disagreement=agg['case_disagreement']<=g['mean_disagreement'],
                every_case_KL=all(r['KL']<=g['case_KL'] for r in rows),
                every_case_disagreement=all(r['disagreement_rate']<=g['case_disagreement'] for r in rows),
                domains=all(d['case_KL']<=g['case_KL'] and d['case_disagreement']<=g['case_disagreement'] for d in agg['domains'].values()))
            flags.append(dict(readout=arm['name'],split=split,gates=gates,pass_all=all(gates.values())))
    assert len(flags)==4
    return ('SOURCE_READOUT_ALIGNED' if all(f['pass_all'] for f in flags) else 'SOURCE_STATE_LABEL_ALIGNMENT_FAIL'),flags


def bind(a):
    assert not a.out.exists()
    pbpath=DOC/'original_joint_history_recovery_binding_20261009.json';pb=json.loads(pbpath.read_bytes())
    hpath=DOC/'original_history_boundaries_finish_result_20261009.json';hr=json.loads(hpath.read_bytes())
    hapath=DOC/'original_history_boundaries_stored_adjudication_20261009.json';ha=json.loads(hapath.read_bytes())
    assert ha['result_sha256']==sha(hpath) and hr['cases']==48 and hr['LM_head_calls']==0
    source=ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008'
    weight=source/'model.safetensors';config_path=source/'config.json';config=json.loads(config_path.read_bytes())
    assert (config['hidden_size'],config['vocab_size'],config['rms_norm_eps'],config['lm_head_multiplier'])==(D,V,1e-5,.01953125)
    weight_extent=extent(weight)
    assert weight_extent['sha256']=='1fb788513f2c58e3abb91af8730afe19e247bce7a7f01a46e452a084e98b2abd'
    ff=fields(weight);records=pb['records'];assert len(records)==48
    targets={r['id']:next(i['h'] for i in r['boundaries'] if i['boundary']==24) for r in hr['records']}
    assert sum(len(r['positions']) for r in records)==8808
    for rec in records:
        assert targets[rec['id']]['bytes']==len(rec['student_input_ids'])*D*2
        assert rec['logits']['bytes']==len(rec['positions'])*V*2
    previous=DOC/'original_latent_readout_attribution_result_20261010.json'
    previous_audit=DOC/'original_latent_readout_attribution_stored_adjudication_20261010.json'
    pr=json.loads(previous.read_bytes());pa=json.loads(previous_audit.read_bytes())
    assert pr['decision']==pa['decision']=='TARGET_DECODER_COUPLING' and pa['result']==extent(previous)
    assert json.loads(previous.with_suffix('.terminal.json').read_bytes())['exit_code']==0
    receipt=previous_audit.with_suffix('.receipt.json');ar=json.loads(receipt.read_bytes())
    assert ar['exit_code']==0 and ar['error'] is None and ar['output']==extent(previous_audit)
    files=[Path(__file__),B/'source_final_readout_launch.py',B/'source_final_readout_audit.py',B/'source_final_readout_audit_launch.py',
        B/'original_latent_readout_attribution.py',B/'original_joint_history_recovery_audit.py',B/'original_packed_capacity.py',
        B/'original_falcon_whole_recovery.py',B/'chatbot_falcon_usability.py',B/'chatbot_falcon_usability_launch.py',
        DOC/'SOURCE_FINAL_READOUT_PROTOCOL_20261010.md',DOC/'PAIRED_OUTPUT_CODEC_NEXT_20261010.md',
        DOC/'ORIGINAL_READOUT_INFORMATION_RESULT_20261009.md',pbpath,hpath,hapath,weight,config_path,
        previous,previous.with_suffix('.terminal.json'),previous_audit,receipt,
        ROOT/'.venv/Lib/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py',
        Path(sys.executable),Path(sys.executable).parent/'python312.dll',SITE/'torch/__init__.py',SITE/'torch/_C.cp312-win_amd64.pyd',
        SITE/'numpy/__init__.py',SITE/'numpy/_core/_multiarray_umath.cp312-win_amd64.pyd',SITE/'psutil/__init__.py']
    files+=sorted((SITE/'numpy.libs').glob('*.dll'))+sorted((SITE/'psutil').glob('*.pyd'))
    for pattern in ('torch_cpu.dll','torch_cuda.dll','c10.dll','c10_cuda.dll','cublas*.dll'):
        files+=sorted((SITE/'torch/lib').glob(pattern))
    for rec in records:files.extend((Path(targets[rec['id']]['path']),Path(rec['logits']['path'])))
    binding=dict(schema='SOURCE_FINAL_READOUT_BINDING_V1',records=records,targets=targets,weights=weight_extent,fields=ff,
        eps=1e-5,multiplier=.01953125,chunk=16,source_dimension=D,vocab=V,
        source_label_gate=dict(mean_KL=.01,mean_disagreement=.01,case_KL=.05,case_disagreement=.05),
        numeric_gates=dict(metric_absolute=1e-10,feature_relative_RMS=.01,scalar_head_absolute=1e-8),
        limits=dict(seconds=1200,OS_bytes=6<<30,GPU_allocated_bytes=2<<30,GPU_reserved_bytes=3<<30,output_bytes=8<<30,log_bytes=8<<20),
        expected=dict(labels=8808,bf16_score_bytes=8808*V*2,f64_score_bytes=8808*V*8,feature_bytes=8808*D*2),
        inputs=[weight_extent if p.resolve()==weight.resolve() else extent(p) for p in dict.fromkeys(files)],runtime=dict(Torch='2.6.0+cu124',NumPy='2.4.6',threads=6),
        scope='Two source final readouts on retained h24 and cached generation labels. No history/model/generation/optimizer/native/RESERVED call. '
            'BF16 norm/head on GPU, F64 head on the SAME saved BF16 feature. No codec fitting or compact/native/quality/speed admission.')
    write(a.out,binding);print(json.dumps(dict(binding=str(a.out),sha256=sha(a.out),inputs=len(binding['inputs']),expected=binding['expected'])),flush=True)


def worker(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import psutil
    start=time.monotonic();proc=psutil.Process();proc.cpu_affinity(list(range(6)))
    assert sha(a.binding)==a.binding_sha;b=json.loads(a.binding.read_bytes());a.directory.mkdir(exist_ok=False)
    assert torch.__version__==b['runtime']['Torch'] and np.__version__==b['runtime']['NumPy']
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    path=b['weights']['path'];hf=b['fields']['lm_head.weight'];nf=b['fields']['model.final_layernorm.weight']
    raw=np.memmap(path,dtype='<u2',mode='r',offset=hf['offset'],shape=(V,D))
    gpu_head=torch.from_numpy(raw.copy()).view(torch.bfloat16).cuda()
    w=((raw.astype('<u4')<<16).view('<f4')).astype('f8');del raw
    gamma_bits=np.fromfile(path,dtype='<u2',count=D,offset=nf['offset'])
    gamma=torch.from_numpy(gamma_bits.copy()).view(torch.bfloat16).cuda()
    rows={'BF16':[],'F64':[]};rounding=[];count=0;stage='loaded';current=None;ref_peak=0.
    def guard():
        assert time.monotonic()-start<=b['limits']['seconds'],'source readout deadline'
        assert proc.memory_info().peak_wset<=b['limits']['OS_bytes'],'OS cap'
        assert torch.cuda.max_memory_allocated()<=b['limits']['GPU_allocated_bytes'],'GPU allocated cap'
        assert torch.cuda.max_memory_reserved()<=b['limits']['GPU_reserved_bytes'],'GPU reserved cap'
        assert sum(p.stat().st_size for p in a.directory.iterdir() if p.is_file())<=b['limits']['output_bytes'],'output cap'
    try:
        with torch.no_grad():
            for rec in b['records']:
                guard();current=rec['id'];stage='norm';n=len(rec['student_input_ids']);m=len(rec['positions'])
                h=np.memmap(b['targets'][current]['path'],dtype='<u2',mode='r',shape=(n,D))
                value=torch.from_numpy(np.array(h[rec['positions']],copy=True)).view(torch.bfloat16).cuda();del h
                f=value.float();value=(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+b['eps'])).bfloat16()*gamma
                feature_bits=value.cpu().view(torch.uint16).numpy().copy();del f
                features=a.directory/(current+'.norm.bf16')
                with features.open('xb') as stream:
                    stream.write(feature_bits.astype('<u2',copy=False).tobytes());stream.flush();os.fsync(stream.fileno())
                feat=((feature_bits.astype('<u4')<<16).view('<f4')).astype('f8')
                teacher=np.memmap(rec['logits']['path'],dtype='<u2',mode='r',shape=(m,V))
                paths={name:a.directory/(current+'.'+name+'.scores') for name in rows};metrics={name:[] for name in rows}
                max_score=0.;round_losses=[];round_dis=0;stage='head_contractions'
                with paths['BF16'].open('xb') as bs,paths['F64'].open('xb') as ds:
                    for first in range(0,m,b['chunk']):
                        guard();last=min(first+b['chunk'],m);scores=(feat[first:last]@w.T)*b['multiplier']
                        ds.write(scores.astype('<f8',copy=False).tobytes())
                        for j in range(first,last):
                            output=F.linear(value[j:j+1],gpu_head)*b['multiplier']
                            bits=output.cpu().view(torch.uint16).numpy()[0].copy();bs.write(bits.astype('<u2',copy=False).tobytes())
                            bf=(bits.astype('<u4')<<16).view('<f4');q=(teacher[j].astype('<u4')<<16).view('<f4')
                            for name,student in (('BF16',bf),('F64',scores[j-first])):
                                packet=row_metrics(q,student);metrics[name].append(packet);ref_peak=max(ref_peak,packet[-1])
                            packet=row_metrics(bf,scores[j-first]);round_losses.append(packet[0]);round_dis+=packet[1];ref_peak=max(ref_peak,packet[-1])
                            max_score=max(max_score,float(np.max(np.abs(bf.astype('f8')-scores[j-first]))));count+=1
                        del scores,output,bits,bf,q
                    for stream in (bs,ds):stream.flush();os.fsync(stream.fileno())
                for name in rows:
                    losses=[p[0] for p in metrics[name]];ent=[p[2] for p in metrics[name]];uniform=[p[3] for p in metrics[name]];bad=sum(p[1] for p in metrics[name])
                    row=dict(readout=name,id=current,split=rec['split'],domain=rec['domain'],history=n,labels=m,positions=rec['positions'],
                        KL=float(np.mean(losses)),KL_per_label=losses,disagreement=bad,disagreement_rate=bad/m,
                        teacher_entropy=float(np.mean(ent)),entropy_per_label=ent,uniform_KL=float(np.mean(uniform)),uniform_KL_per_label=uniform,
                        max_reference_delta=max(p[-1] for p in metrics[name]),scores=extent(paths[name]),features=extent(features),
                        teacher=rec['logits'],h24=b['targets'][current])
                    assert row['scores']['bytes']==m*V*(2 if name=='BF16' else 8)
                    write(a.directory/(current+'.'+name+'.case.json'),row);rows[name].append(row)
                rounding.append(dict(id=current,labels=m,max_score_absolute_delta=max_score,disagreement=round_dis,
                    case_KL_BF16_to_F64=float(np.mean(round_losses)),KL_per_label=round_losses))
                del value,feature_bits,feat,teacher;gc.collect();guard()
                print(json.dumps(dict(stage=stage,id=current,completed=len(rounding),labels=count,seconds=time.monotonic()-start)),flush=True)
        arms=[dict(name=name,records=rr,aggregates={s:aggregate([r for r in rr if r['split']==s]) for s in ('FIT','DEV')}) for name,rr in rows.items()]
        decision,flags=alignment(arms,b['source_label_gate']);guard();assert count==8808
        out=dict(schema='SOURCE_FINAL_READOUT_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,arms=arms,rounding=rounding,
            decision=decision,alignment_flags=flags,reference_max_absolute_delta=ref_peak,BF16_source_head_calls=count,F64_head_contractions=count,
            source_history_forwards=0,source_generations=0,model_instances=0,optimizer_updates=0,native_calls=0,reserved_queries=0,
            GPU_allocated_peak=torch.cuda.max_memory_allocated(),GPU_reserved_peak=torch.cuda.max_memory_reserved(),
            worker_OS_peak_snapshot=proc.memory_info().peak_wset,seconds=time.monotonic()-start,quality_admission=False,speed_admission=False,
            useful_large_n_admission=False,physical_DRAM_bytes=None,scope=b['scope'],
            limits='Cached generation labels versus full-prefill states may differ. Only final operators reconstructed; no whole source or compact model qualification. '
                'Inherited capture/runtime/numerical/quality gaps retained; F64 uses actual saved BF16 normalized features.')
        write(a.out,out);print(json.dumps(dict(stage='complete',decision=decision,seconds=out['seconds'])),flush=True)
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,id=current,error=repr(error),completed_cases=[r['id'] for r in rows['F64']],
            completed_head_pairs=count,seconds=time.monotonic()-start,source_history_forwards=0,optimizer_updates=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bind',action='store_true');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze');p.add_argument('--directory',type=Path)
    a=p.parse_args();bind(a) if a.bind else worker(a)
