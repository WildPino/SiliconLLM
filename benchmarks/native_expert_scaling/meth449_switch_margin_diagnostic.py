"""Saved447/448 decisions: paired source-head state/A16/rounding decomposition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_info,threadpool_limits
import meth449_switch_margin_math as H

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
PROTOCOL=DOC/'METH_449_SWITCH_MARGIN_PROTOCOL_20261005.md'
OUT=ROOT/'results/native_expert_scaling/meth449_switch_margin'
PARENTS={'meth447_switch_i4_result.json':'3025bb7b383a631162c2eca803784e0c2cdc811c47a6abe1f73619a791e6dd29',
         'meth448_switch_block_i4_result.json':'c5d26a3e8192ae6c648567134290783192250ed7a92730cafb25db44440e516c'}
EXPORT=('meth380_switch_base128_export_result.json','6be77281f7ecca73c47eab7f318fed2cb27699f18071b3f3781ab9bce54881ee')
GAP_EDGES=(0.,.001,.01,.1,1.,float('inf'))


def committed(path):
    path=Path(path);rel=path.resolve().relative_to(ROOT).as_posix()
    body=subprocess.check_output(['git','-c','core.autocrlf=false','cat-file','--filters','--path='+rel,'HEAD:'+rel],cwd=ROOT)
    assert path.read_bytes()==body,('uncommitted_physical_source',rel)


def write(path,data):
    path.write_bytes((json.dumps(data,indent=2,allow_nan=False)+'\n').replace('\n','\r\n').encode('utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();numeric=None;peak=hashed=0;stage='bindings'
    result={'experiment':'METH-449-saved-row-block-I4-margin-state-A16-head-decomposition','arms':{}}
    def guard():
        nonlocal peak
        mi=psutil.Process().memory_info();peak=max(peak,mi.rss,getattr(mi,'peak_wset',0))
        size=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
        elapsed=time.monotonic()-start
        assert peak<=512<<20 and size<=16<<20,'512MiB_16MiB'
        assert elapsed<=360 and ((numeric is None and elapsed<=300) or
               (numeric is not None and time.monotonic()-numeric<=60)),'admission300_numeric60_total360'
    def sha(path):
        nonlocal hashed
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while block:=f.read(4<<20):h.update(block);hashed+=len(block);guard()
        return h.hexdigest()
    try:
        result['helper_sha256']={}
        for p in (Path(__file__),Path(H.__file__),PROTOCOL):
            committed(p);result['helper_sha256'][str(p)]=sha(p)
        assert subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'],cwd=ROOT).decode().strip()=='research/native-expert-scaling'
        parents={};seen=dict(result['helper_sha256']);files={};binaries={}
        for name,expected in PARENTS.items():
            p=DOC/name;committed(p);assert sha(p)==expected;parents[name]=r=json.loads(p.read_text(encoding='utf-8'))
            assert all(r['apparatus_gates'].values()) and sum(r['feasibility_gates'].values())==4 and not r['feasibility_gates']['argmax_changed_fraction_le0_01']
            for p,e in r['helper_sha256'].items():
                if p in seen:assert seen[p]==e;continue
                committed(p);assert sha(p)==e;seen[p]=e
            for a in r['output_inventory']:
                assert sha(a['path'])==a['sha256'];files[a['path']]=a['sha256']
            for mapping in (r['runtime']['binary_sha256'],r['preserved_original_binary_sha256']):
                for p,e in mapping.items():
                    if p in binaries:assert binaries[p]==e;continue
                    assert sha(p)==e;binaries[p]=e
        rowparent=parents['meth447_switch_i4_result.json'];blockparent=parents['meth448_switch_block_i4_result.json']
        assert rowparent['controls']['I4_original_ID']['argmax_changed_positions']==4 and blockparent['controls']['I4B64_original_ID']['argmax_changed_positions']==9
        assert rowparent['data']['keys']==blockparent['data']['keys'] and rowparent['artifacts']==blockparent['artifacts']
        result['retained_record_sha256']=PARENTS;result['retained_output_sha256']=files;result['preserved_binary_sha256']=binaries
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['name','cmdline']):
            if process.pid in own:continue
            name=(process.info['name'] or '').lower();argv=process.info['cmdline'] or []
            if name=='pythonw.exe' and len(argv)==2 and Path(argv[1]).resolve()==Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve():
                result.setdefault('preserved_daemons',[]).append(process.pid);continue
            assert not(name.startswith('python') or ('meth' in name and name.endswith('.exe'))),('concurrent_job',process.pid,name)
        psutil.Process().cpu_affinity([0]);assert np.__version__=='2.4.6' and psutil.disk_usage(str(ROOT)).free>=1<<30
        name,expected=EXPORT;p=DOC/name;committed(p);assert sha(p)==expected;export=json.loads(p.read_text(encoding='utf-8'))
        artifact=export['artifact'];assert artifact==rowparent['artifacts']['128']
        assert sha(artifact['payload'])==artifact['sha256'] and sha(artifact['manifest'])==artifact['manifest_sha256']
        result['artifact']=artifact;result['export_record_sha256']={name:expected}
        initial=(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns)
        mapped=np.memmap(artifact['payload'],mode='r',dtype='u1');entry=export['tensors']['lm_head.weight']
        assert entry['shape']==[32128,768] and entry['encoding']==1 and entry['elements']==32128*768
        head=np.frombuffer(mapped,dtype=np.int8,count=entry['elements'],offset=entry['offset']).reshape(32128,768)
        headscales=np.frombuffer(mapped,dtype=np.float32,count=32128,offset=entry['scale_offset'])
        assert not np.any(head==-128) and np.isfinite(headscales).all() and np.all(headscales>0)
        def find(r,name):return next(Path(a['path']) for a in r['output_inventory'] if Path(a['path']).name==name)
        original=np.load(find(rowparent,'original_logits.npy'),mmap_mode='r',allow_pickle=False)
        with np.load(find(rowparent,'original_prefixes.npz'),allow_pickle=False) as z:rows=z['rows'].copy();keys=z['keys'].tolist()
        assert keys==rowparent['data']['keys'] and original.shape==(336,32128) and original.dtype==np.float32
        result['data']={'keys':keys,'books':rowparent['data']['books'],'selected_ids':rowparent['data']['selected_ids'],'positions':336,'consumed_diagnostic':True}
        numeric=time.monotonic();OUT.mkdir(parents=True);result['admission']={'seconds':numeric-start,'bytes_hashed':hashed}
        print(json.dumps({'admission_complete':result['admission']}),flush=True)
        with threadpool_limits(limits=1):
            result['runtime']={'numpy':np.__version__,'CPU_affinity':[0],'BLAS':threadpool_info(),'GPU':False,'Torch_imported':False}
            result['runtime']['binary_sha256']={a['filepath']:sha(a['filepath']) for a in result['runtime']['BLAS']}
            stage='tiny_algebra_qualification';result['tiny_qualification']=H.tiny_qualification()
            arrays={'original_head_input':rows['head_input'].copy(),'original_head_codes':rows['head_codes'].copy(),'original_head_scale':rows['head_scale'].copy()}
            flipsets={}
            for arm,parent,tag in (('rowI4',rowparent,'I4_original_ID'),('block64I4',blockparent,'I4B64_original_ID')):
                stage=arm;candidate=np.load(find(parent,tag+'_logits.npy'),mmap_mode='r',allow_pickle=False)
                with np.load(find(parent,tag+'_states.npz'),allow_pickle=False) as z:hi=z['head_input'].copy()
                assert candidate.shape==original.shape and candidate.dtype==np.float32 and hi.shape==(336,768)
                records=[];weights=np.empty((336,2,768),np.int8);scales=np.empty((336,2),np.float32)
                newcodes=np.empty((336,768),np.int16);newscale=np.empty(336,np.float32)
                for i,(o,c) in enumerate(zip(original,candidate)):
                    a=int(np.argmax(o));winner=int(np.argmax(c));changed=winner!=a
                    temp=o.copy();temp[a]=-np.inf;runner=int(np.argmax(temp));b=winner if changed else runner
                    pair=np.asarray([a,b]);w=head[pair].copy();s=headscales[pair].copy();weights[i]=w;scales[i]=s
                    s0,q0=H.quant(rows['head_input'][i]);assert s0.tobytes()==rows['head_scale'][i].tobytes() and q0.tobytes()==rows['head_codes'][i].tobytes()
                    sc,qc=H.quant(hi[i]);newcodes[i]=qc;newscale[i]=sc
                    no,u0=H.native_rows(w,s,s0,q0);nc,uc=H.native_rows(w,s,sc,qc)
                    assert no.tobytes()==o[pair].tobytes() and nc.tobytes()==c[pair].tobytes(),'selected_native_head_rows'
                    m,d,mgap=H.contrast(o,c,a,b)
                    diff=w[1].astype(np.float64)*np.float64(s[1])-w[0].astype(np.float64)*np.float64(s[0])
                    dh=hi[i].astype(np.float64)-rows['head_input'][i].astype(np.float64)
                    qa=q0.astype(np.float64)*np.float64(s0);qb=qc.astype(np.float64)*np.float64(sc)
                    state=float(np.dot(diff,dh));activation=float(np.dot(diff,(qb-qa)-dh))
                    unrounded=float((uc[1]-uc[0])-(u0[1]-u0[0]));cast=d-unrounded
                    order=unrounded-state-activation
                    assert abs(order)<=1e-10 and abs(d-(state+activation+cast+order))<=1e-10
                    m0float=float(-np.dot(diff,rows['head_input'][i].astype(np.float64)))
                    mcfloat=float(-np.dot(diff,hi[i].astype(np.float64)))
                    assert abs((m0float-mcfloat)-state)<=1e-10
                    info,p=H.information(o,c);assert abs(info['KL']-parent['controls'][tag]['per_position_KL'][i])<=1e-10
                    g=float(np.float64(o[a])-np.float64(o[runner]));pgap=float(p[a]-p[runner]);eps=info['centered_logit_Linf_best']
                    certificates={'centered_logit_range':g>2*eps+1e-12,'posterior_TV':pgap>2*info['TV']+1e-12,
                                  'posterior_Pinsker':pgap>np.sqrt(2*(max(info['KL'],0)+1e-10))}
                    assert not changed or not any(certificates.values())
                    classification='unchanged'
                    if changed:
                        assert mgap<=0 and d>=m-1e-12
                        classification=('source_pair_tie_or_disagrees_shadow' if m0float<=1e-10 else
                                        'state_crosses_shadow_pair' if mcfloat < -1e-10 else
                                        'shadow_candidate_pair_preserved' if mcfloat>1e-10 else 'candidate_shadow_near_tie')
                    records.append({'index':i,'original_winner':a,'candidate_winner':winner,'source_runner_up':runner,'paired_competitor':b,
                        'changed':changed,'source_top2_gap':g,'source_competitor_gap':m,'candidate_native_pair_gap':mgap,'native_perturbation_contrast':d,
                        'state_projection':state,'A16_quantization_projection':activation,'F32_head_cast_contrast':cast,'F64_order_residual':order,
                        'source_smooth_pair_gap':m0float,'candidate_smooth_pair_gap':mcfloat,'source_A16_unrounded_pair_gap':float(u0[0]-u0[1]),
                        'candidate_A16_unrounded_pair_gap':float(uc[0]-uc[1]),'source_probability_gap':pgap,
                        'head_state_delta_norm':float(np.linalg.norm(dh)),'head_A16_delta_error_norm':float(np.linalg.norm((qb-qa)-dh)),
                        'certificates_reference_required':certificates,'classification':classification,**info})
                    guard()
                changed=[r for r in records if r['changed']];flipsets[arm]={r['index'] for r in changed}
                assert len(changed)==parent['controls'][tag]['argmax_changed_positions']
                summary={'changed_count':len(changed),'changed_classifications':{k:sum(v['classification']==k for v in changed) for k in ('source_pair_tie_or_disagrees_shadow','state_crosses_shadow_pair','shadow_candidate_pair_preserved','candidate_shadow_near_tie')},
                    'reference_certificate_counts':{k:sum(r['certificates_reference_required'][k] for r in records) for k in ('centered_logit_range','posterior_TV','posterior_Pinsker')},
                    'source_top2_gap_bins':{f'[{lo},{hi_edge})':{'positions':sum(lo<=r['source_top2_gap']<hi_edge for r in records),'changes':sum(lo<=r['source_top2_gap']<hi_edge and r['changed'] for r in records)} for lo,hi_edge in zip(GAP_EDGES[:-1],GAP_EDGES[1:])},
                    'max_abs_F64_order_residual':max(abs(r['F64_order_residual']) for r in records)}
                result['arms'][arm]={'summary':summary,'per_position':records}
                arrays.update({arm+'_head_input':hi,arm+'_A16_codes':newcodes,arm+'_A16_scale':newscale,arm+'_selected_head_codes':weights,arm+'_selected_head_scales':scales})
                print(json.dumps({'arm':arm,**summary}),flush=True)
            result['overlap']={'both_changed':sorted(flipsets['rowI4']&flipsets['block64I4']),'row_only':sorted(flipsets['rowI4']-flipsets['block64I4']),'block_only':sorted(flipsets['block64I4']-flipsets['rowI4'])}
            np.savez(OUT/'margin_inputs.npz',**arrays)
            with np.load(OUT/'margin_inputs.npz',allow_pickle=False) as saved:
                for key,v in arrays.items():assert saved[key].tobytes()==v.tobytes()
            result['apparatus_gates']={'all_source_parents_complete_archives_helpers_runtime_fresh':True,'same336_original_logits_inputs_pair_keys':True,
                'all_original_A16_codes_scale_byte_exact':True,'all2688_selected_head_logits_I64_native_byte_exact':True,
                'all672_margin_and_KL_cumulant_identities_qualified':True,'all672_state_A16_F32_cast_order_decompositions_le1e_minus10':True,
                'all_reference_required_certificates_exclude_actual_flips':True,'tiny_independent_algebra_manual_primal_and_saved_archive_exact':True}
        assert initial==(Path(artifact['payload']).stat().st_size,Path(artifact['payload']).stat().st_mtime_ns)
        result['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.glob('*')) if p.is_file()]
        result['resource']={'seconds':time.monotonic()-start,'admission_seconds':numeric-start,'numeric_and_reporting_seconds':time.monotonic()-numeric,
            'peak_bytes':peak,'bytes_hashed':hashed,'output_bytes':sum(a['bytes'] for a in result['output_inventory']),'optimizer_updates':0,
            'expert_FFN_forwards':0,'complete_head_forwards':0,'selected_head_rows_replayed':2688}
        result['scope']='Consumed fixed-prefix saved447/448 diagnosis only. Selected native head rows independently replayed, no complete forward/new model/fit/quality-gate change. Shadow classifications concern declared winner/competitor pairs, not full smooth-head argmax. Reference-required certificates not deployed fallback/selector. No native C/LUT/rate/DRAM/useful-n/family100B proof.'
        guard();write(args.out,result);print(json.dumps({'apparatus':result['apparatus_gates'],'overlap':result['overlap'],'resource':result['resource'],'sha256':sha(args.out)}),flush=True)
    except BaseException as error:
        result.update(failure_stage=stage,error=repr(error),seconds=time.monotonic()-start,peak_bytes=peak,bytes_hashed=hashed)
        write(args.out.with_suffix('.failure.json'),result);raise


if __name__=='__main__':main()
