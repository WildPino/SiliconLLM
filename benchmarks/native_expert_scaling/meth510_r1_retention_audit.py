"""Independent510 retained input/full output/AVX cell/cost reconstruction; no native calls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import psutil
import meth510_r1_operations as O

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--main-sha',required=True);args=ap.parse_args();ctx=O.Context('audit')
    try:
        b=ctx.admit(args.binding_sha);path=O.DOC/'meth510_r1_main_result.json';assert ctx.digest(path)==args.main_sha;m=json.loads(path.read_bytes())
        assert m['binding_sha256']==args.binding_sha
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        out=O.ROOT/'results/native_expert_scaling/meth510_r1_main';output=out/'head_outputs.bin';packed=out/'head_inputs.bin'
        assert ctx.digest(output)==m['native_output_sha256'] and ctx.digest(packed)==m['input_sha256']
        assert ctx.digest(out/'meth510.exe')==m['binary_sha256'];assert ctx.digest(out/'libomp.dll')==b['libomp']['sha256']
        oracle=json.loads(Path(b['oracle509']['path']).read_bytes());W=np.memmap(b['source_head']['path'],dtype='<f4',mode='r',offset=b['source_head']['offset'],shape=(32128,768))
        counts={'positions':0,'old_changed_cells':0,'hybrid_changed_cells':0,'selected_ID_mismatches':0,'unselected_tail_changed_cells':0,
                'hybrid_winner_mismatches':0,'recoveries':0,'introductions':0,'remaining_differences':0}
        allpos=[];comparisons=[];scale=np.float32(1/math.sqrt(768));ids=np.arange(32128)
        with packed.open('rb') as fin,output.open('rb') as fout:
            assert fin.read(16)==b'SW510X01'+struct.pack('<2I',996,768)
            assert fout.read(24)==b'SW510H01'+struct.pack('<4I',996,32128,768,8)
            for i,(row,oraclecase,maincase) in enumerate(zip(b['selected'],oracle['cases'],m['cases'])):
                data=Path(row['wire']['path']).read_bytes();assert data[:8]==b'SWR32O01'
                s,t,d,e,de,v,nr=struct.unpack_from('<7I',data,8);assert (s,d,e,de,v,nr)==(29,768,12,12,32128,174+6*t)
                n=len(oraclecase['positions']);encbytes=14*29*768*4;decbytes=t*14*768*4
                assert len(data)==36+encbytes+decbytes+t*32128*4+nr*12
                for j,(p,mp) in enumerate(zip(oraclecase['positions'],maincase['positions'])):
                    assert fin.read(8)==struct.pack('<2I',i,j);statebytes=data[36+encbytes+(j*14+13)*768*4:36+encbytes+(j*14+14)*768*4]
                    assert fin.read(3072)==statebytes
                    expected_old=np.frombuffer(data,dtype='<f4',count=32128,offset=36+encbytes+decbytes+j*32128*4)
                    oldbytes=fout.read(128512);hybytes=fout.read(128512);selectedbytes=fout.read(32);dotbytes=fout.read(64)
                    old=np.frombuffer(oldbytes,dtype='<f4');hy=np.frombuffer(hybytes,dtype='<f4');selected=np.frombuffer(selectedbytes,dtype='<i4');actualdot=np.frombuffer(dotbytes,dtype='<f8')
                    assert old.size==hy.size==32128 and np.isfinite(old).all() and np.isfinite(hy).all()
                    chosen=np.lexsort((ids,-expected_old.astype('f8')))[:8].astype('<i4')
                    assert chosen.tolist()==p['chosen_original_IDs']
                    # Reconstruct native AVX2 exact lane order independently of C/BLAS.
                    x=(np.frombuffer(statebytes,dtype='<f4')*scale).astype('f4')
                    products=W[selected].astype('f8')*x.astype('f8');products=products.reshape(8,96,8)
                    a=np.zeros((8,4),dtype='f8');c=np.zeros((8,4),dtype='f8')
                    for lane_group in range(96):a+=products[:,lane_group,:4];c+=products[:,lane_group,4:]
                    joined=a+c;calc=np.zeros(8,dtype='f8')
                    for lane in range(4):calc+=joined[:,lane]
                    assert calc.astype('<f8').tobytes()==dotbytes,'native_F64_lane_reduction_BYTE'
                    assert hy[selected].tobytes()==calc.astype('<f4').tobytes(),'native_F32_rounding_BYTE'
                    expected=expected_old.copy();expected[chosen]=np.array(p['chosen_refined_F32_logits'],dtype='<f4')
                    ob=int(np.count_nonzero(old.view('u4')!=expected_old.view('u4')));hb=int(np.count_nonzero(hy.view('u4')!=expected.view('u4')))
                    idbad=int(selectedbytes!=chosen.tobytes());mask=np.ones(32128,dtype=bool);mask[chosen]=False
                    tailbad=int(np.count_nonzero(hy.view('u4')[mask]!=old.view('u4')[mask]))
                    oldwinner=int(np.flatnonzero(old==old.max())[0]);winner=int(np.flatnonzero(hy==hy.max())[0]);donor=p['donor_winner']
                    counts['positions']+=1;counts['old_changed_cells']+=ob;counts['hybrid_changed_cells']+=hb;counts['selected_ID_mismatches']+=idbad
                    counts['unselected_tail_changed_cells']+=tailbad;counts['hybrid_winner_mismatches']+=int(winner!=p['hybrid_winner'])
                    counts['recoveries']+=int(oldwinner!=donor and winner==donor);counts['introductions']+=int(oldwinner==donor and winner!=donor);counts['remaining_differences']+=int(winner!=donor)
                    assert (mp['step'],mp['position'],mp['old_cell_mismatches'],mp['hybrid_cell_mismatches'],mp['selected_ID_mismatch'],mp['tail_cell_mismatches'],mp['old_winner'],mp['hybrid_winner'],mp['donor_winner'])==(j,len(allpos),ob,hb,idbad,tailbad,oldwinner,winner,donor)
                    assert hashlib.sha256(oldbytes).hexdigest()==mp['old_logits_sha256'] and hashlib.sha256(hybytes).hexdigest()==mp['hybrid_logits_sha256']
                    assert selected.tolist()==mp['selected_IDs'] and actualdot.tolist()==mp['source_F64_dots'] and hy[selected].tolist()==mp['hybrid_F32_selected']
                    # Independent prefix order-statistic recurrence: rank(k) at
                    # r=max(rank(k) at r-1,min(value[r],rank(k-1) at r-1)).
                    # cummax(min(value,previous rank(k-1))) gives every prefix.
                    actualchosen=np.lexsort((ids,-old.astype('f8')))[:8]
                    assert actualchosen.tolist()==selected.tolist()
                    values=old.astype('f8');rankvalues=np.maximum.accumulate(values);ge=np.zeros(32128,dtype='i4')
                    for rank in range(8):
                        previous=np.empty(32128,dtype='f8');previous[0]=-np.inf;previous[1:]=rankvalues[:-1]
                        ge+=(previous>=values);rankvalues=np.maximum.accumulate(np.minimum(values,previous))
                    count=int(np.minimum(np.minimum(ids,8),ge+1).sum())
                    comparisons.append(count);allpos.append((i,j));ctx.guard()
                assert n==len(maincase['positions'])
                if i%4==3:ctx.log(audited_book=i//4,positions=len(allpos))
            assert fin.read()==fout.read()==b'' and len(allpos)==996
        assert counts==m['summary']['parity']
        rows=[json.loads(line) for line in (out/'paired_head.stdout').read_text(encoding='utf8').splitlines()];terminal=rows.pop();assert terminal==m['native_terminal']
        assert len(rows)==4980 and terminal['I8_calls']==2*len(rows) and terminal['hybrid_calls']==len(rows)
        assert terminal['source_rows']==len(rows)*8 and terminal['source_MACs']==len(rows)*8*768 and terminal['extra_f32_logical_bytes']==len(rows)*8*768*4
        extra=[];oldtimes=[];newtimes=[];scan=[];refine=[];books=[[] for _ in range(24)]
        for k,r in enumerate(rows):
            pass_no=k//996-2;pos=k%996;case,step=allpos[pos]
            assert (r['pass'],r['position'],r['case'],r['step'],r['comparisons'])==(pass_no,pos,case,step,comparisons[pos])
            assert r['order']==('BA' if pass_no in (-1,1) else 'AB') and r['old_seconds']>0 and r['hybrid_seconds']>0
            if pass_no>=0:
                delta=r['hybrid_seconds']-r['old_seconds'];extra.append(delta);oldtimes.append(r['old_seconds']);newtimes.append(r['hybrid_seconds']);scan.append(r['scan_seconds']);refine.append(r['refine_seconds']);books[case//4].append(delta)
        def avg(v):return math.fsum(v)/len(v)
        ordered=sorted(extra)
        def quantile(q):
            index=(len(ordered)-1)*q;lo=math.floor(index);hi=math.ceil(index);return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)
        cost={'paired_samples':len(extra),'mean_old_seconds':avg(oldtimes),'mean_hybrid_seconds':avg(newtimes),'mean_extra_seconds':avg(extra),
              'p50_extra_seconds':quantile(.5),'p95_extra_seconds':quantile(.95),'p99_extra_seconds':quantile(.99),'maximum_extra_seconds':max(extra),
              'mean_ratio':math.fsum(newtimes)/math.fsum(oldtimes),'mean_scan_seconds':avg(scan),'mean_refine_seconds':avg(refine),
              'books':[{'book':i,'samples':len(v),'mean_extra_seconds':avg(v)} for i,v in enumerate(books)]}
        for k,v in cost.items():
            if k!='books':assert abs(v-m['summary']['cost'][k])<=1e-12,('aggregate_receipt',k)
        for book,record in zip(cost['books'],m['summary']['cost']['books']):
            assert book['book']==record['book'] and book['samples']==record['samples'] and abs(book['mean_extra_seconds']-record['mean_extra_seconds'])<=1e-12
        gates={'ALL996_original_I8_logits_BYTE':counts['old_changed_cells']==0,
               'ALL996_hybrid_full_logits_selected_IDs_509_winners_BYTE':counts['hybrid_changed_cells']==counts['selected_ID_mismatches']==counts['hybrid_winner_mismatches']==0,
               'ALL996_unselected_tail_unchanged':counts['unselected_tail_changed_cells']==0,
               'mean_extra_le_1ms':cost['mean_extra_seconds']<=.001,'p95_extra_le_2ms':cost['p95_extra_seconds']<=.002}
        assert gates==m['summary']['local_gates'] and all(gates.values())==m['summary']['local_recipe_pass']
        commands=m['commands'];assert [r['label'] for r in commands]==['compile','positive',*["negative_"+v for v in ('rows','cols','dtype','source','rounding')],'paired_head']
        assert all(r['returncode']==r['expected_exit'] for r in commands) and len(commands)==8
        positive=json.loads((out/'positive.stdout').read_text(encoding='utf8'));assert positive==m['control']
        assert positive['I64_totals']==[sum([-128*32767]*4096),sum([127*(-32767)]*4096),sum([(-128 if j%2 else 127)*(32767 if j%2 else -32767) for j in range(4096)])]
        for r in commands:
            pid=r['process_instance'];alive=False
            try:p=psutil.Process(pid['pid']);alive=abs(p.create_time()-pid['create_time_unix'])<.002
            except psutil.NoSuchProcess:pass
            assert not alive
            if r['label'].startswith('negative_'):assert (out/(r['label']+'.stderr')).read_bytes().replace(b'\r\n',b'\n')==b'switch_reference_error:hybrid_contract510\n'
            else:assert (out/(r['label']+'.stderr')).stat().st_size==0
        own=m['process_instance']
        try:assert abs(psutil.Process(own['pid']).create_time()-own['create_time_unix'])>=.002
        except psutil.NoSuchProcess:pass
        ctx.r['gates'].update(ALL996_independent_original_snapshots_full_cells_AVX_F64_rounding_BYTE=True,
            ALL4980_paired_orders_scan_counts_aggregate_cost_and_decisions=True,native_controls_command_count_exit_and_PID_creation_closure=True)
        ctx.finish({'main_sha256':args.main_sha,'audited_output_sha256':m['native_output_sha256'],'source_F64_rows_independently_audited':7968,
                    'native_compile_model_calls':0,'independent_cost':cost,'summary':m['summary']})
    except BaseException:ctx.fail();raise
    finally:ctx.close()
if __name__=='__main__':main()
