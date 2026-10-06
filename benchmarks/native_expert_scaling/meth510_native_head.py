"""ONE compiled native510 branch: controls, complete retained parity, paired cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import meth510_operations as O

def wire(row,n,np):
    data=Path(row['wire']['path']).read_bytes();assert data[:8]==b'SWR32O01'
    s,t,d,e,de,v,nr=struct.unpack_from('<7I',data,8)
    assert (s,d,e,de,v,nr)==(29,768,12,12,32128,174+6*t) and n<=t<=64
    enc=14*29*768;dec=t*14*768;assert len(data)==36+4*(enc+dec+t*32128)+12*nr
    states=np.frombuffer(data,dtype='<f4',offset=36+4*enc,count=dec).reshape(t,14,768)[:n,13]
    logits=np.frombuffer(data,dtype='<f4',offset=36+4*(enc+dec),count=t*32128).reshape(t,32128)[:n]
    assert np.isfinite(states).all() and np.isfinite(logits).all();return states,logits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);args=ap.parse_args();ctx=O.Context('main')
    try:
        b=ctx.admit(args.binding_sha);os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        import numpy as np
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        oracle=json.loads(Path(b['oracle509']['path']).read_bytes());assert oracle['K']==8 and oracle['summary']['counts']['positions']==996
        packed=ctx.out/'head_inputs.bin';positions=[]
        with packed.open('xb') as f:
            f.write(b'SW510X01'+struct.pack('<2I',996,768))
            for i,(row,c) in enumerate(zip(b['selected'],oracle['cases'])):
                assert (row['book'],row['index'])==(c['book'],c['index'])==divmod(i,4)
                states,_=wire(row,len(c['positions']),np)
                for j,state in enumerate(states):f.write(struct.pack('<2I',i,j));f.write(state.tobytes());positions.append((i,j))
        assert len(positions)==996 and packed.stat().st_size==b['price']['head_input_bytes']
        input_sha=ctx.digest(packed);binary=ctx.out/'meth510.exe';shutil.copyfile(b['libomp']['path'],ctx.out/'libomp.dll')
        argv=[b['compiler']['path'],'-O3','-std=c11','-march=x86-64-v3','-fno-fast-math','-ffp-contract=off','-fopenmp',
              '-DSILICON_SWITCH_HYBRID_HEAD',str(O.ROOT/'benchmarks/phase60/engine.c'),'-lpsapi','-o',str(binary)]
        ctx.run(argv,'compile');binary_sha=ctx.digest(binary)
        positive=ctx.run([binary,'--head-contract','positive'],'positive');assert len(positive)==1
        control=json.loads(positive[0]);assert control['contract'] and control['tail_winner']==8
        assert control['I64_totals']==[-128*32767*4096,-127*32767*4096,-255*32767*2048]
        for mode in ('rows','cols','dtype','source','rounding'):
            lines=ctx.run([binary,'--head-contract',mode],'negative_'+mode,expected=2);assert lines==[]
            assert (ctx.out/('negative_'+mode+'.stderr')).read_bytes().replace(b'\r\n',b'\n')==b'switch_reference_error:hybrid_contract510\n'
        def workers(r):
            assert r['worker_physical_cores']==6
            assert [(v['slot'],v['group'],v['actual_mask']) for v in r['worker_affinity']]==[(0,0,1),(1,0,4),(2,0,16)]
            assert len(set(v['windows_thread_id'] for v in r['worker_affinity']))==3
            assert all(v['group']==0 and v['actual_mask']==[1,4,16][v['slot']] for v in r['worker_binding_events'])
        workers(control);ctx.r['gates']['native_integer_extremes_zero_even_ties_top8_tail_and_five_rejections']=True
        output=ctx.out/'head_outputs.bin'
        lines=ctx.run([binary,'--retained-head',b['source_head']['manifest']['path'],packed,output],'paired_head',timing=True)
        records=[json.loads(line) for line in lines];terminal=records.pop();assert terminal['terminal'];workers(terminal)
        assert all(terminal[k]==v for k,v in {'positions':996,'warmups':2,'repetitions':3,'I8_calls':9960,'hybrid_calls':4980,
                       'source_rows':39840,'source_MACs':30597120,'extra_f32_logical_bytes':122388480}.items())
        assert output.stat().st_size==b['price']['head_output_bytes'];assert len(records)==4980
        measured=[]
        for k,r in enumerate(records):
            p=k//996-2;i=k%996;case,step=positions[i]
            assert (r['pass'],r['position'],r['case'],r['step'])==(p,i,case,step)
            assert r['order']==('BA' if p in (-1,1) else 'AB') and r['old_seconds']>0 and r['hybrid_seconds']>0
            assert 0<r['old_matrix_seconds']<=r['old_seconds']+.000000002
            assert 0<r['hybrid_matrix_seconds']<=r['hybrid_seconds']+.000000002
            assert r['scan_seconds']>=0 and r['refine_seconds']>=0 and 32120<=r['comparisons']<=32128*8
            if p>=0:measured.append(r)
        parity={'positions':0,'old_changed_cells':0,'hybrid_changed_cells':0,'selected_ID_mismatches':0,'unselected_tail_changed_cells':0,
                'hybrid_winner_mismatches':0,'recoveries':0,'introductions':0,'remaining_differences':0};cases=[];offset=0
        with output.open('rb') as f:
            assert f.read(24)==b'SW510H01'+struct.pack('<4I',996,32128,768,8)
            for i,(row,c) in enumerate(zip(b['selected'],oracle['cases'])):
                _,old_expected=wire(row,len(c['positions']),np);ps=[]
                for j,p in enumerate(c['positions']):
                    old=np.frombuffer(f.read(128512),dtype='<f4');hybrid=np.frombuffer(f.read(128512),dtype='<f4')
                    selected=np.frombuffer(f.read(32),dtype='<i4');dots=np.frombuffer(f.read(64),dtype='<f8')
                    assert np.isfinite(old).all() and np.isfinite(hybrid).all() and np.isfinite(dots).all()
                    chosen=np.array(p['chosen_original_IDs'],dtype='<i4');expected=old_expected[j].copy();expected[chosen]=np.array(p['chosen_refined_F32_logits'],dtype='<f4')
                    oldbad=int(np.count_nonzero(old.view('u4')!=old_expected[j].view('u4')))
                    hybad=int(np.count_nonzero(hybrid.view('u4')!=expected.view('u4')))
                    mask=np.ones(32128,dtype=bool);mask[chosen]=False
                    tailbad=int(np.count_nonzero(hybrid.view('u4')[mask]!=old.view('u4')[mask]));winner=int(hybrid.argmax());donor=p['donor_winner']
                    idbad=int(selected.tobytes()!=chosen.tobytes());newbad=winner!=donor;oldbadid=int(old.argmax())!=donor
                    parity['positions']+=1;parity['old_changed_cells']+=oldbad;parity['hybrid_changed_cells']+=hybad
                    parity['selected_ID_mismatches']+=idbad;parity['unselected_tail_changed_cells']+=tailbad;parity['hybrid_winner_mismatches']+=int(winner!=p['hybrid_winner'])
                    parity['recoveries']+=int(oldbadid and not newbad);parity['introductions']+=int(not oldbadid and newbad);parity['remaining_differences']+=int(newbad)
                    ps.append({'step':j,'position':offset,'old_cell_mismatches':oldbad,'hybrid_cell_mismatches':hybad,'selected_ID_mismatch':idbad,
                        'tail_cell_mismatches':tailbad,'old_winner':int(old.argmax()),'hybrid_winner':winner,'donor_winner':donor,
                        'selected_IDs':selected.tolist(),'source_F64_dots':dots.tolist(),'hybrid_F32_selected':hybrid[selected].tolist(),
                        'old_logits_sha256':hashlib.sha256(old.tobytes()).hexdigest(),'hybrid_logits_sha256':hashlib.sha256(hybrid.tobytes()).hexdigest()});offset+=1
                cases.append({'book':row['book'],'index':row['index'],'positions':ps});ctx.guard()
                if i%4==3:ctx.log(book=row['book'],positions=offset)
            assert f.read()==b'' and offset==996
        extra=np.array([r['hybrid_seconds']-r['old_seconds'] for r in measured],dtype='f8')
        oldtimes=np.array([r['old_seconds'] for r in measured],dtype='f8');newtimes=np.array([r['hybrid_seconds'] for r in measured],dtype='f8')
        cost={'paired_samples':2988,'mean_old_seconds':float(oldtimes.mean()),'mean_hybrid_seconds':float(newtimes.mean()),'mean_extra_seconds':float(extra.mean()),
              'p50_extra_seconds':float(np.quantile(extra,.5)),'p95_extra_seconds':float(np.quantile(extra,.95)),'p99_extra_seconds':float(np.quantile(extra,.99)),
              'maximum_extra_seconds':float(extra.max()),'mean_ratio':float(newtimes.sum()/oldtimes.sum()),
              'mean_scan_seconds':float(np.mean([r['scan_seconds'] for r in measured])),'mean_refine_seconds':float(np.mean([r['refine_seconds'] for r in measured])),
              'books':[{'book':book,'samples':sum(r['case']//4==book for r in measured),
                        'mean_extra_seconds':float(np.mean([r['hybrid_seconds']-r['old_seconds'] for r in measured if r['case']//4==book]))} for book in range(24)]}
        gates={'ALL996_original_I8_logits_BYTE':parity['old_changed_cells']==0,
               'ALL996_hybrid_full_logits_selected_IDs_509_winners_BYTE':parity['hybrid_changed_cells']==parity['selected_ID_mismatches']==parity['hybrid_winner_mismatches']==0,
               'ALL996_unselected_tail_unchanged':parity['unselected_tail_changed_cells']==0,
               'mean_extra_le_1ms':cost['mean_extra_seconds']<=.001,'p95_extra_le_2ms':cost['p95_extra_seconds']<=.002}
        summary={'parity':parity,'cost':cost,'local_gates':gates,'local_recipe_pass':all(gates.values()),'goal_complete':False,
                 'scope':'Consumed-state native arithmetic and instrumented paired head cost only; no new quality, model rate, physical DRAM, useful n or family promotion.'}
        ctx.r['gates']['native_worker_layout_command_count_complete_paired_passes_and_counter_counts']=len(ctx.r['commands'])==8
        ctx.finish({'input_sha256':input_sha,'binary_sha256':binary_sha,'native_output_sha256':ctx.digest(output),'control':control,'native_terminal':terminal,'cases':cases,'summary':summary})
    except BaseException:ctx.fail();raise
    finally:ctx.close()
if __name__=='__main__':main()
