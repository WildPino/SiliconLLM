"""576 first-only native calls; source/candidate own histories, within-arm byte controls."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth511_operations as O

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--cohort-sha',required=True);args=ap.parse_args();ctx=O.Context('native')
    try:
        b=ctx.admit(args.binding_sha);prep=ctx.import_result('cohort',args.cohort_sha)
        cohort=json.loads((O.ROOT/'results/native_expert_scaling/meth511_cohort/cohort.json').read_bytes())
        assert ctx.digest(O.ROOT/'results/native_expert_scaling/meth511_cohort/cohort.json')==prep['cohort_sha256']
        import numpy as np
        import meth511_quality as Q
        from threadpoolctl import threadpool_limits
        pool=threadpool_limits(limits=1);np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        records=[]
        def native(case,arm,label,generation,profile,warm,reps):
            prefix=ctx.out/label;spec=b[arm+'_manifest']['path'];binary=b[arm+'_binary']['path']
            argv=[binary,'--generate',spec,','.join(map(str,case['source_ids'])),str(prefix),'3','64','32095',str(profile),str(warm),str(reps),'0'] if generation else [binary,spec,','.join(map(str,case['source_ids'])),','.join(map(str,case['decoder_ids'])),str(prefix),'3',str(profile),str(warm),str(reps),'0']
            rows=ctx.run_native(argv,label);assert [v['repetition'] for v in rows]==list(range(-warm,reps))
            for row in rows:
                O.worker(row);assert row['profile']==profile and row['threads']==3
                p=Path(str(prefix)+f'.{row["repetition"]}.bin');a=Q.read_output(p);assert all(np.isfinite(v).all() for v in a)
                assert a[0].shape==(14,29,768) and a[1].shape[1:]==(14,768) and a[2].shape[1]==32128
                assert a[3].shape==(6*(29+len(a[2])),3) and np.all((a[3][:,0]>=0)&(a[3][:,0]<128))
                assert np.isin(a[3][:,1],[0,1]).all() and np.all((a[3][:,2]>0)&(a[3][:,2]<=1))
                if generation:assert a[2].argmax(-1).tolist()==row['generated_ids'] and row['actual_generated_tokens']==len(row['generated_ids'])
                row['wire']={'path':str(p),'bytes':p.stat().st_size,'sha256':ctx.digest(p)}
            return rows
        for bi,book in enumerate(cohort['items']):
            for case in book['cases']:
                ci=case['index'];record={'book':bi,'index':ci,'source_id':book['source_id'],'case':case,'native':{}}
                order=['source','candidate'] if (4*bi+ci)%2==0 else ['candidate','source']
                for arm in order:
                    label=f'b{bi:02d}c{ci}.{arm}'
                    record['native'][arm]={'teacher':native(case,arm,label+'.teacher',False,0,0,1),
                        'generation':native(case,arm,label+'.gen',True,0,1,3),'profile':native(case,arm,label+'.profile',True,1,0,1)}
                    assert len({r['wire']['sha256'] for r in record['native'][arm]['generation']+record['native'][arm]['profile']})==1
                    record[arm+'_task']=Q.task(record['native'][arm]['generation'][0]['generated_ids'],case['masked_spans_ids'])
                s=Q.read_output(Path(record['native']['source']['teacher'][0]['wire']['path']));c=Q.read_output(Path(record['native']['candidate']['teacher'][0]['wire']['path']))
                assert all(s[i].tobytes()==c[i].tobytes() for i in [0,1,3]),('forced_teacher_upstream_BYTE',bi,ci)
                record['teacher_head_changed_cells']=int(np.count_nonzero(s[2].view('u4')!=c[2].view('u4')))
                O.write(ctx.out/f'b{bi:02d}c{ci}.native.json',record);records.append(record);del s,c
            ctx.log(native_book_terminal=bi);print(json.dumps({'phase':'native','book':bi,'seconds':time.monotonic()-ctx.start}),flush=True)
        assert ctx.r['native_calls']==576 and len(records)==96
        ctx.r['gates'].update(all96_forced_teacher_encoder_decoder_normalized_routes_BYTE=True,all96_each_arm5_generation_wires_BYTE=True,
            all576_native_calls_exit0_and_three_actual_physical_workers=True,continuous_sampled_scientific_isolation_idle_exact_CPU_zero=True)
        ctx.finish({'cases':records,'cohort_sha256':prep['cohort_sha256'],'summary':{'cases':96,'native_calls':576,
            'cross_arm_different_generations':sum(r['source_task']['generated_ids']!=r['candidate_task']['generated_ids'] for r in records),
            'teacher_changed_head_cells':sum(r['teacher_head_changed_cells'] for r in records)}})
    except BaseException:ctx.fail();raise
    finally:ctx.close()
if __name__=='__main__':main()
