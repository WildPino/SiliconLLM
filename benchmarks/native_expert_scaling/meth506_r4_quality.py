"""Donor-only recovery using original qualified4.57.6 APIs and immutable C files."""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import meth506_r4_operations as O

def trim():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessWorkingSetSize.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
    assert kernel.SetProcessWorkingSetSize(kernel.GetCurrentProcess(),ctypes.c_size_t(-1).value,ctypes.c_size_t(-1).value)

def worker(row):
    assert row['worker_physical_cores']==6 and [v['slot'] for v in row['worker_affinity']]==[0,1,2]
    assert [v['actual_mask'] for v in row['worker_affinity']]==[1,4,16]
    assert all(v['group']==0 for v in row['worker_affinity']) and len({v['windows_thread_id'] for v in row['worker_affinity']})==3
    assert all(v['actual_mask']==[1,4,16][v['slot']] and v['group']==0 for v in row['worker_binding_events'])
    assert set(v['slot'] for v in row['worker_binding_events'])=={0,1,2}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--preparation-sha',required=True);args=ap.parse_args()
    prepraw=O.PREPRAW;ctx=O.Context(O.OUT,O.RAW)
    try:
        b=ctx.admit(args.binding_sha)
        assert ctx.digest(prepraw)==args.preparation_sha;ctx.head(prepraw);prep=json.loads(prepraw.read_bytes());assert all(prep['gates'].values())
        for v in prep['output_inventory']:ctx.exact(v)
        ctx.r['preparation_sha256']=args.preparation_sha
        os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits,threadpool_info
        from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
        import meth506_artifact as A
        import meth506_cohort as C
        import meth506_quality as Q
        import meth506_results as R
        pool=threadpool_limits(limits=1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and not torch.cuda.is_initialized()
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.r['numerical_imports']=True;ctx.r['runtime']={'pools':threadpool_info(),'torch':torch.__version__,'torch_num_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'GPU_initialized':torch.cuda.is_initialized(),'environment':b['runtime_environment']}
        ctx.import_diagnostic();ctx.parent_modules()
        assert __import__('transformers').__version__=='4.57.6'
        import inspect
        assert ctx.digest(inspect.getfile(SwitchTransformersForConditionalGeneration))=='5acb527d50a07e4bce3a5dec4db4f0c476cb4a01e8eb05219f26607034a12a83'
        artifact=prep['artifact'];cohort=json.loads((O.PREP/'cohort.json').read_bytes());assert ctx.digest(O.PREP/'cohort.json')==prep['cohort_sha256']
        failed=json.loads((O.DOC/'meth506_whole_result.failure.json').read_bytes());assert len(failed['commands'])==578 and all(v['returncode']==0 for v in failed['commands'])
        controls=json.loads((O.NATIVE/'column-controls.stdout').read_text(encoding='utf8'));worker(controls);assert controls['controls']
        cases=[]
        for bi in range(24):
            for ci in range(4):
                record=json.loads((O.NATIVE/f'b{bi:02d}c{ci}.native.json').read_bytes());assert record['book']==bi and record['index']==ci and record['case']==cohort['items'][bi]['cases'][ci]
                assert record['native']['source']['teacher'][0]['wire']['sha256']==record['native']['candidate']['teacher'][0]['wire']['sha256']
                sha={v['wire']['sha256'] for arm in record['native'].values() for v in arm['generation']+arm['profile']};assert len(sha)==1
                for arm in record['native'].values():
                    for rows in arm.values():
                        for row in rows:ctx.exact(row['wire']);worker(row)
                cases.append(record)
        ctx.r['gates']['all96_immutable_original_native_cases_and_repetitions']=True
        # Native timings precede all F32 donor inference; no mapped donor competes with timings.
        ctx.phase='fresh-original-F32-canonical-source';bound=json.loads((O.DOC/'meth379_switch_tensor_binding_result.json').read_bytes());sourcecfg=json.loads((O.DOC/'meth380_switch_base128_export_result.json').read_bytes())['original_config']
        os.environ.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**sourcecfg))
        namespace=model.state_dict();assert set(namespace)==set(bound['tensors']);seen=set();shards=[]
        for path in sorted(Path(b['donor_directory']).glob('pytorch_model-*-of-*.bin')):
            start=time.monotonic();state=torch.load(path,weights_only=True,mmap=True,map_location='cpu');assert not seen.intersection(state)
            for ordinal,(name,value) in enumerate(sorted(state.items(),key=lambda item:item[1].data_ptr())):
                expected=bound['tensors'][name];assert value.dtype==torch.float32 and value.is_contiguous() and list(value.shape)==expected['shape']==list(namespace[name].shape)
                assert hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()==expected['sha256'],name;seen.add(name)
                if ordinal%32==31:trim()
                ctx.guard()
            missing=model.load_state_dict(state,strict=False,assign=True);assert not missing.unexpected_keys
            del state;gc.collect();trim();shards.append({'path':str(path),'tensor_names_so_far':len(seen),'seconds':time.monotonic()-start});ctx.log(canonical_shard=path.name,tensors=len(seen))
        assert len(seen)==3320;model.tie_weights();model.requires_grad_(False);model.eval();assert sum(p.numel() for p in model.parameters())==7415217408
        assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters())
        ctx.r['gates']['all3320_actual_original_F32_coefficients_and_unique_parameter_count']=True
        ctx.phase='fresh-donor-own-states-quality'
        for ordinal,record in enumerate(cases):
            case=record['case'];source=torch.tensor([case['source_ids']]);decoder=torch.tensor([case['decoder_ids']]);label=f'b{record["book"]:02d}c{record["index"]}'
            ref=Q.torch_reference(model,source,decoder);ctx.guard();official_teacher=Q.official_teacher_cached(model,source,case['decoder_ids']);err=Q.relative(official_teacher,ref[2])
            assert err<=1e-6 and np.array_equal(official_teacher.argmax(-1),ref[2].argmax(-1))
            donor_ids,gen=Q.cached_greedy(model,source,64,32095);ctx.guard();official_ids,official_gen=Q.official_greedy(model,source,64,32095);gerr=Q.relative(official_gen,gen[2])
            assert donor_ids==official_ids and gerr<=1e-6 and np.array_equal(official_gen.argmax(-1),gen[2].argmax(-1))
            assert ref[0].tobytes()==gen[0].tobytes()
            saved=ctx.out/(label+'.donor.npz');assert not saved.exists()
            np.savez(saved,teacher_encoder=ref[0],teacher_decoder=ref[1],teacher_logits=ref[2],teacher_routes=ref[3],official_teacher_logits=official_teacher,generation_encoder=gen[0],generation_decoder=gen[1],generation_logits=gen[2],generation_routes=gen[3],official_generation_logits=official_gen)
            actual=Q.read_output(Path(record['native']['candidate']['teacher'][0]['wire']['path']))
            record.update(donor_reference={'path':str(saved),'bytes':saved.stat().st_size,'sha256':ctx.digest(saved)},donor_prediction=Q.metrics(ref[2],case['target_ids'],case['masked_spans_ids']),candidate_prediction=Q.metrics(actual[2],case['target_ids'],case['masked_spans_ids']),donor_task=Q.task(donor_ids,case['masked_spans_ids']),top1_agreement=float(np.mean(actual[2].argmax(-1)==ref[2].argmax(-1))),masked_top1_agreement=float(np.mean(actual[2].argmax(-1)[Q.MASK]==ref[2].argmax(-1)[Q.MASK])),changed_donor_route_choices=int(np.sum(actual[3][:,0]!=ref[3][:,0])),official_teacher_relative_l2=err,official_generation_relative_l2=gerr)
            record['normalized_prose_edit']=Q.edit_distance(record['candidate_task']['prose_ids'],record['donor_task']['prose_ids'])/max(len(record['candidate_task']['prose_ids']),len(record['donor_task']['prose_ids']),1)
            O.write(ctx.out/(label+'.quality.json'),record);del ref,gen,actual,official_teacher,official_gen;gc.collect();trim();ctx.guard()
            if ordinal%4==3:ctx.log(donor_book_terminal=record['book']);print(json.dumps({'phase':ctx.phase,'book':record['book'],'seconds':time.monotonic()-ctx.start}),flush=True)
        ctx.r['gates']['all96_official_teacher_and_own_generate_bridges_and_fresh_tasks']=True
        ctx.phase='whole-frozen-decision';summary=R.summarize(cases);del model,namespace;gc.collect();trim();assert not torch.cuda.is_initialized()
        summary['observed_numerical_recipe_gates_pass']=summary['whole_recipe_eligible']
        summary['whole_recipe_eligible']=False
        ctx.parent_modules()
        r=ctx.finish({'artifact':artifact,'cohort_sha256':prep['cohort_sha256'],'controls':controls,'binary_sha256':ctx.digest(O.NATIVE/'meth506.exe'),'source_shards':shards,'commands':failed['commands'],'original_native_process_scope':'Inherited original sole578 processes; ZERO executed by R4; their actual original Windows terminal record is bound separately.','cases':cases,'summary':summary,'new_compile_native_calls':0,'original_failure_sha256':ctx.digest(O.DOC/'meth506_whole_result.failure.json'),'timing_isolation':{'strictly_verified':False,'precall_waits':failed['isolation_waits'],'reason':'Original pre-call guard waited on two transient Python groups; command/create/end identity was not retained for those groups. No continuous in-call foreign-process monitor. Strict no-foreign-scientific-timing provenance is unverified; retain observed rates/costs, do not promote to final goal rate.'},'physical_DRAM':{'verified':False,'reason':'No qualified actual memory-controller counter available in this frozen path; logical code/scale byte counters are not DRAM.'},'scope':'Exploratory ALL12 source128 whole artifact, unchanged505 integer kernel/precision/block; fixed3 physical workers; fresh project-excluded tasks; F32 original donor. Source-sized WI and unchanged128 parents are not compact-core/useful-n/LUT/general-family completion.'})
        print(json.dumps({'raw':str(O.RAW),'sha256':ctx.digest(O.RAW),'gates':r['gates'],'quality':summary['quality_gates'],'economics':summary['economic_gates'],'candidate_rates':summary['rates']['candidate'],'resource':r['resource']}),flush=True)
    except BaseException as e:ctx.fail(e);raise
    finally:ctx.done.set();ctx.watchdog.cancel();ctx.watcher.join(timeout=1);ctx.progress.close();ctx.fatal.close()

if __name__=='__main__':main()
