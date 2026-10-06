"""Original full F32 donor, official API bridges, unchanged quality/economic conjunction."""
import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meth511_r4_operations as O
def trim():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
    k.SetProcessWorkingSetSize.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
    assert k.SetProcessWorkingSetSize(k.GetCurrentProcess(),ctypes.c_size_t(-1).value,ctypes.c_size_t(-1).value)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding-sha',required=True);ap.add_argument('--native-sha',required=True);args=ap.parse_args();ctx=O.Context('donor')
    try:
        b=ctx.admit(args.binding_sha);native=ctx.import_result('native',args.native_sha);assert native['native_calls']==576
        import numpy as np
        import torch
        from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
        from transformers.models.switch_transformers import modeling_switch_transformers
        from threadpoolctl import threadpool_limits,threadpool_info
        import meth506_quality as Q
        import meth511_results as R
        pool=threadpool_limits(limits=1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and not torch.cuda.is_initialized()
        assert ctx.digest(modeling_switch_transformers.__file__)==b['reference_model_code']['sha256']
        np.seterr(over='raise',invalid='raise',divide='raise',under='ignore');ctx.modules()
        ctx.r['runtime']={'pools':threadpool_info(),'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'GPU_initialized':False}
        bound=json.loads((O.DOC/'meth379_switch_tensor_binding_result.json').read_bytes());cfg=json.loads((O.DOC/'meth380_switch_base128_export_result.json').read_bytes())['original_config']
        with torch.device('meta'):model=SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**cfg))
        namespace=model.state_dict();assert set(namespace)==set(bound['tensors']);seen=set();shards=[];canonical_hashed_bytes=0
        for path in sorted(Path(b['donor_directory']).glob('pytorch_model-*-of-*.bin')):
            start=time.monotonic();state=torch.load(path,weights_only=True,mmap=True,map_location='cpu');assert not seen.intersection(state)
            for ordinal,(name,value) in enumerate(sorted(state.items(),key=lambda item:item[1].data_ptr())):
                expected=bound['tensors'][name];assert value.dtype==torch.float32 and value.is_contiguous() and list(value.shape)==expected['shape']==list(namespace[name].shape)
                assert hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()==expected['sha256'],name;seen.add(name)
                canonical_hashed_bytes+=value.numel()*value.element_size()
                if ordinal%32==31:trim()
                ctx.guard()
            missing=model.load_state_dict(state,strict=False,assign=True);assert not missing.unexpected_keys
            del state;gc.collect();trim();shards.append({'path':str(path),'tensor_names_so_far':len(seen),'seconds':time.monotonic()-start})
            ctx.log(canonical_shard=path.name,tensors=len(seen));print(json.dumps({'phase':'donor_load','shard':path.name,'tensors':len(seen)}),flush=True)
        assert len(seen)==3320;model.tie_weights();model.requires_grad_(False);model.eval();assert sum(p.numel() for p in model.parameters())==7415217408
        assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters())
        ctx.r['gates']['all3320_actual_original_F32_coefficients7415217408_unique_parameters']=True
        ctx.r['canonical_F32_bytes_hashed_before_first_inference']=canonical_hashed_bytes
        cases=[]
        for ordinal,record in enumerate(native['cases']):
            case=record['case'];source=torch.tensor([case['source_ids']]);decoder=torch.tensor([case['decoder_ids']]);label=f'b{record["book"]:02d}c{record["index"]}'
            ctx.r['model_calls']+=1;ref=Q.torch_reference(model,source,decoder);ctx.guard()
            ctx.r['model_calls']+=1;official_teacher=Q.official_teacher_cached(model,source,case['decoder_ids']);err=Q.relative(official_teacher,ref[2])
            assert err<=1e-6 and np.array_equal(official_teacher.argmax(-1),ref[2].argmax(-1))
            ctx.r['model_calls']+=1;donor_ids,gen=Q.cached_greedy(model,source,64,32095);ctx.guard()
            ctx.r['model_calls']+=1;official_ids,official_gen=Q.official_greedy(model,source,64,32095);gerr=Q.relative(official_gen,gen[2])
            assert donor_ids==official_ids and gerr<=1e-6 and np.array_equal(official_gen.argmax(-1),gen[2].argmax(-1))
            assert ref[0].tobytes()==gen[0].tobytes();saved=ctx.out/(label+'.donor.npz');assert not saved.exists()
            np.savez(saved,teacher_encoder=ref[0],teacher_decoder=ref[1],teacher_logits=ref[2],teacher_routes=ref[3],official_teacher_logits=official_teacher,
                generation_encoder=gen[0],generation_decoder=gen[1],generation_logits=gen[2],generation_routes=gen[3],official_generation_logits=official_gen)
            actual=Q.read_output(Path(record['native']['candidate']['teacher'][0]['wire']['path']))
            record.update(donor_reference={'path':str(saved),'bytes':saved.stat().st_size,'sha256':ctx.digest(saved)},
                donor_prediction=Q.metrics(ref[2],case['target_ids'],case['masked_spans_ids']),candidate_prediction=Q.metrics(actual[2],case['target_ids'],case['masked_spans_ids']),
                donor_task=Q.task(donor_ids,case['masked_spans_ids']),top1_agreement=float(np.mean(actual[2].argmax(-1)==ref[2].argmax(-1))),
                masked_top1_agreement=float(np.mean(actual[2].argmax(-1)[Q.MASK]==ref[2].argmax(-1)[Q.MASK])),
                changed_donor_route_choices=int(np.sum(actual[3][:,0]!=ref[3][:,0])),official_teacher_relative_l2=err,official_generation_relative_l2=gerr)
            record['normalized_prose_edit']=Q.edit_distance(record['candidate_task']['prose_ids'],record['donor_task']['prose_ids'])/max(len(record['candidate_task']['prose_ids']),len(record['donor_task']['prose_ids']),1)
            O.write(ctx.out/(label+'.quality.json'),record);cases.append(record);del ref,gen,actual,official_teacher,official_gen;gc.collect();trim();ctx.guard()
            if ordinal%4==3:ctx.log(donor_book_terminal=record['book']);print(json.dumps({'phase':'donor_quality','book':record['book'],'seconds':time.monotonic()-ctx.start}),flush=True)
        summary=R.summarize(cases);del model,namespace;gc.collect();trim();assert not torch.cuda.is_initialized()
        ctx.r['gates'].update(all96_official_teacher_and_own_generate_bridges=True,quality_rate_same_candidate_binary_manifest_payload=True)
        ctx.finish({'cohort_sha256':native['cohort_sha256'],'cases':cases,'summary':summary,'source_shards':shards,
            'physical_DRAM':{'verified':False,'reason':'No qualified physical memory-controller counter in this recipe; logical matrix counters are not DRAM.'},
            'scope':'Fixed128 original parents/source-sized WI, exact conditional WO, existing shared F32 head rows. Batch1 bounded infilling, not chat; compact core/useful n/LUT/multiple families remain open.'})
    except BaseException:ctx.fail();raise
    finally:ctx.close()
if __name__=='__main__':main()
