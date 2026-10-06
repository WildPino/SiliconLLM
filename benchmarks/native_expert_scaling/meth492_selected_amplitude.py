"""Development-only exact constant amplitude selection and complete source joins."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import struct
import traceback
from meth492_operations import Context, DOC, ROOT, write

OUT = ROOT / 'results/native_expert_scaling/meth492_selected_amplitude'
RAW = DOC / 'meth492_selected_amplitude_result.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--binding-sha', required=True)
    args = parser.parse_args(); assert args.out.resolve() == RAW.resolve()
    ctx = Context(OUT, RAW, 180, 256 << 20)
    try:
        binding = ctx.admit(args.binding_sha); ctx.binding = binding
        import numpy as np
        import threadpoolctl
        import meth492_amplitude_math as math
        ctx.r['numerical_imports'] = True
        threadpoolctl.threadpool_limits(limits=1)
        pools = threadpoolctl.threadpool_info()
        assert all(p['num_threads'] == 1 for p in pools)
        ctx.r['threadpools'] = pools
        ctx.phase = 'new_exact_rational_controls'
        write(OUT / 'math_controls.json', math.controls())
        ctx.r['gates']['new_exact_rational_boundary_subnormal_empty_and_F32_gap_controls'] = True
        U, Q = 238872, 387036
        ctx.r['source_scan_bytes'] = 0
        def rows(name, magic, width, reserved, count, dtype, block=2048):
            with Path(binding['data'][name]['path']).open('rb') as stream:
                assert stream.read(24) == struct.pack('<8sIIQ', magic, width, reserved, count)
                ctx.r['source_scan_bytes'] += 24
                for start in range(0, count, block):
                    payload = stream.read(min(block, count-start)*width)
                    chunk = np.frombuffer(payload, dtype)
                    assert len(chunk) == min(block, count-start)
                    ctx.r['source_scan_bytes'] += len(payload); ctx.guard()
                    yield start, chunk
                assert not stream.read(1)
        unique_dtype = np.dtype([('meta','<u4',(12,)), ('hashes','u1',(64,)),
                                ('value','<f8',(3,)), ('x','<f4',(768,)), ('scores','<f4',(128,))])
        meta = np.empty((U,12),'<u4'); pbits = np.empty(U,'<u4')
        ctx.phase = 'ALL_original_canonical_selected_probability_and_metadata'
        for start, chunk in rows('unique_inputs.bin', b'M479UNI1', 3720, 768, U, unique_dtype):
            meta[start:start+len(chunk)] = chunk['meta']
            p = chunk['value'][:,1].astype('<f4')
            assert p.astype('<f8').tobytes() == chunk['value'][:,1].tobytes()
            pbits[start:start+len(chunk)] = p.view('<u4')
        del chunk, p
        owner = np.empty((U,8),'<u4'); links = np.empty((Q,13),'<u4')
        for start, chunk in rows('ownership.bin', b'M479OWN1', 32, 0, U, np.dtype(('<u4',(8,)))):
            owner[start:start+len(chunk)] = chunk
        for start, chunk in rows('query_links.bin', b'M479LNK1', 52, 0, Q, np.dtype(('<u4',(13,)))):
            links[start:start+len(chunk)] = chunk
        assert np.array_equal(meta[:,0],np.arange(U)) and np.array_equal(owner[:,0],meta[:,0])
        assert np.array_equal(owner[:,1],meta[:,2]) and np.all(meta[:,2]<12) and np.all(meta[:,7]<128)
        assert np.array_equal(links[:,0],np.arange(Q)) and np.all(links[:,12]<U)
        assert np.all(links[:,2]<192) and np.all(links[:,4]<2) and np.all(links[:,5]<3)
        assert np.array_equal(links[:,5],np.where(links[:,2]<64,0,np.where(links[:,2]<128,1,2)))
        assert np.isin(links[:,10],[0,1]).all()
        assert np.array_equal(links[:,6],meta[links[:,12],2]) and np.array_equal(links[:,9],meta[links[:,12],7])
        source_dtype = np.dtype([('meta','<u4',(12,)),('value','<f8',(3,))])
        ctx.phase = 'ALL_occurrence_source_fields_BYTE_probability_join'
        for start, chunk in rows('source_fields.bin', b'M479SRC1', 72, 0, Q, source_dtype):
            at = links[start:start+len(chunk)]
            assert chunk['meta'].tobytes() == at[:,:12].tobytes()
            p = chunk['value'][:,1].astype('<f4')
            assert p.astype('<f8').tobytes() == chunk['value'][:,1].tobytes()
            assert p.view('<u4').tobytes() == pbits[at[:,12]].tobytes()
        del chunk, p, at
        roles = np.zeros(U,'<u4'); modes = np.zeros(U,'<u4'); accepts = np.zeros(U,'<u4')
        np.bitwise_or.at(roles,links[:,12],np.left_shift(np.uint32(1),links[:,5]))
        np.bitwise_or.at(modes,links[:,12],np.left_shift(np.uint32(1),links[:,4]))
        np.bitwise_or.at(accepts,links[:,12],np.left_shift(np.uint32(1),links[:,10]))
        assert np.array_equal(roles,owner[:,2]) and np.array_equal(modes,owner[:,3]) and np.array_equal(accepts,owner[:,4])
        assert np.array_equal(np.bincount(links[:,12],minlength=U),owner[:,5])
        del roles, modes, accepts
        dev = (owner[:,2]&5)!=0; val = (owner[:,2]&2)!=0
        assert (int(dev.sum()),int(val.sum())) == (159414,79458) and not np.any(dev&val)
        assert np.all(dev|val)
        ctx.r['gates']['ALL_UID_bank_ID_roles_modes_accepted_occurrence_and_probability_BYTE_joins'] = True
        valid = (pbits>0)&(pbits<=math.ONE)
        cellkey = meta[:,2]*128+meta[:,7]
        constants = np.zeros(1536,'<u4'); limits = np.zeros((1536,2),'<u4')
        cells=[]; ctx.r['cells']=cells
        ctx.phase = 'ONE_exact_development_extrema_interval_per_original_cell'
        for bank in range(12):
            ctx.quiet('cell_bank'+str(bank))
            for expert in range(128):
                key=bank*128+expert; ids=np.flatnonzero((cellkey==key)&dev)
                vids=np.flatnonzero((cellkey==key)&val)
                item={'bank':bank,'source_ID':expert,'development_count':len(ids),
                      'consumed_validation_count':len(vids), 'invalid_development_count':int(np.count_nonzero(~valid[ids]))}
                if len(ids) and not np.all(valid[ids]):
                    item.update(decision='INVALID_DEVELOPMENT_PROBABILITY',ideal_constant_exists=None,
                                candidate_bits=None, first_invalid_UID=int(ids[~valid[ids]][0]))
                elif not len(ids):
                    item.update(math.select(None,None))
                else:
                    minimum,maximum=int(pbits[ids].min()),int(pbits[ids].max())
                    item.update(minimum_bits=minimum,maximum_bits=maximum,
                        minimum_UID=int(ids[pbits[ids]==minimum][0]),maximum_UID=int(ids[pbits[ids]==maximum][0]))
                    item.update(math.select(minimum,maximum))
                    if item['candidate_bits'] is not None:
                        constants[key]=item['candidate_bits']; limits[key]=math.accepted_bits(item['candidate_bits'])
                cells.append(item)
            ctx.log(completed_cell_bank=bank);ctx.guard()
        assert len(cells)==1536
        ctx.r['gates']['ALL1536_development_only_exact_intervals_fixed_F32_candidates_and_unsupported_cells'] = True
        # State0 PASS,1 below allowable p (candidate too large),2 above p,
        # state3 no scalar,4 invalid p. Zero candidate bits mean unavailable.
        state=np.full(U,3,'<u4'); available=constants[cellkey]!=0
        state[available&valid]=0
        state[available&valid&(pbits<limits[cellkey,0])]=1
        state[available&valid&(pbits>limits[cellkey,1])]=2
        state[~valid]=4
        uid_dtype=np.dtype([('uid','<u4'),('bank','<u4'),('source_ID','<u4'),('roles','<u4'),
                            ('source_p_bits','<u4'),('candidate_bits','<u4'),('state','<u4')])
        saved=np.empty(U,uid_dtype)
        for field,array in (('uid',meta[:,0]),('bank',meta[:,2]),('source_ID',meta[:,7]),('roles',owner[:,2]),
                            ('source_p_bits',pbits),('candidate_bits',constants[cellkey]),('state',state)):
            saved[field]=array
        with (OUT/'uid_amplitude.bin').open('xb') as stream:
            stream.write(struct.pack('<8sIIQ',b'M492AMP1',28,0,U));stream.write(saved.tobytes())
        del saved
        ctx.phase='ALL_unique_occurrence_book_ID_accepted_reports'
        reports={k:[] for k in ('unique','occurrence','book','source_ID','accepted')}
        def summary(ids):
            values=state[ids]
            return {'count':len(ids),'state_counts':np.bincount(values,minlength=5).tolist(),
                    'first_UID_by_state':[int(ids[values==s].min()) if np.any(values==s) else None for s in range(5)]}
        for bank in [-1,*range(12)]:
            for role,mask in (('ALL',np.ones(U,bool)),('development',dev),('consumed_validation',val)):
                at=mask if bank==-1 else mask&(meta[:,2]==bank)
                reports['unique'].append({'bank':bank,'role':role,**summary(np.flatnonzero(at))})
        occstate=state[links[:,12]]
        base=links[:,6]*6+links[:,5]*2+links[:,4]
        keysets={'occurrence':(base,72), 'book':(links[:,6]*384+links[:,2]*2+links[:,4],4608),
                 'source_ID':(base*128+links[:,9],9216),'accepted':(base*2+links[:,10],144)}
        for kind,(keys,size) in keysets.items():
            counts=np.empty((size,5),'<u4');first=np.full((size,5),0xffffffff,'<u4')
            for s in range(5):
                take=occstate==s
                counts[:,s]=np.bincount(keys[take],minlength=size)
                np.minimum.at(first[:,s],keys[take],links[take,12])
            for key in range(size):
                if kind=='book':
                    bank,remainder=divmod(key,384);book,mode=divmod(remainder,2)
                    labels={'bank':bank,'book':book,'mode':mode}
                else:
                    parent=key;labels={}
                    if kind=='source_ID':parent,labels['source_ID']=divmod(parent,128)
                    if kind=='accepted':parent,labels['accepted']=divmod(parent,2)
                    bank,remainder=divmod(parent,6);role,mode=divmod(remainder,2)
                    labels.update(bank=bank,role=role,mode=mode)
                reports[kind].append({**labels,'count':int(counts[key].sum()),'state_counts':counts[key].tolist(),
                    'first_UID_by_state':[None if v==0xffffffff else int(v) for v in first[key]]})
            ctx.guard()
        assert [len(reports[k]) for k in reports]==[39,72,4608,9216,144]
        assert all(sum(v['count'] for v in reports[k])==Q for k in ('occurrence','book','source_ID','accepted'))
        outcome_counts={name:sum(v['decision']==name for v in cells) for name in sorted({v['decision'] for v in cells})}
        exclusions=sum(v['decision']=='IDEAL_CONSTANT_EXCLUDED' for v in cells)
        all_pass=bool(np.all(state==0))
        decision=('CONSTANT_AMPLITUDE_CLASS_EXCLUDED_ON_DEVELOPMENT' if exclusions else
                  'FIXED_F32_SOURCE_DOMAIN_ELIGIBLE' if all_pass else 'INCOMPLETE_CONSTANT_AMPLITUDE_ELIGIBILITY')
        ctx.r['gates']['ALL_saved_UID_flags_and_complete39_72_4608_9216_144_reports'] = True
        value={'completed_cells':1536,'unique_inputs':U,'occurrences':Q,'development_inputs':159414,
            'consumed_validation_inputs':79458,'rho_exact':[101,100],'outcome_counts':outcome_counts,
            'state_labels':['PASS','SOURCE_P_BELOW_CONSTANT_RANGE','SOURCE_P_ABOVE_CONSTANT_RANGE','NO_SCALAR','INVALID_SOURCE_P'],
            'reports':reports,'decision':decision,'candidate_whole_source_amplitude_pass':all_pass,
            'native_calls':0,'optimizer_updates':0,'SVD_calls':0,
            'scope':'Uniform selected-amplitude constant eligibility only; no functional/task/rate/DRAM or large-n promotion.'}
        result=ctx.finish(value)
        print(json.dumps({'decision':decision,'outcome_counts':outcome_counts,'gates':result['gates'],
                          'resource':ctx.terminal_resources['resource']}))
    except BaseException as exc:
        ctx.fail(exc)
        print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True)
        raise

if __name__=='__main__':main()
