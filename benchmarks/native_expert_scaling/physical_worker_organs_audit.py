"""Separate full stored replay against qualified original serial operators."""
import ctypes
from pathlib import Path
import time
from falcon_auxiliary_s6_transport import extent
import contextual_generator_cost as cg


def audit(r,b,gb,rb,np,guard):
    from physical_worker_organs import ARMS,ORDERS,reference,summary,function,exact
    started=time.monotonic();gdll,_=cg.native(gb);weights=cg.weights(gb,np)
    mup=np.asarray(gb['MuP_bits'],'u2')
    original=ctypes.CDLL(rb['compile']['dll']['path'])
    readout=function(original,'ir_history',[ctypes.c_int]+[ctypes.c_void_p]*7)
    expected_records=[(mode,rec['id']) for mode in b['modes'] for rec in gb['records']]
    assert [(x['mode'],x['id']) for x in r['records']] == expected_records
    pins={e['path']:e for e in b['inputs']}
    from native_24_cost import parameter_sets
    parameters=parameter_sets(rb,np)
    gcalls=rcalls=gwords=rwords=0
    for mode_index,mode in enumerate(b['modes']):
        for record_index,rec in enumerate(gb['records']):
            guard();entry=next(q for q in r['records'] if q['mode']==mode and q['id']==rec['id'])
            assert extent(entry['file']['path']) == entry['file']
            ref=reference(rec,mode,rb,np);h,truth=cg.record_inputs(rec,np)
            with np.load(entry['file']['path'],allow_pickle=False) as z:
                assert set(z.files)=={'g_raw','g_gg','g_fields','r_dots','r_raw','r_bits'}
                for key in z.files:
                    dtype='f4' if key=='g_raw' else 'f8' if key=='r_raw' else 'i8' if key=='r_dots' else 'u2'
                    expected_shape=(3,23,16,6704) if key.startswith('g_') else (3,24,16,2048)
                    assert z[key].dtype==np.dtype(dtype) and z[key].shape==expected_shape
                raw=np.empty(6704,'f4');gg=np.empty(6704,'u2');fields=np.empty(6704,'u2')
                generator={k:z[k] for k in ('g_raw','g_gg','g_fields')}
                for site,w in enumerate(weights):
                    for t in range(16):
                        guard()
                        assert gdll.cg_step(w.ctypes.data,h[site,t].ctypes.data,mup.ctypes.data,
                            raw.ctypes.data,gg.ctypes.data,fields.ctypes.data)==0
                        for ai in range(3):
                            for key,value in (('g_raw',raw),('g_gg',gg),('g_fields',fields)):
                                exact(generator[key][ai,site,t],value,np);gwords+=value.size
                        assert max(cg.controls(fields[None],truth[site,t:t+1],np).values()) <= gb['gates']['field_relative']
                        gcalls+=1
                readouts={k:z[k] for k in ('r_dots','r_raw','r_bits')}
                for site,(p,_,_) in enumerate(parameters):
                    dots=np.empty((16,2048),'i8');values=np.empty((16,2048),'f8');bits=np.empty((16,2048),'u2')
                    assert readout(16,p['q'].ctypes.data,p['scale'].ctypes.data,
                        ref['r'][site].ctypes.data,ref['a'][site].ctypes.data,
                        dots.ctypes.data,values.ctypes.data,bits.ctypes.data)==0
                    for ai in range(3):
                        for key,value in (('r_dots',dots),('r_raw',values),('r_bits',bits)):
                            exact(readouts[key][ai,site],value,np);rwords+=value.size
                    for key,value in (('dots',dots),('raw',values),('output',bits)):
                        exact(ref[key][site],value,np)
                    rcalls+=16
            rows=[q for q in r['samples'] if q['id']==rec['id'] and q['mode']==mode]
            assert len(rows)==48
            expected=[(pos,arm) for t,pos in enumerate(rec['positions'])
                for arm in ORDERS[(t+record_index*16+mode_index*64)%6]]
            assert [(q['position'],q['workers']) for q in rows]==expected
            for index,q in enumerate(rows):
                t=index//3
                assert q['order']==list(ORDERS[(t+record_index*16+mode_index*64)%6])
                assert len(q['generator_site_ns'])==23 and len(q['readout_site_ns'])==24
                assert q['generator_ns']==sum(q['generator_site_ns'])
                assert q['readout_ns']==sum(q['readout_site_ns'])
                assert q['whole_ns']>=q['generator_ns']+q['readout_ns']
                assert all(x>=0 for x in q['generator_site_ns']+q['readout_site_ns'])
                assert q['flush_ns']>=0 and (mode!='resident' or q['flush_ns']==0)
    assert len(r['samples'])==384 and gcalls==2944 and rcalls==3072
    assert r['summary']==summary(r['samples'],np)
    for key in ('initial_worker_binding','final_worker_binding'):
        rows=r[key]
        assert len(rows)==6 and len({x[0] for x in rows})==6
        assert len({x[1] for x in rows})==6
        assert all(x[1]>0 and x[1]&(x[1]-1)==0 and x[2]==0 and x[3]>=6 for x in rows)
    assert r['FP_control']==8064 and r['negative_contracts_pass']
    assert r['GPU_calls']==0 and r['RESERVED_queries']==0
    return dict(schema='PHYSICAL_WORKER_ORGANS_AUDIT_V1',serial_generator_calls=gcalls,
        serial_readout_positions=rcalls,generator_cells_compared=gwords,readout_cells_compared=rwords,
        candidate_generator_calls=gcalls*3,candidate_readout_positions=rcalls*3,
        measured_samples=384,all_numeric_bits_exact=True,all_coverage_accounting_exact=True,
        decision='FULL_STORED_SERIAL_PARITY_AND_ACCOUNTING_PASS',seconds=time.monotonic()-started,
        scope='Full scheduling parity against qualified original serial kernels and same operands. '
              'Original independent leaf-arithmetic controls inherited; no new whole-model quality or TPS/DRAM admission.')
