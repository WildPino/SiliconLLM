"""Independent codec reconstruction/F64 contraction and complete stored replay."""
import ctypes,time
import contextual_generator_cost as cg
from falcon_auxiliary_s6_transport import extent


def audit(r,b,gb,np,dll,guard):
    from contextual_generator_codec import mapped,function,exact,ORDERS,summarize
    start=time.monotonic();source=cg.weights(gb,np);mup=np.asarray(gb['MuP_bits'],'u2')
    codec=function(dll,'qc_step',[ctypes.c_int]+[ctypes.c_void_p]*7)
    assert function(dll,'po_threads',[ctypes.c_int])(3)==0
    counts=dict(coefficients=0,calls=0,raw_words=0,BF_words=0,BF_control_calls=0);max_raw=0.;all_errors=[]
    baseline=function(dll,'po_generator',[ctypes.c_void_p]*6);parent=cg.load(b['parent_report']['path'])
    packets={}
    for p in b['packets']:
        guard();fmt=p['format'];site=p['site'];codes,scales=mapped(p,np)
        for key in ('codes','scales'):assert extent(p[key]['path'])==p[key]
        q=np.asarray(codes).copy() if fmt==8 else np.empty((6704,2048),'i1')
        if fmt==4:
            assert ((codes&15)<=14).all() and ((codes>>4)<=14).all()
            q[:,::2]=(codes&15).astype('i2')-7;q[:,1::2]=(codes>>4).astype('i2')-7
        else:assert q.min()>=-127 and q.max()<=127
        for lo in range(0,6704,128):
            guard();hi=min(lo+128,6704);x=cg.decode(source[site-1][lo:hi],np).astype('f8')
            groups=x if fmt==8 else x.reshape(hi-lo,16,128)
            expected_scale=(np.max(np.abs(groups),axis=-1)/(127 if fmt==8 else 7)).astype('f4')
            exact(np.ascontiguousarray(scales[lo:hi]),expected_scale,np)
            ratio=np.divide(groups,expected_scale[...,None].astype('f8'),out=np.zeros_like(groups),where=expected_scale[...,None]!=0)
            absolute=np.abs(ratio);base=np.floor(absolute);fraction=absolute-base
            rounded=np.copysign(base+((fraction>.5)|((fraction==.5)&((base.astype('i8')&1)!=0))),ratio)
            manual=np.clip(rounded,-127 if fmt==8 else -7,127 if fmt==8 else 7).astype('i1').reshape(hi-lo,2048)
            exact(q[lo:hi],manual,np);counts['coefficients']+=manual.size
        packets[(fmt,site)]=(codes,scales,q)
    assert counts['coefficients']==2*315785216
    raw=np.empty((16,6704),'f4');gg=np.empty((16,6704),'u2');fields=np.empty((16,6704),'u2')
    for mode in b['modes']:
        for rec in gb['records']:
            guard();entry=next(x for x in r['records'] if x['id']==rec['id'] and x['mode']==mode)
            assert extent(entry['file']['path'])==entry['file'];h,truth=cg.record_inputs(rec,np)
            with np.load(entry['file']['path'],allow_pickle=False) as archive:
                assert set(archive.files)=={f'{key}_{fmt}' for fmt in (8,4) for key in ('raw','gg','fields')}
                for fmt in (8,4):
                    saved={key:archive[f'{key}_{fmt}'] for key in ('raw','gg','fields')}
                    assert all(x.shape==(23,16,6704) for x in saved.values())
                    error=[]
                    for site in range(1,24):
                        guard();codes,scales,q=packets[(fmt,site)]
                        if fmt==8:w=q.astype('f8')*scales.astype('f8')[:,None]
                        else:w=(q.reshape(6704,16,128).astype('f8')*scales.astype('f8')[...,None]).reshape(6704,2048)
                        reference=cg.decode(h[site-1],np).astype('f8')@w.T
                        discrepancy=cg.relative(saved['raw'][site-1],reference,np);max_raw=max(max_raw,discrepancy)
                        assert discrepancy<=b['gates']['decoded_F64_raw_relative']
                        rounded=cg.bf(saved['raw'][site-1],np);exact(rounded,saved['gg'][site-1],np)
                        expected_fields=cg.bf(cg.decode(rounded,np)*cg.decode(mup,np),np)
                        exact(expected_fields,saved['fields'][site-1],np)
                        for t in range(16):
                            assert codec(fmt,codes.ctypes.data,scales.ctypes.data,h[site-1,t].ctypes.data,mup.ctypes.data,
                                raw[t].ctypes.data,gg[t].ctypes.data,fields[t].ctypes.data)==0
                        for key,value in (('raw',raw),('gg',gg),('fields',fields)):exact(value,saved[key][site-1],np)
                        counts['calls']+=16;counts['raw_words']+=raw.size;counts['BF_words']+=2*gg.size
                        error.append(cg.controls(fields,truth[site-1],np))
                    assert error==entry['source_errors'][str(fmt)];all_errors.extend(error)
            assert entry['BF_control_exact']
            original=next(x for x in parent['records'] if x['id']==rec['id'] and x['mode']==mode)
            assert extent(original['file']['path'])==original['file']
            with np.load(original['file']['path'],allow_pickle=False) as z:
                for i,w in enumerate(source):
                    guard()
                    for t in range(16):
                        assert baseline(w.ctypes.data,h[i,t].ctypes.data,mup.ctypes.data,
                            raw[t].ctypes.data,gg[t].ctypes.data,fields[t].ctypes.data)==0
                    for key,value in (('raw',raw),('gg',gg),('fields',fields)):exact(value,z[key][i],np)
                    counts['BF_control_calls']+=16
    assert counts['calls']==5888 and counts['raw_words']==39473152 and counts['BF_words']==78946304
    assert counts['BF_control_calls']==2944
    expected=[(mode,rec['id'],rec['positions'][t],fmt,list(ORDERS[(t+ri*16+mi*64)%6]))
        for mi,mode in enumerate(b['modes']) for ri,rec in enumerate(gb['records']) for t in range(16)
        for fmt in ORDERS[(t+ri*16+mi*64)%6]]
    assert [(x['mode'],x['id'],x['position'],x['format'],x['order']) for x in r['samples']]==expected
    for x in r['samples']:
        assert len(x['site_ns'])==23 and x['whole_ns']>=sum(x['site_ns'])
        assert all(v>=0 for v in x['site_ns']) and x['flush_ns']>=0
        assert x['mode']!='resident' or x['flush_ns']==0
    assert r['summary']==summarize(r['samples'],np)
    for key in ('initial_workers','final_workers'):
        rows=r[key];assert len(rows)==6 and len({x[0] for x in rows})==6 and len({x[1] for x in rows})==6
        assert all(x[1]>0 and x[1]&(x[1]-1)==0 and x[2]==0 and x[3]>=6 for x in rows)
    assert r['GPU_calls']==r['RESERVED_queries']==0
    assert all(r['controls'][k] for k in ('reserved_codes_rejected','bad_format_rejected','nonfinite_h_rejected',
        'changed_FP_rejected_without_output_mutation','zero_row_scale_exact'))
    assert r['controls']['I8_codes']==255 and r['controls']['I4_codes']==15
    return dict(schema='CONTEXTUAL_GENERATOR_CODEC_AUDIT_V1',counts=counts,max_decoded_F64_raw_relative=max_raw,
        source_field_max_relative=max(v for e in all_errors for v in e.values()),all_coefficients_codes_scales_exact=True,
        all_raw_GG_fields_replayed_exact=True,all_coverage_accounting_exact=True,
        historical_one_percent_source_field_screen_all_pass=all(v<=.01 for e in all_errors for v in e.values()),
        decision='FULL_CODEC_ARITHMETIC_AND_ACCOUNTING_PASS',seconds=time.monotonic()-start,
        scope='Full source-coefficient codec reconstruction and decoded-weight arithmetic; source approximation and whole own-history model quality remain distinct.')
