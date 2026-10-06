"""Complete single-payload transform. NumPy is imported only by the admitted main."""
import hashlib
import struct
from pathlib import Path
import numpy as np

def manifest(path):
    data=Path(path).read_bytes();assert data[:8] in [b'SWI8A001',b'SWI8C001']
    config=struct.unpack_from('<13If',data,8);nf,nt=struct.unpack_from('<2I',data,64);offset=72
    def string():
        nonlocal offset
        n,=struct.unpack_from('<I',data,offset);offset+=4
        s=data[offset:offset+n].decode('utf8');offset+=n;return s
    files=[string() for _ in range(nf)];tensors=[]
    for _ in range(nt):
        name=string();record=offset;fi,nd,r,c,enc,o,so,n=struct.unpack_from('<5I3Q',data,offset);offset+=44
        assert nd in [1,2] and r*c==n and enc in [0,1,2]
        tensors.append(dict(name=name,file=fi,dims=nd,rows=r,cols=c,encoding=enc,offset=o,scale_offset=so,elements=n,record=record))
    assert offset==len(data) and nf==1 and nt==3320
    return data,config,files,tensors

def sparse_wo(t):return '.mlp.experts.expert_' in t['name'] and t['name'].endswith('.wo.weight')

def export(ctx,source_manifest):
    data,config,files,tensors=manifest(source_manifest);source=Path(files[0]);dest=ctx.out/'weights.bin'
    assert source.stat().st_size==7541946880 and data[:8]==b'SWI8A001'
    with source.open('rb') as f,dest.open('xb') as g:
        while block:=f.read(16<<20):g.write(block);ctx.guard()
    transformed=[]
    with source.open('rb') as f,dest.open('r+b') as g:
        for t in tensors:
            if sparse_wo(t):
                assert (t['rows'],t['cols'],t['encoding'])==(768,3072,1)
                f.seek(t['offset']);old=f.read(t['elements']);assert len(old)==t['elements']
                new=np.frombuffer(old,dtype='i1').reshape(768,3072).T.copy().tobytes()
                g.seek(t['offset']);g.write(new)
                transformed.append({**t,'source_sha256':hashlib.sha256(old).hexdigest(),'column_sha256':hashlib.sha256(new).hexdigest()});ctx.guard()
    assert len(transformed)==1536
    # Reconstruct the manifest, changing only magic, one path, and WO encoding.
    p=str(dest.resolve()).encode('utf8');old_end=76+len(files[0].encode('utf8'))
    tail=bytearray(data[old_end:]);delta=old_end
    for t in transformed:struct.pack_into('<I',tail,t['record']-delta+16,2)
    target=ctx.out/'manifest.bin'
    with target.open('xb') as f:f.write(b'SWI8C001'+data[8:72]+struct.pack('<I',len(p))+p+tail)
    return {'payload':str(dest),'bytes':dest.stat().st_size,'sha256':ctx.digest(dest),'manifest':str(target),'manifest_sha256':ctx.digest(target),'transformed':transformed,'distinct_original_parameters':7415217408,'additional_distinct_parameters':0}

def invert_all(ctx,source_manifest,candidate_manifest):
    a,ca,fa,ta=manifest(source_manifest);b,cb,fb,tb=manifest(candidate_manifest)
    assert a[:8]==b'SWI8A001' and b[:8]==b'SWI8C001' and ca==cb
    assert len(ta)==len(tb)==3320
    changed=[]
    for x,y in zip(ta,tb):
        assert {k:v for k,v in x.items() if k not in ['encoding','record']}=={k:v for k,v in y.items() if k not in ['encoding','record']}
        assert y['encoding']==(2 if sparse_wo(x) else x['encoding'])
        if sparse_wo(x):changed.append(x)
    assert len(changed)==1536
    end=0
    with Path(fa[0]).open('rb') as f,Path(fb[0]).open('rb') as g:
        def gap(stop):
            nonlocal end
            while end<stop:
                n=min(16<<20,stop-end);assert f.read(n)==g.read(n);end+=n;ctx.guard()
        for t in sorted(changed,key=lambda v:v['offset']):
            gap(t['offset']);old=f.read(t['elements']);new=g.read(t['elements'])
            restored=np.frombuffer(new,dtype='i1').reshape(3072,768).T.copy().tobytes()
            assert restored==old;end+=t['elements'];ctx.guard()
        gap(7541946880);assert f.read(1)==g.read(1)==b''
    return {'all_transformed_tensors':1536,'all_original_tensors':3320,'all_payload_bytes_including_scales_padding_aliases':7541946880,'all_inverse_and_unchanged_bytes_exact':True}
