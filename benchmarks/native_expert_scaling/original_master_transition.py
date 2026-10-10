"""Lossless, bounded F32 master transitions; no floating-point update arithmetic."""
import hashlib,json,os,zlib
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def bits(value):
    import numpy as np
    if hasattr(value,'detach'):value=value.detach().cpu().contiguous().numpy()
    value=np.asarray(value)
    assert value.dtype==np.dtype('<f4') and value.flags.c_contiguous
    return value.view('<u4').reshape(-1),list(value.shape)


def encode(source,target,directory,chunk_bytes=4<<20,progress=None):
    """Each payload chunk is copy, zlib XOR, or raw new bits (bounded fallback)."""
    import numpy as np
    assert set(source)==set(target) and source and chunk_bytes>0 and chunk_bytes%4==0
    directory=Path(directory);directory.mkdir(exist_ok=False);blob=directory/'transition.bin'
    rows=[];counts=dict(copy=0,xor_zlib=0,raw=0);moved=0;total=0
    with blob.open('xb') as output:
        for name in sorted(source):
            old,shape=bits(source[name]);new,new_shape=bits(target[name]);assert shape==new_shape
            old_h=hashlib.sha256();new_h=hashlib.sha256();chunks=[]
            for first in range(0,old.size,chunk_bytes//4):
                last=min(first+chunk_bytes//4,old.size);a=old[first:last];b=new[first:last]
                av=memoryview(a).cast('B');bv=memoryview(b).cast('B');old_h.update(av);new_h.update(bv)
                changed=int(np.count_nonzero(a!=b));moved+=changed;length=len(av);total+=length
                if not changed:codec='copy';data=b''
                else:
                    delta=np.bitwise_xor(a,b);compressed=zlib.compress(delta.tobytes(),1)
                    codec,data=('xor_zlib',compressed) if len(compressed)<length else ('raw',bv.tobytes())
                offset=output.tell();output.write(data);counts[codec]+=1
                chunks.append(dict(raw_offset=first*4,raw_bytes=length,encoded_offset=offset,
                    encoded_bytes=len(data),codec=codec,source_sha256=digest(av),target_sha256=digest(bv),
                    encoded_sha256=digest(data),changed_words=changed))
                if progress:progress(name,last,old.size)
            rows.append(dict(name=name,shape=shape,dtype='<f4',raw_bytes=old.size*4,
                source_sha256=old_h.hexdigest(),target_sha256=new_h.hexdigest(),chunks=chunks))
        output.flush();os.fsync(output.fileno())
    index=dict(schema='ORIGINAL_MASTER_TRANSITION_V1',chunk_bytes=chunk_bytes,
        blob_bytes=blob.stat().st_size,blob_sha256=file_digest(blob),raw_bytes=total,
        changed_words=moved,codec_counts=counts,parameters=rows,
        scope='Exact F32 bits; source chunk custody required. No optimization or quantization.')
    path=directory/'transition.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(index,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return index


def decoded_chunks(source,index_path):
    """Yield name, byte offset, exact target bytes; consume fully to verify all hashes."""
    import numpy as np
    path=Path(index_path);index=json.loads(path.read_bytes());blob=path.parent/'transition.bin'
    assert index['schema']=='ORIGINAL_MASTER_TRANSITION_V1'
    assert index['chunk_bytes']>0 and index['chunk_bytes']%4==0
    assert blob.stat().st_size==index['blob_bytes'] and file_digest(blob)==index['blob_sha256']
    rows=index['parameters'];assert [r['name'] for r in rows]==sorted(source)
    offset=0;total=0;moved=0;counts=dict(copy=0,xor_zlib=0,raw=0)
    with blob.open('rb') as stream:
        for row in rows:
            old,shape=bits(source[row['name']]);assert shape==row['shape'] and row['dtype']=='<f4'
            assert old.size*4==row['raw_bytes'];raw_offset=0;oh=hashlib.sha256();nh=hashlib.sha256()
            for chunk in row['chunks']:
                n=chunk['raw_bytes'];assert 0<n<=index['chunk_bytes'] and n%4==0
                assert chunk['raw_offset']==raw_offset and chunk['encoded_offset']==offset
                assert raw_offset+n<=row['raw_bytes'] and 0<=chunk['encoded_bytes']<=n
                a=memoryview(old).cast('B')[raw_offset:raw_offset+n]
                assert digest(a)==chunk['source_sha256'];oh.update(a)
                data=stream.read(chunk['encoded_bytes']);assert len(data)==chunk['encoded_bytes']
                assert digest(data)==chunk['encoded_sha256'];codec=chunk['codec']
                if codec=='copy':assert not data;out=a.tobytes()
                elif codec=='raw':assert len(data)==n;out=data
                else:
                    assert codec=='xor_zlib' and 0<len(data)<n
                    decompressor=zlib.decompressobj();delta=decompressor.decompress(data,n+1)
                    assert decompressor.eof and not decompressor.unused_data and not decompressor.unconsumed_tail and len(delta)==n
                    out=np.bitwise_xor(np.frombuffer(a,'<u4'),np.frombuffer(delta,'<u4')).tobytes()
                assert digest(out)==chunk['target_sha256'];nh.update(out)
                changed=int(np.count_nonzero(np.frombuffer(a,'<u4')!=np.frombuffer(out,'<u4')))
                assert changed==chunk['changed_words'];moved+=changed;counts[codec]+=1
                yield row['name'],raw_offset,out
                raw_offset+=n;offset+=len(data);total+=n
            assert raw_offset==row['raw_bytes'] and oh.hexdigest()==row['source_sha256'] and nh.hexdigest()==row['target_sha256']
        assert not stream.read(1) and offset==index['blob_bytes']
    assert total==index['raw_bytes'] and moved==index['changed_words'] and counts==index['codec_counts']
