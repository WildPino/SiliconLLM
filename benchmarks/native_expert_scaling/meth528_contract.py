"""528 shared wire/I/O only; masks, quantizers and metrics are separate."""
from pathlib import Path
import struct
import numpy as np
from meth528_operations import wire

N,D,H,E = 17540,768,3072,128
UID=np.dtype([('m','<u4',(13,)),('hash','u1',(128,))])
INPUT=np.dtype([('e','<u4'),('accept','<u4'),('alpha','<f4'),('x','<f4',(D,)),('q','<i2',(D,)),('p','<f4')])
ARRAYS={'physical':((N,2,D),'<f4'),'integer_WO':((N,2,D),'<i8'),
        'continuous':((N,D),'<f8'),'negative_fold':((N,D),'<f8'),
        'hidden_alpha':((N,2),'<f4'),'metrics':((N,24),'<f8')}


def records(ctx,key,ids):
    magic,width,reserved,dtype,count={
        'uid':(b'M493U001',180,0,UID,N),'occurrences':(b'M493O001',68,0,np.dtype(('<u4',(17,))),19962),
        'inputs':(b'M499INP1',4624,D,INPUT,N),'unweighted':(b'M499F001',3072,D,np.dtype(('<f4',(D,))),N),
        'targets':(b'M493Y001',3072,D,np.dtype(('<f4',(D,))),N)}[key]
    a=wire(np,ctx.data(key),magic,width,reserved,dtype,count); result=np.array(a[ids],copy=True); a._mmap.close(); return result


def old(ctx,key,ids):
    a=np.load(ctx.data(key),mmap_mode='r',allow_pickle=False); result=np.array(a[ids],copy=True); a._mmap.close(); return result


def metadata(ctx):
    m=records(ctx,'uid',slice(None))['m']; occ=records(ctx,'occurrences',slice(None))
    assert np.array_equal(m[:,0],np.arange(N)) and np.all(m[:,2]==11) and np.all(m[:,6]==2)
    dev=(m[:,4]&5)!=0; val=(m[:,4]&2)!=0
    assert (int(dev.sum()),int(val.sum()))==(11721,5819) and np.all(dev^val)
    assert np.array_equal(occ[:,0],np.arange(19962)) and np.all(occ[:,1]<N) and np.all(occ[:,12]==1)
    assert np.all(occ[:,3]==E) and np.all(occ[:,8]==11)
    for oi,mi in ((14,1),(11,3),(15,12),(16,10)): assert np.array_equal(occ[:,oi],m[occ[:,1],mi])
    assert np.array_equal(np.bincount(occ[:,1],minlength=N),m[:,7])
    counts=np.bincount(m[dev,3],minlength=E); assert counts[0]==0 and np.all(counts[1:]>0)
    return m,occ,dev,counts


def weights(ctx,e):
    entry=next(v for v in ctx.b['parents'] if v['parent']==e); out=[]
    with Path(ctx.b['payload']['path']).open('rb') as f:
        for name,shape in (('wi',(H,D)),('wo',(D,H))):
            v=entry[name]; f.seek(v['offset']); raw=f.read(v['bytes']); assert len(raw)==v['bytes']
            f.seek(v['scale_offset']); s=f.read(v['scale_bytes']); assert len(s)==v['scale_bytes']
            out.extend((np.frombuffer(raw,'i1').reshape(shape),np.frombuffer(s,'<f4')))
    assert all(np.all(v>0) and np.isfinite(v).all() for v in (out[1],out[3])); return out


def pack(path,e,arm,ids,wi,si,wo,so):
    b=len(ids); assert 512<=b<=H and np.array_equal(ids,np.unique(ids))
    with Path(path).open('xb') as f:
        f.write(struct.pack('<8s6I',b'M528ATM1',e,arm,D,b,H,1))
        for a in (ids.astype('<u2'),si[ids].astype('<f4'),so.astype('<f4'),wi[ids],wo[:,ids]): f.write(a.tobytes(order='C'))


def unpack(path,e,arm):
    raw=Path(path).read_bytes(); magic,ee,aa,d,b,h,flags=struct.unpack('<8s6I',raw[:32])
    assert (magic,ee,aa,d,h,flags)==(b'M528ATM1',e,arm,D,H,1) and 512<=b<=H
    assert len(raw)==32+6*b+4*D+2*D*b; at=32; vals=[]
    for shape,dtype in (((b,),'<u2'),((b,),'<f4'),((D,),'<f4'),((b,D),'i1'),((D,b),'i1')):
        dt=np.dtype(dtype); count=int(np.prod(shape)); vals.append(np.frombuffer(raw,dt,count,at).reshape(shape)); at+=count*dt.itemsize
    return vals


def create(folder):
    for name,(shape,dtype) in ARRAYS.items():
        a=np.lib.format.open_memmap(Path(folder)/(name+'.npy'),mode='w+',dtype=dtype,shape=shape); a._mmap.close()


def put(folder,name,ids,value):
    a=np.load(Path(folder)/(name+'.npy'),mmap_mode='r+',allow_pickle=False); a[ids]=value; a.flush(); a._mmap.close()


def get(folder,name,ids):
    a=np.load(Path(folder)/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    assert a.shape==ARRAYS[name][0] and a.dtype==np.dtype(ARRAYS[name][1]); result=np.array(a[ids],copy=True); a._mmap.close(); return result


def domain_ids(m,occ,counts):
    dev=(m[:,4]&5)!=0
    for name,mask in (('development',dev),('consumed_validation',~dev)): yield dict(kind='uid',split=name),np.flatnonzero(mask)
    for label,lo,hi in (('0',0,0),('1..4',1,4),('5..15',5,15),('>=16',16,N)):
        for name,mask in (('development',dev),('consumed_validation',~dev)):
            yield dict(kind='rare',development_class=label,split=name),np.flatnonzero(mask&(counts[m[:,3]]>=lo)&(counts[m[:,3]]<=hi))
    for e in range(E):
        for name,mask in (('development',dev),('consumed_validation',~dev)): yield dict(kind='parent',expert=e,split=name),np.flatnonzero(mask&(m[:,3]==e))
    for book in range(192):
        for mode in range(2): yield dict(kind='book',book=book,role=book//64,mode=mode),occ[(occ[:,4]==book)&(occ[:,6]==mode),1]
    for role in range(3):
        for mode in range(2): yield dict(kind='role',role=role,mode=mode),occ[(occ[:,7]==role)&(occ[:,6]==mode),1]
