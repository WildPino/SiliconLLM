"""Stored audit adapter: Windows longdouble alias resolved by accurate fsum rows."""
import argparse,json,math,sys
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import original_categorical_trust_step as frozen

def main(a):
    import numpy as np
    original_sum=np.sum
    def independent_rows(value,axis=None,dtype=None,out=None,keepdims=False,initial=None,where=True):
        x=np.asarray(value)
        if axis==1 and x.ndim==2 and x.dtype==np.dtype('float64') and dtype is None and out is None and initial is None and where is True:
            rows=np.asarray([math.fsum(row) for row in x],dtype='f8')
            return rows[:,None] if keepdims else rows
        kwargs=dict(axis=axis,dtype=dtype,out=out,keepdims=keepdims)
        if where is not True:kwargs['where']=where
        if initial is not None:kwargs['initial']=initial
        return original_sum(value,**kwargs)
    b=json.loads(a.binding.read_bytes());adapter_record=dict(code=frozen.extent(__file__),protocol=frozen.extent(Path(b['parent_binding']['path']).parent/'ORIGINAL_CATEGORICAL_TRUST_FSUM_AUDIT_20261010.md'),NumPy=np.__version__,longdouble_bytes=np.dtype(np.longdouble).itemsize,longdouble_mantissa=np.finfo(np.longdouble).nmant,method='math.fsum on every 2D float64 axis1 reduction in fresh stored audit process; original on other reductions',model_forwards=0,backward_calls=0,native_calls=0,thresholds_unchanged=True)
    np.sum=independent_rows
    original_emit=frozen.emit
    def audit_emit(path,value):
        if value.get('schema')=='ORIGINAL_CATEGORICAL_TRUST_STEP_AUDIT_V1':
            value['independent_reduction_adapter']=adapter_record;value['all_group_updates_independent_fsum_verified']=value.pop('all_group_updates_longdouble_norms_verified');value['scope']='Stored arrays/weights only: math.fsum row-norm/dot proposal verification, full original packing/witness checks and shifted logaddexp metrics. longdouble aliases F64 on this Windows runtime; no extended-precision claim, original history/backward/native replay or changed thresholds.'
        original_emit(path,value)
    frozen.emit=audit_emit
    try:frozen.audit(a)
    finally:np.sum=original_sum

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-worker',action='store_true',required=True)
    for k in ('binding','directory','out','source-result'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('binding-sha','freeze'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
