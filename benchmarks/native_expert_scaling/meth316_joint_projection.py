"""Equivalent fixed-span projection: conditioned Cholesky, SVD fallback."""
import numpy as np
import scipy.linalg

def svd_span(matrix):
    if not matrix.shape[1]:return np.zeros((matrix.shape[0],0))
    u,s,_=scipy.linalg.svd(matrix,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    rank=int(np.sum(s>np.finfo(np.float64).eps*max(matrix.shape)*s[0]))
    return u[:,:rank]

def factor(matrix):
    if not matrix.shape[1]:return ('svd',svd_span(matrix),None,0.)
    gram=matrix.T@matrix
    try:
        cholesky=scipy.linalg.cho_factor(gram,lower=True,check_finite=False)
        condition,status=scipy.linalg.lapack.dpocon(cholesky[0],float(np.linalg.norm(gram,1)),uplo='L')
        if status==0 and np.isfinite(condition) and condition>=1e-6:
            return ('cholesky',matrix,cholesky,float(condition))
    except scipy.linalg.LinAlgError:pass
    return ('svd',svd_span(matrix),None,0.)

def project(record,vector):
    kind,space,cholesky,_=record
    if kind=='cholesky':
        coefficients=scipy.linalg.cho_solve(cholesky,space.T@vector,check_finite=False)
        return space@coefficients,space.shape[1]
    return space@(space.T@vector),space.shape[1]

def controls():
    values=np.arange(1,9,dtype=np.float64)
    independent=np.eye(8)[:,:2];record=factor(independent);actual,rank=project(record,values)
    assert record[0]=='cholesky' and rank==2 and np.array_equal(actual,np.asarray([1.,2.,0.,0.,0.,0.,0.,0.]))
    duplicate=np.eye(8)[:,[0,0,1]];fallback=factor(duplicate);other,rank=project(fallback,values)
    assert fallback[0]=='svd' and rank==2 and np.max(np.abs(other-actual))<=1e-12
    assert np.linalg.norm(actual)>1e-6,'zero projection fault'
    return {'independent_cholesky_control':True,'duplicate_column_SVD_fallback_control':True,'zero_projection_fault_detected':True}
