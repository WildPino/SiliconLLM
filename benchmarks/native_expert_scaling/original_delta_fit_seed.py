"""Single-factor initialization from FIT activation directions not in old drives.

This helper does not run a model, update an optimizer, or edit a checkpoint.
Its zero-read partner must remain zero until an explicitly recorded new step.
"""
def seed_from_fit(x,old_u,count=32,max_norm_multiple=10.,rcond=1e-12):
    import numpy as np
    x=np.asarray(x,dtype='f8');old_u=np.asarray(old_u,dtype='f8')
    assert x.ndim==old_u.ndim==2 and x.shape[1]==old_u.shape[1]
    assert np.isfinite(x).all() and np.isfinite(old_u).all() and count>0
    # Least squares in time removes responses already represented by old delta
    # functions. This is response orthogonality on FIT, not weight orthogonality.
    old_drive=x@old_u.T
    coef,_,rank,_=np.linalg.lstsq(old_drive,x,rcond=rcond)
    assert rank==old_u.shape[0],'old FIT drive rank'
    residual=x-old_drive@coef
    gram=(residual.T@residual)/x.shape[0]
    eigenvalues,vectors=np.linalg.eigh((gram+gram.T)/2)
    assert eigenvalues[0]>=-1e-10*max(float(eigenvalues[-1]),1e-24),'PSD roundoff'
    selected=eigenvalues[-count:][::-1]
    original_energy=float(np.sum(x*x)/x.shape[0])
    assert len(selected)==count and selected[-1]>rcond*max(original_energy,1e-24),'new FIT rank'
    principal=vectors[:,-count:][:,::-1].T
    # Fold the response residualization back into legal x_proj rows. No new
    # bias or history-dependent runtime operator is introduced by calibration.
    rows=principal-(principal@coef.T)@old_u
    for i in range(count):
        index=int(np.argmax(np.abs(rows[i])))
        if rows[i,index]<0:rows[i]*=-1
    target=float(np.median(np.sqrt(np.mean(old_drive**2,axis=0))))
    limit=float(max_norm_multiple*np.median(np.linalg.norm(old_u,axis=1)))
    assert target>0 and limit>0
    response=x@rows.T;rms=np.sqrt(np.mean(response**2,axis=0))
    factor=np.minimum(target/rms,limit/np.linalg.norm(rows,axis=1))
    rows*=factor[:,None];packed=rows.astype('<f4')
    new_drive=x@packed.astype('f8').T
    orth=float(np.linalg.norm(old_drive.T@new_drive)/max(np.linalg.norm(old_drive)*np.linalg.norm(new_drive),1e-24))
    new_sv=np.linalg.svd(new_drive,compute_uv=False)
    assert orth<=1e-6 and new_sv[-1]>rcond*new_sv[0]>0
    assert np.linalg.norm(packed.astype('f8'),axis=1).max()<=limit*(1+1e-6)
    return packed,dict(samples=x.shape[0],features=x.shape[1],old_rank=int(rank),new_rank=count,
        residual_eigenvalues=selected.tolist(),new_response_RMS=np.sqrt(np.mean(new_drive**2,axis=0)).tolist(),
        old_response_median_RMS=target,new_row_L2_limit=limit,FIT_response_cross_relative=orth,
        scope='FIT activation residual PCA and legal linear-row fold;source information or DEV quality preservation not proved.')
