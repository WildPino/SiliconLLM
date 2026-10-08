"""FIT-only quota-constrained parent-key proposal; no work at import.

Balanced construction assignments are not inference routes. Actual selected4
support/masses must be measured under the unchanged compact routing operator.
The greedy constrained assignment is deterministic, not an optimal transport
solver or a monotonic descent/whole-quality certificate. Not yet executed.
"""


def squared_distance_tree(query,centers):
    """Declared F64 difference/square then five adjacent-pair addition levels."""
    import numpy as np
    delta=query[:,None,:]-centers[None,:,:];value=delta*delta
    assert value.shape[-1]==32
    while value.shape[-1]>1:value=value[...,::2]+value[...,1::2]
    return np.ascontiguousarray(value[...,0])


def balanced_centers(query,initial_centers=None,guard=None):
    """Twenty fixed greedy quota-assignment/centroid iterations, F64.

    query: saved actual F32 projection representatives promoted to F64, [N,32].
    Warm centers, if provided, keep their labels as starting inputs. Otherwise
    seed16 distinct farthest-first query points. No DEV input or tuning ladder.
    """
    import numpy as np
    query=np.array(query,dtype=np.float64,order='C',copy=True)
    assert query.ndim==2 and query.shape[1]==32 and len(query)>=16 and np.isfinite(query).all()
    assert np.array_equal(query,query.astype(np.float32).astype(np.float64)), 'Queries must be actual F32 representatives'
    n=len(query);parents=16
    if initial_centers is None:
        mean=np.zeros((1,32),dtype=np.float64)
        np.add.at(mean,np.zeros(n,dtype=np.int64),query);mean/=n
        first=int(np.argmax(squared_distance_tree(query,mean)[:,0]))
        chosen=[first];nearest=squared_distance_tree(query,query[first:first+1])[:,0]
        for _ in range(1,parents):
            index=int(np.argmax(nearest));assert index not in chosen
            chosen.append(index);nearest=np.minimum(nearest,squared_distance_tree(query,query[index:index+1])[:,0])
        centers=query[chosen].copy(order='C')
    else:
        centers=np.array(initial_centers,dtype=np.float64,order='C',copy=True)
        assert centers.shape==(16,32) and np.isfinite(centers).all();chosen=None
    quota=np.full(parents,n//parents,dtype=np.int64);quota[:n%parents]+=1
    assert quota.min()>0 and quota.sum()==n
    assignments=[];history=[centers.copy()];costs=[]
    for _ in range(20):
        distance=squared_distance_tree(query,centers)
        assert distance.shape==(n,parents) and np.isfinite(distance).all()
        # C-order stable sort breaks exact ties by row first, then parent ID.
        order=np.argsort(distance.reshape(-1),kind='stable')
        assignment=np.full(n,-1,dtype=np.int64);remaining=quota.copy();assigned=0
        for flat in order:
            row=int(flat)//parents;parent=int(flat)%parents
            if assignment[row]<0 and remaining[parent]>0:
                assignment[row]=parent;remaining[parent]-=1;assigned+=1
                if assigned==n:break
        assert assigned==n and np.all(remaining==0) and np.array_equal(np.bincount(assignment,minlength=16),quota)
        costs.append(float(np.add.accumulate(distance[np.arange(n),assignment],dtype=np.float64)[-1]))
        centers=np.zeros((parents,32),dtype=np.float64)
        np.add.at(centers,assignment,query);centers/=quota[:,None]
        assert np.isfinite(centers).all();assignments.append(assignment);history.append(centers.copy())
        if guard is not None:guard()
    final=np.ascontiguousarray(centers,dtype=np.float32);assert np.isfinite(final).all()
    trace=dict(query_F32=np.ascontiguousarray(query,dtype=np.float32),quotas_int64=quota,
        assignments_int64=np.stack(assignments),centers_F64=np.stack(history),
        assignment_costs_F64=np.array(costs,dtype=np.float64))
    record=dict(method='fixed20_F64_greedy_quota_assignments_and_centroid_means',rows=n,parents=16,
        quota_min=int(quota.min()),quota_max=int(quota.max()),iterations=20,
        starting_centers='warm_BYTE_input' if chosen is None else 'NEW_FIT_farthest_first',
        distance_arithmetic='F64 separate difference/square; fixed five-level adjacent-pair tree',
        mean_arithmetic='F64 row-order unbuffered adds then division by exact integer quota',
        cost_arithmetic='F64 row-order accumulation before centroid update',
        farthest_seed_indices=chosen,constrained_optimality_or_monotonicity_claim=False,
        balanced_assignments_are_NOT_actual_inference_support=True)
    return final,trace,record


def make_geometry(fit,device,warm_input,guard):
    """NEW balanced keys; reuse warm projection/starting labels where supplied."""
    import numpy as np
    import torch
    if warm_input is not None:
        warm=torch.load(warm_input['path'],map_location='cpu',weights_only=True)
        assert warm['projection'].dtype==warm['parent_centers'].dtype==torch.float32
        projection=warm['projection'].to(device).contiguous()
        initial=warm['parent_centers'].numpy().astype(np.float64);values=None
        del warm
    else:
        unique=fit['x'][fit['unique_index']].astype(np.float64);centered=unique-unique.mean(0)
        covariance=torch.from_numpy(np.ascontiguousarray(centered.T@centered/len(unique)))
        eigenvalues,vectors=torch.linalg.eigh(covariance);eigenvalues=eigenvalues[-32:].flip(0)
        vectors=vectors[:,-32:].flip(1).T.contiguous()
        for row in vectors:
            if row[row.abs().argmax()]<0:row.neg_()
        assert torch.isfinite(eigenvalues).all() and bool((eigenvalues>0).all())
        projection=(vectors/eigenvalues.sum().sqrt()).float().contiguous().to(device)
        values=eigenvalues.tolist();initial=None
    assert tuple(projection.shape)==(32,896) and torch.isfinite(projection).all()
    x=fit['tx'][torch.as_tensor(fit['unique_index'],device=device)]
    query=torch.nn.functional.linear(x,projection).cpu().numpy()
    assert query.dtype==np.float32
    parents,trace,record=balanced_centers(query,initial,guard)
    geometry=dict(origin='NEW_balanced_FIT_keys',projection_origin='BYTE_warm_projection' if warm_input is not None else 'NEW_unique_FIT_covariance',
        projection_input=warm_input,eigenvalues=values,parents=record)
    guard()
    return projection,torch.from_numpy(parents).to(device),trace,geometry
