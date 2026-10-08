"""FIT-only quota-constrained parent-key proposal; no work at import.

Balanced construction assignments are not inference routes. Actual selected4
support/masses must be measured under the unchanged compact routing operator.
The greedy constrained assignment is deterministic, not an optimal transport
solver or a monotonic descent/whole-quality certificate. Not yet executed.
"""


def balanced_centers(query,initial_centers=None):
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
        first=int(np.argmax(np.square(query-query.mean(0)).sum(1)))
        chosen=[first];nearest=np.square(query-query[first]).sum(1)
        for _ in range(1,parents):
            index=int(np.argmax(nearest));assert index not in chosen
            chosen.append(index);nearest=np.minimum(nearest,np.square(query-query[index]).sum(1))
        centers=query[chosen].copy(order='C')
    else:
        centers=np.array(initial_centers,dtype=np.float64,order='C',copy=True)
        assert centers.shape==(16,32) and np.isfinite(centers).all();chosen=None
    quota=np.full(parents,n//parents,dtype=np.int64);quota[:n%parents]+=1
    assert quota.min()>0 and quota.sum()==n
    assignments=[];history=[centers.copy()];costs=[]
    for _ in range(20):
        distance=np.square(query[:,None,:]-centers[None,:,:]).sum(2)
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
        costs.append(float(distance[np.arange(n),assignment].sum(dtype=np.float64)))
        centers=np.stack([query[assignment==parent].mean(0) for parent in range(parents)])
        assert np.isfinite(centers).all();assignments.append(assignment);history.append(centers.copy())
    final=np.ascontiguousarray(centers,dtype=np.float32);assert np.isfinite(final).all()
    trace=dict(query_F32=np.ascontiguousarray(query,dtype=np.float32),quotas_int64=quota,
        assignments_int64=np.stack(assignments),centers_F64=np.stack(history),
        assignment_costs_F64=np.array(costs,dtype=np.float64))
    record=dict(method='fixed20_F64_greedy_quota_assignments_and_centroid_means',rows=n,parents=16,
        quota_min=int(quota.min()),quota_max=int(quota.max()),iterations=20,
        starting_centers='warm_BYTE_input' if chosen is None else 'NEW_FIT_farthest_first',
        farthest_seed_indices=chosen,constrained_optimality_or_monotonicity_claim=False,
        balanced_assignments_are_NOT_actual_inference_support=True)
    return final,trace,record
