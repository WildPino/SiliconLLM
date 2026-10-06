"""ONE deterministic exact support-packing attempt, parameterized for small controls."""
import numpy as np
def pack(support,uids,C,B):
    H=support.shape[1];size=support.sum(axis=1)
    order=np.lexsort((uids,-size.astype('<i8')));unions=np.zeros((C,H),bool)
    assigned=np.full(len(uids),-1,'<i4');steps=[];first=None
    for k in order:
        k=int(k);before=unions.sum(axis=1);added=np.sum(support[k]&~unions,axis=1);enlarged=before+added
        feasible=np.flatnonzero(enlarged<=B)
        if not len(feasible):
            first=int(uids[k]);steps.append([int(uids[k]),-1,*map(int,before),*map(int,added)]);break
        c=min(map(int,feasible),key=lambda c:(int(added[c]),int(enlarged[c]),c))
        steps.append([int(uids[k]),c,*map(int,before),*map(int,added)]);assigned[k]=c;unions[c]|=support[k]
    return order,assigned,unions,steps,first
def fill(unions,B):
    C,H=unions.shape;result=unions.copy();base=int(unions.sum());unique=int(unions.any(axis=0).sum())
    feasible=H-unique<=C*B-base
    if not feasible:return None,{'base_slots':base,'base_union':unique,'duplicate_slots':base-unique,'missing_source_leaves':H-unique,'available_slots':C*B-base,'filled':False}
    assert np.all(unions.sum(axis=1)<=B) and B<=H
    missing=np.flatnonzero(~result.any(axis=0));copies=[]
    for j in missing:
        sizes=result.sum(axis=1);c=min((c for c in range(C) if sizes[c]<B),key=lambda c:(int(sizes[c]),c));result[c,j]=True;copies.append([int(j),c])
    for c in range(C):
        need=B-int(result[c].sum());result[c,np.flatnonzero(~result[c])[:need]]=True
    assert np.all(result.sum(axis=1)==B) and np.all(result.any(axis=0))
    return result,{'base_slots':base,'base_union':unique,'duplicate_slots':base-unique,'missing_source_leaves':H-unique,'available_slots':C*B-base,'filled':True,'missing_leaf_placements':copies}
def controls():
    s=np.zeros((5,8),bool)
    for i,j in enumerate([[0,1],[1,2],[3,4],[5,6],[6,7]]):s[i,j]=True
    order,a,u,steps,fail=pack(s,np.arange(5),4,4);f,counts=fill(u,4)
    assert fail is None and a.tolist()==[0,0,1,2,2] and np.array_equal(order,np.arange(5))
    assert np.all(np.logical_or(~s,f[a])) and np.all(f.sum(axis=1)==4) and np.all(f.any(axis=0))
    s=np.array([[1,1,0,0],[0,0,1,1]],bool);_,a,u,steps,fail=pack(s,np.arange(2),1,2);assert fail==1 and a.tolist()==[0,-1]
    u=np.zeros((4,8),bool);u[:,:4]=True;f,counts=fill(u,4);assert f is None and counts['missing_source_leaves']==4 and counts['available_slots']==0
    return {'fixed_tie_pack_complete_cover':True,'first_no_feasible_child_retained':True,'global_union_fill_duplicate_budget_obstruction':True}
