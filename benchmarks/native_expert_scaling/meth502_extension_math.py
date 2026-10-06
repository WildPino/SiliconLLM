"""Monotone continuation from retained prefixes; integer sets, no old pack replay."""
def extend(unions, remaining, support, B, H):
    u=unions.copy();steps=[];packing_new=0
    for k in remaining:
        s=support[k];assert s.bit_count()<=B,('single_support_above_active_ceiling',k)
        feasible=[c for c,v in enumerate(u) if (v|s).bit_count()<=B]
        created=not feasible
        if created:c=len(u);u.append(0);packing_new+=1
        else:c=min(feasible,key=lambda c:((s&~u[c]).bit_count(),(s|u[c]).bit_count(),c))
        steps.append({'UID':int(k),'child':c,'created':created,'before':u[c].bit_count(),'added':(s&~u[c]).bit_count()})
        u[c]|=s
    S=sum(v.bit_count() for v in u);whole=0
    for v in u:whole|=v
    U=whole.bit_count();fill_new=0
    for j in range(H):
        if whole&(1<<j):continue
        feasible=[c for c,v in enumerate(u) if v.bit_count()<B]
        if not feasible:u.append(0);fill_new+=1;c=len(u)-1
        else:c=min(feasible,key=lambda c:(u[c].bit_count(),c))
        u[c]|=1<<j
    assert sum(v.bit_count() for v in u)==H+S-U
    return u,steps,{'S':S,'U':U,'missing':H-U,'copies':H+S-U,'packing_new_children':packing_new,'global_union_new_children':fill_new}

def controls():
    # Resume from a full retained child; append for a disjoint support, then fill.
    f,t,n=extend([3],[7],{7:12},2,6)
    assert f==[3,12,48] and t==[{'UID':7,'child':1,'created':True,'before':0,'added':2}]
    assert n=={'S':4,'U':4,'missing':2,'copies':6,'packing_new_children':1,'global_union_new_children':1}
    # Existing-child tie remains lowest ID; variable widths are not padded.
    f,t,n=extend([1,2],[8],{8:4},3,5)
    assert t[0]['child']==0 and f==[21,10] and n['copies']==5
    return {'resume_append_fill_exact_cost':True,'deterministic_tie_and_no_padding':True}
