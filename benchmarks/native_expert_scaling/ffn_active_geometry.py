"""Source remainder derivative metric and balanced FIT input factors."""
import numpy as np


def sigmoid(x): return np.exp(-np.logaddexp(0,-x))


def balanced_geometry(U,G,D,alpha,beta,center,fit_records,data,route,guard):
    moment = np.zeros((2048,2048),'f8'); total = 0; delta_sum = np.zeros(2048,'f8')
    aa = np.zeros(4608,'f8'); bb = aa.copy(); ab = aa.copy()
    weight = np.sum((beta*D)**2,axis=0)
    rng = np.random.default_rng(0); action_directions = rng.normal(size=(2048,4))
    action_directions /= np.linalg.norm(action_directions,axis=0)
    UV,GV = U@action_directions,alpha*(G@action_directions)
    action_gram = np.zeros((4,4),'f8'); fd = None
    for rec in fit_records:
        guard(); X = data(rec); labels = route(rec); delta = X-center['centers'][labels]
        u,g = X@U.T,alpha*(X@G.T); q = sigmoid(g); sx = g*q; derivative = q+g*q*(1-q)
        uc,gc,s,first = [center[k][labels] for k in ('u_center','g_center','s','derivative')]
        a,b = sx-s,u*derivative-uc*first
        aa += np.sum(a*a,axis=0); bb += np.sum(b*b,axis=0); ab += np.sum(a*b,axis=0)
        for left in range(4):
            lv = a*UV[:,left]+b*GV[:,left]
            for right in range(left,4):
                rv = a*UV[:,right]+b*GV[:,right]
                action_gram[left,right] += np.sum(weight*lv*rv)
        moment += delta.T@delta; delta_sum += delta.sum(axis=0); total += len(X)
        if fd is None:
            indices = np.linspace(0,len(X)-1,16,dtype='i8')
            directions = rng.normal(size=(16,2048)); directions /= np.linalg.norm(directions,axis=1)[:,None]
            analytic,numeric,denominators = [],[],[]
            for index,v in zip(indices,directions):
                guard(); pu,pg = U@v,alpha*(G@v)
                analytic.append(a[index]*pu+b[index]*pg)
                def remainder(sign):
                    uu,gg = u[index]+sign*1e-4*pu,g[index]+sign*1e-4*pg
                    ss = gg*sigmoid(gg)
                    return uc[index]*(ss-s[index]-first[index]*(gg-gc[index]))+(uu-uc[index])*(ss-s[index])
                numeric.append((remainder(1)-remainder(-1))/(2e-4))
                denominators.append(np.linalg.norm(np.sqrt(weight)*(sx[index]*pu+u[index]*derivative[index]*pg)))
            analytic,numeric,denominators = np.array(analytic),np.array(numeric),np.array(denominators)
            errors = np.linalg.norm((analytic-numeric)*np.sqrt(weight),axis=1)/np.maximum(denominators,1e-30)
            assert np.isfinite(errors).all() and errors.max() <= 1e-5
            fd = dict(FD_indices=indices,FD_directions=directions,FD_analytic=analytic,FD_numeric=numeric,
                      FD_denominators=denominators,FD_relative_errors=errors)
    aa,bb,ab = aa/total,bb/total,ab/total
    action_gram /= total; action_gram = action_gram+np.triu(action_gram,1).T
    guard(); metric = U.T@((weight*aa)[:,None]*U+(alpha*weight*ab)[:,None]*G)
    metric += G.T@((alpha*weight*ab)[:,None]*U+(alpha**2*weight*bb)[:,None]*G)
    metric = (metric+metric.T)/2
    observed = action_directions.T@metric@action_directions
    assert np.linalg.norm(observed-action_gram)/max(np.linalg.norm(action_gram),1e-30) <= 1e-10
    moment /= total; mean = delta_sum/total; covariance = moment-np.outer(mean,mean)
    guard(); sigma_values,sigma_basis = np.linalg.eigh(moment)
    sigma_values,sigma_basis = sigma_values[::-1].copy(),sigma_basis[:,::-1].copy()
    assert sigma_values[-1] >= -1e-10*max(float(sigma_values[0]),1e-30)
    keep = sigma_values > 1e-12*sigma_values[0]; assert int(keep.sum()) >= 512
    supported = np.where(keep,sigma_values,0)
    root = (sigma_basis*np.sqrt(supported))@sigma_basis.T
    inv_values = np.zeros_like(sigma_values); inv_values[keep] = 1/np.sqrt(sigma_values[keep])
    inv_root = (sigma_basis*inv_values)@sigma_basis.T
    guard(); metric_values = np.linalg.eigvalsh(metric)
    assert metric_values[0] >= -1e-10*max(float(metric_values[-1]),1e-30)
    balanced = root@metric@root; balanced = (balanced+balanced.T)/2
    guard(); values,vectors = np.linalg.eigh(balanced)
    values,vectors = values[::-1].copy(),vectors[:,::-1].copy()
    for j in range(2048):
        if vectors[np.argmax(np.abs(vectors[:,j])),j] < 0: vectors[:,j] *= -1
        if sigma_basis[np.argmax(np.abs(sigma_basis[:,j])),j] < 0: sigma_basis[:,j] *= -1
    assert values[-1] >= -1e-10*max(float(values[0]),1e-30)
    encoder,decoder = inv_root@vectors[:,:512],root@vectors[:,:512]
    assert np.linalg.norm(encoder.T@decoder-np.eye(512))/np.sqrt(512) <= 1e-10
    result = dict(moment=moment,covariance=covariance,mean=mean,eigenvalues=sigma_values,basis=sigma_basis,
        metric=metric,metric_eigenvalues=metric_values,balanced=balanced,balanced_eigenvalues=values,
        balanced_basis=vectors,encoder=encoder,decoder=decoder,
        U_projected=U@decoder,G_projected=alpha*(G@decoder),column_energy=weight,
        coefficient_aa=aa,coefficient_bb=bb,coefficient_ab=ab,action_directions=action_directions,
        action_gram=action_gram,**fd)
    return result,dict(FIT_rows=total,supported_rank=int(keep.sum()),
        dropped_trace=float(np.maximum(sigma_values[~keep],0).sum()),
        sigma_support_condition=float(sigma_values[0]/sigma_values[keep][-1]),
        maximum_FD=float(fd['FD_relative_errors'].max()))
