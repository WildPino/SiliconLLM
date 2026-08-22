import numpy as np
rng = np.random.default_rng(1)
D,T,dv,dk = 24, 400, 8, 6

def causal_softmax(Q,K,scale,T):
    S=(Q.T@K)*scale; m=np.triu(np.ones((T,T)),1).astype(bool)
    S=np.where(m,-np.inf,S); S=S-S.max(1,keepdims=True); E=np.exp(S)
    return E/E.sum(1,keepdims=True)
def elu1(x): return np.where(x>0,x+1,np.exp(np.clip(x,-30,30)))
def rownorm(A): return A/np.maximum(A.sum(1,keepdims=True),1e-12)

X=rng.standard_normal((D,T))
Wq,Wk=rng.standard_normal((dk,D)),rng.standard_normal((dk,D))
Wv0=rng.standard_normal((dv,D))
A_soft=causal_softmax(Wq@X,Wk@X,dk**-0.5,T)
A_lin =rownorm(elu1(Wq@X).T@elu1(Wk@X)*np.tril(np.ones((T,T))))
Zs=X@A_soft.T

def solve(Z,Zstar,lam_scale=1e-4,sym=True):
    H=Z@Z.T; lam=lam_scale*np.trace(H)/Z.shape[1]
    Hr=H+lam*np.eye(Z.shape[0])
    C=Wv0@Zstar@Z.T + (lam*Wv0 if sym else 0)
    return C@np.linalg.inv(Hr)
def err(W,Z): return np.linalg.norm(W@Z-Wv0@Zs,'fro')/np.linalg.norm(Wv0@Zs,'fro')
def recovery(Z):
    e_dn=err(Wv0,Z); e_sol=err(solve(Z,A_soft@0+Zs if False else Zs),Z)  # target always donor
    return (e_dn-e_sol)/e_dn, e_dn, e_sol

print("=== A. conditioning: cond(XX^T) vs cond(ZZ^T), and identity-control deviation ===")
for name,A in [("A_soft",A_soft),("A_lin(elu+1)",A_lin),("causal-uniform",rownorm(np.tril(np.ones((T,T)))))]:
    Z=X@A.T; H=Z@Z.T
    ev=np.linalg.eigvalsh(H); c=ev.max()/max(ev.min(),1e-300)
    lam=1e-4*np.trace(H)/T
    dev_brief=np.abs((Wv0@H)@np.linalg.inv(H+lam*np.eye(D))-Wv0).max()
    print("  %-16s cond=%.3e  lam=%.3e  IDENTITY dev (brief form)=%.3e" % (name,c,lam,dev_brief))
print("  cond(XX^T) = %.3e" % (lambda e:e.max()/e.min())(np.linalg.eigvalsh(X@X.T)))

print("\n=== B. recovery fraction: real A_lin vs nulls (in-sample, T>>D) ===")
nulls={"A_lin (elu+1)":A_lin,
       "causal-uniform (no content at all)":rownorm(np.tril(np.ones((T,T)))),
       "shuffled A_lin cols (breaks causality)":None,
       "A_lin from UNRELATED sequence":None,
       "random causal row-stochastic":rownorm(np.tril(rng.random((T,T))))}
P=rng.permutation(T); nulls["shuffled A_lin cols (breaks causality)"]=rownorm(A_lin[:,P])
X2=rng.standard_normal((D,T))
nulls["A_lin from UNRELATED sequence"]=rownorm(elu1(Wq@X2).T@elu1(Wk@X2)*np.tril(np.ones((T,T))))
for k,A in nulls.items():
    Z=X@A.T; W=solve(Z,Zs); e_dn=err(Wv0,Z); e_s=err(W,Z)
    print("  %-40s err(DN)=%.4f err(SOLVED)=%.4f  recovery=%.3f"%(k,e_dn,e_s,(e_dn-e_s)/e_dn))

print("\n=== C. T vs D: the interpolation degeneracy (in-sample recovery) ===")
for Tt in (12, 24, 40, 100, 400):
    Xs=X[:,:Tt]; As=causal_softmax(Wq@Xs,Wk@Xs,dk**-0.5,Tt)
    Al=rownorm(elu1(Wq@Xs).T@elu1(Wk@Xs)*np.tril(np.ones((Tt,Tt))))
    Zss=Xs@As.T; Z=Xs@Al.T
    H=Z@Z.T; lam=1e-4*np.trace(H)/Tt
    W=(Wv0@Zss@Z.T+lam*Wv0)@np.linalg.inv(H+lam*np.eye(D))
    n=np.linalg.norm(Wv0@Zss,'fro')
    e_dn=np.linalg.norm(Wv0@Z-Wv0@Zss,'fro')/n; e_s=np.linalg.norm(W@Z-Wv0@Zss,'fro')/n
    print("  T=%4d (D=%d)  err(DN)=%.4f err(SOLVED)=%.4f  recovery=%.3f"%(Tt,D,e_dn,e_s,(e_dn-e_s)/max(e_dn,1e-15)))

print("\n=== D. GQA: is the joint solve still closed-form? (G=3 q-heads share one Wv) ===")
G=3
Wqs=[rng.standard_normal((dk,D)) for _ in range(G)]
Asofts=[causal_softmax(Wq_@X,Wk@X,dk**-0.5,T) for Wq_ in Wqs]
Alins=[rownorm(elu1(Wq_@X).T@elu1(Wk@X)*np.tril(np.ones((T,T)))) for Wq_ in Wqs]
Zg=[X@A.T for A in Alins]; Zsg=[X@A.T for A in Asofts]
Hsum=sum(z@z.T for z in Zg); Csum=sum(Wv0@zs@z.T for zs,z in zip(Zsg,Zg))
W_joint=Csum@np.linalg.inv(Hsum)
# brute force lstsq over stacked system: [W Z_1 | W Z_2 | W Z_3] vs targets
Zcat=np.concatenate(Zg,axis=1); Ycat=np.concatenate([Wv0@zs for zs in Zsg],axis=1)
W_ls=np.linalg.lstsq(Zcat.T,Ycat.T,rcond=None)[0].T
print("  unweighted joint closed form vs lstsq: max diff = %.3e" % np.abs(W_joint-W_ls).max())
# W_o-weighted version
Wo=[rng.standard_normal((D,dv)) for _ in range(G)]
def obj_w(W): return sum(np.linalg.norm(Wo[i]@(W@Zg[i]-Wv0@Zsg[i]),'fro')**2 for i in range(G))
# true minimiser by vec/Kronecker
Amat=sum(np.kron((Zg[i]@Zg[i].T).T, Wo[i].T@Wo[i]) for i in range(G))
bvec=sum(((Wo[i].T@Wo[i])@Wv0@Zsg[i]@Zg[i].T).flatten(order='F') for i in range(G))
W_true=np.linalg.solve(Amat,bvec).reshape(dv,D,order='F')
print("  Wo-weighted GQA: obj(single-inverse)=%.6e  obj(true Kronecker)=%.6e  ratio=%.4f"%(
      obj_w(W_joint),obj_w(W_true),obj_w(W_joint)/obj_w(W_true)))
print("  max|W_joint - W_true| = %.3e   (0 would mean weighting is free)"%np.abs(W_joint-W_true).max())
# MHA control: single head, does Wo weighting change the answer?
Z1,Zs1=Zg[0],Zsg[0]
W_unw=(Wv0@Zs1@Z1.T)@np.linalg.inv(Z1@Z1.T)
G1=Wo[0].T@Wo[0]
W_w=np.linalg.solve(np.kron((Z1@Z1.T).T,G1),(G1@Wv0@Zs1@Z1.T).flatten(order='F')).reshape(dv,D,order='F')
print("  MHA (1 head): max|unweighted - Wo-weighted| = %.3e  (expect ~0)"%np.abs(W_unw-W_w).max())

print("\n=== E. bias term: Qwen2 has q/k/v bias. Does it survive commuting? ===")
b=rng.standard_normal(dv)
for name,A in [("A_soft (rows sum to 1)",A_soft),("A_lin normalised",A_lin),
               ("A_lin UNnormalised",elu1(Wq@X).T@elu1(Wk@X)*np.tril(np.ones((T,T))))]:
    lhs=(Wv0@X+b[:,None])@A.T          # true: mix the biased values
    rhs=Wv0@(X@A.T)+b[:,None]          # brief's assumption
    print("  %-24s rowsum range=[%.3f,%.3f]  max|true-brief| = %.3e"%(
        name,A.sum(1).min(),A.sum(1).max(),np.abs(lhs-rhs).max()))
