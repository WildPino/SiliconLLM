import numpy as np
np.random.seed(0)
D, T, dv, dk = 17, 64, 6, 5   # tens, fp64

def causal_softmax(Q,K,scale):
    S = (Q.T@K)*scale                      # [T,T]
    m = np.triu(np.ones((T,T)),1).astype(bool)
    S = np.where(m, -np.inf, S)
    S = S - S.max(1,keepdims=True)
    E = np.exp(S); return E/E.sum(1,keepdims=True)

X  = np.random.randn(D,T)
Wq, Wk = np.random.randn(dk,D), np.random.randn(dk,D)
Wv0 = np.random.randn(dv,D)
A_soft = causal_softmax(Wq@X, Wk@X, dk**-0.5)

# a "wrong" linear attention: elu+1 feature map, causal, row-normalised
def elu1(x): return np.where(x>0,x+1,np.exp(np.clip(x,-30,30)))
Qf, Kf = elu1(Wq@X), elu1(Wk@X)
Araw = (Qf.T@Kf)*np.tril(np.ones((T,T)))
A_lin = Araw/Araw.sum(1,keepdims=True)

Z  = X@A_lin.T          # D x T
Zs = X@A_soft.T
Y  = Wv0@Zs             # donor target

print("=== C0. is the head output really Wv @ (X A^T)? ===")
V = Wv0@X
out_direct = V@A_soft.T
print("   max|Wv(XA^T) - (WvX)A^T| =", np.abs(Wv0@Zs - out_direct).max())

print("\n=== C1. brief's closed form vs lstsq, lam=0 ===")
brief0 = (Wv0@Zs@Z.T)@np.linalg.inv(Z@Z.T)
ls = np.linalg.lstsq(Z.T, Y.T, rcond=None)[0].T
print("   max|brief(lam=0) - lstsq| =", np.abs(brief0-ls).max())
def err(W): return np.linalg.norm(W@Z - Y,'fro')
print("   err(brief0)=%.6e  err(lstsq)=%.6e  err(donor/DO-NOTHING)=%.6e" % (err(brief0),err(ls),err(Wv0)))
# perturbation test: is it a true minimum?
worse = [err(brief0 + 1e-6*np.random.randn(dv,D)) for _ in range(5)]
print("   perturbed errs all >= err(brief0):", all(w>=err(brief0) for w in worse))

print("\n=== C2. identity case A_lin := A_soft, with the brief's lam ===")
H = Zs@Zs.T; N = T
for lam_scale in (1e-4, 0.0):
    lam = lam_scale*np.trace(H)/N
    brief = (Wv0@Zs@Zs.T)@np.linalg.inv(Zs@Zs.T + lam*np.eye(D))       # brief's form
    sym   = (Wv0@(Zs@Zs.T + lam*np.eye(D)))@np.linalg.inv(Zs@Zs.T + lam*np.eye(D))  # D4 convention
    print("   lam_scale=%g lam=%.4g" % (lam_scale,lam))
    print("     brief form : max_abs_weight_deviation = %.6e   rel = %.3e" % (
        np.abs(brief-Wv0).max(), np.linalg.norm(brief-Wv0)/np.linalg.norm(Wv0)))
    print("     D4 symmetric: max_abs_weight_deviation = %.6e" % np.abs(sym-Wv0).max())

print("\n=== C3. does the D4-symmetric form solve a stated objective? ===")
lam = 1e-4*np.trace(Z@Z.T)/N
sym = (Wv0@(Zs@Z.T + lam*np.eye(D)))@np.linalg.inv(Z@Z.T + lam*np.eye(D))
# claim: sym minimises ||WZ - Wv0 Zs||^2 + lam||W - Wv0||^2
def obj_sym(W): return np.linalg.norm(W@Z-Y,'fro')**2 + lam*np.linalg.norm(W-Wv0,'fro')**2
g = 2*(sym@Z-Y)@Z.T + 2*lam*(sym-Wv0)
print("   grad-norm at sym of [||WZ-Y||^2 + lam||W-Wv0||^2] =", np.linalg.norm(g))
brief = (Wv0@Zs@Z.T)@np.linalg.inv(Z@Z.T+lam*np.eye(D))
g2 = 2*(brief@Z-Y)@Z.T + 2*lam*brief
print("   grad-norm at brief of [||WZ-Y||^2 + lam||W||^2]    =", np.linalg.norm(g2))
print("   err(sym)=%.6e err(brief)=%.6e err(donor)=%.6e" % (err(sym),err(brief),err(Wv0)))
