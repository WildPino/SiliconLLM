"""One fixed affine conditional-mass recipe. No winner fitting or data selection."""
import numpy as np

STEPS, RATE = 32, .03
ETA = float(np.log1p(.01)) / 7

def pair(z):
    t = np.exp(-np.abs(z))
    small = t / (1 + t)
    large = 1 / (1 + t)
    return np.column_stack((np.where(z >= 0, small, large), np.where(z >= 0, large, small)))

def loss_gradient(x, y, omega, theta):
    z = x @ theta[:-1] + theta[-1]
    loss = np.sum(omega * (y * np.logaddexp(0, -z) + (1-y) * np.logaddexp(0, z)), dtype='<f8')
    r = omega * (pair(z)[:, 1] - y)
    gradient = np.r_[x.T @ r, r.sum(dtype='<f8')]
    return float(loss), gradient

def fit(x, y, omega, guard):
    assert x.dtype == np.dtype('<f4') and x.shape[1] == 768
    assert y.shape == omega.shape == (len(x),) and np.all((y >= 0) & (y <= 1))
    assert np.all(omega > 0) and abs(omega.sum()-1) < 1e-12
    raw = x.astype('<f8')
    mu = omega @ raw
    constant = np.all(raw == raw[0], axis=0)
    mu[constant] = raw[0, constant]
    raw -= mu
    centered = raw
    sigma = np.sqrt(omega @ (centered * centered))
    sigma[constant] = 1.
    assert np.all(sigma > 0) and np.isfinite(sigma).all()
    centered /= sigma
    standardized = centered
    theta = np.zeros(769, '<f8')
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    history = {k:np.empty((STEPS,769),'<f8') for k in ('before','gradient','first','second','after')}
    losses = np.empty(STEPS, '<f8')
    for step in range(1,STEPS+1):
        losses[step-1], gradient = loss_gradient(standardized,y,omega,theta)
        history['before'][step-1] = theta
        first = .9*first + .1*gradient
        second = .999*second + .001*gradient*gradient
        theta = theta - RATE*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
        for key,v in (('gradient',gradient),('first',first),('second',second),('after',theta)):
            history[key][step-1] = v
        guard()
    coefficient = theta[:-1]/sigma
    correction = 0.
    for j in range(768):correction += float(coefficient[j]*mu[j])
    bias = theta[-1] - correction
    head = np.r_[coefficient,bias].astype('<f4')
    assert np.isfinite(head).all()
    return {**history,'loss':losses,'mu':mu,'sigma':sigma,'constant':constant,'head':head,
            'final_loss':np.array(loss_gradient(standardized,y,omega,theta)[0],'<f8')}

def ordered_logits(x, head):
    # Source AVX dot order, plus the new F32 bias in F64, then ONE F32 cast.
    accum = np.zeros((len(x),8),'<f8')
    for offset in range(0,768,8):
        accum += x[:,offset:offset+8].astype('<f8') * head[offset:offset+8].astype('<f8')
    halves = accum[:,:4] + accum[:,4:]
    total = np.zeros(len(x),'<f8')
    for lane in range(4): total += halves[:,lane]
    return (total + float(head[-1])).astype('<f4')

def controls():
    x=np.array([[.2,-.4],[1.,.3],[-.7,.1]],'<f8')
    y=np.array([0.,.3,1.]);omega=np.array([.2,.3,.5]);theta=np.array([.1,-.2,.4])
    loss,g=loss_gradient(x,y,omega,theta);errors=[]
    for j in range(3):
        lo,hi=theta.copy(),theta.copy();lo[j]-=1e-6;hi[j]+=1e-6
        fd=(loss_gradient(x,y,omega,hi)[0]-loss_gradient(x,y,omega,lo)[0])/2e-6
        assert abs(fd-g[j])<1e-9;errors.append(abs(fd-g[j]))
    p=pair(np.array([-1000.,-80.,-1.,0.,1.,80.,1000.]))
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.all(np.abs(p.sum(axis=1)-1)<3e-16)
    assert p[0].tolist()==[1.,0.] and p[-1].tolist()==[0.,1.] and p[3].tolist()==[.5,.5]
    return {'soft_target_BCE_gradient_max_error':max(errors),'endpoint_targets_not_clipped':True,
            'stable_pair_zero_one_and_extreme_controls':True,'loss':loss}
