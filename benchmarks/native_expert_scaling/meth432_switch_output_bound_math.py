"""Convex output-distribution bound, not a trainable model or native artifact."""
from decimal import Decimal, localcontext
import numpy as np


def log_mixture(logp, logr, t):
    assert t >= 1 and np.isfinite(t)
    return np.logaddexp(logp+np.log(t), logr)-np.log1p(t)


def book_metrics(p, r, logp, logr, t):
    logq = log_mixture(logp, logr, t)
    norms = np.sum(np.exp(logq), axis=1)
    assert np.max(np.abs(norms-1)) <= 5e-13
    h256 = -np.sum(p*logp, axis=1); h128 = -np.sum(r*logr, axis=1)
    ce256 = -np.sum(p*logq, axis=1); ce128 = -np.sum(r*logq, axis=1)
    kl256 = np.sum(p*(logp-logq), axis=1); kl128 = np.sum(r*(logr-logq), axis=1)
    assert min(float(np.min(kl256)), float(np.min(kl128))) >= -1e-12
    assert max(float(np.max(np.abs(ce256-h256-kl256))), float(np.max(np.abs(ce128-h128-kl128)))) <= 1e-10
    return dict(h256=h256, h128=h128, ce256=ce256, ce128=ce128, mix=.5*(ce256+ce128), kl256=kl256, kl128=kl128)


def kl_mean(book, t):
    p, r, logp, logr = book
    return float(np.mean(np.sum(p*(logp-log_mixture(logp, logr, t)), axis=1)))


def root_feasible(function, cap, guard):
    """Return the feasible end of a deterministic monotone bracket."""
    first = function(1.); guard()
    if first <= cap: return 1., dict(active=False, evaluations=1, feasible_value=first, bracket=[1., 1.])
    lo, hi = 1., 2.; evaluations = 1
    for _ in range(32):
        value = function(hi); evaluations += 1; guard()
        if value <= cap: break
        lo = hi; hi *= 2
    else: raise AssertionError('constraint_bracket_did_not_close_32_doublings')
    for _ in range(60):
        if hi-lo <= 1e-12*max(1., hi): break
        mid = .5*(lo+hi); value = function(mid); evaluations += 1; guard()
        if value <= cap: hi = mid
        else: lo = mid
    else: raise AssertionError('constraint_bisection_did_not_close_60_iterations')
    value = function(hi); evaluations += 1; guard(); assert value <= cap+1e-12
    return hi, dict(active=True, evaluations=evaluations, feasible_value=value, bracket=[lo, hi])


def solve(books, mean_cap, book_cap, guard):
    assert books and all(len(book[0]) == len(books[0][0]) for book in books)
    floors = []; book_roots = []
    for book in books:
        t, detail = root_feasible(lambda t: kl_mean(book, t), book_cap, guard)
        floors.append(t); book_roots.append(detail)
    def global_constraint(t):
        return float(np.mean([kl_mean(book, max(t, floor)) for book, floor in zip(books, floors)]))
    global_t, global_root = root_feasible(global_constraint, mean_cap, guard)
    final_t = np.maximum(floors, global_t); metrics = [book_metrics(*book, float(t)) for book, t in zip(books, final_t)]
    kl = np.asarray([np.mean(m['kl256']) for m in metrics]); primal = float(np.mean([np.mean(m['mix']) for m in metrics]))
    lam = (global_t-1)/2; mu = (final_t-global_t)/2
    assert lam >= 0 and np.min(mu) >= 0 and np.mean(kl) <= mean_cap+1e-12 and np.max(kl) <= book_cap+1e-12
    dual = primal+lam*(float(np.mean(kl))-mean_cap)+float(np.mean(mu*(kl-book_cap)))
    gap = primal-dual; assert -1e-12 <= gap <= 1e-8, ('primal_dual_gap', gap)
    # Independent coefficient check of simplex stationarity at weighted optimum.
    maximum_stationarity = 0.
    for book, t, u in zip(books, final_t, mu):
        p, r, lp, lr = book; lq = log_mixture(lp, lr, float(t))
        # Ratio in log space avoids artificial 0/0 in saturated vocabulary tails.
        log_coefficient = np.logaddexp(np.log(.5+lam+float(u))+lp, np.log(.5)+lr)
        ratio = np.exp(log_coefficient-lq)
        error = float(np.max(np.abs(ratio-(1+lam+float(u))))) / max(1., 1+lam+float(u))
        maximum_stationarity = max(maximum_stationarity, error)
    assert maximum_stationarity <= 1e-10
    return dict(global_t=global_t, book_t=final_t.tolist(), book_minimum_t=floors, lambda_global=lam, mu_books=mu.tolist(),
                mean_kl256=float(np.mean(kl)), book_kl256=kl.tolist(), primal_CE=primal, dual_lower_CE=dual,
                primal_dual_gap=gap, maximum_stationarity_relative=maximum_stationarity, mean_cap=mean_cap, book_cap=book_cap,
                global_root=global_root, book_roots=book_roots), metrics


def fixture(guard, path):
    # Known multipliers t_global=2 and t_book=[4,2], independently constructed.
    p = np.asarray([[.7, .2, .1], [.2, .7, .1], [.3, .3, .4], [.4, .4, .2]], np.float64)
    r = np.asarray([[.1, .2, .7], [.1, .2, .7], [.35, .25, .4], [.35, .45, .2]], np.float64)
    with localcontext() as ctx:
        ctx.prec = 80; decp = [[Decimal(float(v)) for v in row] for row in p]; decr = [[Decimal(float(v)) for v in row] for row in r]
        q = []; dp = []; ce = []
        for i, (a, b) in enumerate(zip(decp, decr)):
            t = Decimal(4 if i < 2 else 2); c = [(t*x+y)/(t+1) for x, y in zip(a, b)]; q.append([float(v) for v in c])
            dp.append(sum(x*(x.ln()-z.ln()) for x, z in zip(a, c)))
            ce.append(-sum((x+y)/2*z.ln() for x, y, z in zip(a, b, c)))
        cap = float(sum(dp[:2])/2); mean_cap = float(sum(dp)/4); expected_ce = float(sum(ce)/4)
    assert float(sum(dp[2:])/2) < cap
    books = [(p[i:i+2], r[i:i+2], np.log(p[i:i+2]), np.log(r[i:i+2])) for i in (0, 2)]
    solved, metrics = solve(books, mean_cap, cap, guard); expected = np.asarray(q)
    actual = np.concatenate([np.exp(log_mixture(book[2], book[3], t)) for book, t in zip(books, solved['book_t'])])
    assert np.max(np.abs(actual-expected)) <= 1e-10 and abs(solved['primal_CE']-expected_ce) <= 1e-10
    assert abs(solved['global_t']-2) <= 1e-8 and np.max(np.abs(np.asarray(solved['book_t'])-[4, 2])) <= 1e-8
    # Unconstrained optimum identity and a wrong arithmetic/geometric-mixture control.
    mixture = .5*(p+r); expected_entropy = -np.sum(mixture*np.log(mixture), axis=1)
    mixed = np.concatenate([book_metrics(*book, 1.)['mix'] for book in books])
    assert np.max(np.abs(mixed-expected_entropy)) <= 1e-12
    wrong = np.sqrt(p*r); wrong /= np.sum(wrong, axis=1, keepdims=True)
    wrong_CE = -.5*np.sum((p+r)*np.log(wrong), axis=1)
    assert np.mean(wrong_CE-expected_entropy) > 1e-3
    # Missing /2 in multipliers must violate stationarity, not silently certify.
    wrong_lam = solved['global_t']-1; wrong_mu = np.asarray(solved['book_t'])-solved['global_t']
    defect = 0.
    for i in range(4):
        ratio = ((.5+wrong_lam+wrong_mu[i//2])*p[i]+.5*r[i])/actual[i]
        defect = max(defect, float(np.ptp(ratio)))
    assert defect > 1e-3
    np.savez(path, p=p, r=r, decimal80_known_q=expected, solved_q=actual, unconstrained_q=mixture,
             wrong_geometric_q=wrong, caps=np.asarray([mean_cap, cap]), book_t=np.asarray(solved['book_t']))
    return dict(solver=solved, Decimal80_expected_CE=expected_ce, max_probability_error=float(np.max(np.abs(actual-expected))),
                entropy_identity_max_error=float(np.max(np.abs(mixed-expected_entropy))), geometric_control_harm=float(np.mean(wrong_CE-expected_entropy)),
                wrong_multiplier_stationarity_defect=defect)
