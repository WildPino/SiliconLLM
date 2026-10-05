"""Fixed convex support2/root partition16; no validation-driven selection."""
import numpy as np

STEPS, RATE = 32, .03
N, D, GROUPS, PARAMETERS = 128, 768, 62, 3344
UNIQUE = np.dtype([('meta', '<u4', (12,)), ('input_sha', 'u1', (32,)), ('score_sha', 'u1', (32,)),
                   ('value', '<f8', (3,)), ('input', '<f4', (768,)), ('score', '<f4', (128,))])
NODE_TARGET = np.dtype([('maximum', '<f4', (2,)), ('winner', '<u2', (2,)), ('mass', '<f8', (2,))])
TARGET = np.dtype([('unique_id', '<u4'), ('nodes', NODE_TARGET, (127,))])
PRED = np.dtype([('meta', '<u4', (12,)), ('value', '<f4', (3,)), ('root', '<f8', (4,)),
                 ('vector_ids', '<u2', (42,)), ('score', '<f4', (42,)), ('direction', '<u4', (7,))])
assert (UNIQUE.itemsize, TARGET.itemsize, PRED.itemsize) == (3720, 3560, 372)

def simplex(row):
    assert row.ndim == 1 and np.isfinite(row).all()
    ordered = np.sort(row)[::-1]
    means = (np.cumsum(ordered, dtype='<f8') - 1) / np.arange(1, len(row) + 1)
    last = np.flatnonzero(ordered > means)[-1]
    projected = np.maximum(row - means[last], 0)
    projected /= projected.sum(dtype='<f8')
    assert np.min(projected) >= 0 and abs(projected.sum(dtype='<f8') - 1) <= 2e-15
    return projected

def project(matrix):
    return np.array([simplex(row) for row in matrix], '<f8')

def support(score, maximum, omega, a):
    values = score @ a.T
    selected = np.argmax(values, axis=1)
    prediction = values[np.arange(len(score)), selected]
    error = prediction - maximum
    derivative = 2 * omega * error
    gradient = np.array([score.T @ (derivative * (selected == j)) for j in range(2)], '<f8')
    return float(np.sum(omega * error * error, dtype='<f8')), gradient

def partition(score, target, omega, a, theta):
    shifted = theta - theta.max()
    probability_b = np.exp(shifted)
    probability_b /= probability_b.sum(dtype='<f8')
    log_b = np.log(N) + theta - (theta.max() + np.log(np.exp(shifted).sum(dtype='<f8')))
    values = score @ a.T + log_b
    top = values.max(axis=1)
    terms = np.exp(values - top[:, None])
    total = terms.sum(axis=1, dtype='<f8')
    rho = terms / total[:, None]
    prediction = top + np.log(total)
    error = prediction - target
    derivative = 2 * omega * error
    gradient_a = (derivative[:, None] * rho).T @ score
    gradient_theta = (derivative[:, None] * rho).sum(axis=0, dtype='<f8') - probability_b * derivative.sum(dtype='<f8')
    return float(np.sum(omega * error * error, dtype='<f8')), gradient_a, gradient_theta, log_b

def adam(parameter, gradient, first, second, step):
    first = .9 * first + .1 * gradient
    second = .999 * second + .001 * gradient * gradient
    update = RATE * (first / (1 - .9 ** step)) / (np.sqrt(second / (1 - .999 ** step)) + 1e-8)
    return parameter - update, first, second

def groups(tree):
    result = []
    internal = [v for v in tree if v['count'] > 1]
    for target_index, node in enumerate(internal):
        for child, name in enumerate(('left', 'right')):
            group = tree[node[name]]
            if group['count'] >= 4:
                result.append({'group': group['node'], 'target_index': target_index,
                               'child': child, 'ids': group['ids']})
    assert len(result) == 62 and sum(2 * len(v['ids']) for v in result) == 1280
    return result

def fit(score, targets, omega, tree, guard, checkpoint):
    assert score.dtype == np.dtype('<f8') and score.shape[1] == 128
    assert abs(omega.sum(dtype='<f8') - 1) <= 1e-12 and np.min(omega) > 0
    group_list = groups(tree)
    initial = np.zeros(PARAMETERS, '<f8')
    parameter = np.zeros((STEPS, PARAMETERS), '<f8')
    gradients, firsts, seconds = [np.zeros_like(parameter) for _ in range(3)]
    loss_before = np.zeros((STEPS, 63), '<f8')
    loss_final = np.zeros(63, '<f8')
    offset = 0
    for k, group in enumerate(group_list):
        ids = group['ids']
        s = np.ascontiguousarray(score[:, ids])
        maximum = targets['nodes']['maximum'][:, group['target_index'], group['child']].astype('<f8')
        assert np.array_equal(s.max(axis=1), maximum)
        a = np.zeros((2, len(ids)), '<f8')
        a[0, 0] = a[1, -1] = 1
        size = a.size
        initial[offset:offset + size] = a.ravel()
        first, second = np.zeros_like(a), np.zeros_like(a)
        for step in range(1, STEPS + 1):
            loss, gradient = support(s, maximum, omega, a)
            loss_before[step - 1, k] = loss
            value, first, second = adam(a, gradient, first, second, step)
            a = project(value)
            parameter[step - 1, offset:offset + size] = a.ravel()
            gradients[step - 1, offset:offset + size] = gradient.ravel()
            firsts[step - 1, offset:offset + size] = first.ravel()
            seconds[step - 1, offset:offset + size] = second.ravel()
            guard()
        loss_final[k] = support(s, maximum, omega, a)[0]
        group.update(offset=offset, size=size, vector_offset=128 + 2 * k)
        offset += size
        if (k + 1) % 8 == 0:
            checkpoint(support_groups_fitted=k + 1)
    assert offset == 1280
    mass_groups = [v for v in tree if v['count'] == 8]
    assert len(mass_groups) == 16
    a = np.zeros((16, 128), '<f8')
    for j, group in enumerate(mass_groups):
        a[j, group['ids']] = 1 / 8
    theta = np.zeros(16, '<f8')
    initial[1280:3328] = a.ravel()
    return {'groups': group_list, 'mass_initial': a, 'theta_initial': theta,
            'initial': initial, 'parameter': parameter, 'gradient': gradients,
            'first': firsts, 'second': seconds, 'loss_before': loss_before, 'loss_final': loss_final}

def fit_mass(score, root_target, omega, history, guard):
    a, theta = history.pop('mass_initial'), history.pop('theta_initial')
    af, av, tf, tv = np.zeros_like(a), np.zeros_like(a), np.zeros_like(theta), np.zeros_like(theta)
    for step in range(1, STEPS + 1):
        loss, ga, gt, _ = partition(score, root_target, omega, a, theta)
        history['loss_before'][step - 1, 62] = loss
        value, af, av = adam(a, ga, af, av, step)
        a = project(value)
        theta, tf, tv = adam(theta, gt, tf, tv, step)
        theta -= theta.mean(dtype='<f8')
        for key, value_a, value_t in (('parameter', a, theta), ('gradient', ga, gt),
                                     ('first', af, tf), ('second', av, tv)):
            history[key][step - 1, 1280:3328] = value_a.ravel()
            history[key][step - 1, 3328:3344] = value_t
        guard()
    loss, _, _, log_b = partition(score, root_target, omega, a, theta)
    history['loss_final'][62] = loss
    return log_b

def exported_vectors(weights, history):
    vectors = np.zeros((268, 768), '<f4')
    vectors[:128] = weights
    def blend(a, ids):
        value = np.zeros((len(a), 768), '<f8')
        for k, expert in enumerate(ids):
            value += a[:, k, None] * weights[expert].astype('<f8')
        return value.astype('<f4')
    last = history['parameter'][-1]
    for group in history['groups']:
        start, size, ids = group['offset'], group['size'], group['ids']
        a = last[start:start + size].reshape(2, len(ids))
        vectors[group['vector_offset']:group['vector_offset'] + 2] = blend(a, ids)
    vectors[252:268] = blend(last[1280:3328].reshape(16, 128), list(range(128)))
    assert np.isfinite(vectors).all()
    return vectors

def controls():
    # Fixed projections and independent small finite differences before any fit.
    for row, expected in (([-1., 2., 0.], [0., 1., 0.]), ([.2, .3, .5], [.2, .3, .5]),
                          ([0., 0., 0.], [1/3, 1/3, 1/3]), ([2., 2.], [.5, .5])):
        assert np.allclose(simplex(np.array(row)), expected, rtol=0, atol=1e-15)
    score = np.array([[2., -1., .3], [-.4, 1.5, .7], [1.2, .5, -.8]], '<f8')
    omega = np.array([.2, .3, .5], '<f8')
    a = np.array([[.8, .1, .1], [.1, .8, .1]], '<f8')
    maximum = score.max(axis=1)
    _, g = support(score, maximum, omega, a)
    epsilon, worst = 1e-6, 0.
    for index in np.ndindex(a.shape):
        left, right = a.copy(), a.copy()
        left[index] -= epsilon
        right[index] += epsilon
        estimate = (support(score, maximum, omega, right)[0] - support(score, maximum, omega, left)[0]) / (2 * epsilon)
        worst = max(worst, abs(estimate - g[index]))
        assert abs(estimate - g[index]) <= 1e-8
    theta = np.array([.1, -.2], '<f8')
    target = np.array([3., 3.2, 3.4], '<f8')
    _, ga, gt, _ = partition(score, target, omega, a, theta)
    for index in np.ndindex(a.shape):
        left, right = a.copy(), a.copy()
        left[index] -= epsilon
        right[index] += epsilon
        estimate = (partition(score, target, omega, right, theta)[0] - partition(score, target, omega, left, theta)[0]) / (2 * epsilon)
        worst = max(worst, abs(estimate - ga[index]))
        assert abs(estimate - ga[index]) <= 1e-8
    for j in range(2):
        left, right = theta.copy(), theta.copy()
        left[j] -= epsilon
        right[j] += epsilon
        estimate = (partition(score, target, omega, a, right)[0] - partition(score, target, omega, a, left)[0]) / (2 * epsilon)
        worst = max(worst, abs(estimate - gt[j]))
        assert abs(estimate - gt[j]) <= 1e-8
    assert abs(gt.sum()) <= 1e-12
    parameter = np.array([.5, .5], '<f8')
    gradient = np.array([.2, -.2], '<f8')
    updated, first, second = adam(parameter, gradient, np.zeros(2), np.zeros(2), 1)
    assert np.allclose(first, .1 * gradient, rtol=0, atol=1e-15)
    assert np.allclose(second, .001 * gradient * gradient, rtol=0, atol=1e-15)
    assert np.allclose(updated, parameter - RATE * gradient / (np.abs(gradient) + 1e-8), rtol=0, atol=1e-14)
    return {'simplex_controls': 4, 'gradient_controls': 14, 'maximum_gradient_absolute_difference': worst,
            'Adam_first_step_control': True, 'support_prototype_tie': 'first prototype',
            'root_theta_shift_invariance_gradient': True}
