"""Only zero-readout numerical qualification repair; fitted arithmetic unchanged."""
import numpy as np
import meth439_switch_shared_readout_fd as P
H = P.H
G, R, L = H.G, H.R, H.L
forward, parameters = H.forward, H.parameters


def qualify(label, feature, base, scores, chosen, buffers, fn, head, params, target, tiny, path, expected=None):
    if tiny:
        return H.qualify(label, feature, base, scores, chosen, buffers, fn, head, params, target, tiny, path, expected)
    record = P.diagnose(label, feature, base, scores, chosen, buffers, fn, head, params, target, path, expected, label.startswith('adapter'))
    with np.load(path, allow_pickle=False) as z:
        wrong = z['continuation_logits']-z['offset_logits']; native = z['native_logits']; error = R.relative(wrong, native)
        assert error > 1e-12 or np.max(np.abs(wrong-native)) > 1e-9
    record['negatives'] = {'dropped_head_offset_detected': True, 'relative': error}
    return record
