"""CPU native-value forward with explicitly approximate smooth backwards.

No gradient is claimed through native rounding or hard expert selection.
The saved-primal straight-through rules are qualified by a separate controller.
"""
import numpy as np
import torch
import meth418_switch_function_capture as C


def array(x):
    assert x.device.type == 'cpu' and x.dtype == torch.float32
    return x.detach().numpy()


def result(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32).copy())


class I8Operator:
    def __init__(self, weights, scales):
        assert weights.dtype == np.int8 and weights.ndim == 2
        assert np.all(weights != -128)
        assert scales.dtype == np.float32 and scales.shape == (weights.shape[0],)
        assert np.isfinite(scales).all() and np.all(scales > 0)
        self.weights = weights
        self.scales = scales
        self.integers = weights.astype(np.int64)
        self._smooth = None

    def native(self, x):
        scale, codes = C.quant(x)
        sums = self.integers @ codes.astype(np.int64)
        return ((sums.astype(np.float64)*self.scales.astype(np.float64))*np.float64(scale)).astype(np.float32)

    def smooth_weight(self):
        if self._smooth is None:
            self._smooth = torch.from_numpy(self.weights.astype(np.float64)*self.scales.astype(np.float64)[:, None])
        return self._smooth


class NativeI8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op):
        ctx.op = op
        return result(op.native(array(x)))

    @staticmethod
    def backward(ctx, dy):
        # Identity STE for A16 activation quantization. Frozen I8 weights/scales.
        return (ctx.op.smooth_weight().T @ dy.to(torch.float64)).to(torch.float32), None


class NativeFloat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return result(array(weight).astype(np.float64) @ array(x).astype(np.float64))

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        g = dy.to(torch.float64)
        return ((weight.to(torch.float64).T @ g).to(torch.float32),
                torch.outer(g, x.to(torch.float64)).to(torch.float32))


class NativeRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        v = array(x)
        ctx.save_for_backward(x)
        ctx.weight = weight
        d = v.size
        # Switch: square in F32, sum in F64, mean/inverse/sqrt/multiply F32.
        mean = np.float32(np.sum((v*v).astype(np.float64))/d)
        scale = np.float32(1)/np.sqrt(np.float32(mean+np.float32(1e-6)))
        return result((v*scale)*weight)

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        v = x.to(torch.float64)
        w = torch.from_numpy(ctx.weight.astype(np.float64))
        g = dy.to(torch.float64)*w
        r = torch.rsqrt(torch.mean(v*v)+1e-6)
        return (g*r-v*(r*r*r)*(torch.dot(g, v)/v.numel())).to(torch.float32), None


class NativeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        v = array(x)
        return result(np.where(v < 0, np.float32(0), v))

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return dy*(x > 0).to(dy.dtype)


class NativeProbability(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, chosen):
        v = array(scores)
        assert np.isfinite(v).all() and int(np.argmax(v)) == chosen
        ctx.save_for_backward(scores)
        ctx.chosen = chosen
        exponential = np.exp((v-v[chosen]).astype(np.float64)).astype(np.float32).astype(np.float64)
        return result(np.float32(1/np.sum(exponential)))

    @staticmethod
    def backward(ctx, dy):
        scores, = ctx.saved_tensors
        p = torch.softmax(scores.to(torch.float64), dim=0)
        gradient = -p[ctx.chosen]*p
        gradient[ctx.chosen] += p[ctx.chosen]
        return (dy.to(torch.float64)*gradient).to(torch.float32), None


def corrections(d, rank, seed, zero=True):
    rng = np.random.default_rng(seed)
    values = {}
    for name, shape in (('A', (d, rank)), ('B', (rank, d)), ('C', (d, rank)), ('D', (rank, d))):
        value = np.zeros(shape, np.float32) if zero and name in ('A', 'C') else rng.normal(0, 1/np.sqrt(d), shape).astype(np.float32)
        if not zero: value *= np.float32(.02)
        values[name] = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    return values


def forward(pre, scores, chosen, ffnorm, finalnorm, wi, wo, head, parameters=None):
    x = NativeRMS.apply(pre, ffnorm)
    corrected_input = x
    if parameters is not None:
        corrected_input = x+NativeFloat.apply(NativeFloat.apply(x, parameters['B']), parameters['A'])
    up_raw = NativeI8.apply(corrected_input, wi)
    up = NativeReLU.apply(up_raw)
    down = NativeI8.apply(up, wo)
    corrected_output = down
    if parameters is not None:
        corrected_output = down+NativeFloat.apply(NativeFloat.apply(down, parameters['D']), parameters['C'])
    probability = NativeProbability.apply(scores, chosen)
    post = pre+probability*corrected_output
    final = NativeRMS.apply(post, finalnorm)
    head_input = final*np.float32(1/np.sqrt(pre.numel()))
    logits = NativeI8.apply(head_input, head)
    return {'input': x, 'up_raw': up_raw, 'up': up, 'down': down, 'probability': probability,
            'post': post, 'final': final, 'head_input': head_input, 'logits': logits}


def smooth_forward(pre, scores, chosen, ffnorm, finalnorm, wi, wo, head, parameters):
    """F64 smooth twin: dequantized fixed weights, no activation rounding."""
    x = pre*torch.rsqrt(torch.mean(pre*pre)+1e-6)*ffnorm
    x = x+parameters['A'] @ (parameters['B'] @ x)
    up = torch.relu(wi @ x)
    down = wo @ up
    down = down+parameters['C'] @ (parameters['D'] @ down)
    post = pre+torch.softmax(scores, dim=0)[chosen]*down
    final = post*torch.rsqrt(torch.mean(post*post)+1e-6)*finalnorm
    return head @ (final/np.sqrt(pre.numel()))


def prediction_loss(logits, target):
    z = logits.to(torch.float64)
    return torch.logsumexp(z, dim=0)-torch.dot(target, z)
