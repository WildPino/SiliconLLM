"""Storage-only SSD schedule. Full original reduction axes remain intact.

This module is separate from the donor BF16 qualification. It changes neither
model geometry nor precision and installs only in the current worker process.
"""
import inspect
import textwrap
from chatbot_falcon_ssd_tiles import rewrite_source as three_contractions

OPERATIONS = ('products', 'states', 'boundaries', 'outputs')


def rewrite_source(text):
    text = three_contractions(text)
    old = 'new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)'
    assert text.count(old) == 1
    return text.replace(old, 'new_states = _ssd_group_boundaries(decay_chunk, states)')


def contract(name, a, b, tiled):
    import torch
    if name == 'products':
        if not tiled:
            return (a[:, :, :, None, :, :] * b[:, :, None, :, :, :]).sum(-1)
        return torch.cat([(a[:, i:i+1, :, None, :, :] * b[:, i:i+1, None, :, :, :]).sum(-1)
                          for i in range(a.shape[1])], 1)
    if name == 'states':
        if not tiled:
            return (a[..., None, :] * b[..., None]).sum(2)
        return torch.cat([(a[:, i:i+1, ..., None, :] * b[:, i:i+1, ..., None]).sum(2)
                          for i in range(a.shape[1])], 1)
    if name == 'boundaries':
        if not tiled:
            return (a[..., None, None] * b[:, :, None, ...]).sum(1)
        # Tile destination j. Every output still reduces ALL source chunks i.
        return torch.cat([(a[:, :, j:j+1, ..., None, None] * b[:, :, None, ...]).sum(1)
                          for j in range(a.shape[2])], 1)
    if name == 'outputs':
        if not tiled:
            return (a[..., None, :] * b[:, :, None, ...]).sum(-1)
        return torch.cat([(a[:, i:i+1, ..., None, :] * b[:, i:i+1, None, ...]).sum(-1)
                          for i in range(a.shape[1])], 1)
    raise ValueError(name)


def install(source_code, observer=None):
    original = source_code.FalconH1Mixer.torch_forward
    text = rewrite_source(textwrap.dedent(inspect.getsource(original)))
    def wrapped(name):
        def call(a, b):
            y = contract(name, a, b, True)
            if observer is not None:
                observer(name, (a, b), y)
            return y
        return call
    namespace = dict(vars(source_code))
    namespace.update({'_ssd_group_'+name: wrapped(name) for name in OPERATIONS})
    exec(compile(text, '<target-F32-SSD-storage>', 'exec'), namespace)
    source_code.FalconH1Mixer.torch_forward = namespace['torch_forward']
    return text
