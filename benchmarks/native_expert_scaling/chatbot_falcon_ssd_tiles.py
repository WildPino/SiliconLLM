"""Bound temporary storage while retaining source SSD products and reductions."""
import inspect
import textwrap


def install(source_code):
    import torch

    def group_products(C, B):
        return torch.cat([(C[:, i:i+1, :, None, :, :] * B[:, i:i+1, None, :, :, :]).sum(-1)
                          for i in range(C.shape[1])], dim=1)

    def group_states(B, x):
        return torch.cat([(B[:, i:i+1, ..., None, :] * x[:, i:i+1, ..., None]).sum(dim=2)
                          for i in range(B.shape[1])], dim=1)

    def group_outputs(C, states):
        return torch.cat([(C[:, i:i+1, ..., None, :] * states[:, i:i+1, None, ...]).sum(-1)
                          for i in range(C.shape[1])], dim=1)

    original = source_code.FalconH1Mixer.torch_forward
    text = textwrap.dedent(inspect.getsource(original))
    replacements = (
        ('    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)\n'
         '    G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)\n',
         '    G = _ssd_group_products(C, B)\n'),
        ('    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)\n',
         '    states = _ssd_group_states(B_decay, hidden_states)\n'),
        ('    C_times_states = (C[..., None, :] * states[:, :, None, ...])\n',
         '    C_reduced_states = _ssd_group_outputs(C, states)\n'),
        ('    Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])\n',
         '    Y_off = (C_reduced_states * state_decay_out_permuted[..., None])\n'))
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    namespace = dict(vars(source_code), _ssd_group_products=group_products,
                     _ssd_group_states=group_states, _ssd_group_outputs=group_outputs)
    exec(compile(text, '<bound-source-SSD-tiles>', 'exec'), namespace)
    source_code.FalconH1Mixer.torch_forward = namespace['torch_forward']
    return dict(source_method=original.__qualname__, changed_contractions=3, tile_axis='source chunk axis',
                source_chunk_size_unchanged=True, generated_method=text)
