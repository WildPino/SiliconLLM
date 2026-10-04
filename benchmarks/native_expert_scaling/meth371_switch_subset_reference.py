"""Explicit actualn64/128 META namespace/control closure; unchanged356 integer math."""
import numpy as np
import torch
import meth356_switch_all_a16_reference as R

compact_reference=R.compact_reference


def target_control_model(model, entries, payload):
    """Official META architecture with actual target F32 controls only.

    Quantized Linear parameters remain shape descriptors on META; every such
    forward is replaced by serialized integer projection in compact_reference.
    """
    n=model.config.num_experts
    assert n in (64,128) and model.config.d_model==768 and model.config.d_ff==3072
    assert model.config.num_layers==model.config.num_decoder_layers==12
    namespace = model.state_dict()
    assert set(namespace) == set(entries) and len(namespace) == 24*n+248
    assert sum(value.numel() for value in namespace.values())-3*32128*768=={64:3790748928,128:7415217408}[n]
    for name, descriptor in namespace.items():
        assert list(descriptor.shape) == entries[name]['shape']
        assert descriptor.dtype == torch.float32 and descriptor.device.type == 'meta'
    mapped = np.memmap(payload, dtype=np.uint8, mode='r')
    controls = {}; cache = {}
    for name, entry in entries.items():
        if entry['encoding'] == 0:
            key = (entry['offset'], entry['elements'])
            if key not in cache:
                view = np.ndarray(tuple(entry['shape']), dtype='<f4', buffer=mapped, offset=entry['offset'])
                cache[key] = torch.from_numpy(view.copy())
            controls[name] = cache[key]
    incompatible = model.load_state_dict(controls, strict=False, assign=True)
    quantized = {name for name, entry in entries.items() if entry['encoding'] == 1}
    assert not incompatible.unexpected_keys and set(incompatible.missing_keys) == quantized
    model.tie_weights(); model.requires_grad_(False); model.eval()
    meta = {name for name, value in model.state_dict().items() if value.device.type == 'meta'}
    assert meta == quantized - {'lm_head.weight'}
    assert all(value.dtype == torch.float32 for value in model.parameters())
    assert all(value.device.type == 'cpu' for name, value in model.state_dict().items() if name not in meta)
    linear_names = {name + '.weight' for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    assert quantized <= linear_names
    return {'namespace_shapes_exact': len(namespace), 'actual_f32_control_names': len(controls),
            'quantized_linear_names': len(quantized), 'remaining_meta_descriptors': len(meta),
            'f32_unique_loaded_bytes': sum(value.numel()*4 for value in cache.values()),
            'official_architecture_target_only_reference': True}
