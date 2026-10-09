"""Two common plus six selected private functions: eight active H128 functions.

Reuses original compact target operators and shared/private state mapping.
This is a new unqualified architecture; zero common output does not preserve
the old eight-private response after reducing private selection to six.
"""
import torch
from torch import nn
import chatbot_hybrid_target as base
import chatbot_hybrid_shared_private_target as shared

SCHEMA = 'COMPACT_FALCON_FIXED_WORK_TARGET_V2'
COMMON_COUNT = 2
PRIVATE_K = 6
PARAMETERS = shared.PARAMETERS
COMMON_SEED_PRIVATE_IDS = shared.COMMON_SEED_PRIVATE_IDS
assert COMMON_COUNT + PRIVATE_K == base.K == 8


class PrivateBanks(base.Banks):
    def __init__(self, existing):
        # Adopt the existing six arrays/router without allocating another bank.
        nn.Module.__init__(self)
        names = ('gate', 'up', 'down', 'gate_scale', 'up_scale', 'down_scale')
        assert tuple(dict(existing.named_parameters(recurse=False))) == names
        for name in names:
            self.register_parameter(name, getattr(existing, name))
        self.router = existing.router

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, base.D)
        scores = self.router(x)
        ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :PRIVATE_K]
        mass = torch.softmax(torch.gather(scores, 1, ids), dim=-1)
        y = torch.zeros_like(x)
        for e in range(base.E):
            positions, slots = (ids == e).nonzero(as_tuple=True)
            if positions.numel() == 0:
                continue
            inputs = x[positions]
            g = base.ternary_linear(inputs, self.gate[e], self.gate_scale[e])
            u = base.ternary_linear(inputs, self.up[e], self.up_scale[e])
            z = torch.nn.functional.silu(g) * u
            out = base.ternary_linear(z, self.down[e], self.down_scale[e])
            y = y.index_add(0, positions, out * mass[positions, slots, None])
        return y.reshape(shape)


class Target(shared.Target):
    def __init__(self, source, common_precision='ternary'):
        super().__init__(source, common_precision)
        for layer in self.layers:
            layer.banks.private = PrivateBanks(layer.banks.private)


# Parameter enumeration and storage are identical to the 2+8 shared/private
# variant; only private selection changes. No response equivalence is claimed.
canonical_private_name = shared.canonical_private_name
mapped_model = shared.mapped_model
mapped_optimizer = shared.mapped_optimizer
make_optimizer = shared.make_optimizer
