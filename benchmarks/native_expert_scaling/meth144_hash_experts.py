"""Token-context grandchild choice inside the sparse expert forward path."""

import numpy as np
import torch

from meth136_sparse_experts import CpuSparseExperts
from meth142_token_hash_route_screen import hash_grandchildren


class TokenHashSparseExperts(CpuSparseExperts):
    def __init__(self, parent, layer_id, cpu_bank, shared_a):
        # The parent sparse class owns the shared-A/B gather path. Its seeded
        # projection/key buffers are unused by this deterministic router.
        projection = torch.zeros((32, 896), dtype=torch.float32)
        keys = torch.zeros((1280, 10, 32), dtype=torch.float32)
        super().__init__(parent, layer_id, cpu_bank, shared_a, projection, keys)
        self.layer_id = layer_id
        self.token_context = None
        self.last_children = None
        self.last_parents = None
        self.last_scores = None

    def routes(self, flat):
        parents, scores = super().routes(flat)
        self.last_parents = parents.detach()
        self.last_scores = scores.detach()
        return parents, scores

    def set_token_context(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0 or np.any(ids < 0):
            raise ValueError("token context must be a nonempty 1D ID sequence")
        self.token_context = (
            ids.copy(), np.r_[0, ids[:-1]], np.arange(ids.size, dtype=np.int64))

    def selected_ids(self, flat, children):
        if self.token_context is None:
            raise RuntimeError("token context must be set before expert forward")
        tokens, previous, positions = self.token_context
        if flat.shape[0] != tokens.size or children.shape != (tokens.size, 4):
            raise RuntimeError("token context shape differs from expert route")
        self.last_children = children.detach()
        chosen = hash_grandchildren(tokens, previous, positions,
                                    children.detach().cpu().numpy().astype(np.int64),
                                    self.layer_id)
        result = torch.from_numpy(chosen).to(device=children.device)
        if not torch.equal(result.div(10, rounding_mode="floor"), children):
            raise AssertionError("hash grandchild escaped its source child")
        return result
