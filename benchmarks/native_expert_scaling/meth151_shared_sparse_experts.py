"""METH-150 shared/context route inside the CPU-master sparse expert."""

import numpy as np
import torch

from meth144_hash_experts import TokenHashSparseExperts
import meth150_shared_structure_route as R150


TABLE = R150.load_table()


class SharedStructureSparseExperts(TokenHashSparseExperts):
    def __init__(self, parent, layer_id, cpu_bank, shared_a):
        super().__init__(parent, layer_id, cpu_bank, shared_a)
        self.last_shared = None

    def selected_ids(self, flat, children):
        if self.token_context is None:
            raise RuntimeError("token context must be set before expert forward")
        tokens, previous, positions = self.token_context
        if flat.shape[0] != tokens.size or children.shape != (tokens.size, 4):
            raise RuntimeError("token context shape differs from expert route")
        self.last_children = children.detach()
        selected, shared = R150.route(tokens, previous, positions,
                                      children.detach().cpu().numpy().astype(np.int64),
                                      self.layer_id, TABLE)
        result = torch.from_numpy(selected).to(device=children.device)
        if not torch.equal(result.div(10, rounding_mode="floor"), children):
            raise AssertionError("shared grandchild escaped its source child")
        self.last_shared = shared
        return result
