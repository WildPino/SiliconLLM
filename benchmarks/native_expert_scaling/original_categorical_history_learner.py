"""Original causal operators with a fixed, coherent categorical supervision head.

Training-only F64 readout evaluates the fixed F32 coefficients as real numbers.
Packed inference keeps the original F32 head and RMSNorm. This adapter still
requires its own GPU/native preflight before optimization.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from original_wide_learner import SourceLearner
from original_tensor_learner import norm


class CategoricalHistoryLearner(SourceLearner):
    @torch.no_grad()
    def install_fixed_head(self, head):
        """Call after loading the immutable source core/bank model state."""
        assert head.dtype == torch.float32 and head.shape == self.head.shape
        assert torch.isfinite(head).all() and torch.count_nonzero(head[:, -1]) == 0
        assert not hasattr(self, 'supervision_head')
        self.head.copy_(head.to(self.head.device))
        self.final_norm.fill_(1.)
        self.head.requires_grad_(False)
        self.final_norm.requires_grad_(False)
        self.register_buffer('supervision_head', self.head.detach().to(torch.float64).clone())

    def forward_features(self, ids, positions):
        x = F.embedding(ids, self.embed)
        for block in self.layers:
            x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True) if self.use_checkpoint and torch.is_grad_enabled() else block(x)
        return norm(x, self.final_norm)[:, positions]

    @torch.no_grad()
    def validate_fixed_geometry(self):
        assert hasattr(self, 'supervision_head') and self.supervision_head.dtype == torch.float64
        assert self.head.dtype == self.final_norm.dtype == torch.float32
        assert not self.head.requires_grad and not self.final_norm.requires_grad
        assert torch.equal(self.head, self.supervision_head.to(torch.float32))
        assert torch.equal(self.final_norm, torch.ones_like(self.final_norm))

    def categorical_objective(self, features, moments, negative_entropy, weights):
        """Return real fixed-head logits, per-label KL and weighted case loss."""
        self.validate_fixed_geometry()
        assert features.ndim == 3 and features.shape[0] == 1 and features.shape[-1] == 256
        assert moments.shape == features.shape and negative_entropy.shape == weights.shape == features.shape[:2]
        assert moments.dtype == negative_entropy.dtype == weights.dtype == torch.float64
        assert not moments.requires_grad and not negative_entropy.requires_grad and not weights.requires_grad
        f = features.to(torch.float64)
        logits = F.linear(f, self.supervision_head)
        losses = torch.logsumexp(logits, dim=-1) - (f * moments).sum(-1) + negative_entropy
        return logits, losses, (losses * weights).sum()
