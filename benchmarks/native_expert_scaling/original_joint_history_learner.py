"""Training-only residual supervision at the unchanged original operator target.

Prepared adapter, not a qualified conversion or executed training recipe.
Student trajectories use their own states; teacher residuals enter only loss.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from original_wide_learner import SourceLearner
from original_tensor_learner import norm


class JointHistoryLearner(SourceLearner):
    def forward(self, ids, positions=None, boundary_targets=None):
        """Return usual logits, or logits plus six normalized auxiliary losses.

        Targets are h4,h8,...,h24 projected through the fixed inherited P256,
        with shape (1,T,256). No target replaces a student input or state.
        Whole residual error = mean error + centered error, with a shared
        teacher energy denominator; reporting decomposition costs no inference
        parameter or exported field.
        """
        x=F.embedding(ids,self.embed)
        terms=[]; summaries=[]
        if boundary_targets is not None:
            assert len(boundary_targets)==len(self.layers)==6
        for site,block in enumerate(self.layers):
            x=checkpoint(block,x,use_reentrant=False,preserve_rng_state=True) if self.use_checkpoint and torch.is_grad_enabled() else block(x)
            if boundary_targets is not None:
                target=boundary_targets[site]
                assert target.shape==x.shape and target.dtype==x.dtype and target.device==x.device
                assert not target.requires_grad
                denominator=target.square().mean().clamp_min(1e-24)
                error=x-target
                loss=error.square().mean()/denominator
                terms.append(loss)
                with torch.no_grad():
                    mean_error=error.mean(dim=1,keepdim=True)
                    mean_term=mean_error.square().mean()/denominator
                    centered_term=(error-mean_error).square().mean()/denominator
                    teacher_center=(target-target.mean(dim=1,keepdim=True)).square().mean().clamp_min(1e-24)
                    centered_relative=(error-mean_error).square().mean()/teacher_center
                    summaries.append(dict(site=site,boundary=4*(site+1),total=loss.detach(),
                        mean=mean_term,centered= centered_term,centered_relative=centered_relative))
        x=norm(x,self.final_norm)
        if positions is not None:x=x[:,positions]
        logits=F.linear(x,self.head)
        if boundary_targets is None:return logits
        return logits,torch.stack(terms).mean(),summaries
