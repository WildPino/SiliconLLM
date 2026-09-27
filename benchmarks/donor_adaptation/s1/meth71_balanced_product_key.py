"""METH-71 dense product keys with a training-only axis balance objective."""

import torch
import torch.nn.functional as F

import meth15_zero_residual_expert_smoke as M15
import meth70_dense_product_key as M70


class BalancedProductKeyExperts(M70.DenseProductKeyExperts):
    def routes(self, flat):
        width = M15.M13.D
        projection_size = M70.ROUTER_RANK * width
        a_size = self.axis_a * M70.ROUTER_RANK
        p = self.router[:projection_size].view(M70.ROUTER_RANK, width)
        a_keys = self.router[projection_size:projection_size + a_size].view(self.axis_a,
                                                                            M70.ROUTER_RANK)
        b_keys = self.router[projection_size + a_size:].view(self.axis_b, M70.ROUTER_RANK)
        q = F.linear(flat.float(), p)
        a_scores = F.linear(q, a_keys)
        b_scores = F.linear(q, b_keys)
        a_values, a_ids = torch.topk(a_scores, M15.K, dim=-1)
        b_values, b_ids = torch.topk(b_scores, M15.K, dim=-1)
        candidate_scores = (a_values.unsqueeze(-1) + b_values.unsqueeze(-2)).flatten(1)
        candidate_ids = (a_ids.unsqueeze(-1) * self.axis_b + b_ids.unsqueeze(-2)).flatten(1)
        selected_scores, order = torch.topk(candidate_scores, M15.K, dim=-1)
        chosen = candidate_ids.gather(1, order)
        if self.oracle_checks:
            dense_scores = (a_scores.unsqueeze(-1) + b_scores.unsqueeze(-2)).flatten(1)
            exact = torch.topk(dense_scores, M15.K, dim=-1).indices
            if not torch.equal(chosen.sort(dim=1).values, exact.sort(dim=1).values):
                raise AssertionError("product-key route differs from exhaustive score")
        if self.collect_load:
            self.route_counts += torch.bincount(chosen.detach().flatten(),
                                                minlength=self.a.shape[0]).cpu()
        self.balance_loss = None
        if self.training and torch.is_grad_enabled():
            def axis_balance(scores, ids, axis_width):
                frequency = torch.bincount(ids.detach().flatten(),
                                           minlength=axis_width).float()
                frequency = (frequency / ids.numel()).detach()
                temperature = scores.detach().std(unbiased=False).clamp_min(1e-3)
                probability = F.softmax(scores / temperature, dim=-1).mean(dim=0)
                return axis_width * (frequency * probability).sum() - 1.0

            self.balance_loss = (axis_balance(a_scores, a_ids, self.axis_a)
                                 + axis_balance(b_scores, b_ids, self.axis_b))
        return chosen, selected_scores
