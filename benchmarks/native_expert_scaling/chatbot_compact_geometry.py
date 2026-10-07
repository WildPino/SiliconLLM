"""Price one finite complete geometry; optionally construct its actual Torch block.

No model or tensor imports in the CLI. Cost outputs are dimension deductions,
never measured DRAM/rate or distinct/useful capacity. The block's functions read
full x, while its frozen projection is used only for routing.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CENSUS_SHA = 'a4293870cb38706fc60d94562d9579653ddff1a1387c366c307a56b4b262bbc2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(spec):
    assert spec['schema'] == 'QWEN_JOINT_SWIGLU_GEOMETRY_V1'
    assert (spec['hidden_size'], spec['layers'], spec['vocabulary_size'], spec['source_ffn_width']) == (896, 24, 151936, 4864)
    assert (spec['shared_width'], spec['function_width'], spec['selected_parents'], spec['parents'], spec['query_width']) == (512, 128, 4, 16, 32)
    assert spec['children_per_parent'] == [1, 10]
    assert spec['function_input'] == 'full_normalized_x' and spec['activation'] == 'silu_gate_times_up' and spec['bias'] is False
    assert spec['router'] == 'fixed_projected_squared_distance_then_stable_top4_parent_and_one_nearest_child'
    assert spec['mass'] == 'softmax_of_four_selected_parent_negative_squared_distances'
    assert spec['coefficient_storage_bytes'] == 2 and spec['router_norm_storage_bytes'] == 4 and spec['native_cache_storage_bytes'] == 4
    assert spec['maximum_matrix_ratio'] == spec['maximum_logical_coefficient_ratio'] == [3,5]
    assert spec['head'] == 'complete_source_tied_BF16_embedding_head'
    assert spec['attention'] == 'complete_source_GQA_projections_and_context_attention'
    return spec


def price(spec):
    validate(spec)
    d, l, h0, h1, k, p, q = (spec[n] for n in ('hidden_size', 'layers', 'shared_width', 'function_width', 'selected_parents', 'parents', 'query_width'))
    source_ffn = l * 3 * d * spec['source_ffn_width']
    source_attention = l * (d*d + 2*d*128 + d*d)
    head = d * spec['vocabulary_size']
    source_matrix = source_ffn + source_attention + head
    norm_bias_embed = 2*(l*2*d+d + l*(d+128+128) + d)
    source_payload = 2*(source_matrix + l*2*d+d + l*(d+128+128))
    ff_active = l * 3 * d * (h0+k*h1)
    shared_stored = l * 3*d*h0
    rows = []
    for c in spec['children_per_parent']:
        # Fixed squared distances use q*(2q-c) dot products plus separate norm/
        # reduction/scalar work. Keys are actually [P,C,Q], not full D queries.
        parent_score = l*p*q
        child_score = l*k*c*q if c > 1 else 0
        projection = l*d*q
        router_active = projection + parent_score + child_score
        # Parent/child centroid norms are stored scalars, charged in bytes.
        router_stored = l*(d*q+p*q + (p*c*q if c > 1 else 0))
        router_norms = l*(p + (p*c if c > 1 else 0))
        router_scalar_bytes = spec['router_norm_storage_bytes']*l*(p + (k*c if c > 1 else 0))
        matrix = source_attention+head+ff_active+router_active
        coefficients = 2*matrix+norm_bias_embed+router_scalar_bytes
        elements = (source_payload//2-source_ffn)+shared_stored+l*3*d*h1*p*c+router_stored+router_norms
        rows.append(dict(children=c,stored_function_slots=l*p*c,functions_per_layer=p*c,
            shared_FFN_elements=shared_stored,conditional_FFN_elements=l*3*d*h1*p*c,
            matrix_MAC_per_token=matrix,active_FFN_MAC=ff_active,router_matrix_MAC=router_active,
            head_MAC=head,attention_projection_MAC=source_attention,
            stored_named_elements=elements,proposed_mixed_payload_bytes=2*(elements-router_norms)+spec['router_norm_storage_bytes']*router_norms,
            logical_coefficient_bytes_per_token=coefficients,
            matrix_ratio_exact=str(Fraction(matrix, source_matrix)),
            logical_ratio_exact=str(Fraction(coefficients, 2*source_matrix+norm_bias_embed)),
            necessary_cost_gate=(Fraction(matrix, source_matrix)<=Fraction(*spec['maximum_matrix_ratio']) and
                Fraction(coefficients, 2*source_matrix+norm_bias_embed)<=Fraction(*spec['maximum_logical_coefficient_ratio'])),
            additional_work='RMS, SiLU/products, query norm, centroid scalar reads, distance/topK/argmin/softmax, accumulation, cache and control',
            weight_uniqueness='UNMEASURED; these are proposed slots, not actual distinct useful functions'))
    return dict(scope='PROPOSED_COMPLETE_DIMENSION_BUDGET_NOT_WEIGHTS_QUALITY_DRAM_OR_RATE',
        source_matrix_MAC=source_matrix,source_FFN_MAC=source_ffn,
        source_payload_bytes=source_payload,source_logical_coefficient_bytes=2*source_matrix+norm_bias_embed,
        source_attention_MAC_per_context_token=l*2*14*64,
        native_cache_bytes_at_capacity=spec['native_max_context']*l*2*128*spec['native_cache_storage_bytes'],
        active_gated_channels_per_layer=h0+k*h1,arms=rows,
        decision='ELIGIBLE_FOR_SOURCE_EXPOSURE_AND_ONE_FINITE_FIT' if all(a['necessary_cost_gate'] for a in rows) else 'CLOSE_FINITE_GEOMETRY_ON_COMPLETE_COST',
        physical_DRAM_bytes='UNKNOWN',accepted_IDs_per_second='UNKNOWN',compact_chatbot_qualified=False)


def build_block(spec, children, projection, parent_centers, child_centers):
    """Actual trainable nonlinear block; caller owns imports/device/frozen inputs.

    Routing arrays are buffers. Bias-free shared and every leaf G/U/B are jointly
    trainable; no source4864-channel basis is evaluated by this module.
    All constructor tensors must be finite float32 and C contiguous; shape checks
    reject silent flattening/broadcasts. FP32 is the fit contract. Export and BF16
    native arithmetic qualification are separate missing stages.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    validate(spec)
    assert children in spec['children_per_parent']
    d, q, p, k, h0, h1 = (spec[n] for n in ('hidden_size', 'query_width', 'parents', 'selected_parents', 'shared_width', 'function_width'))
    for value, shape in ((projection,(q,d)),(parent_centers,(p,q)),(child_centers,(p,children,q))):
        assert value.shape == shape and value.dtype == torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
    assert projection.device == parent_centers.device == child_centers.device

    class JointBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('projection', projection.detach().clone())
            self.register_buffer('parent_centers', parent_centers.detach().clone())
            self.register_buffer('child_centers', child_centers.detach().clone())
            # Squared distance q^2 - 2q*c + c^2: common q^2 cancels in choices/
            # selected-parent softmax. Store actual center norms explicitly.
            self.register_buffer('parent_norms', parent_centers.square().sum(-1))
            self.register_buffer('child_norms', child_centers.square().sum(-1))
            self.shared_g = nn.Parameter(torch.zeros(h0,d,device=projection.device))
            self.shared_u = nn.Parameter(torch.zeros(h0,d,device=projection.device))
            self.shared_b = nn.Parameter(torch.zeros(d,h0,device=projection.device))
            # Separate Parameters make an unselected leaf genuinely absent from
            # autograd/optimizer work. Indexing one giant Parameter would scatter
            # a dense E*H*D gradient on every selected-leaf call. The mathematical
            # functions and eventual contiguous export order are unchanged.
            self.leaf_g = nn.ParameterList([nn.Parameter(torch.zeros(h1,d,device=projection.device)) for _ in range(p*children)])
            self.leaf_u = nn.ParameterList([nn.Parameter(torch.zeros(h1,d,device=projection.device)) for _ in range(p*children)])
            self.leaf_b = nn.ParameterList([nn.Parameter(torch.zeros(d,h1,device=projection.device)) for _ in range(p*children)])
            self.initialized = False

        @torch.no_grad()
        def initialize_from_source(self,gate,up,down,shared_indices,leaf_indices):
            source_h = spec['source_ffn_width']
            for value,shape in ((gate,(source_h,d)),(up,(source_h,d)),(down,(d,source_h))):
                assert value.shape == shape and value.dtype == torch.float32 and value.is_contiguous() and value.device == self.projection.device and torch.isfinite(value).all()
            for indices,shape in ((shared_indices,(h0,)),(leaf_indices,(p*children,h1))):
                assert indices.shape == shape and indices.dtype == torch.int64 and indices.device == gate.device and indices.is_contiguous()
                assert (indices >= 0).all() and (indices < source_h).all()
            assert torch.unique(shared_indices).numel() == h0
            assert all(torch.unique(row).numel() == h1 for row in leaf_indices)
            assert not torch.isin(leaf_indices,shared_indices).any()
            self.shared_g.copy_(gate[shared_indices]); self.shared_u.copy_(up[shared_indices]); self.shared_b.copy_(down[:,shared_indices])
            for leaf,indices in enumerate(leaf_indices):
                self.leaf_g[leaf].copy_(gate[indices]); self.leaf_u[leaf].copy_(up[indices])
                self.leaf_b[leaf].copy_(down[:,indices])
            self.initialized = True

        def routes(self,x):
            assert x.ndim == 2 and x.shape[1] == d and x.dtype == torch.float32 and x.device == self.projection.device
            query = F.linear(x,self.projection)
            score = 2*F.linear(query,self.parent_centers)-self.parent_norms
            parents = torch.argsort(score,dim=-1,descending=True,stable=True)[:,:k]
            mass = torch.softmax(score.gather(1,parents),dim=-1)
            if children == 1:
                child = torch.zeros_like(parents)
            else:
                centers = self.child_centers[parents]
                child_scores = 2*torch.einsum('nq,nkcq->nkc',query,centers)-self.child_norms[parents]
                child = child_scores.argmax(dim=-1) # first/lower ID on exact tie
            return parents*children+child,mass

        def forward(self,x):
            assert self.initialized, 'Source-informed initializer must run before evaluating/fitting the zero constructor'
            assert x.ndim == 2 and x.shape[1] == d and x.dtype == torch.float32 and x.is_contiguous()
            ids,mass = self.routes(x)
            return self.evaluate_selected(x,ids,mass)

        def evaluate_selected(self,x,ids,mass):
            """Evaluate fixed choices; also enables a declared within-parent intervention."""
            assert self.initialized and x.ndim == 2 and x.shape[1] == d
            assert x.dtype == torch.float32 and x.is_contiguous() and x.device == self.projection.device
            assert ids.shape == mass.shape == (len(x),k) and ids.dtype == torch.int64 and mass.dtype == torch.float32
            assert ids.device == mass.device == x.device
            assert bool(((ids >= 0)&(ids < p*children)).all()) and bool(torch.isfinite(mass).all())
            result = F.linear(F.silu(F.linear(x,self.shared_g))*F.linear(x,self.shared_u),self.shared_b)
            # Only the selected nonlinear functions are evaluated. Grouping avoids
            # expanding a distinct DxH coefficient tensor for every token.
            for leaf in torch.unique(ids,sorted=True).tolist():
                locations = (ids == leaf).nonzero(as_tuple=False)
                tokens,slots = locations[:,0],locations[:,1]
                part = x.index_select(0,tokens)
                activation = F.silu(F.linear(part,self.leaf_g[leaf]))*F.linear(part,self.leaf_u[leaf])
                output = F.linear(activation,self.leaf_b[leaf])*mass[tokens,slots,None]
                result = result.index_add(0,tokens,output)
            return result

    return JointBlock()


def main(args):
    assert not args.out.exists()
    assert digest(args.spec) == args.spec_sha and digest(args.census) == CENSUS_SHA
    spec = json.loads(args.spec.read_bytes())
    # Existing completed census is reused by SHA; no header/value recount.
    result = price(spec)
    result.update(spec_SHA256=args.spec_sha,census_SHA256=CENSUS_SHA,source_freeze=args.freeze,
                  command_scope='New finite geometry arithmetic only; no imported scientific packages or model calls.')
    with args.out.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,separators=(',',':'),allow_nan=False);stream.write('\n')
    print(json.dumps(result,separators=(',',':')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--spec-sha',required=True)
    parser.add_argument('--census',type=Path,required=True)
    parser.add_argument('--freeze',required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
