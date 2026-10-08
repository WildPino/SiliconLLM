"""Hydrate ALL24 qualified balanced blocks with explicit new-buffer precedence.

No work at import, no source/full-model forward or optimizer update. Actual
complete hydration/installation must be frozen and measured by its caller.
"""
import json
from pathlib import Path

BUFFERS=('projection','parent_centers','child_centers','parent_norms','child_norms')
COEFFICIENTS=('shared_g','shared_u','shared_b','leaf_g','leaf_u','leaf_b')


def hydrate_blocks(initializer_path,audit_path,audit_binding_path,spec,device):
    import torch
    from chatbot_compact_geometry import build_block,validate
    from chatbot_interaction_launch import sha
    validate(spec)
    raw=json.loads(Path(initializer_path).read_bytes());audit=json.loads(Path(audit_path).read_bytes())
    audit_binding=json.loads(Path(audit_binding_path).read_bytes())
    assert audit_binding['job']['name']=='balanced_initializer_audit'
    assert Path(audit_binding['initializer_result_path']).resolve()==Path(initializer_path).resolve()
    adopted=next(v for v in audit_binding['inputs'] if Path(v['path']).resolve()==Path(initializer_path).resolve())
    assert sha(initializer_path)==adopted['sha256'] and Path(initializer_path).stat().st_size==adopted['bytes']
    assert raw['schema']=='QWEN_BALANCED_INITIALIZER_RESULT_V1' and audit['schema']=='QWEN_BALANCED_INITIALIZER_AUDIT_RESULT_V1'
    assert raw['decision']=='BALANCED_ALL24_INITIALIZED_REQUIRE_FIRST_AUDIT'
    assert audit['decision']=='BALANCED_ALL24_INITIALIZER_INDEPENDENTLY_VERIFIED'
    assert raw['all24_source_initialized_and_NEW_supported'] and audit['all24_source_initialized_and_NEW_supported']
    assert all(raw['procedure_gates'].values()) and all(audit['procedure_gates'].values())
    assert [v['layer'] for v in raw['layers']]==[v['layer'] for v in audit['layers']]==list(range(24))
    blocks=[];receipts=[]
    for record in raw['layers']:
        assert record['support_eligible']
        for name in ('witness','initial_checkpoint'):
            entry=record[name];assert sha(entry['path'])==entry['sha256'] and Path(entry['path']).stat().st_size==entry['bytes']
        routing=torch.load(record['witness']['path'],map_location='cpu',weights_only=True)
        initial=torch.load(record['initial_checkpoint']['path'],map_location='cpu',weights_only=True)
        for name in BUFFERS:
            value=routing[name];assert value.dtype==torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
        expected_shapes=dict(shared_g=(512,896),shared_u=(512,896),shared_b=(896,512),
            leaf_g=(16,128,896),leaf_u=(16,128,896),leaf_b=(16,896,128))
        for name,shape in expected_shapes.items():
            value=initial[name];assert tuple(value.shape)==shape and value.dtype==torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
        block=build_block(spec,1,routing['projection'].to(device),routing['parent_centers'].to(device),routing['child_centers'].to(device))
        with torch.no_grad():
            for name in BUFFERS:getattr(block,name).copy_(routing[name].to(device))
            for name in ('shared_g','shared_u','shared_b'):getattr(block,name).copy_(initial[name].to(device))
            for name in ('leaf_g','leaf_u','leaf_b'):
                for leaf in range(16):getattr(block,name)[leaf].copy_(initial[name][leaf].to(device))
        block.initialized=True
        assert sum(v.numel() for v in block.parameters())==6881280
        if record['initial_origin']=='BYTE_warm_GUD_only_with_NEW_routing':
            assert record['old_routing_buffers_MUST_NOT_override_NEW_geometry'] is True
        else:assert record['initial_origin']=='NEW_FIT_source_rows'
        blocks.append(block);receipts.append(dict(layer=record['layer'],routing=record['witness'],GUD=record['initial_checkpoint'],
            coefficient_origin=record['initial_origin'],routing_buffers_always_from_NEW_witness=True))
        del routing,initial
    assert len({id(v) for v in blocks})==24 and sum(p.numel() for b in blocks for p in b.parameters())==165150720
    return blocks,dict(schema='QWEN_WHOLE_CHECKPOINT_HYDRATION_V1',layers=receipts,trainable_F32_elements=165150720,
        no_source_dense_FFN_fallback=True,scope='Actual parameter hydration only; no full-model forward, optimizer allocation, quality or native admission')
