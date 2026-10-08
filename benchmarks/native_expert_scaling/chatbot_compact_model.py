"""Decode an independently byte-adopted native artifact into a Torch reference.

No original dense source FFN values are read or allocated. BF16 deployed G/U/D
are decoded to F32 for the declared compact operator. No work at import.
"""
import json
from pathlib import Path


def load_exported_reference(export_path, codec_audit_path, spec, device):
    import torch
    from safetensors import safe_open
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from chatbot_interaction_launch import sha
    from chatbot_compact_geometry import build_block,validate
    from chatbot_whole_transfer_model import install_compact_model
    validate(spec)
    export_path=Path(export_path);codec_audit_path=Path(codec_audit_path)
    export=json.loads(export_path.read_bytes());audit=json.loads(codec_audit_path.read_bytes())
    terminals=[]
    for path in (export_path,codec_audit_path):
        terminal=json.loads(path.with_suffix('.terminal.json').read_bytes())
        assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(path) and all(terminal['gates'].values())
        terminals.append(terminal)
    assert export['decision']=='COMPACT_NATIVE_EXPORT_REQUIRE_FIRST_CODEC_AUDIT'
    assert audit['decision']=='COMPACT_EXPORT_BYTES_INDEPENDENTLY_VERIFIED'
    assert all(export['procedure_gates'].values()) and all(audit['procedure_gates'].values())
    assert audit['RTNE_BF16_GUD_elements_checked']==165150720 and audit['fields_and_catalog_entries_checked']==435
    terminal=terminals[1];binding_path=Path(terminal['command'][terminal['command'].index('--binding')+1])
    assert sha(binding_path)==terminal['command'][terminal['command'].index('--binding-sha')+1]
    binding=json.loads(binding_path.read_bytes());assert Path(binding['export_result_path']).resolve()==export_path.resolve()
    receipt=next(v for v in binding['inputs'] if Path(v['path']).resolve()==export_path.resolve())
    assert receipt['sha256']==sha(export_path) and receipt['bytes']==export_path.stat().st_size
    artifact=export['export']['archive'];archive_path=Path(artifact['path'])
    assert sha(archive_path)==artifact['sha256'] and archive_path.stat().st_size==artifact['bytes']
    source_config=next(v for v in export['interaction_files'] if Path(v['path']).name=='source_config.json')
    config_path=Path(source_config['path']);assert sha(config_path)==source_config['sha256']
    raw_config=json.loads(config_path.read_bytes());assert raw_config['model_type']=='qwen2'
    config=Qwen2Config(**raw_config);config._attn_implementation='eager'
    with torch.device('meta'):
        model=Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
    # Meta dense FFNs establish HF source class/shape only; they never acquire
    # data storage and are replaced before any forward.
    assert all(value.is_meta for value in model.parameters())
    blocks=[]
    with safe_open(str(archive_path),framework='pt',device='cpu') as archive:
        names=set(archive.keys());assert len(names)==435
        core={name:archive.get_tensor(name).to(device) for name in names if '.mlp.compact.' not in name and name!='native.rope.inv_freq'}
        assert len(core)==218 and all(value.dtype==torch.bfloat16 for value in core.values())
        missing,unexpected=model.load_state_dict(core,strict=False,assign=True)
        allowed={f'model.layers.{li}.mlp.{part}_proj.weight' for li in range(24) for part in ('gate','up','down')}
        assert set(missing)==allowed|{'lm_head.weight'} and not unexpected
        model.tie_weights();del core
        for li in range(24):
            prefix=f'model.layers.{li}.mlp.compact.'
            projection=archive.get_tensor(prefix+'projection').to(device)
            parents=archive.get_tensor(prefix+'parent_centers').to(device)
            norms=archive.get_tensor(prefix+'parent_norms').to(device)
            block=build_block(spec,1,projection,parents,parents[:,None].contiguous())
            with torch.no_grad():
                block.parent_norms.copy_(norms);block.child_norms.copy_(norms[:,None])
                for name in ('shared_g','shared_u','shared_b'):
                    value=archive.get_tensor(prefix+name);assert value.dtype==torch.bfloat16
                    getattr(block,name).copy_(value.to(device=device,dtype=torch.float32))
                for name in ('leaf_g','leaf_u','leaf_b'):
                    value=archive.get_tensor(prefix+name);assert value.dtype==torch.bfloat16
                    for leaf in range(16):getattr(block,name)[leaf].copy_(value[leaf].to(device=device,dtype=torch.float32))
            block.initialized=True;blocks.append(block)
        inv=archive.get_tensor('native.rope.inv_freq').to(device);assert inv.dtype==torch.float32 and inv.shape==(32,)
        model.model.rotary_emb.inv_freq=inv;model.model.rotary_emb.original_inv_freq=inv.clone()
    installation=install_compact_model(model,blocks,spec)
    model.gradient_checkpointing_disable();model.config.use_cache=True
    for value in model.parameters():value.requires_grad_(False)
    assert all(not value.is_meta for value in list(model.parameters())+list(model.buffers()))
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert sum(v.numel() for v in model.parameters())==345397120
    return model.eval(),blocks,dict(schema='QWEN_DECODED_NATIVE_REFERENCE_V1',archive=artifact,
        installation=installation,source_dense_FFN_values_read_or_allocated=False,
        decoded_BF16_GUD_to_F32=True,scope='Hydration only; complete output/C/fresh behavior and rate unqualified')
