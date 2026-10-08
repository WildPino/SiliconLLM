"""Export an independently eligible fixed whole checkpoint to one native format.

AVAILABLE code only until a new bound export protocol executes it. Native
qualification and source-relative fresh quality/rate are separate mandatory work.
"""
import hashlib
import json
from pathlib import Path
import struct


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def export_compact(source_path,fit_path,audit_path,directory,assembly_manifest_path,guard=None):
    """Produce BF16 core/GUD, F32 router and source-matched RoPE frequencies.

    Caller freezes hashes/runtime/resources first. No source/student forward or
    new optimizer execution. Exact source/core and approximate trained GUD codec
    are recorded separately. Directory must be fresh.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    source_path=Path(source_path);directory=Path(directory)
    assert not directory.exists()
    fit=json.loads(Path(fit_path).read_bytes());audit=json.loads(Path(audit_path).read_bytes())
    fit_terminal=json.loads(Path(fit_path).with_suffix('.terminal.json').read_bytes())
    audit_terminal=json.loads(Path(audit_path).with_suffix('.terminal.json').read_bytes())
    for terminal,path in ((fit_terminal,fit_path),(audit_terminal,audit_path)):
        assert terminal['actual_worker_exit_code']==0 and terminal['result_sha256']==sha(path) and all(terminal['gates'].values())
    audit_binding_path=Path(audit_terminal['command'][audit_terminal['command'].index('--binding')+1])
    assert sha(audit_binding_path)==audit_terminal['command'][audit_terminal['command'].index('--binding-sha')+1]
    audit_binding=json.loads(audit_binding_path.read_bytes())
    assert Path(audit_binding['fit_result_path']).resolve()==Path(fit_path).resolve()
    adopted=next(v for v in audit_binding['inputs'] if Path(v['path']).resolve()==Path(fit_path).resolve())
    assert adopted['sha256']==sha(fit_path) and adopted['bytes']==Path(fit_path).stat().st_size
    assert fit['schema']=='QWEN_WHOLE_OUTPUT_FIT_RESULT_V1' and audit['schema']=='QWEN_WHOLE_OUTPUT_AUDIT_RESULT_V1'
    assert fit['decision']=='WHOLE_OUTPUT_FIT_REQUIRE_FIRST_AUDIT' and all(fit['transfer_gates'].values())
    assert audit['decision']=='WHOLE_OUTPUT_FIT_INDEPENDENTLY_VERIFIED' and audit['candidate_transfer_eligible']
    assert all(fit['procedure_gates'].values()) and all(audit['procedure_gates'].values())
    assert fit['new_optimizer_updates']==1280 and audit['final_trainable_elements_checked']==165150720
    assert [v['layer'] for v in fit['final_checkpoints']]==list(range(24))
    assembled=json.loads(Path(assembly_manifest_path).read_bytes());tensors={};codec=[]
    def check():
        if guard:guard()
    with safe_open(str(source_path),framework='pt',device='cpu') as archive:
        names=[n for n in archive.keys() if '.mlp.' not in n]
        assert len(names)==218
        for name in names:
            tensor=archive.get_tensor(name);assert tensor.dtype==torch.bfloat16 and tensor.is_contiguous() and torch.isfinite(tensor).all()
            h=hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()
            assert h==assembled['parameters'][name]['sha256'];tensors[name]=tensor;check()
    assert sum(t.numel() for t in tensors.values())==180246400
    for item in fit['final_checkpoints']:
        assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
        values=torch.load(item['path'],map_location='cpu',weights_only=True);li=item['layer']
        prefix=f'model.layers.{li}.mlp.compact.'
        for field in ('projection','parent_centers','parent_norms'):
            value=values[field];assert value.dtype==torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
            tensors[prefix+field]=value
        # C=1 child choice is constantly0, so redundant child buffers are absent
        # from this deployed profile, with exact equality of their parent values.
        assert torch.equal(values['child_centers'][:,0],values['parent_centers'])
        assert torch.equal(values['child_norms'][:,0],values['parent_norms'])
        for field in ('shared_g','shared_u','shared_b','leaf_g','leaf_u','leaf_b'):
            value=values[field];assert value.dtype==torch.float32 and value.is_contiguous() and torch.isfinite(value).all()
            encoded=value.to(torch.bfloat16).contiguous();assert torch.isfinite(encoded).all()
            delta=encoded.float()-value
            squared=float(delta.double().square().sum());energy=float(value.double().square().sum())
            codec.append(dict(layer=li,field=field,elements=value.numel(),squared_error=squared,source_F32_energy=energy,
                relative_RMS=None if energy==0 else (squared/energy)**.5,maximum_absolute_error=float(delta.abs().max())))
            tensors[prefix+field]=encoded;check()
        del values
    # Reproduce the installed local HF default frequency formula and bind its
    # complete bytes to the actual assembly witness, rather than assuming libm.
    inv=1.0/torch.pow(1000000.0,torch.arange(0,64,2,dtype=torch.float32)/64.0)
    inv_sha=hashlib.sha256(memoryview(inv.view(torch.uint8).numpy())).hexdigest()
    assert inv_sha==assembled['buffers']['model.rotary_emb.inv_freq']['sha256']
    tensors['native.rope.inv_freq']=inv
    assert len(tensors)==435
    payload=sum(v.numel()*v.element_size() for v in tensors.values());assert payload==693597568
    metadata=dict(format='QWEN_CHAT_COMPACT_SHARED_SELECTED_BF16_GUD_F32_ROUTER_V1',
        source_weights_sha256=sha(source_path),fixed_fit_sha256=sha(fit_path),first_fit_audit_sha256=sha(audit_path),
        tied_head='model.embed_tokens.weight',config_json=json.dumps(dict(hidden_size=896,layers=24,vocabulary=151936,
            shared_width=512,private_width=128,parents=16,selected=4,query=32,heads=14,KV_heads=2,head_dimension=64,
            rms_epsilon=1e-6,rope_theta=1000000.,EOS=[151645,151643]),separators=(',',':')),
        arithmetic='BF16 core/GUD weights; source BF16 stream; F32 compact SwiGLU/router; C reference profile requires separate qualification',
        deployment_scope='Mixed-codec conversion; no native quality/rate admission')
    directory.mkdir();archive_path=directory/'model.safetensors'
    save_file(tensors,str(archive_path),metadata=metadata);check()
    catalog_path=directory/'catalog.h';catalog=catalog_text(archive_path)
    catalog_path.write_text(catalog,encoding='utf8',newline='\n')
    report=dict(schema='QWEN_COMPACT_EXPORT_MANIFEST_V1',archive=dict(path=str(archive_path.resolve()),sha256=sha(archive_path),bytes=archive_path.stat().st_size),
        catalog=dict(path=str(catalog_path.resolve()),sha256=sha(catalog_path),bytes=catalog_path.stat().st_size),
        fields=435,payload_bytes=payload,source_core_exact_BF16_elements=180246400,encoded_GUD_elements=165150720,
        router_F32_bytes=2803200,RoPE_F32_bytes=128,codec=codec,metadata=metadata,
        scope='Exact source core/NEW router + approximate trained BF16 GUD codec; whole encoded/C behavior and physical throughput unmeasured')
    report_path=directory/'export.json';report_path.write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8',newline='\n')
    check();return report


def catalog_text(archive_path):
    """Generate complete static offsets/hash for the one actual exported blob."""
    archive_path=Path(archive_path)
    with archive_path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length));base=8+length
    assert header.pop('__metadata__')['format']=='QWEN_CHAT_COMPACT_SHARED_SELECTED_BF16_GUD_F32_ROUTER_V1'
    assert len(header)==435
    rows=[];ranges=[]
    for name,item in sorted(header.items()):
        width={'F32':4,'BF16':2}[item['dtype']];kind=1 if item['dtype']=='F32' else 2
        shape=item['shape'];assert 1<=len(shape)<=3 and all(type(i) is int and i>0 for i in shape)
        elements=1
        for size in shape:elements*=size
        begin,end=item['data_offsets'];assert end-begin==elements*width and (base+begin)%width==0
        ranges.append((begin,end));rows.append('    {'+json.dumps(name)+','+str(kind)+','+str(len(shape))+', {'+
            ','.join(map(str,shape+[0]*(3-len(shape))))+'}, '+str(base+begin)+'ULL, '+str(end-begin)+'ULL},')
    ranges.sort();assert ranges[0][0]==0 and all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
    assert ranges[-1][1]==693597568 and base+ranges[-1][1]==archive_path.stat().st_size
    return ('/* One fixed actual compact archive; generated only after eligible audited fit. */\n'
        '#define CB_D 896\n#define CB_L 24\n#define CB_V 151936\n#define CB_P 16\n#define CB_Q 32\n'
        '#define CB_HS 512\n#define CB_HF 128\n#define CB_NH 14\n#define CB_NKV 2\n#define CB_HD 64\n'
        '#define CB_FIELD_COUNT 435\n#define CB_ARCHIVE_BYTES '+str(archive_path.stat().st_size)+'ULL\n'
        '#define CB_ARCHIVE_SHA "'+sha(archive_path)+'"\n'
        'typedef struct {const char *name;int dtype,rank;uint64_t shape[3],offset,bytes;} CbField;\n'
        'static const CbField cb_fields[CB_FIELD_COUNT]={\n'+'\n'.join(rows)+'\n};\n')
