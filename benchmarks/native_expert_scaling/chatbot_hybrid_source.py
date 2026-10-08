"""Pinned public hybrid-source package. Metadata first, bounded values second."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
WIDTH={'BF16':2,'F16':2,'F32':4,'I64':8,'I32':4,'BOOL':1}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def main(a):
    start=time.monotonic()
    out=a.directory.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True,exist_ok=False)
    completed=[]
    def guard():
        if time.monotonic()-start > (90 if a.mode=='metadata' else 600):
            raise TimeoutError('source stage deadline')
    def stream(url):
        guard()
        return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SiliconLLM-hybrid-source'}),timeout=20)
    def small(url,name,limit=16<<20):
        with stream(url) as response:
            raw=response.read(limit+1)
        assert len(raw)<=limit, name
        with (out/name).open('xb') as f:f.write(raw)
        return dict(name=name,path=str((out/name).resolve()),bytes=len(raw),sha256=sha(out/name),url=url)
    try:
        if a.mode=='metadata':
            assert a.model.startswith('tiiuae/Falcon-H1-')
            api_url=f'https://huggingface.co/api/models/{a.model}?blobs=true'
            entry=small(api_url,'producer_api.json',2<<20)
            api=json.loads((out/'producer_api.json').read_bytes())
            rev=api['sha']
            assert len(rev)==40 and all(c in '0123456789abcdef' for c in rev)
            siblings={v['rfilename']:v for v in api['siblings']}
            weights=[v for n,v in siblings.items() if n.endswith('.safetensors') and '/' not in n]
            assert weights and sum(v['size'] for v in weights)<=4<<30, '4GiB weight envelope'
            assert all(v['lfs']['sha256'] and v['size']>0 for v in weights)
            for name in ('config.json','generation_config.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','model.safetensors.index.json'):
                if name not in siblings:
                    assert name not in ('config.json','generation_config.json','tokenizer_config.json'), name
                    continue
                rec=small(f'https://huggingface.co/{a.model}/resolve/{rev}/{name}',name)
                assert rec['bytes']==siblings[name]['size']
                completed.append(rec)
            assert 'tokenizer.json' in siblings
            c=json.loads((out/'config.json').read_bytes())
            assert c['model_type']=='falcon_h1', 'supported local source class required'
            d,l,v,h=(c[k] for k in ('hidden_size','num_hidden_layers','vocab_size','intermediate_size'))
            m=c.get('mamba_d_ssm') or int(c['mamba_expand']*d)
            n,g,heads,hd=(c[k] for k in ('mamba_d_state','mamba_n_groups','mamba_n_heads','mamba_d_head'))
            assert m==heads*hd
            conv=m+2*g*n
            q=c['num_attention_heads']*c['head_dim']
            kv=c['num_key_value_heads']*c['head_dim']
            terms=dict(head=v*d,ffn=3*l*d*h,ssm_projection=l*d*(m+conv+heads+m),attention=l*(2*d*q+2*d*kv))
            result=dict(schema='HYBRID_SOURCE_METADATA_V1',model=a.model,revision=rev,freeze=a.freeze,
               api=entry,files=completed,download_files=[dict(name=rec['rfilename'],bytes=rec['size'],
               producer_lfs_sha256=rec.get('lfs',{}).get('sha256'),
               url=f'https://huggingface.co/{a.model}/resolve/{rev}/{rec["rfilename"]}')
               for rec in [siblings['tokenizer.json'],*sorted(weights,key=lambda v:v['rfilename'])]],
               config_shape=dict(D=d,L=l,V=v,ffn_h=h,ssm_D=m,N=n,groups=g,heads=heads,head_dim=hd,
                      gated_rms=c['mamba_rms_norm'],norm_before_gate=c['mamba_norm_before_gate']),
               matrix_products_per_decode=terms,total_matrix_products=sum(terms.values()),
               BF16_weight_file_bytes=sum(rec['size'] for rec in weights),
               dense_F32_Adam_matrix_lower_bound_bytes=16*sum(terms.values()),
               recurrent_elements=l*m*n,conv_elements=l*conv*c['mamba_d_conv'],
               cached_kv_elements_per_context_token=l*2*kv,
               scope='Pinned public metadata/operator counts, no weight values/header completeness/quality/rate',
               elapsed_seconds=time.monotonic()-start)
            write(out/'metadata.json',result)
            print(json.dumps(result),flush=True)
        else:
            assert a.metadata and a.metadata_sha and sha(a.metadata)==a.metadata_sha
            meta=json.loads(a.metadata.read_bytes())
            assert meta['schema']=='HYBRID_SOURCE_METADATA_V1'
            for rec in [meta['api'],*meta['files']]:
                assert sha(rec['path'])==rec['sha256']
                shutil.copyfile(rec['path'],out/rec['name'])
                completed.append(dict(rec,path=str((out/rec['name']).resolve())))
            assert sum(v['bytes'] for v in meta['download_files'] if v['name'].endswith('.safetensors'))<=4<<30
            headers=[]
            for rec in meta['download_files']:
                name=rec['name']
                size=rec['bytes']
                assert 0<size<=(4<<30 if name.endswith('.safetensors') else 32<<20)
                h=hashlib.sha256()
                written=0
                with stream(rec['url']) as response,(out/name).open('xb') as f:
                    if name.endswith('.safetensors'):
                        first=response.read(8)
                        assert len(first)==8
                        length=struct.unpack('<Q',first)[0]
                        assert 0<length<=2<<20
                        body=response.read(length)
                        assert len(body)==length
                        header=json.loads(body)
                        ranges=[]
                        elements=0
                        for key,t in header.items():
                            if key=='__metadata__':continue
                            count=1
                            for dim in t['shape']:
                                assert type(dim)is int and dim>0
                                count*=dim
                            begin,end=t['data_offsets']
                            assert end-begin==count*WIDTH[t['dtype']]
                            elements+=count
                            ranges.append((begin,end))
                        offset=0
                        for begin,end in sorted(ranges):
                            assert begin==offset
                            offset=end
                        assert offset+8+length==size
                        prefix=first+body
                        f.write(prefix);h.update(prefix);written=len(prefix)
                        header_name=name+'.validated_header.json'
                        write(out/header_name,dict(tensors=header,named_elements=elements,header_bytes=len(prefix),
                                file_bytes=size,header_sha256=hashlib.sha256(prefix).hexdigest()))
                        headers.append(dict(path=str((out/header_name).resolve()),sha256=sha(out/header_name),
                            named_elements=elements,tensors=len(header)-('__metadata__' in header)))
                    while True:
                        guard()
                        block=response.read(1<<20)
                        if not block:break
                        written+=len(block)
                        assert written<=size
                        f.write(block);h.update(block)
                assert written==size
                digest=h.hexdigest()
                assert not rec['producer_lfs_sha256'] or digest==rec['producer_lfs_sha256']
                completed.append(dict(name=name,path=str((out/name).resolve()),bytes=size,sha256=digest,
                   producer_lfs_sha_verified=bool(rec['producer_lfs_sha256']),url=rec['url']))
                print(json.dumps(dict(file=name,bytes=size,seconds=time.monotonic()-start,sha256=digest)),flush=True)
            result=dict(schema='HYBRID_ORIGINAL_SOURCE_PACKAGE_V1',model=meta['model'],revision=meta['revision'],
               freeze=a.freeze,metadata_sha256=a.metadata_sha,files=completed,headers=headers,
               total_named_elements=sum(v['named_elements'] for v in headers),
               header_validated_before_payload=True,elapsed_seconds=time.monotonic()-start,
               scope='Original bytes/producer weight SHA, no model call, no through-exit resource or quality admission')
            write(out/'source_package.json',result)
            print(json.dumps(dict(package=str(out/'source_package.json'),seconds=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        write(out/'first_failure.json',dict(fault=repr(error),completed=completed,elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['metadata','weights'])
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--model')
    p.add_argument('--metadata',type=Path)
    p.add_argument('--metadata-sha')
    main(p.parse_args())
