"""Pinned original Falcon package acquisition; header validated before payload."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
MODEL='tiiuae/Falcon-H1-Tiny-90M-Instruct'
REV='e6389502a0b12cd8da894b395ba5bf7436873b16'
FILES=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
       'special_tokens_map.json','chat_template.jinja','model.safetensors')
WIDTH={'BF16':2,'F32':4,'F16':2,'I64':8,'I32':4,'BOOL':1}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',required=True,type=Path)
    ap.add_argument('--freeze',required=True)
    a=ap.parse_args()
    out=a.directory.absolute();out.relative_to(ROOT);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();records=[];header=None
    def guard():
        if time.monotonic()-start>300:raise TimeoutError('300s acquisition cap')
    def stream(url):
        guard()
        return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SiliconLLM-source-acquisition'}),timeout=20)
    def create(name,raw):
        with (out/name).open('xb') as f:f.write(raw)
    try:
        api_url=f'https://huggingface.co/api/models/{MODEL}/revision/{REV}?blobs=true'
        with stream(api_url) as response:api_raw=response.read((2<<20)+1)
        if len(api_raw)>2<<20:raise ValueError('API cap')
        api=json.loads(api_raw)
        if api['sha']!=REV:raise ValueError('revision mismatch')
        create('producer_api.json',api_raw)
        siblings={s['rfilename']:s for s in api['siblings']}
        small_total=0
        for name in FILES:
            meta=siblings[name];size=meta['size']
            limit=256<<20 if name=='model.safetensors' else 16<<20
            if type(size)is not int or not 0<size<=limit:raise ValueError(('file cap',name,size))
            if name!='model.safetensors':
                small_total+=size
                if small_total>16<<20:raise ValueError('interaction aggregate cap')
            url=f'https://huggingface.co/{MODEL}/resolve/{REV}/{name}'
            h=hashlib.sha256();written=0
            with stream(url) as response,(out/name).open('xb') as f:
                if name=='model.safetensors':
                    first=response.read(8)
                    if len(first)!=8:raise ValueError('short header length')
                    length=struct.unpack('<Q',first)[0]
                    if not 0<length<=2<<20:raise ValueError('header cap')
                    body=response.read(length)
                    if len(body)!=length:raise ValueError('short header')
                    header=json.loads(body);ranges=[];elements=0
                    for key,t in header.items():
                        if key=='__metadata__':continue
                        shape=t['shape'];begin,end=t['data_offsets'];count=1
                        for dim in shape:
                            if type(dim)is not int or dim<=0:raise ValueError('invalid shape')
                            count*=dim
                        if end-begin!=count*WIDTH[t['dtype']]:raise ValueError('tensor byte count')
                        ranges.append((begin,end));elements+=count
                    offset=0
                    for begin,end in sorted(ranges):
                        if begin!=offset:raise ValueError('noncontiguous payload')
                        offset=end
                    if offset+8+length!=size:raise ValueError('header/file size mismatch')
                    prefix=first+body;f.write(prefix);h.update(prefix);written=len(prefix)
                    create('validated_header.json',json.dumps({'tensors':header,'named_elements':elements,
                        'header_bytes':length+8,'file_bytes':size,'header_sha256':hashlib.sha256(prefix).hexdigest()},indent=2).encode())
                while True:
                    guard();block=response.read(1<<20)
                    if not block:break
                    written+=len(block)
                    if written>size:raise ValueError('unexpected excess bytes')
                    f.write(block);h.update(block)
            if written!=size:raise ValueError('incomplete source file')
            digest=h.hexdigest()
            if 'lfs' in meta and digest!=meta['lfs']['sha256']:raise ValueError('producer LFS SHA mismatch')
            records.append({'name':name,'path':str(out/name),'bytes':size,'sha256':digest,
                            'producer_lfs_sha_verified':'lfs' in meta,'url':url})
            print(json.dumps({'file':name,'bytes':size,'sha256':digest,'seconds':time.monotonic()-start}),flush=True)
        report={'schema':'FALCON_ORIGINAL_SOURCE_PACKAGE_V1','model':MODEL,'revision':REV,'freeze':a.freeze,
                'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,
                'header_validated_before_payload':True,'seconds':time.monotonic()-start,
                'weight_values_executed':False,'interaction_or_quality_qualified':False,
                'scope':'original package bytes and producer weight SHA; not model/native/resource-through-exit admission'}
        create('source_package.json',(json.dumps(report,indent=2)+'\n').encode())
        print(json.dumps({'package':str(out/'source_package.json'),'seconds':report['seconds']}),flush=True)
    except Exception as e:
        create('acquisition_failure.json',json.dumps({'error':repr(e),'completed_files':records,'seconds':time.monotonic()-start},indent=2).encode())
        raise


if __name__=='__main__':main()
