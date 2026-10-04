"""Bounded complete original Switch128 acquisition with whole-file SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests
import psutil
import meth324_switch_reference as M

MANIFEST=M.DOC/'meth325_switch_source_headers_result.json'
MANIFEST_SHA='f49bd77012045ce2ba2d1ad20d29f86833d229674a644ee0d250e622c39d71e8'
PROTOCOL=M.DOC/'METH_377_SWITCH_BASE128_ACQUISITION_PROTOCOL_20261003.md'
OUT=M.ROOT/'results/native_expert_scaling/meth377_switch_base128_source'
SIDE=('config.json','pytorch_model.bin.index.json','generation_config.json','special_tokens_map.json',
      'spiece.model','tokenizer.json','tokenizer_config.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    start=time.monotonic();downloaded=0;maximum=0;stage='bindings'
    result={'experiment':'METH-377-complete-original-Switch128-acquisition','files':[]}
    try:
        for path in (Path(__file__),PROTOCOL,MANIFEST,M.PRIOR,Path(M.__file__)):M.committed(path)
        assert M.digest(MANIFEST)==MANIFEST_SHA and M.digest(M.PRIOR)==M.PRIOR_SHA
        manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
        model=next(v for v in manifest['models'] if v['model']=='google/switch-base-128')
        assert model['passed'] and manifest['decision']=='eligible_for_separate_bounded_full_payload_acquisition_and_function_hashes'
        api_path=M.ROOT/'results/native_expert_scaling/meth325_switch_headers/google_switch-base-128_api.json'
        assert M.digest(api_path)==model['metadata_sha256']
        api=json.loads(api_path.read_text(encoding='utf-8'));siblings={v['rfilename']:v for v in api['siblings']}
        files=[{'name':name,'bytes':siblings[name]['size'],'lfs_sha256':siblings[name].get('lfs',{}).get('sha256'),
                'git_blob_sha1':siblings[name].get('blobId'),
                'url':f"https://huggingface.co/{model['model']}/resolve/{model['revision']}/{name}"} for name in SIDE]
        files.extend(model['shards'])
        expected=sum(v['bytes'] for v in files)
        assert expected<40<<30 and shutil.disk_usage(OUT.parent).free>=expected+(32<<30)
        OUT.mkdir(parents=True)
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),
                       'manifest_sha256':MANIFEST_SHA,'model':model['model'],'revision':model['revision'],
                       'expected_acquisition_bytes':expected,'available_disk_before_bytes':shutil.disk_usage(OUT).free,'available_RAM_before_bytes':psutil.virtual_memory().available})
        for item in files:
            name=item['name'];stage=name
            target=OUT/name;partial=OUT/(name+'.partial')
            assert not target.exists() and not partial.exists()
            assert shutil.disk_usage(OUT).free>=expected-downloaded+(32<<30)
            sha=hashlib.sha256();blob=hashlib.sha1()
            blob.update(f"blob {item['bytes']}\0".encode())
            count=0;next_report=time.monotonic()+30
            row={'name':name,'expected_bytes':item['bytes'],'expected_lfs_sha256':item.get('lfs_sha256'),
                 'expected_git_blob_sha1':item.get('git_blob_sha1'),'path':str(target),'complete':False}
            result['files'].append(row)
            with requests.get(item['url'],stream=True,timeout=(15,30)) as response:
                response.raise_for_status();assert response.status_code==200
                origin=next(({k.lower():v for k,v in r.headers.items()} for r in response.history if 'x-repo-commit' in {k.lower():v for k,v in r.headers.items()}),{k.lower():v for k,v in response.headers.items()})
                assert origin.get('x-repo-commit')==model['revision']
                if item.get('lfs_sha256'):assert origin.get('x-linked-etag','').strip('"')==item['lfs_sha256']
                with partial.open('xb') as stream:
                    for block in response.iter_content(1<<20):
                        stream.write(block);sha.update(block);blob.update(block)
                        count+=len(block);downloaded+=len(block)
                        maximum=max(maximum,psutil.Process().memory_info().rss);assert maximum<2<<30
                        row['acquired_bytes']=count
                        assert count<=item['bytes'] and downloaded<=40<<30 and time.monotonic()-start<=5400
                        if time.monotonic()>=next_report:
                            print(json.dumps({'stage':stage,'file_bytes':count,'total_bytes':downloaded,'seconds':time.monotonic()-start}),flush=True)
                            assert shutil.disk_usage(OUT).free>=expected-downloaded+(32<<30)
                            next_report=time.monotonic()+30
            assert count==item['bytes']
            if item.get('lfs_sha256'):assert sha.hexdigest()==item['lfs_sha256']
            else:assert blob.hexdigest()==item['git_blob_sha1']
            if name in ('config.json','pytorch_model.bin.index.json'):
                prior=next(v for v in json.loads(M.PRIOR.read_text(encoding='utf-8'))['models'] if v['model']==model['model'])
                binding=next(v for v in prior['bindings'] if Path(v['path']).name.endswith('_'+name))
                assert sha.hexdigest()==binding['sha256']
            partial.rename(target)
            row.update({'complete':True,'bytes':count,'sha256':sha.hexdigest(),'git_blob_sha1':blob.hexdigest()})
            print(json.dumps({'completed':name,'bytes':count,'sha256':sha.hexdigest(),'seconds':time.monotonic()-start}),flush=True)
        assert downloaded==expected
        result.update({'downloaded_bytes':downloaded,'seconds':time.monotonic()-start,'passed':True,
            'decision':'eligible_for_separate_all_tensor_and_distinct_function_binding',
            'resource':{'maximum_checked_RSS_bytes':maximum,'available_disk_end_bytes':shutil.disk_usage(OUT).free,'available_RAM_end_bytes':psutil.virtual_memory().available},
            'scope':'Whole original source archives verified by official expected LFS SHA256; small files by Git blob SHA1 and fixed config/index SHA256. No loaded model, learned expert uniqueness or quality/native cost/rate proof.'})
        M.write(args.out,result)
        print(json.dumps({'sha256':M.digest(args.out),'seconds':result['seconds'],'downloaded_bytes':downloaded,'passed':True}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'stage':stage,'downloaded_bytes':downloaded,'maximum_checked_RSS_bytes':maximum,'seconds':time.monotonic()-start})
        M.write(args.out.with_suffix('.failure.json'),result)
        raise


if __name__=='__main__':main()
