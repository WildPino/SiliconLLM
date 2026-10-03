"""Bounded immutable source manifest and actual remote ZIP tensor headers."""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import time
import zipfile
import requests
import psutil
import meth324_switch_reference as M

PROTOCOL = M.DOC/'METH_325_SWITCH_SOURCE_HEADERS_PROTOCOL_20261003.md'
REFERENCE = M.DOC/'meth324_switch_reference_result.json'
REFERENCE_SHA = '9420d46235cdcf2e48a1153ae250aac59deecee12d6d243e0319ed33fb91f082'
OUT = M.ROOT/'results/native_expert_scaling/meth325_switch_headers'


class Network:
    def __init__(self):
        self.start = time.monotonic()
        self.bytes = 0
        self.calls = 0
        self.session = requests.Session()

    def get(self,url,maximum,headers=None):
        self.calls += 1
        assert self.calls <= 200 and time.monotonic()-self.start<=600
        with self.session.get(url,headers=headers or {},stream=True,timeout=(15,30)) as response:
            response.raise_for_status()
            # Range requests must never fall back to an unbounded whole shard.
            if headers and 'Range' in headers: assert response.status_code==206, ('range_not_supported',response.status_code)
            data = bytearray()
            for block in response.iter_content(1<<16):
                data.extend(block)
                self.bytes += len(block)
                assert len(data)<=maximum and self.bytes<=32<<20 and time.monotonic()-self.start<=600
            history = [dict(v.headers) for v in response.history]
            return bytes(data),dict(response.headers),history


class RemoteFile(io.RawIOBase):
    def __init__(self,net,url,size,revision,sha):
        self.net,self.url,self.size,self.revision,self.sha = net,url,size,revision,sha
        self.position=0
        self.cache={}
        self.ranges=[]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        assert 0<=target<=self.size
        self.position=target
        return target

    def read(self,size=-1):
        if size==-1: size=self.size-self.position
        assert 0<=size<=2<<20, ('oversized_metadata_read',size)
        size=min(size,self.size-self.position)
        if not size: return b''
        start=self.position
        end=start+size
        # Fixed64KiB window reduces tiny ZIP record reads; no weight record is requested.
        window_start=(start//65536)*65536
        window_end=min(self.size,((end+65535)//65536)*65536)
        key=(window_start,window_end)
        if key not in self.cache:
            assert window_end-window_start<=2<<20
            data,headers,history=self.net.get(self.url,window_end-window_start,
                {'Range':f'bytes={window_start}-{window_end-1}','Accept-Encoding':'identity'})
            lower={k.lower():v for k,v in headers.items()}
            assert lower.get('content-range')==f'bytes {window_start}-{window_end-1}/{self.size}'
            assert len(data)==window_end-window_start
            origin=next(({k.lower():v for k,v in h.items()} for h in history if 'x-repo-commit' in {k.lower():v for k,v in h.items()}),lower)
            assert origin.get('x-repo-commit')==self.revision
            assert origin.get('x-linked-etag','').strip('"')==self.sha
            self.cache[key]=data
            self.ranges.append({'start':window_start,'end_exclusive':window_end,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        self.position=end
        return self.cache[key][start-window_start:end-window_start]


def rebuild(storage,offset,shape,stride,*rest):
    assert isinstance(storage,dict) and storage.get('kind')=='storage'
    assert type(offset) is int and offset>=0 and len(shape)==len(stride) and 1<=len(shape)<=4
    assert all(type(v) is int and v>0 for v in shape) and all(type(v) is int and v>=0 for v in stride)
    maximum=offset+sum((dimension-1)*step for dimension,step in zip(shape,stride))
    assert maximum<storage['elements']
    return {'kind':'tensor','storage':storage,'offset':offset,'shape':list(shape),'stride':list(stride)}


class HeaderUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'): return OrderedDict
        if module=='torch._utils' and name in ('_rebuild_tensor','_rebuild_tensor_v2'): return rebuild
        if module=='torch' and name in ('FloatStorage','BFloat16Storage','HalfStorage','LongStorage','IntStorage'):
            return {'FloatStorage':('F32',4),'BFloat16Storage':('BF16',2),'HalfStorage':('F16',2),'LongStorage':('I64',8),'IntStorage':('I32',4)}[name]
        raise pickle.UnpicklingError(('unsupported_global',module,name))

    def persistent_load(self,value):
        assert type(value) is tuple and len(value)==5 and value[0]=='storage'
        _,dtype,key,device,elements=value
        assert type(dtype) is tuple and len(dtype)==2 and type(key) is str and key.isdigit()
        assert device=='cpu' and type(elements) is int and elements>0
        return {'kind':'storage','key':key,'dtype':dtype[0],'item_bytes':dtype[1],'elements':elements}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists() and not OUT.exists()
    result={'experiment':'METH-325-actual-Switch-remote-source-headers','models':[]}
    net=Network()
    stage='bindings'
    try:
        for path in (Path(__file__),PROTOCOL,REFERENCE,M.PRIOR,Path(M.__file__)): M.committed(path)
        assert M.digest(REFERENCE)==REFERENCE_SHA and M.digest(M.PRIOR)==M.PRIOR_SHA
        reference=json.loads(REFERENCE.read_text())
        assert reference['passed'] and all(reference['qualification']['gates'].values())
        prior=json.loads(M.PRIOR.read_text())
        result.update({'controller_sha256':M.digest(__file__),'protocol_sha256':M.digest(PROTOCOL),
                       'reference_sha256':REFERENCE_SHA,'prior_sha256':M.PRIOR_SHA})
        OUT.mkdir(parents=True)
        # Fixed primary14.7B first, genuine n128 comparison second. All source shards.
        for model_id in ('google/switch-base-256','google/switch-base-128'):
            previous=next(v for v in prior['models'] if v['model']==model_id)
            revision=previous['revision']
            stage=model_id+'_manifest'
            index_binding=next(v for v in previous['bindings'] if 'index.json' in v['path'])
            assert M.digest(index_binding['path'])==index_binding['sha256']
            index=json.loads(Path(index_binding['path']).read_text())
            url=f'https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true'
            raw,_,_=net.get(url,4<<20)
            info=json.loads(raw)
            assert info['sha']==revision
            metadata_path=OUT/(model_id.replace('/','_')+'_api.json');metadata_path.write_bytes(raw)
            siblings={v['rfilename']:v for v in info['siblings']}
            shards=sorted(set(index['weight_map'].values()))
            entry={'model':model_id,'revision':revision,'metadata_sha256':M.digest(metadata_path),'shards':[],
                   'actual_tensor_headers':{},'declared_index_bytes':index['metadata']['total_size']}
            result['models'].append(entry)
            for shard in shards:
                stage=model_id+'/'+shard
                item=siblings[shard];lfs=item['lfs']
                assert item['size']==lfs['size'] and re.fullmatch('[0-9a-f]{64}',lfs['sha256'])
                url=f'https://huggingface.co/{model_id}/resolve/{revision}/{shard}'
                remote=RemoteFile(net,url,item['size'],revision,lfs['sha256'])
                shard_entry={'name':shard,'bytes':item['size'],'lfs_sha256':lfs['sha256'],'url':url}
                entry['shards'].append(shard_entry)
                with zipfile.ZipFile(remote) as archive:
                    listing=archive.infolist()
                    assert all(v.compress_type==zipfile.ZIP_STORED and '..' not in Path(v.filename).parts for v in listing)
                    choices=[v for v in listing if v.filename.endswith('/data.pkl')]
                    assert len(choices)==1 and choices[0].file_size<=2<<20
                    pkl=archive.read(choices[0])
                    prefix=choices[0].filename[:-len('data.pkl')]
                    assert len(pkl)==choices[0].file_size
                    header_path=OUT/(model_id.replace('/','_')+'_'+shard+'.data.pkl');header_path.write_bytes(pkl)
                    state=HeaderUnpickler(io.BytesIO(pkl)).load()
                    assert isinstance(state,(dict,OrderedDict))
                    expected={name for name,file in index['weight_map'].items() if file==shard}
                    assert set(state)==expected,('actual_shard_namespace',shard)
                    storages={}
                    for name,tensor in state.items():
                        assert isinstance(tensor,dict) and tensor['kind']=='tensor'
                        assert tensor['shape']==previous['tensors'][name] and tensor['storage']['dtype']=='F32'
                        storage=tensor['storage'];storage_key=storage['key']
                        if storage_key in storages: assert storages[storage_key]==storage
                        else: storages[storage_key]=storage
                        record=archive.getinfo(prefix+'data/'+storage_key)
                        assert record.file_size==storage['elements']*storage['item_bytes']
                        entry['actual_tensor_headers'][name]={**tensor,'shard':shard}
                    data_records={v.filename for v in listing if v.filename.startswith(prefix+'data/')}
                    assert data_records=={prefix+'data/'+key for key in storages}
                    shard_entry.update({'pickle_bytes':len(pkl),'pickle_sha256':M.digest(header_path),
                        'tensor_names':len(state),'distinct_storage_records':len(storages),
                        'storage_payload_bytes':sum(s['elements']*s['item_bytes'] for s in storages.values()),
                        'metadata_ranges':remote.ranges,'full_payload_hash_verified':False})
            assert set(entry['actual_tensor_headers'])==set(previous['tensors'])
            elements=sum(__import__('math').prod(v['shape']) for v in entry['actual_tensor_headers'].values())
            assert elements*4==entry['declared_index_bytes']
            entry.update({'serialized_tensor_count':len(entry['actual_tensor_headers']),
                'actual_serialized_tensor_bytes':elements*4,'source_archive_bytes':sum(v['bytes'] for v in entry['shards']),
                'source_storage_bytes':sum(v['storage_payload_bytes'] for v in entry['shards']),
                'passed':True,'learned_tensor_payload_or_expert_hashes_verified':False})
            print(json.dumps({k:entry[k] for k in ('model','serialized_tensor_count','source_archive_bytes','actual_serialized_tensor_bytes','source_storage_bytes','passed')}),flush=True)
        result['resource']={'seconds':time.monotonic()-net.start,'downloaded_body_bytes':net.bytes,'HTTP_calls':net.calls,
                            'end_rss_bytes':psutil.Process().memory_info().rss}
        assert result['resource']['seconds']<=600 and result['resource']['end_rss_bytes']<=1<<30
        result['decision']='eligible_for_separate_bounded_full_payload_acquisition_and_function_hashes'
        result['scope']='Actual tensor metadata decoded by restricted structural parser, all shapes/dtypes/names/storage record sizes. Range windows can include incidental adjacent weight bytes; no full weight tensor or function reconstructed. LFS manifest SHA is expected whole-file hash, not verified full payload. No source quality/uniqueness/native cost/rate proof.'
        M.write(args.out,result)
        print(json.dumps({'sha256':M.digest(args.out),'resource':result['resource'],'decision':result['decision']}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'stage':stage,'seconds':time.monotonic()-net.start,
                       'downloaded_body_bytes':net.bytes,'HTTP_calls':net.calls})
        M.write(args.out.with_suffix('.failure.json'),result)
        raise


if __name__=='__main__': main()
