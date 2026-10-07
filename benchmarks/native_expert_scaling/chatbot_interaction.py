"""Plain-text Qwen chat adapter and source-relative offline qualification."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT/'results/native_expert_scaling/chatbot_interaction_runtime/site'
sys.path.insert(0,str(SITE))


def serialize(tokenizer,messages,mode):
    """Reusable adapter; modes distinguish a new assistant message from its continuation."""
    if mode not in ('sealed','generate','continue') or not messages:
        raise ValueError('nonempty plain conversation and explicit mode required')
    if any(set(m)!={'role','content'} or m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str) for m in messages):
        raise ValueError('scope: plain system/user/assistant strings without tools')
    if mode=='continue' and messages[-1]['role']!='assistant':
        raise ValueError('assistant continuation requires a final assistant message')
    return tokenizer.apply_chat_template(messages,tokenize=False,
        add_generation_prompt=mode=='generate',continue_final_message=mode=='continue')


def terminate_generated(ids,eos,max_new_tokens):
    """Only newly generated IDs enter this adapter, never serialized prompt/history IDs."""
    if type(max_new_tokens) is not int or max_new_tokens<=0:
        raise ValueError('positive generation limit required')
    accepted=[]
    for token in ids:
        if type(token) is not int or token<0:
            raise ValueError('nonnegative integer token IDs required')
        accepted.append(token)
        if token in eos:
            return dict(accepted_ids=accepted,termination='eos',stop_id=token)
        if len(accepted)==max_new_tokens:
            return dict(accepted_ids=accepted,termination='length',stop_id=None)
    return dict(accepted_ids=accepted,termination='input_exhausted',stop_id=None)


class IndependentBPE:
    """Python source-JSON BPE witness, separate from both HF and Rust implementations."""
    def __init__(self,data):
        import regex
        assert data['normalizer']=={'type':'NFC'}
        assert data['truncation'] is None and data['padding'] is None
        model=data['model']
        assert model['type']=='BPE' and model['dropout'] is None
        assert not model.get('byte_fallback') and not model.get('fuse_unk')
        assert not model.get('ignore_merges') and model.get('unk_token') is None
        assert model.get('continuing_subword_prefix') in ('',None) and model.get('end_of_word_suffix') in ('',None)
        self.vocab=model['vocab']
        merges=[tuple(v.split(' ')) if isinstance(v,str) else tuple(v) for v in model['merges']]
        assert all(len(v)==2 for v in merges)
        self.rank={v:i for i,v in enumerate(merges)}
        assert len(self.rank)==len(merges)
        pre=data['pre_tokenizer']['pretokenizers']
        assert pre[0]['type']=='Split' and pre[0]['behavior']=='Isolated' and pre[0]['invert'] is False
        assert pre[1]['type']=='ByteLevel' and not pre[1]['add_prefix_space'] and not pre[1]['use_regex']
        self.pattern=regex.compile(pre[0]['pattern']['Regex'])
        ordinary=list(range(33,127))+list(range(161,173))+list(range(174,256))
        extra=[b for b in range(256) if b not in ordinary]
        self.alphabet=dict(zip(ordinary+extra,[chr(b) for b in ordinary]+[chr(256+i) for i in range(len(extra))]))
        self.special={v['content']:v['id'] for v in data['added_tokens']}
        assert all(not any(v[k] for k in ('lstrip','rstrip','single_word','normalized')) for v in data['added_tokens'])
        self.special_pattern=re.compile('('+ '|'.join(re.escape(s) for s in sorted(self.special,key=lambda s:(-len(s),s)))+')')

    def encode(self,text):
        out=[]
        for block in self.special_pattern.split(text):
            if block in self.special:
                out.append(self.special[block]); continue
            normalized=unicodedata.normalize('NFC',block)
            pieces=[m.group() for m in self.pattern.finditer(normalized)]
            assert ''.join(pieces)==normalized, 'reference pretokenizer coverage'
            for piece in pieces:
                symbols=[self.alphabet[b] for b in piece.encode('utf8')]
                while len(symbols)>1:
                    choices=[(self.rank[pair],i,pair) for i in range(len(symbols)-1) if (pair:=(symbols[i],symbols[i+1])) in self.rank]
                    if not choices: break
                    _,_,pair=min(choices)
                    merged=[]; i=0
                    while i<len(symbols):
                        if i+1<len(symbols) and (symbols[i],symbols[i+1])==pair:
                            merged.append(symbols[i]+symbols[i+1]); i+=2
                        else:
                            merged.append(symbols[i]); i+=1
                    symbols=merged
                out.extend(self.vocab[s] for s in symbols)
        return out


def write(path,value):
    data=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
    assert len(data)<=2<<20
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def main(args):
    import psutil
    proc=psutil.Process(); proc.cpu_affinity([10])
    start=time.monotonic()
    binding=json.loads(args.binding.read_bytes())
    assert hashlib.sha256(args.binding.read_bytes()).hexdigest()==args.binding_sha
    assert sys.version_info[:3]==(3,12,10)
    assert all(importlib.util.find_spec(v) is None for v in ('torch','tensorflow','jax'))
    r=dict(scope='CANONICAL_PLAIN_QWEN_CHAT_CONTRACT_NOT_MODEL_QUALITY',
           started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
           process_instance=dict(pid=proc.pid,create_time_unix=proc.create_time()),
           binding_sha256=args.binding_sha,source_commit=args.freeze,
           new_model_native_or_tensor_weight_calls=0,pipeline_complete=False,gates={})
    try:
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
        for package,version in binding['versions'].items():
            assert md.version(package)==version,(package,md.version(package))
        source=Path(binding['donor_source'])
        config=json.loads((source/'tokenizer_config.json').read_bytes())
        gen=json.loads((source/'generation_config.json').read_bytes())
        eos=gen['eos_token_id']
        assert eos==[151645,151643] and config['add_bos_token'] is False
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
        assert tokenizer.chat_template==config['chat_template']
        assert tokenizer.bos_token is None and tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643
        raw=Tokenizer.from_file(str(source/'tokenizer.json'))
        reference=IndependentBPE(json.loads((source/'tokenizer.json').read_bytes()))
        spec=json.loads(args.fixtures.read_bytes())
        assert spec['schema']=='QWEN_PLAIN_CHAT_GOLDENS_V1'
        r['cases']=[]
        for case in spec['cases']:
            rendered=serialize(tokenizer,case['messages'],case['mode'])
            assert rendered==case['expected_text'],('render golden mismatch',case['name'])
            canonical=tokenizer.apply_chat_template(case['messages'],tokenize=True,
                add_generation_prompt=case['mode']=='generate',continue_final_message=case['mode']=='continue')
            encoded=tokenizer.encode(rendered,add_special_tokens=False)
            original=raw.encode(rendered,add_special_tokens=False).ids
            independent=reference.encode(case['expected_text'])
            assert canonical==encoded==original==independent,('token-ID mismatch',case['name'])
            assert all(0<=t<151936 for t in canonical) and canonical[0]==151644
            decoded=tokenizer.decode(canonical,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert decoded==unicodedata.normalize('NFC',rendered),('NFC roundtrip',case['name'])
            wire=struct.pack('<'+'I'*len(canonical),*canonical)
            r['cases'].append(dict(name=case['name'],mode=case['mode'],messages=case['messages'],
                rendered_text=rendered,rendered_UTF8_SHA256=hashlib.sha256(rendered.encode('utf8')).hexdigest(),
                prompt_ids=canonical,prompt_U32LE_SHA256=hashlib.sha256(wire).hexdigest(),
                canonical_raw_and_independent_ID_parity=True,NFC_decode_roundtrip=True))
        r['gates']['all_frozen_text_goldens_and_three_ID_paths']=True
        for spelling,want in [('<|endoftext|>',151643),('<|im_start|>',151644),('<|im_end|>',151645)]:
            assert tokenizer.encode(spelling,add_special_tokens=False)==[want]
            assert raw.token_to_id(spelling)==reference.special[spelling]==want
        r['gates']['source_special_tokens_no_automatic_BOS']=True
        try:
            tokenizer.apply_chat_template([{'role':'assistant','content':'x'}],add_generation_prompt=True,continue_final_message=True)
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible final-message modes must reject')
        try:
            tokenizer.apply_chat_template([])
        except ValueError:
            pass
        else:
            raise AssertionError('empty conversation must reject')
        r['gates']['canonical_invalid_mode_and_empty_input_rejected']=True
        r['stop_tests']=[]
        for case in spec['stop_cases']:
            result=terminate_generated(case['new_ids'],eos,case['max_new_tokens'])
            assert result==case['expected'],('stop golden mismatch',case['name'])
            r['stop_tests'].append(dict(name=case['name'],result=result,prompt_ids_never_examined=True))
        r['gates']['both_EOS_length_and_generated_only_stops']=True
        assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
        assert not proc.children(recursive=True)
        used=[]
        for name,module in sorted(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path and Path(path).is_file():
                p=Path(path).resolve()
                if '.venv' in str(p) or 'chatbot_interaction' in str(p):
                    used.append(dict(module=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        r.update(decision='PLAIN_QWEN_INTERACTION_QUALIFIED_PIPELINE_NOT_QUALIFIED',
            decoding_policy=dict(do_sample=False,max_new_tokens=128,eos_token_ids=eos,
                source_sampling_defaults=gen,count_generated_EOS_as_accepted_ID=True,
                displayed_text_skips_special_tokens=True,scope='deterministic plain-chat comparison; source sampling defaults retained separately'),
            tools_and_multimodal_qualified=False,native_chat_adapter_integrated=False,
            runtime=dict(python=sys.version,versions=binding['versions'],Torch_imported=False,used_dependency_files=used),
            resource_before_final_serialization=dict(seconds=time.monotonic()-start,OS_peak_bytes=proc.memory_info().peak_wset),
            ended_compute_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(args.out,r)
        print(json.dumps(dict(decision=r['decision'],cases=len(r['cases']),gates=r['gates'])),flush=True)
    except BaseException as error:
        r.update(fault=repr(error),ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 resource_snapshot=dict(seconds=time.monotonic()-start,OS_peak_bytes=proc.memory_info().peak_wset))
        write(args.out.with_suffix('.failure.json'),r)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--binding-sha',required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
