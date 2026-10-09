"""Canonical plain chat client for the packed ternary engine stream, no donor runtime."""
import argparse
import json
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SITE=ROOT/'results/native_expert_scaling/chatbot_source_runtime/site'
sys.path.insert(0,str(SITE))
MAGIC=0x31434853


class ChatTokenizer:
    def __init__(self,source):
        from tokenizers import Tokenizer
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        source=Path(source)
        self.tokenizer=Tokenizer.from_file(str(source/'tokenizer.json'))
        self.tokenizer.no_truncation();self.tokenizer.no_padding()
        config=json.loads((source/'tokenizer_config.json').read_bytes())
        self.bos=config['bos_token']
        self.template=ImmutableSandboxedEnvironment(undefined=StrictUndefined,trim_blocks=True,lstrip_blocks=True).from_string((source/'chat_template.jinja').read_text())
        assert self.tokenizer.token_to_id(self.bos)==17
        assert self.tokenizer.token_to_id('<|end_of_text|>')==11
        assert self.tokenizer.token_to_id('<|im_end|>')==228

    def encode(self,messages):
        assert messages and all(m['role'] in ('system','user','assistant') and isinstance(m['content'],str) for m in messages)
        assert all(m['role']!='system' for m in messages[1:])
        text=self.template.render(bos_token=self.bos,messages=messages,tools=None,add_generation_prompt=True)
        expected=self.bos+''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
        assert text==expected,'source template mismatch'
        return self.tokenizer.encode(text,add_special_tokens=False).ids

    def decode(self,ids):
        return self.tokenizer.decode(ids,skip_special_tokens=True)


class EngineClient:
    def __init__(self,exe,model,stderr,timeout=100):
        self.timeout=timeout;self.messages=queue.Queue()
        self.process=subprocess.Popen([str(exe),'--stream',str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                     stderr=stderr,creationflags=0x08000000)
        threading.Thread(target=self._reader,daemon=True).start()
        try:self.hello=self._receive()
        except BaseException:self.abort();raise

    def _read(self,n):
        parts=[]
        while n:
            part=self.process.stdout.read(n)
            if not part:raise EOFError('engine pipe ended')
            parts.append(part);n-=len(part)
        return b''.join(parts)

    def _reader(self):
        try:
            hello=struct.unpack('<6I3Qd',self._read(56))
            assert hello[:3]==(MAGIC,1,65537) and hello[4:6]==(11,228)
            self.messages.put(hello)
            while True:
                header=struct.unpack('<10I8d',self._read(104))
                assert header[:2]==(MAGIC,1) and header[2]<=512
                ids=list(struct.unpack('<'+str(header[2])+'I',self._read(header[2]*4)))
                logits=self._read(65537*4)
                self.messages.put((header,ids,logits))
        except BaseException as error:self.messages.put(error)

    def _receive(self):
        value=self.messages.get(timeout=self.timeout)
        if isinstance(value,BaseException):raise value
        return value

    def request(self,ids,max_new=64,force_reset=False):
        begin=time.monotonic()
        assert 1<=len(ids)<=8192 and 0<=max_new<=512 and all(0<=i<65537 for i in ids)
        raw=struct.pack('<4I',MAGIC,2 if force_reset else 1,len(ids),max_new)+struct.pack('<'+str(len(ids))+'I',*ids)
        self.process.stdin.write(raw);self.process.stdin.flush()
        header,output,logits=self._receive()
        names=('request_seconds','prefill_seconds','decode_seconds','core_seconds','mlp_seconds','head_seconds','forward_seconds','load_seconds')
        result=dict(input_ids=len(ids),generated_ids=output,reused_prefix_ids=header[4],state_reset=bool(header[5]),
                    eos=header[6],core_calls=header[7],head_calls=header[8],cached_ids=header[9],
                    **dict(zip(names,header[10:],strict=True)),pipe_request_seconds=time.monotonic()-begin)
        assert header[3]==len(ids) and len(output)<=max_new
        assert result['core_calls']==len(ids)-result['reused_prefix_ids']+len(output)
        assert result['head_calls']==int(len(ids)>result['reused_prefix_ids'])+len(output)
        assert result['cached_ids']==len(ids)+len(output)
        assert result['eos']==(output[-1] if output and output[-1] in (11,228) else 0)
        assert not any(i in (11,228) for i in output[:-1])
        return result,logits

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(struct.pack('<4I',MAGIC,0,0,0));self.process.stdin.flush()
            self.process.wait(timeout=5)
        assert self.process.returncode==0

    def abort(self):
        if self.process.poll() is None:self.process.kill();self.process.wait()


def main(a):
    tokenizer=ChatTokenizer(a.source);messages=[]
    if a.system:messages.append(dict(role='system',content=a.system))
    client=EngineClient(a.engine,a.model,sys.stderr)
    try:
        while True:
            try:line=input('You: ')
            except EOFError:break
            if line.strip()=='/exit':break
            messages.append(dict(role='user',content=line))
            row,_=client.request(tokenizer.encode(messages),a.max_new)
            text=tokenizer.decode(row['generated_ids']);print('Assistant: '+text)
            messages.append(dict(role='assistant',content=text))
        client.close()
    finally:client.abort()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--system');p.add_argument('--max-new',type=int,default=64)
    main(p.parse_args())
