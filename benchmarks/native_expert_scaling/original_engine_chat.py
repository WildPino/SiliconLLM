"""Canonical interactive client for the unchanged, full-head original kernels."""
import argparse
import struct
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot_hybrid_engine_chat import ChatTokenizer, EngineClient, MAGIC


class OriginalEngineClient(EngineClient):
    def request(self, ids, max_new=64, force_reset=False):
        begin=time.monotonic()
        assert 1<=len(ids)<=8192 and 0<=max_new<=512 and all(0<=i<65537 for i in ids)
        packet=struct.pack('<4I',MAGIC,2 if force_reset else 1,len(ids),max_new)
        self.process.stdin.write(packet+struct.pack('<'+str(len(ids))+'I',*ids));self.process.stdin.flush()
        header,output,logits=self._receive()
        names=('request_seconds','prefill_seconds','decode_seconds','core_seconds','mlp_seconds',
               'head_seconds','forward_seconds','load_seconds')
        row=dict(input_ids=len(ids),generated_ids=output,reused_prefix_ids=header[4],state_reset=bool(header[5]),
            eos=header[6],core_calls=header[7],head_calls=header[8],cached_ids=header[9],
            **dict(zip(names,header[10:],strict=True)),pipe_request_seconds=time.monotonic()-begin)
        assert header[3]==len(ids) and len(output)<=max_new
        assert row['core_calls']==len(ids)-row['reused_prefix_ids']+len(output)
        assert row['head_calls']==row['core_calls'], 'Original kernels compute every full head'
        assert row['cached_ids']==len(ids)+len(output)
        assert row['eos']==(output[-1] if output and output[-1] in (11,228) else 0)
        assert not any(i in (11,228) for i in output[:-1])
        for name in ('core_seconds','mlp_seconds','head_seconds'):
            assert row[name]==0;row[name]=None
        return row,logits


def main(a):
    tokenizer=ChatTokenizer(a.source);messages=[]
    if a.system:messages.append(dict(role='system',content=a.system))
    client=OriginalEngineClient(a.engine,a.model,sys.stderr)
    try:
        while True:
            try:line=input('You: ')
            except EOFError:break
            if line.strip()=='/exit':break
            messages.append(dict(role='user',content=line))
            row,_=client.request(tokenizer.encode(messages),a.max_new)
            answer=tokenizer.decode(row['generated_ids']);print('Assistant: '+answer)
            messages.append(dict(role='assistant',content=answer))
        client.close()
    finally:client.abort()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--system');p.add_argument('--max-new',type=int,default=64);main(p.parse_args())
