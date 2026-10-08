"""Persistent native compact-chat protocol; caller binds runtime and artifact.

No model/process/tokenizer work at import. This component is UNEXECUTED until
the eligible fixed fit is exported and the new native profile is qualified.
"""
import hashlib
from pathlib import Path
import struct
import subprocess
import time


def read_exact(stream, count):
    pieces=[];remaining=count
    while remaining:
        part=stream.read(remaining)
        if not part:raise EOFError(f'Native response ended with {remaining} bytes missing')
        pieces.append(part);remaining-=len(part)
    return b''.join(pieces)


class CompactNativeClient:
    """One loaded archive with separate generation and operator-probe modes.

    Caller supplies exact catalog/blob/executable receipts, a finite watchdog,
    isolated ownership and a binary stderr log. Timings here include request
    transport; source-relative behavioral scoring is performed by the caller.
    """
    def __init__(self, executable, archive, capacity, stderr, mode='stream'):
        if type(capacity) is not int or not 2<=capacity<=4096 or mode not in ('stream','probe'):
            raise ValueError('Declared capacity/mode required')
        self.capacity=capacity;self.mode=mode;self.started=time.perf_counter()
        self.process=subprocess.Popen([str(Path(executable).resolve()),str(Path(archive).resolve()),mode,str(capacity)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,creationflags=8,bufsize=0)
        self.requests=0;self.cold_first_seconds=None

    def _ids(self, ids):
        if not isinstance(ids,list) or not ids or any(type(i) is not int or not 0<=i<151936 for i in ids):
            raise ValueError('Nonempty complete-vocabulary integer IDs required')
        if self.process.poll() is not None:raise RuntimeError(f'Native process exited: {self.process.returncode}')

    def _write(self, payload):
        remaining=memoryview(payload)
        while remaining:
            written=self.process.stdin.write(remaining)
            if written is None or written<=0:raise BrokenPipeError('Native request write')
            remaining=remaining[written:]
        self.process.stdin.flush()

    def generate(self, prompt_ids, max_new):
        if self.mode!='stream':raise ValueError('Generation requires stream mode')
        self._ids(prompt_ids)
        if type(max_new) is not int or not 1<=max_new<=128 or len(prompt_ids)+max_new>self.capacity:
            raise ValueError('Generation/context limits')
        started=time.perf_counter()
        payload=struct.pack('<8s2I',b'QWCR0001',len(prompt_ids),max_new)+struct.pack('<'+'i'*len(prompt_ids),*prompt_ids)
        self._write(payload);header=read_exact(self.process.stdout,24)
        magic,count,stop,compute=struct.unpack('<8s2Id',header)
        if magic!=b'QWCO0001' or not 1<=count<=max_new or stop not in (0,1) or not 0<compute<float('inf'):
            raise ValueError('Native response header')
        ids=list(struct.unpack('<'+'i'*count,read_exact(self.process.stdout,count*4)))
        if any(not 0<=i<151936 for i in ids):raise ValueError('Native output ID range')
        if any(i in (151645,151643) for i in ids[:-1]):raise ValueError('Output after EOS')
        if bool(stop)!=(ids[-1] in (151645,151643)) or (not stop and count!=max_new):
            raise ValueError('Native EOS/length policy')
        elapsed=time.perf_counter()-started;self.requests+=1
        if self.requests==1:self.cold_first_seconds=time.perf_counter()-self.started
        return dict(generated_ids=ids,termination='eos' if stop else 'length',stop_id=ids[-1] if stop else None,
            native_compute_seconds=compute,transport_and_compute_seconds=elapsed,
            cold_first_seconds=self.cold_first_seconds if self.requests==1 else None)

    def probe(self, input_ids, positions):
        if self.mode!='probe':raise ValueError('Operator qualification requires probe mode')
        self._ids(input_ids)
        if len(input_ids)>self.capacity or not isinstance(positions,list) or not 1<=len(positions)<=16:
            raise ValueError('Probe context/position limits')
        if any(type(i) is not int or not 0<=i<len(input_ids) for i in positions) or positions!=sorted(set(positions)):
            raise ValueError('Probe positions must be strictly increasing')
        payload=struct.pack('<8s2I',b'QWCP0001',len(input_ids),len(positions))
        payload+=struct.pack('<'+'I'*len(positions),*positions)+struct.pack('<'+'i'*len(input_ids),*input_ids)
        started=time.perf_counter();self._write(payload)
        header=read_exact(self.process.stdout,24)
        if struct.unpack('<8s4I',header)!=(b'QWCT0001',151936,len(positions),24,896):raise ValueError('Probe response header')
        frames=[]
        for position in positions:
            actual=struct.unpack('<I',read_exact(self.process.stdout,4))[0]
            if actual!=position:raise ValueError('Probe position mismatch')
            # Complete little-endian arrays remain bytes for independent typed
            # decoding; no numerical comparison or qualification occurs here.
            frame=dict(position=position)
            for name,size in [('selected_I32',24*4*4),('mass_F32',24*4*4),('query_F32',24*32*4),
                              ('compact_input_BF16',24*896*2),('compact_output_F32',24*896*4),('logits_BF16',151936*2)]:
                frame[name]=read_exact(self.process.stdout,size)
            frames.append(frame)
        self.requests+=1
        return dict(frames=frames,transport_and_compute_seconds=time.perf_counter()-started)

    def close(self):
        if self.process.poll() is None:self.process.stdin.close()
        code=self.process.wait(timeout=30);self.process.stdout.close()
        if code!=0:raise RuntimeError(f'Native process exit: {code}')
        return code

    def abort(self):
        if self.process.poll() is None:self.process.kill()
        return self.process.wait(timeout=30)


def respond(client, tokenizer, messages, mode='generate', max_new=64):
    """Canonical source HF API -> native -> visible text, measured together.

    Canonical rendering/IDs use the qualified interaction component. Caller
    separately adopts exact tokenizer source/runtime hashes and scores outputs.
    Own-history dialogue caller appends this returned assistant text, never a
    donor answer. Tools are outside the supported plain-text contract.
    """
    from chatbot_interaction import serialize
    started=time.perf_counter();rendered=serialize(tokenizer,messages,mode)
    prompt=tokenizer.encode(rendered,add_special_tokens=False)
    result=client.generate(prompt,max_new)
    text=tokenizer.decode(result['generated_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    special=set(tokenizer.all_special_ids)
    visible_ids=[i for i in result['generated_ids'] if i not in special]
    result.update(assistant_text=text,prompt_ids=prompt,mode=mode,
        rendered_UTF8_sha256=hashlib.sha256(rendered.encode('utf8')).hexdigest(),
        visible_generated_ids=visible_ids,visible_ID_count=len(visible_ids),
        end_to_end_seconds=time.perf_counter()-started,
        cold_complete_response_seconds=time.perf_counter()-client.started if client.requests==1 else None,
        acceptance_scope='Unscored output; visible IDs count as useful only after frozen per-case quality scoring')
    return result
