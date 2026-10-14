"""Structural depth intervention; source residual width/operators stay intact.

Experimental quality bridge, not an admitted CPU model or conditional router.
Omitted blocks are identity maps. Kept blocks consume their NEW own-history
hidden, use the qualified activation backend and have independent native state.
No token-varying routing: reactivating a skipped stateful block is unsupported.
"""
import ctypes
import hashlib
import os
import types

from full_width_native_chat import NativeBridge
from bf_activation_backend import load_qualified_backend
from falcon_auxiliary_s6_transport import extent


def kept_sites(count):
    """Fixed evenly spaced source sites including first/last; no data-dependent fit."""
    if count not in (3,6,12,24):
        raise ValueError('Frozen depth family is 3, 6, 12, 24')
    return tuple((46*i+count-1)//(2*(count-1)) for i in range(count))


class FixedDepthBridge(NativeBridge):
    def __init__(self,model,binding,np,torch,count,activation_closure):
        from token_state import arrays,ptrs
        self.active=kept_sites(count)
        self.np,self.torch,self.model=np,torch,model
        self.dll,self.create,self.destroy,self.tokens,self.rows,self.snapshot=load_qualified_backend(activation_closure)
        self.last=self.dll.ts_last;self.last.argtypes=[ctypes.c_void_p];self.last.restype=ctypes.c_uint32
        self.table=np.memmap(binding['table']['path'],dtype='u2',mode='r',shape=(65537,6704))
        self.params,self.weights,self.conv,self.outputs=([None]*24 for _ in range(4))
        self.contexts=[];self.token=None;self.steps=0;self.cache=None;self.streams=None;self.consumed=[]
        self.block_invocations=[0]*24;self.original_forwards=[];self.original_mamba_forwards=[]
        assert len(model.model.layers)==24 and len(binding['exports'])==24
        for i,layer in enumerate(model.model.layers):
            self.original_mamba_forwards.append(layer.mamba.forward)
            if i in self.active:
                export=binding['exports'][i];assert export['site']==i
                assert extent(export['file']['path'])==export['file']
                with np.load(export['file']['path'],allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
                self.params[i]=p;self.weights[i]=ptrs([p[k] for k in ('Ws','Vs','Alog','D','gamma')])
                self.conv[i]=ptrs([p[k] for k in ('cw','cb','db')]);self.outputs[i]=arrays(1,np)
                def native(m,hidden_states,cache_params=None,attention_mask=None,site=i):
                    assert site in self.active
                    return self.mix(site,m,hidden_states,cache_params,attention_mask)
                layer.mamba.forward=types.MethodType(native,layer.mamba)
            original=layer.forward;self.original_forwards.append(original)
            if i in self.active:
                def kept(m,hidden_states,*args,source_forward=original,site=i,**kwargs):
                    assert hidden_states.shape==(1,1,2048) and hidden_states.dtype==torch.bfloat16
                    self.block_invocations[site]+=1
                    return source_forward(hidden_states,*args,**kwargs)
                layer.forward=types.MethodType(kept,layer)
            else:
                def omitted(m,hidden_states,*args,site=i,**kwargs):
                    assert hidden_states.shape==(1,1,2048) and hidden_states.dtype==torch.bfloat16
                    self.block_invocations[site]+=1
                    # No norm, generator, state update, attention/cache or FFN calls.
                    return (hidden_states,)
                layer.forward=types.MethodType(omitted,layer)

    def restore(self):
        """Restore source methods before another arm; cannot discard open traces."""
        assert self.streams is None
        self.close_contexts()
        for i,layer in enumerate(self.model.model.layers):
            layer.forward=self.original_forwards[i]
            layer.mamba.forward=self.original_mamba_forwards[i]

    def close_contexts(self):
        for context in self.contexts:
            if context is not None:self.destroy(context)
        self.contexts=[];self.cache=None

    def reset(self,directory,label):
        assert self.streams is None
        self.close_contexts()
        self.contexts=[self.create() if i in self.active else None for i in range(24)]
        assert all(self.contexts[i] for i in self.active)
        self.steps=0;self.consumed=[];self.token=None;self.block_invocations=[0]*24
        self.trace_paths=[directory/f'{label}.site{i:02d}.trace.bf16' if i in self.active else None for i in range(24)]
        self.streams=[p.open('xb') if p is not None else None for p in self.trace_paths]

    def finish(self,directory,label):
        np=self.np
        assert self.streams is not None and self.steps>0
        assert self.block_invocations==[self.steps]*24
        for stream in self.streams:
            if stream is not None:
                stream.flush();os.fsync(stream.fileno());stream.close()
        self.streams=None;records=[]
        for i in self.active:
            context=self.contexts[i];state=np.empty((3072,96),'f4');ring=np.empty((3,3584),'u2');seen=np.zeros(1,'u8')
            assert self.snapshot(context,state.ctypes.data,ring.ctypes.data,seen.ctypes.data)==0
            assert seen[0]==self.steps and self.last(context)==self.consumed[-1]
            path=directory/f'{label}.site{i:02d}.state.npz'
            np.savez(path,state=state,ring=ring,seen=seen,last=np.array([self.last(context)],'u4'))
            width=2048 if i==0 else 10800
            assert self.trace_paths[i].stat().st_size==self.steps*width*2
            records.append(dict(site=i,trace=dict(**extent(self.trace_paths[i]),shape=[self.steps,width],dtype='BF16'),
                                final=extent(path)))
        attention=[]
        for i in range(24):
            cache_layer=self.cache.layers[i]
            for key in ('keys','values'):
                tensor=getattr(cache_layer,key,None)
                if i not in self.active:
                    assert tensor is None or tensor.numel()==0,(i,key)
                    continue
                assert tensor is not None and tensor.shape[-2]==self.steps
                bits=tensor.detach().cpu().contiguous().view(self.torch.uint16).numpy()
                attention.append(dict(site=i,field=key,shape=list(tensor.shape),dtype=str(tensor.dtype),
                                      sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
        assert len(attention)==2*len(self.active)
        result=dict(label=label,consumed_ids=self.consumed.copy(),seen=self.steps,sites=records,attention=attention,
            active_sites=list(self.active),omitted_sites=[i for i in range(24) if i not in self.active],
            block_invocations=self.block_invocations.copy(),selected_functions_per_step=len(self.active),
            selection_policy='Frozen source site schedule, selected masses exactly1, omitted masses0. '
                             'Omitted blocks are identity; not token-varying selective capacity.',
            native_parameter_payload_bytes=sum(v.nbytes for p in self.params if p is not None for v in p.values()))
        self.close_contexts()
        return result
