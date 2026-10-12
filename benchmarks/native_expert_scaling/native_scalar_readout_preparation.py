"""Compile-only original engine head bodies plus explicit scalar RMS extension."""
import argparse,hashlib,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'benchmarks/native_expert_scaling';DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0,str(B))
from falcon_auxiliary_s6_transport import extent,write


def prepare(a):
    start=time.monotonic();assert not a.out.exists() and not a.directory.exists()
    engine=ROOT/'benchmarks/phase60/engine.c';original=extent(engine)
    assert original['sha256']=='5f948fc0dcd28b1647a2a2c73067bdc0a76a7d41ae7f34395ede8120aeaa83ce'
    source=engine.read_text();bodies={}
    for name in ['hsum256','dotf','matvec','rmsnorm']:
        match=re.search(r'^static[^\n]*\b'+name+r'\(',source,re.M);assert match
        lo=source.index('{',match.start());depth=1;hi=lo+1
        while depth:depth+=(source[hi]=='{')-(source[hi]=='}');hi+=1
        bodies[name]=source[match.start():hi]
    body=bodies['rmsnorm'];expected='static inline void rmsnorm(const float* in,const float* w,float* out)'
    assert body.startswith(expected) and body.count('1e-5f')==1
    extended=body.replace(expected,'static inline float scalar_rmsnorm(const float* in,const float* w,float* out,float epsilon)',1).replace('1e-5f','epsilon',1)
    extended=extended[:-1]+' return r; }'
    exports='''
__declspec(dllexport) int scalar_readout_abi(int which){return which?V:D;}
__declspec(dllexport) int scalar_readout(const float* head,const float* bias,const float* states,int count,float epsilon,float* normalized,float* inverse,float* scores){
    if(count<1||count>128||!isfinite(epsilon)||epsilon<=0.0f)return 1;
    float gamma[D];for(int i=0;i<D;i++)gamma[i]=1.0f;
    for(int j=0;j<count;j++){
        inverse[j]=scalar_rmsnorm(states+(size_t)j*D,gamma,normalized+(size_t)j*D,epsilon);
        if(!isfinite(inverse[j])||inverse[j]<=0.0f)return 2;
        matvec(head,normalized+(size_t)j*D,scores+(size_t)j*V,V,D);
        for(int i=0;i<V;i++){
            scores[(size_t)j*V+i]+=bias[i]*inverse[j];
            if(!isfinite(scores[(size_t)j*V+i]))return 3;
        }
    }return 0;
}
'''
    text='#include <stddef.h>\n#include <math.h>\n#include <immintrin.h>\n#define D 256\n#define V 65537\n#define OMP_PFOR\n'
    text+='\n\n'.join(bodies.values())+'\n'+extended+exports
    compiler=Path(json.loads((DOC/'chatbot_hybrid_engine_probe_binding_20261009.json').read_bytes())['compiler'])
    before=extent(compiler);a.directory.mkdir();c=a.directory/'native_scalar_readout.c';dll=a.directory/'native_scalar_readout.dll';log=a.directory/'compile.log'
    with c.open('x',encoding='utf8',newline='\n') as f:f.write(text)
    command=[str(compiler),'-O3','-mavx2','-mfma','-march=znver2','-shared',str(c.resolve()),'-o',str(dll.resolve()),'-lm']
    p=subprocess.run(command,capture_output=True,timeout=45,creationflags=0x08000000);log.write_bytes(p.stdout+p.stderr)
    assert p.returncode==0,log.read_text(errors='replace');assert extent(engine)==original and extent(compiler)==before
    write(a.out,dict(schema='NATIVE_SCALAR_READOUT_PREPARATION_V1',source_tool=extent(__file__),freeze=a.freeze,command=sys.orig_argv,
        engine=original,compiler=before,compile_command=command,compile_exit_code=p.returncode,source=extent(c),dll=extent(dll),log=extent(log),
        functions=[dict(name=k,original_body_sha256=hashlib.sha256(v.encode()).hexdigest()) for k,v in bodies.items()],
        extension=dict(name='scalar_rmsnorm',body_sha256=hashlib.sha256(extended.encode()).hexdigest(),
            changes='Original scalar ms loop/gamma multiplication unchanged;epsilon becomes argument;return inverse RMS. Original rmsnorm also retained verbatim.'),
        additional_reader_bias='scores+=bias*inverse_RMS;not constant postnorm bias.',seconds=time.monotonic()-start,
        scope='Compile-only actual original hsum256/dotf/matvec/rmsnorm bodies unchanged and named epsilon/inverse RMS extension. '
            'No DLL load/readout input/call or source model/GPU/T4;no native precision or timing/quality admission yet. '
            'OMP_PFOR empty,original engine file immutable.'))
    print(json.dumps(dict(preparation=extent(a.out),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',required=True);prepare(p.parse_args())
