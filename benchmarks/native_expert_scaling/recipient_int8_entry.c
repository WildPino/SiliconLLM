/* Original Qwen2.5-0.5B-Instruct cost profile on the existing complete engine.
   This wrapper changes neither model arithmetic nor the QWENDON1 loader. */
#define main recipient_legacy_main
#include "../donor_adaptation/engine/donor_engine.c"
#undef main
#include <errno.h>
#include <fenv.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>

static void require(int ok,const char *what){if(!ok){fprintf(stderr,"recipient-int8: %s\n",what);exit(2);}}
static unsigned char *alloc(size_t n,size_t align){(void)align;unsigned char *p=(unsigned char*)malloc(n);require(p!=NULL,"worker allocation");return p;}
#include "meth388_switch_thread_binding.h"

static long ri_number(const char *s,long low,long high,const char *name){
    char *end=NULL;errno=0;long v=strtol(s,&end,10);
    require(!errno&&end&&!*end&&v>=low&&v<=high,name);return v;
}
static FILE *ri_output(const char *path){
    int fd=_open(path,_O_WRONLY|_O_BINARY|_O_CREAT|_O_EXCL,_S_IREAD|_S_IWRITE);
    require(fd>=0,"exclusive output open");FILE *f=_fdopen(fd,"wb");
    if(!f)_close(fd);require(f!=NULL,"output stream");return f;
}
static int32_t *ri_ids(const char *path,int *n){
    FILE *f=fopen(path,"rb");require(f!=NULL,"ids open");
    require(!_fseeki64(f,0,SEEK_END),"ids seek end");long long sz=_ftelli64(f);
    require(sz>=4&&sz%4==0&&sz/4<=2048,"ids length");
    require(!_fseeki64(f,0,SEEK_SET),"ids rewind");
    int32_t *ids=(int32_t*)xmalloc((size_t)sz);require(fread(ids,1,(size_t)sz,f)==(size_t)sz,"ids read");
    require(fclose(f)==0,"ids close");*n=(int)(sz/4);return ids;
}
static void ri_geometry(const model_t *m){
    require(m->D==896&&m->F==4864&&m->L==24&&m->V==151936&&m->NH==14&&m->NKV==2&&m->HD==64,"original recipient geometry");
    require(m->tied==0&&m->quant==5,"untied int8 deployment head/codec");
    require(m->rms_eps==1.0e-6f&&m->rope_theta==1.0e6f,"RMS/RoPE parameters");
    require(!g_fuse&&!g_lut&&!g_lutblk&&!g_lutdiag&&!g_lutnodown&&!g_lutnohead&&g_clip==0,"optional matrix path");
    require(!g_factors.blob&&!g_head_shortlist.blob&&!g_sw&&g_carvek==-1&&!g_carvedump,"optional bank/router/sweep");
    require(g_mvacc==4&&g_attn==ATTN_AVX4&&g_attnr==ATTNR_NONE&&g_fexp==0,"kernel selection");
    for(int l=0;l<m->L;l++)require(!m->lay[l].carved,"original dense FFN only");
}
static uint64_t ri_matrix(const mat_t *m,int rows,int cols){
    require(m->out==rows&&m->in==cols&&m->code&&m->scale&&!m->f32&&!m->packed&&!m->rank&&!m->tposed&&!m->tm,"plain int8 matrix");
    return (uint64_t)rows*(uint64_t)cols+(uint64_t)rows*4u;
}
static uint64_t ri_matrix_bytes(const model_t *m){
    uint64_t bytes=ri_matrix(&m->head,m->V,m->D);
    for(int i=0;i<m->L;i++){
        const layer_t *l=&m->lay[i];
        bytes+=ri_matrix(&l->q,m->D,m->D)+ri_matrix(&l->k,128,m->D)
             + ri_matrix(&l->v,128,m->D)+ri_matrix(&l->o,m->D,m->D)
             + ri_matrix(&l->gate,m->F,m->D)+ri_matrix(&l->up,m->F,m->D)
             + ri_matrix(&l->down,m->D,m->F);
    }
    return bytes;
}
static void ri_finite(const float *x,int n,const char *what){for(int i=0;i<n;i++)require(isfinite(x[i]),what);}
static int ri_argmax(const float *x,int n){int best=0;for(int i=1;i<n;i++)if(x[i]>x[best])best=i;return best;}
static void ri_write(FILE *f,const void *p,size_t n){require(fwrite(p,1,n,f)==n,"output write");}
static void ri_binding_receipt(void){
    for(int i=0;i<worker_requested;i++){
        const WorkerBinding b=worker_latest[i];require(b.tid&&b.slot==i&&b.group==0&&b.mask==worker_masks[i],"worker binding identity");
        HANDLE thread=OpenThread(THREAD_QUERY_LIMITED_INFORMATION,FALSE,b.tid);require(thread!=NULL,"worker liveness");
        GROUP_AFFINITY actual;memset(&actual,0,sizeof actual);int ok=GetThreadGroupAffinity(thread,&actual);CloseHandle(thread);
        require(ok&&actual.Group==0&&actual.Mask==b.mask,"worker binding drift");
    }
}
int main(int argc,char **argv){
    require(argc==7,"usage: engine WEIGHTS IDS NEW_TOKENS THREADS probe|generate OUTPUT");
    int new_tokens=(int)ri_number(argv[3],0,128,"new_tokens"),threads=(int)ri_number(argv[4],1,6,"threads");
    require(threads==1||threads==3||threads==6,"threads must be 1,3,6");
    int probe=!strcmp(argv[5],"probe"),generate=!strcmp(argv[5],"generate");
    require(probe||generate,"mode");require((probe&&new_tokens==0)||(generate&&new_tokens>=1),"mode/new_tokens");
    require(fesetround(FE_TONEAREST)==0&&fegetround()==FE_TONEAREST,"rounding mode");
    require(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma"),"AVX2/FMA required");
    g_mvacc=4;g_attn=ATTN_AVX4;g_attnr=ATTNR_NONE;g_fexp=0;g_wit=0;g_prof=0;
    worker_setup(threads);for(int i=0;i<threads;i++)require(worker_masks[i]==((DWORD_PTR)1<<(2*i)),"physical slots 0,2,4,6,8,10");
    int n=0;int32_t *ids=ri_ids(argv[2],&n);require(!probe||n<=32,"probe limited to 32 positions");
    require(n+new_tokens<=2176,"total context bound");
    FILE *out=ri_output(argv[6]);
    model_t m={0};load(&m,argv[1]);ri_geometry(&m);
    for(int i=0;i<n;i++)require(ids[i]>=0&&ids[i]<m.V,"input token range");
    uint64_t active_matrix_bytes=ri_matrix_bytes(&m);
    state_t s={0};state_init(&s,&m,2176);
    double start=now_s();
    if(probe){
        const char magic[8]={'Q','W','I','P','0','0','0','1'};uint32_t header[2]={(uint32_t)n,(uint32_t)m.V};
        ri_write(out,magic,8);ri_write(out,header,8);
        for(int i=0;i<n;i++){forward(&m,&s,ids[i],i);ri_finite(s.logits,m.V,"nonfinite probe head");
            uint32_t pos=(uint32_t)i;ri_write(out,&pos,4);ri_write(out,s.logits,(size_t)m.V*4);}
        double elapsed=now_s()-start;require(fflush(out)==0&&fclose(out)==0,"probe output close");ri_binding_receipt();
        printf("{\"mode\":\"probe\",\"input_tokens\":%d,\"head_positions\":%d,\"prefill_s\":%.9f,\"decode_s\":0,\"decode_steps\":[],\"output_bytes\":%llu,\"active_matrix_bytes_per_forward\":%llu,\"embedding_lookup_bytes_per_forward\":%d,\"kv_written_bytes_per_forward\":%d,\"peak_wset_bytes\":%llu,\"pid\":%lu",
            n,n,elapsed,(unsigned long long)(16u+(uint64_t)n*(4u+(uint64_t)m.V*4u)),(unsigned long long)active_matrix_bytes,m.D*4,m.L*2*128*4,(unsigned long long)peak_rss_bytes(),(unsigned long)GetCurrentProcessId());
        worker_json();printf("}\n");
    }else{
        for(int i=0;i<n;i++){forward(&m,&s,ids[i],i);ri_finite(s.logits,m.V,"nonfinite prefill head");}
        double prefill=now_s()-start,decode=0;int32_t emitted[128];int count=0,eos=0,charged=0;
        /* Each non-EOS emitted token gets a native forward, including the final one. */
        double step_seconds[128];int step_context[128],step_forward[128];
        for(int i=0;i<new_tokens;i++){
            double step=now_s();int id=ri_argmax(s.logits,m.V);emitted[count]=id;
            step_context[count]=n+i;step_forward[count]=id!=151645&&id!=151643;
            if(step_forward[count]){forward(&m,&s,id,n+i);ri_finite(s.logits,m.V,"nonfinite decode head");charged++;}
            else eos=1;
            step_seconds[count]=now_s()-step;decode+=step_seconds[count];count++;
            if(eos)break;
        }
        ri_write(out,emitted,(size_t)count*4);require(fflush(out)==0&&fclose(out)==0,"generation output close");
        ri_binding_receipt();
        printf("{\"mode\":\"generate\",\"input_tokens\":%d,\"prefill_s\":%.9f,\"decode_s\":%.9f,\"emitted_count\":%d,\"eos_count\":%d,\"native_decode_forwards\":%d,\"emitted_ids\":[",
            n,prefill,decode,count,eos,charged);
        for(int i=0;i<count;i++)printf("%s%d",i?",":"",emitted[i]);
        printf("],\"decode_steps\":[");for(int i=0;i<count;i++)printf("%s{\"index\":%d,\"generated_id\":%d,\"context\":%d,\"native_forward\":%s,\"seconds\":%.9f}",
            i?",":"",i,emitted[i],step_context[i],step_forward[i]?"true":"false",step_seconds[i]);
        printf("],\"output_bytes\":%d,\"active_matrix_bytes_per_forward\":%llu,\"embedding_lookup_bytes_per_forward\":%d,\"kv_written_bytes_per_forward\":%d,\"kv_logical_read_bytes_per_forward_at_context_t\":\"24*14*(t+1)*64*2*4\",\"peak_wset_bytes\":%llu,\"pid\":%lu",
            count*4,(unsigned long long)active_matrix_bytes,m.D*4,m.L*2*128*4,(unsigned long long)peak_rss_bytes(),(unsigned long)GetCurrentProcessId());
        worker_json();printf("}\n");
    }
    require(fflush(stdout)==0,"receipt flush");free(ids);return 0;
}
