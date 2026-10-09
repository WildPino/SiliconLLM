/* Packed compact Falcon target, compiled THROUGH phase60/engine.c.
   Reuses its original LUT bodies and the existing qualified-scope operator
   implementation. Deployment apparatus; the old quality/parity failures remain. */
#define main silicon_hybrid_prefix_main
#include "chatbot_hybrid_native.c"
#undef main

#define CHAT_MAGIC 0x31434853u
#define CHAT_LIMIT 16384u
static uint32_t chat_ids[CHAT_LIMIT],chat_count;

static uint32_t winner(const float *logits){
    uint32_t id=0;for(uint32_t j=1;j<V;j++)if(logits[j]>logits[id])id=j;return id;
}

int main(int argc,char **argv){
    if(argc>1&&!strcmp(argv[1],"--prefix"))return silicon_hybrid_prefix_main(argc-1,argv+1);
    if(argc!=3||strcmp(argv[1],"--stream")){
        fputs("engine --stream packed_model.bin | --prefix model queries logits trace states report\n",stderr);return 1;
    }
    if(_setmode(_fileno(stdin),_O_BINARY)<0||_setmode(_fileno(stdout),_O_BINARY)<0)fail("binary pipes");
    if(fesetround(FE_TONEAREST))fail("round mode");
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    const char *limit_env=getenv("SILICON_CHAT_SECONDS");double deadline=limit_env?atof(limit_env):0;
    double start=now_s();fixtures();load_model(argv[2]);double load_s=now_s()-start;
    uint32_t hello[6]={CHAT_MAGIC,1,V,CHAT_LIMIT,11,228};uint64_t bytes[3]={blob_bytes,code_bytes,state_bytes};
    write_exact(stdout,hello,sizeof(hello));write_exact(stdout,bytes,sizeof(bytes));write_exact(stdout,&load_s,8);fflush(stdout);
    float *logits=malloc(V*4);if(!logits)fail("logits OOM");
    for(;;){
        uint32_t req[4];size_t got=fread(req,4,4,stdin);if(!got&&feof(stdin))break;if(got!=4)fail("request header");
        if(req[0]!=CHAT_MAGIC||req[1]>2)fail("request contract");if(req[1]==0)break;
        uint32_t n=req[2],max_new=req[3],input[8192],output[512];
        if(n<1||n>8192||max_new>512||n+max_new>CHAT_LIMIT)fail("request dimensions");
        read_exact(stdin,input,n*4);for(uint32_t i=0;i<n;i++)if(input[i]>=V)fail("request ID");
        double t0=now_s();uint32_t reused=chat_count,was_reset=0;
        if(req[1]==2||n<chat_count||memcmp(input,chat_ids,chat_count*4)){
            reset();chat_count=0;reused=0;was_reset=1;
        }
        Timings tm={0};uint32_t core_calls=0,head_calls=0;
        double pre0=now_s();
        for(uint32_t i=chat_count;i<n;i++){
            forward(input[i],(int)i,logits,i+1==n,NULL,&tm);core_calls++;head_calls+=i+1==n;
            chat_ids[chat_count++]=input[i];if(deadline>0&&now_s()-start>deadline)fail("stream deadline");
        }
        double pre_s=now_s()-pre0,dec0=now_s();uint32_t generated=0,eos=0;
        while(generated<max_new){
            uint32_t id=winner(logits);output[generated++]=id;
            /* Consume every emitted ID, including the last/EOS. Its full head
               is charged; cached logits permit a subsequent identical prefix. */
            forward(id,(int)chat_count,logits,1,NULL,&tm);core_calls++;head_calls++;
            chat_ids[chat_count++]=id;
            if(deadline>0&&now_s()-start>deadline)fail("stream deadline");
            if(id==11||id==228){eos=id;break;}
        }
        double dec_s=now_s()-dec0;
        uint32_t reply[10]={CHAT_MAGIC,1,generated,n,reused,was_reset,eos,core_calls,head_calls,chat_count};
        double timing[8]={now_s()-t0,pre_s,dec_s,tm.core,tm.mlp,tm.head,tm.total,load_s};
        write_exact(stdout,reply,sizeof(reply));write_exact(stdout,timing,sizeof(timing));
        write_exact(stdout,output,generated*4);write_exact(stdout,logits,V*4);fflush(stdout);
    }
    free(logits);return 0;
}
