/* Original runtime-n packed engine, persistent batch1 canonical chat consumer.
   Build includes a fresh original wide packed prefix wrapper, with exactly the
   same20 computational bodies. No donor inference or extra deployed weights. */
#define main original_prefix_main
#include "original_wide_prefix.c"
#undef main
#define CHAT_MAGIC 0x31434853u
#define CHAT_LIMIT 16384u
static uint32_t chat_ids[CHAT_LIMIT],chat_count;
static void read_exact(FILE *f,void *p,size_t n){if(fread(p,1,n,f)!=n)fail("short stream");}
static void write_exact(FILE *f,const void *p,size_t n){put(f,p,n);}
static void capture_routes(uint8_t *p,uint32_t position){
    if(!p)return;p+=(size_t)position*L*KTOP*8;
    for(int l=0;l<L;l++){memcpy(p,actual_ids[l],KTOP*4);p+=KTOP*4;
        memcpy(p,actual_mass[l],KTOP*4);p+=KTOP*4;}
}
static void check_routes(void){
    for(int l=0;l<L;l++){double sum=0;
        for(int k=0;k<KTOP;k++){
            if(actual_ids[l][k]<0||actual_ids[l][k]>=E||!isfinite(actual_mass[l][k])||actual_mass[l][k]<0)fail("route support");
            for(int j=0;j<k;j++)if(actual_ids[l][k]==actual_ids[l][j])fail("duplicate route");
            sum+=actual_mass[l][k];
        }if(fabs(sum-1)>1e-6)fail("route mass");
    }
}
static uint32_t consume_head(const float *logits){
    uint32_t best=0;for(int v=0;v<V;v++){
        if(!isfinite(logits[v]))fail("nonfinite head");
        if(logits[v]>logits[best])best=(uint32_t)v;
    }return best;
}
int main(int argc,char **argv){
    if(!SetProcessAffinityMask(GetCurrentProcess(),(DWORD_PTR)1))fail("affinity");
    if(argc>1&&!strcmp(argv[1],"--prefix"))return original_prefix_main(argc-1,argv+1);
    if(argc!=3||strcmp(argv[1],"--stream"))fail("usage: --stream model | --prefix model queries logits routes witnesses report");
    if(_setmode(_fileno(stdin),_O_BINARY)<0||_setmode(_fileno(stdout),_O_BINARY)<0)fail("binary pipes");
    if(fesetround(FE_TONEAREST))fail("round mode");
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    const char *limit_env=getenv("SILICON_CHAT_SECONDS");double deadline=limit_env?atof(limit_env):0;
    const char *score_prefix=getenv("SILICON_CHAT_SCORE_PREFIX");uint32_t request_index=0;
    const char *route_prefix=getenv("SILICON_CHAT_ROUTE_PREFIX");
    double start=now_s();load_model(argv[2]);double load_s=now_s()-start;
    uint32_t hello[6]={CHAT_MAGIC,1,(uint32_t)V,CHAT_LIMIT,11,228};
    uint64_t bytes[3]={blob_bytes,code_bytes,state_bytes};
    write_exact(stdout,hello,sizeof(hello));write_exact(stdout,bytes,sizeof(bytes));write_exact(stdout,&load_s,8);fflush(stdout);
    float *logits=malloc((size_t)V*4);if(!logits)fail("head allocation");
    for(;;){
        uint32_t req[4];size_t got=fread(req,4,4,stdin);
        if(!got&&feof(stdin))break;if(got!=4||req[0]!=CHAT_MAGIC||req[1]>2)fail("request header");
        if(req[1]==0)break;
        uint32_t n=req[2],max_new=req[3],input[8192],out[512];
        if(n<1||n>8192||max_new>512||n+max_new>CHAT_LIMIT)fail("request dimensions");
        read_exact(stdin,input,(size_t)n*4);
        float *scores=score_prefix&&max_new?malloc((size_t)max_new*V*4):NULL;
        if(score_prefix&&max_new&&!scores)fail("score capture allocation");
        uint8_t *routes=route_prefix?malloc((size_t)(n+max_new)*L*KTOP*8):NULL;
        if(route_prefix&&!routes)fail("route capture allocation");
        for(uint32_t i=0;i<n;i++)if(input[i]>=(uint32_t)V)fail("request ID");
        double t0=now_s();uint32_t reused=chat_count,was_reset=0;
        if(req[1]==2||n<chat_count||memcmp(input,chat_ids,(size_t)chat_count*4)){
            state_reset();chat_count=0;reused=0;was_reset=1;
        }
        uint32_t core_calls=0,head_calls=0;double forward_s=0,pre0=now_s();
        for(uint32_t i=chat_count;i<n;i++){
            double call=now_s();
            forward_token(input[i],logits,1,0,1,NULL);check_routes();consume_head(logits);
            capture_routes(routes,core_calls);
            forward_s+=now_s()-call;core_calls++;head_calls++;chat_ids[chat_count++]=input[i];
            if(deadline>0&&now_s()-start>deadline)fail("stream deadline");
        }
        double pre_s=now_s()-pre0,dec0=now_s();uint32_t generated=0,eos=0;
        while(generated<max_new){
            if(scores)memcpy(scores+(size_t)generated*V,logits,(size_t)V*4);
            uint32_t id=consume_head(logits);out[generated++]=id;
            /* Consume final/EOS too; full vocabulary head work is charged. */
            double call=now_s();forward_token(id,logits,1,0,1,NULL);check_routes();consume_head(logits);
            capture_routes(routes,core_calls);
            forward_s+=now_s()-call;core_calls++;head_calls++;chat_ids[chat_count++]=id;
            if(deadline>0&&now_s()-start>deadline)fail("stream deadline");
            if(id==11||id==228){eos=id;break;}
        }
        double dec_s=now_s()-dec0;
        uint32_t reply[10]={CHAT_MAGIC,1,generated,n,reused,was_reset,eos,core_calls,head_calls,chat_count};
        /* Component timings unavailable: no profiling in this consumer. */
        double timing[8]={now_s()-t0,pre_s,dec_s,0,0,0,forward_s,load_s};
        if(scores){char path[4096];snprintf(path,sizeof(path),"%s.%04u.scores.f32",score_prefix,request_index);
            FILE *f=output(path);put(f,scores,(size_t)generated*V*4);fclose(f);free(scores);}
        if(routes){char path[4096];snprintf(path,sizeof(path),"%s.%04u.routes",route_prefix,request_index);
            FILE *f=output(path);put(f,routes,(size_t)core_calls*L*KTOP*8);fclose(f);free(routes);}
        request_index++;
        write_exact(stdout,reply,sizeof(reply));write_exact(stdout,timing,sizeof(timing));
        write_exact(stdout,out,(size_t)generated*4);write_exact(stdout,logits,(size_t)V*4);fflush(stdout);
    }
    free(logits);return 0;
}
