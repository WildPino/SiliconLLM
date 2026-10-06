/* Complete forced-decoder CPU cost; not accepted-generation quality/rate. */
#include "meth486_shared_integer_model.c"

static void cost_save(const char *path,Config c,int s,int t,const float *enc,const float *dec,const float *logits){
    FILE *f=fopen(path,"wb");require(f!=NULL,"cost_output");fwrite("SWR32O01",1,8,f);
    uint32_t header[]={s,t,c.d,c.enc,c.dec,c.vocab,route_count};fwrite(header,4,7,f);
    fwrite(enc,4,(size_t)(c.enc+2)*s*c.d,f);fwrite(dec,4,(size_t)t*(c.dec+2)*c.d,f);fwrite(logits,4,(size_t)t*c.vocab,f);
    require(sizeof(Route)==12,"route_layout");fwrite(routes,sizeof(Route),route_count,f);require(fclose(f)==0,"cost_output_close");
}
static void cost_free_cache(Cache *cache,int layers){
    for(int l=0;l<layers;l++){free(cache[l].selfk);free(cache[l].selfv);free(cache[l].crossk);free(cache[l].crossv);}free(cache);
}
static void cost_json_counts(void){
    printf("[");for(int phase=0;phase<2;phase++){if(phase)printf(",");printf("[");
        for(int kind=0;kind<4;kind++){CostCounter a=cost_counts[phase][kind];
            printf("%s{\"calls\":%llu,\"code_bytes\":%llu,\"f32_bytes\":%llu,\"scale_bytes\":%llu,\"matrix_seconds\":%.9f}",kind?",":"",(unsigned long long)a.calls,(unsigned long long)a.codes,(unsigned long long)a.f32,(unsigned long long)a.scales,a.matrix_seconds);
        }printf("]");}printf("]");
}
int main(int argc,char **argv){
    if(argc==6||argc==4||argc==2)return meth486_contract_entry(argc,argv);
    require(argc==10,"cost:manifest source_csv decoder_csv output_prefix threads profile warmups repetitions reserved0");
    int threads=atoi(argv[5]),profile=atoi(argv[6]),warmups=atoi(argv[7]),repetitions=atoi(argv[8]);
    require((threads==1||threads==3||threads==6)&&(profile==0||profile==1)&&warmups>=0&&warmups<=2&&repetitions>=1&&repetitions<=3&&atoi(argv[9])==0,"cost_configuration");
    require(fegetround()==FE_TONEAREST,"cost_rounding");LARGE_INTEGER frequency;require(QueryPerformanceFrequency(&frequency),"performance_frequency");cost_frequency=(double)frequency.QuadPart;
    omp_set_dynamic(0);omp_set_num_threads(threads);worker_setup(threads);double initial=cost_clock();Model *m=load(argv[1]);double load_seconds=cost_clock()-initial;
    Config c=m->c;int encids[CTX],decids[CTX];int s=ids(argv[2],encids,c.vocab),t=ids(argv[3],decids,c.vocab);fault=0;
    route_bound=c.enc*s+c.dec*t;routes=alloc(route_bound,sizeof(Route));
    float *encsnap=alloc((size_t)(c.enc+2)*s*c.d,4),*decsnap=alloc((size_t)t*(c.dec+2)*c.d,4),*logits=alloc((size_t)t*c.vocab,4);
    for(int repetition=-warmups;repetition<repetitions;repetition++){
        g486_reset();memset(cost_counts,0,sizeof(cost_counts));route_count=0;cost_enabled=1;cost_profile=profile;cost_phase=0;cost_kind=0;
        double start=cost_clock();float *encoder=encode(m,encids,s,encsnap);double encoded=cost_clock();Cache *cache=cache_init(m,encoder,s);double prepared=cost_clock();
        cost_phase=1;double steps[CTX];for(int step=0;step<t;step++){double before=cost_clock();decode(m,cache,decids[step],step,s,decsnap+(size_t)step*(c.dec+2)*c.d,logits+(size_t)step*c.vocab);steps[step]=cost_clock()-before;}
        double finished=cost_clock();cost_enabled=0;
        for(size_t i=0;i<(size_t)t*c.vocab;i++)require(isfinite(logits[i]),"cost_logits_finite");
        char path[1200];require(snprintf(path,sizeof(path),"%s.%d.bin",argv[4],repetition)<(int)sizeof(path),"cost_output_name");cost_save(path,c,s,t,encsnap,decsnap,logits);
        int sparse_enc=0,sparse_dec=0;for(int l=0;l<c.enc;l++)sparse_enc+=m->enc[l].sparse;for(int l=0;l<c.dec;l++)sparse_dec+=m->dec[l].sparse;
        require(route_count==sparse_enc*s+sparse_dec*t,"cost_route_count");
        printf("{\"repetition\":%d,\"threads\":%d,\"profile\":%d,\"source_tokens\":%d,\"decoder_positions\":%d,\"experts_per_bank\":%d,\"load_seconds\":%.9f,\"encode_seconds\":%.9f,\"cross_kv_seconds\":%.9f,\"decode_seconds\":%.9f,\"full_seconds\":%.9f,\"routes\":%d,\"embedding_lookup_bytes\":%llu,\"counters\":",repetition,threads,profile,s,t,c.n,load_seconds,encoded-start,prepared-encoded,finished-prepared,finished-start,route_count,(unsigned long long)(s+t)*c.d*4);
        cost_json_counts();printf(",\"decode_step_seconds\":[");for(int step=0;step<t;step++)printf("%s%.9f",step?",":"",steps[step]);printf("]");worker_json();g486_json();printf("}\n");fflush(stdout);
        cost_free_cache(cache,c.dec);free(encoder);
    }
    free(encsnap);free(decsnap);free(logits);free(routes);unload(m);return 0;
}
