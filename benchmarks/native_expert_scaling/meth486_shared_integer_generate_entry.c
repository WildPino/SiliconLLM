/* METH-356: all I8 matrix inputs A16, natural greedy multi-span task. */
#define main meth486_forced_entry
#include "meth486_shared_integer_cost_entry.c"
#undef main

int meth486_generate_entry(int argc,char **argv){
    if(argc==6||argc==10||argc==4||argc==2)return meth486_forced_entry(argc,argv);
    require(argc==12&&!strcmp(argv[1],"--generate"),"generate:manifest source_csv output_prefix threads cap closing_id profile warmups repetitions reserved0");
    int threads=atoi(argv[5]),cap=atoi(argv[6]),closing=atoi(argv[7]),profile=atoi(argv[8]),warmups=atoi(argv[9]),repetitions=atoi(argv[10]);
    require((threads==1||threads==3||threads==6)&&cap>0&&cap<=64&&(profile==0||profile==1)&&warmups>=0&&warmups<=2&&repetitions>=1&&repetitions<=3&&atoi(argv[11])==0,"generation_configuration");
    require(fegetround()==FE_TONEAREST,"generation_rounding");omp_set_dynamic(0);omp_set_num_threads(threads);worker_setup(threads);fault=0;
    LARGE_INTEGER frequency;require(QueryPerformanceFrequency(&frequency),"generation_clock_frequency");cost_frequency=(double)frequency.QuadPart;
    double load_start=cost_clock();Model *m=load(argv[2]);double load_seconds=cost_clock()-load_start;Config c=m->c;
    require(closing>=0&&closing<c.vocab&&closing!=0,"generation_closing_id");int source[CTX],chosen[64];int s=ids(argv[3],source,c.vocab);
    route_bound=c.enc*s+c.dec*cap;routes=alloc(route_bound,sizeof(Route));
    float *encsnap=alloc((size_t)(c.enc+2)*s*c.d,4),*decsnap=alloc((size_t)cap*(c.dec+2)*c.d,4),*logits=alloc((size_t)cap*c.vocab,4);
    for(int repetition=-warmups;repetition<repetitions;repetition++){
        g486_reset();memset(cost_counts,0,sizeof(cost_counts));route_count=0;cost_enabled=1;cost_profile=profile;cost_phase=0;cost_kind=0;
        int used=0,id=0;double steps[64],first_token_seconds=0.;
        double start=cost_clock();float *encoder=encode(m,source,s,encsnap);double encoded=cost_clock();Cache *cache=cache_init(m,encoder,s);double prepared=cost_clock();cost_phase=1;
        for(int position=0;position<cap;position++){
            double before=cost_clock();float *head=logits+(size_t)position*c.vocab;
            decode(m,cache,id,position,s,decsnap+(size_t)position*(c.dec+2)*c.d,head);int best=0;
            for(int token=0;token<c.vocab;token++){require(isfinite(head[token]),"generation_head_finite");if(head[token]>head[best])best=token;}
            chosen[position]=best;id=best;used=position+1;steps[position]=cost_clock()-before;
            if(position==0)first_token_seconds=cost_clock()-start;
            if(best==1||best==closing)break;
        }
        double finished=cost_clock();cost_enabled=0;
        char path[1200];require(snprintf(path,sizeof(path),"%s.%d.bin",argv[4],repetition)<(int)sizeof(path),"generation_output_name");cost_save(path,c,s,used,encsnap,decsnap,logits);
        printf("{\"repetition\":%d,\"threads\":%d,\"profile\":%d,\"source_tokens\":%d,\"actual_generated_tokens\":%d,\"closing_id\":%d,\"stopped_on_EOS\":%s,\"stopped_on_closing\":%s,\"stopped_at_cap\":%s,\"load_seconds\":%.9f,\"encoder_seconds\":%.9f,\"cross_kv_seconds\":%.9f,\"decode_greedy_seconds\":%.9f,\"full_generation_seconds\":%.9f,\"first_token_seconds\":%.9f,\"generated_ids\":[",repetition,threads,profile,s,used,closing,chosen[used-1]==1?"true":"false",chosen[used-1]==closing?"true":"false",chosen[used-1]!=1&&chosen[used-1]!=closing?"true":"false",load_seconds,encoded-start,prepared-encoded,finished-prepared,finished-start,first_token_seconds);
        for(int position=0;position<used;position++)printf("%s%d",position?",":"",chosen[position]);printf("],\"step_seconds\":[");
        for(int position=0;position<used;position++)printf("%s%.9f",position?",":"",steps[position]);printf("],\"counters\":");cost_json_counts();worker_json();g486_json();printf("}\n");fflush(stdout);
        cost_free_cache(cache,c.dec);free(encoder);
    }
    free(encsnap);free(decsnap);free(logits);free(routes);unload(m);return 0;
}
