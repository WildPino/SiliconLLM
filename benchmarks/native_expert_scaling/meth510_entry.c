/* METH-356: all I8 matrix inputs A16, natural greedy multi-span task. */
#define main meth506_forced_entry
#include "meth510_cost_entry.c"
#undef main
#include "meth510_control.h"

static int controls506(void){
    require(fegetround()==FE_TONEAREST&&SetProcessAffinityMask(GetCurrentProcess(),21),"506_control_rounding_pool");worker_setup(3);
    ExactColumnWork *w=alloc(1,sizeof(ExactColumnWork));int8_t *columns=alloc((size_t)3072*768,1);float scales[768],output[768];
    for(int r=0;r<768;r++)scales[r]=r%3==0?.5f:r%3==1?2.f:.25f;
    for(int j=0;j<3072;j++)for(int r=0;r<768;r++)columns[(size_t)j*768+r]=r%3==0?-128:r%3==1?127:(j%2?-128:127);
    for(int mode=0;mode<6;mode++){
        for(int j=0;j<3072;j++)w->codes[j]=mode==0?32767:mode==1?-32767:mode==2?(j%2?-32767:32767):mode==3?0:mode==4?(j==3071?-32767:0):(j%3==0?32767:0);
        uint32_t nz=exact_i8_columns(columns,scales,.25f,output,w);require(nz==(mode==3?0:mode==4?1:mode==5?1024:3072),"506_control_count");
        for(uint32_t j=1;j<nz;j++)require(w->indices[j]>w->indices[j-1],"506_control_index_order");
        for(int r=0;r<768;r++){int64_t sum=0;for(int j=0;j<3072;j++)sum+=(int64_t)columns[(size_t)j*768+r]*w->codes[j];float gold=(float)(((double)sum*(double)scales[r])*.25);require(sum==w->total[r]&&!memcmp(&gold,output+r,4),"506_control_integer_F_BYTE");}
    }
    int8_t weights[4096];int16_t code[4096];for(int mode=0;mode<3;mode++){
        int64_t sum=0;for(int j=0;j<4096;j++){weights[j]=mode==0?-128:mode==1?127:(j%2?-128:127);code[j]=mode==0?32767:mode==1?-32767:(j%2?32767:-32767);sum+=(int64_t)weights[j]*code[j];}
        require(head_integer_dot(weights,code,4096)==sum,"506_control_original_I64_bound");
    }
    float input[768]={0};require(head_activation_codes(input,code,768)==1.f,"506_zero_scale");for(int j=0;j<768;j++)require(code[j]==0,"506_zero_codes");
    float ties[]={-2.5f,-1.5f,-.5f,.5f,1.5f,2.5f,32767.f};int16_t expected[]={-2,-2,0,0,2,2,32767};memcpy(input,ties,sizeof(ties));
    require(head_activation_codes(input,code,768)==1.f&&!memcmp(code,expected,sizeof(expected)),"506_ties_to_even");
    printf("{\"controls\":true,\"sparse_modes\":6,\"original_I64_extremes\":3,\"zero_and_ties\":true,\"workspace_bytes\":%llu",(unsigned long long)sizeof(ExactColumnWork));worker_json();printf("}\n");free(columns);free(w);return 0;
}
int main(int argc,char **argv){
    if(argc==3&&!strcmp(argv[1],"--head-contract"))return contract510(argv[2]);
    if(argc==5&&!strcmp(argv[1],"--retained-head"))return retained_head510(argv[2],argv[3],argv[4]);
    if(argc==2&&!strcmp(argv[1],"--column-controls"))return controls506();
    if(argc==10)return meth506_forced_entry(argc,argv);
    require(argc==12&&!strcmp(argv[1],"--generate"),"generate:manifest source_csv output_prefix threads cap closing_id profile warmups repetitions reserved0");
    int threads=atoi(argv[5]),cap=atoi(argv[6]),closing=atoi(argv[7]),profile=atoi(argv[8]),warmups=atoi(argv[9]),repetitions=atoi(argv[10]);
    require(threads==3&&cap>0&&cap<=64&&(profile==0||profile==1)&&warmups>=0&&warmups<=2&&repetitions>=1&&repetitions<=3&&atoi(argv[11])==0,"generation_configuration");
    require(fegetround()==FE_TONEAREST,"generation_rounding");omp_set_dynamic(0);omp_set_num_threads(threads);require(SetProcessAffinityMask(GetCurrentProcess(),21),"506_declared_native_pool");worker_setup(threads);fault=0;
    LARGE_INTEGER frequency;require(QueryPerformanceFrequency(&frequency),"generation_clock_frequency");cost_frequency=(double)frequency.QuadPart;
    double load_start=cost_clock();Model *m=load(argv[2]);double load_seconds=cost_clock()-load_start;Config c=m->c;
    require(closing>=0&&closing<c.vocab&&closing!=0,"generation_closing_id");int source[CTX],chosen[64];int s=ids(argv[3],source,c.vocab);
    route_bound=c.enc*s+c.dec*cap;routes=alloc(route_bound,sizeof(Route));
    float *encsnap=alloc((size_t)(c.enc+2)*s*c.d,4),*decsnap=alloc((size_t)cap*(c.dec+2)*c.d,4),*logits=alloc((size_t)cap*c.vocab,4);
    for(int repetition=-warmups;repetition<repetitions;repetition++){
        memset(&hybrid_counts510,0,sizeof(hybrid_counts510));memset(cost_counts,0,sizeof(cost_counts));memset(column_counts506,0,sizeof(column_counts506));route_count=0;cost_enabled=1;cost_profile=profile;cost_phase=0;cost_kind=0;
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
        for(int position=0;position<used;position++)printf("%s%.9f",position?",":"",steps[position]);printf("],\"counters\":");cost_json_counts();columns_json506();worker_json();printf("}\n");fflush(stdout);
        cost_free_cache(cache,c.dec);free(encoder);
    }
    free(encsnap);free(decsnap);free(logits);free(routes);unload(m);return 0;
}
