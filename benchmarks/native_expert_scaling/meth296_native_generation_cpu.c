// Prepared native generation assay: original285 profile, full head, real KV.
#include "meth285_model_operator.h"
static void m296_finite(CcState *s){
    for(int v=0;v<CC_V;v++)if(!isfinite(s->logits[v]))die("nonfinite generated full head");
    for(int d=0;d<D;d++)if(!isfinite(s->norm[d]) || s->norm[d]!=cc_bf(s->norm[d]))die("nonfinite/non-BF16 state");
}
static void m296_state(FILE *out,int32_t position,int32_t choice,CcState *s){
    uint16_t bits[D];for(int d=0;d<D;d++)bits[d]=cc_bf_bits(s->norm[d]);
    cc_write(out,&position,4);cc_write(out,&choice,4);cc_write(out,bits,sizeof bits);
}
int main(int argc,char **argv){
    if(argc!=4)die("usage: native_generation ARCHIVE PROMPTS OUTPUT");
    if(fesetround(FE_TONEAREST) || !__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))die("rounding/ISA");
    omp_set_dynamic(0);omp_set_num_threads(6);integer_edge_guards();cc_load(argv[1]);
    size_t bytes;uint8_t *raw=read_file(argv[2],&bytes);if(bytes<12 || memcmp(raw,"M296IDS1",8))die("prompt magic");
    uint32_t count=u32le(raw+8);if(count!=24)die("prompt count");size_t at=12;
    const int32_t *ids[24];uint32_t lengths[24];
    for(int c=0;c<24;c++){
        if(bytes-at<4)die("prompt header");uint32_t n=u32le(raw+at);at+=4;
        if(n<1 || n+128>4096 || (size_t)n*4>bytes-at)die("prompt length");
        ids[c]=(const int32_t*)(raw+at);lengths[c]=n;at+=(size_t)n*4;
        for(uint32_t j=0;j<n;j++)if(ids[c][j]<0 || ids[c][j]>=CC_V)die("prompt ID");
    }
    if(at!=bytes)die("trailing prompts");FILE *out=fopen(argv[3],"wb");if(!out)die("generation output");
    cc_write(out,"M296GEN1",8);uint32_t dims[3]={24,D,CC_V};cc_write(out,dims,12);
    CcState *s=malloc(sizeof *s),*fresh=malloc(sizeof *fresh);if(!s || !fresh)die("state allocation");
    cc_state_init(s,4096);cc_state_init(fresh,4096);int sequence[4096];uint32_t fresh_checks=0,negative_checks=0;
    for(int c=0;c<24;c++){
        int n=(int)lengths[c];memcpy(sequence,ids[c],(size_t)n*4);uint32_t header[2]={(uint32_t)c,(uint32_t)n};cc_write(out,header,8);
        for(int j=0;j<n;j++){
            cc_forward(s,ids[c][j],j,1,1);m296_finite(s);
            if(j==0)cc_write(out,s->logits,sizeof s->logits);
            m296_state(out,j,cc_argmax(s->logits),s);
        }
        uint16_t states[128][D];int32_t choices[128];float chosen_logits[128];uint32_t generated=0;
        for(int step=0;step<128;step++){
            int position=n+step-1;m296_finite(s);
            if((c==0 || c==8 || c==16) && step<3){
                for(int j=0;j<=position;j++)cc_forward(fresh,sequence[j],j,1,j==position);
                if(memcmp(s->norm,fresh->norm,sizeof s->norm) || memcmp(s->logits,fresh->logits,sizeof s->logits))die("native generated cached/fresh byte closure");
                fresh_checks++;
                if(step==0){
                    memset(fresh->kc,0,(size_t)L*4096*128*4);memset(fresh->vc,0,(size_t)L*4096*128*4);
                    cc_forward(fresh,sequence[position],position,1,1);
                    if(!memcmp(s->norm,fresh->norm,sizeof s->norm) || !memcmp(s->logits,fresh->logits,sizeof s->logits))die("erased-history negative control");
                    negative_checks++;
                }
            }
            int token=cc_argmax(s->logits);choices[step]=token;chosen_logits[step]=s->logits[token];
            for(int d=0;d<D;d++)states[step][d]=cc_bf_bits(s->norm[d]);generated++;
            if(token==151645 || step==127)break;
            sequence[n+step]=token;cc_forward(s,token,n+step,1,1);
        }
        uint32_t gen_header[2]={generated,choices[generated-1]==151645};cc_write(out,gen_header,8);
        for(uint32_t step=0;step<generated;step++){
            cc_write(out,&step,4);cc_write(out,choices+step,4);cc_write(out,chosen_logits+step,4);cc_write(out,states[step],sizeof states[step]);
        }
        if(fflush(out))die("generation flush");printf("{\"source_complete\":%d,\"total\":24,\"generated\":%u}\n",c+1,generated);fflush(stdout);
    }
    if(fclose(out))die("generation close");
    printf("{\"sources\":24,\"fresh_byte_checks\":%u,\"erased_history_controls\":%u,\"timing_qualification\":false}\n",fresh_checks,negative_checks);return 0;
}
