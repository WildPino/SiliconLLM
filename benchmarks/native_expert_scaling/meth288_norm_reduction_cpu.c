// Whole-prefix/cache qualification; only norm square accumulation changes; all other285 equations preserved.
#include "meth288_model_operator.h"
static void cc_free_state(CcState *s){free(s->kc);free(s->vc);free(s->scores);free(s);}
int main(int argc,char **argv){
    if(argc!=4)die("usage: complete_prefix ARCHIVE PREFIX_BUNDLE OUT");
    if(fesetround(FE_TONEAREST) || fegetround()!=FE_TONEAREST)die("rounding mode");
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))die("ISA binding");
    omp_set_dynamic(0);omp_set_num_threads(6);integer_edge_guards();cc_load(argv[1]);
    size_t bytes;uint8_t *bundle=read_file(argv[2],&bytes);
    if(bytes<12 || memcmp(bundle,"M285ID01",8) || u32le(bundle+8)!=6)die("prefix bundle binding");
    size_t at=12;const int32_t *ids[6];uint32_t lengths[6];
    for(int i=0;i<6;i++){
        if(at>bytes || bytes-at<4)die("short prefix length");lengths[i]=u32le(bundle+at);at+=4;
        if(lengths[i]<8 || lengths[i]>4096 || (size_t)lengths[i]*4>bytes-at)die("prefix range");
        ids[i]=(const int32_t*)(bundle+at);at+=(size_t)lengths[i]*4;
        for(uint32_t j=0;j<lengths[i];j++)if(ids[i][j]<0 || ids[i][j]>=CC_V)die("prefix token ID");
    }
    if(at!=bytes)die("prefix trailing data");
    FILE *out=fopen(argv[3],"wb");if(!out)die("prefix output open");
    uint32_t header[5]={2,6,8,D,CC_V};cc_write(out,"M285LG01",8);cc_write(out,header,sizeof header);
    CcState *s=malloc(sizeof *s),*fresh=malloc(sizeof *fresh);
    if(!s || !fresh)die("prefix state allocation");cc_state_init(s,4096);cc_state_init(fresh,4096);
    size_t row=(size_t)D+CC_V;float *tail=malloc(8*row*4);if(!tail)die("tail allocation");
    int cache_failures=0,negative_failures=0;
    for(int arm=0;arm<2;arm++)for(int i=0;i<6;i++){
        int n=(int)lengths[i];
        for(int j=0;j<n;j++){
            cc_forward(s,ids[i][j],j,arm,j>=n-8);
            if(j>=n-8){float *record=tail+(size_t)(j-(n-8))*row;memcpy(record,s->norm,D*4);memcpy(record+D,s->logits,CC_V*4);}
        }
        // Independent cache allocation is rebuilt from this complete prefix.
        for(int j=0;j<n;j++)cc_forward(fresh,ids[i][j],j,arm,j==n-1);
        const float *last=tail+7*row;
        int exact=!memcmp(last,fresh->norm,D*4) && !memcmp(last+D,fresh->logits,CC_V*4);
        if(!exact)cache_failures++;
        // Deliberately erase history,then rerun only the last real token.
        memset(fresh->kc,0,(size_t)L*fresh->maxseq*128*4);memset(fresh->vc,0,(size_t)L*fresh->maxseq*128*4);
        cc_forward(fresh,ids[i][n-1],n-1,arm,1);
        int detected=memcmp(last,fresh->norm,D*4)!=0 || memcmp(last+D,fresh->logits,CC_V*4)!=0;
        if(!detected)negative_failures++;
        uint32_t meta[5]={(uint32_t)arm,(uint32_t)i,(uint32_t)n,(uint32_t)exact,(uint32_t)detected};
        cc_write(out,meta,sizeof meta);cc_write(out,tail,8*row*4);
        cc_write(out,fresh->norm,D*4);cc_write(out,fresh->logits,CC_V*4);
    }
    if(fclose(out))die("prefix output close");free(tail);free(bundle);cc_free_state(s);cc_free_state(fresh);
    printf("{\"arms\":2,\"sources\":6,\"positions_per_arm\":48,\"cache_failures\":%d,\"erased_history_not_detected\":%d,\"timing_qualification\":false}\n",cache_failures,negative_failures);
    return 0;
}
