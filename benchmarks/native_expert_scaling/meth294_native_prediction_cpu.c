// Full-head native quality entry; original285 operators with absolute-window RoPE.
#include "meth285_model_operator.h"
#include "meth294_window_forward.h"
static double m294_loss(const float *logits,int target,int *top){
    *top=cc_argmax(logits);double total=0,maximum=logits[*top];
    for(int v=0;v<CC_V;v++){if(!isfinite(logits[v]))die("nonfinite full-head row");total+=exp((double)logits[v]-maximum);}
    double loss=target<0?0:log(total)+maximum-(double)logits[target];if(!isfinite(loss))die("nonfinite loss");return loss;
}
int main(int argc,char **argv){
    if(argc!=4)die("usage: native_quality ARCHIVE BUNDLE OUTPUT");
    if(fesetround(FE_TONEAREST) || !__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))die("rounding/ISA");
    omp_set_dynamic(0);omp_set_num_threads(6);integer_edge_guards();cc_load(argv[1]);
    size_t bytes;uint8_t *raw=read_file(argv[2],&bytes);if(bytes<12 || memcmp(raw,"M294IN01",8))die("bundle magic");
    uint32_t count=u32le(raw+8);if(count<1 || count>1024)die("case count");size_t at=12;
    const int32_t **ids=calloc(count,sizeof *ids),**targets=calloc(count,sizeof *targets);uint32_t *meta=calloc((size_t)count*3,4);
    if(!ids || !targets || !meta)die("case allocation");
    for(uint32_t c=0;c<count;c++){
        if(at>bytes || bytes-at<12)die("case header");uint32_t n=u32le(raw+at),offset=u32le(raw+at+4),first=u32le(raw+at+8);at+=12;
        if(n<1 || n>4096 || first>=n || offset>65536 || (size_t)(2*n-first)*4>bytes-at)die("case range");
        meta[c*3]=n;meta[c*3+1]=offset;meta[c*3+2]=first;ids[c]=(const int32_t*)(raw+at);at+=(size_t)n*4;
        targets[c]=(const int32_t*)(raw+at);at+=(size_t)(n-first)*4;
        for(uint32_t j=0;j<n;j++)if(ids[c][j]<0 || ids[c][j]>=CC_V)die("input ID");
        for(uint32_t j=0;j<n-first;j++)if(targets[c][j]<-1 || targets[c][j]>=CC_V)die("target ID");
    }
    if(at!=bytes)die("trailing bundle data");FILE *out=fopen(argv[3],"wb");if(!out)die("output open");
    cc_write(out,"M294OUT1",8);uint32_t dims[2]={2,count};cc_write(out,dims,8);
    CcState *s=malloc(sizeof *s);if(!s)die("state allocation");cc_state_init(s,4096);
    uint64_t scored=0,prefill=0;
    for(int arm=0;arm<2;arm++)for(uint32_t c=0;c<count;c++){
        int n=(int)meta[c*3],offset=(int)meta[c*3+1],first=(int)meta[c*3+2];uint32_t header[5]={(uint32_t)arm,c,(uint32_t)n,(uint32_t)offset,(uint32_t)first};cc_write(out,header,20);
        for(int j=0;j<n;j++){
            cc_forward_window(s,ids[c][j],j,arm,j>=first,offset);
            if(j<first){prefill++;continue;}
            int32_t top,target=targets[c][j-first];double loss=m294_loss(s->logits,target,&top);
            if(j==first)cc_write(out,s->logits,sizeof s->logits);
            int32_t absolute_position=j+offset;cc_write(out,&absolute_position,4);cc_write(out,&top,4);cc_write(out,&target,4);cc_write(out,&loss,8);scored++;
        }
    }
    if(fclose(out))die("output close");printf("{\"arms\":2,\"cases\":%u,\"scored_rows\":%llu,\"prefill_rows\":%llu,\"timing_qualification\":false}\n",count,(unsigned long long)scored,(unsigned long long)prefill);return 0;
}
