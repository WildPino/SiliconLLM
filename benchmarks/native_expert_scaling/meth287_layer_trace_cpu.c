// Original propagation plus identical-input remaining-operator controls.
#include "meth285_model_operator.h"
#include "meth287_trace_forward.h"
int main(int argc,char **argv){
    if(argc!=6)die("usage: trace ARCHIVE IDS FIXTURE TRACE OUT");
    if(fesetround(FE_TONEAREST))die("rounding");omp_set_dynamic(0);omp_set_num_threads(6);cc_load(argv[1]);
    size_t bytes;uint8_t *idsraw=read_file(argv[2],&bytes);if(bytes!=147*4)die("IDs binding");int32_t *ids=(int32_t*)idsraw;
    CcState *s=malloc(sizeof *s);if(!s)die("state allocation");cc_state_init(s,147);
    cc_trace=calloc((size_t)147*24*5*D,4);if(!cc_trace)die("trace allocation");
    for(int pos=0;pos<147;pos++)cc_forward_trace(s,ids[pos],pos,0,pos==146);
    FILE *file=fopen(argv[4],"wb");if(!file)die("trace open");cc_write(file,"M287TR01",8);cc_write(file,cc_trace,(size_t)147*24*5*D*4);
    cc_write(file,s->norm,D*4);cc_write(file,s->logits,CC_V*4);if(fclose(file))die("trace close");
    uint8_t *raw=read_file(argv[3],&bytes);if(bytes!=8+(size_t)24*147*6*D*4 || memcmp(raw,"M287IN01",8))die("fixture binding");
    file=fopen(argv[5],"wb");if(!file)die("isolated open");cc_write(file,"M287OP01",8);
    for(int li=0;li<24;li++){
        const float *cursor=(const float*)(raw+8)+(size_t)li*147*6*D;
        const float *x=cursor;cursor+=147*D;const float *att=cursor;cursor+=147*D;const float *o=cursor;cursor+=147*D;
        const float *postatt=cursor;cursor+=147*D;const float *postnorm=cursor;cursor+=147*D;const float *ffnbf=cursor;
        float *out=calloc((size_t)147*5*D,4);if(!out)die("isolated allocation");
        float *oo=out,*on=out+147*D,*of=out+2*147*D,*ra=out+3*147*D,*rf=out+4*147*D;
        for(int pos=0;pos<147;pos++){
            cc_mat(cc_layers[li].o,att+(size_t)pos*D,NULL,oo+(size_t)pos*D,D,D);
            cc_norm(postatt+(size_t)pos*D,cc_layers[li].post_norm,on+(size_t)pos*D);
            uint16_t bits[D];for(int d=0;d<D;d++)bits[d]=cc_bf_bits(postnorm[(size_t)pos*D+d]);row_forward(li,bits,of+(size_t)pos*D);
            for(int d=0;d<D;d++){size_t at=(size_t)pos*D+d;ra[at]=cc_bf(x[at]+o[at]);rf[at]=cc_bf(postatt[at]+ffnbf[at]);}
        }
        cc_write(file,out,(size_t)147*5*D*4);free(out);
    }
    if(fclose(file))die("isolated close");puts("{\"trace_positions\":147,\"layers\":24,\"diagnostic_only\":true}");return 0;
}
