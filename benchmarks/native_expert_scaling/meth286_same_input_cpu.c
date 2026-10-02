// Diagnostic oracle inputs only; frozen deployed285 operators are unchanged.
#include "meth285_model_operator.h"
static void attention_probe(const float *q,const float *k,const float *v,float *out,int n,int rounded){
    #pragma omp parallel for schedule(static)
    for(int h=0;h<14;h++){
        float *scores=malloc((size_t)n*4);if(!scores)die("scores allocation");
        for(int pos=0;pos<n;pos++){
            float maximum=-INFINITY;
            for(int t=0;t<=pos;t++){
                float value=cc_dot(k+(size_t)t*128+(h/7)*64,q+(size_t)pos*D+h*64,64)*.125f;
                scores[t]=value;if(value>maximum)maximum=value;
            }
            float total=0;for(int t=0;t<=pos;t++){scores[t]=expf(scores[t]-maximum);total+=scores[t];}
            for(int d=0;d<64;d++){
                float value=0;for(int t=0;t<=pos;t++){
                    float probability=scores[t]/total;if(rounded)probability=cc_bf(probability);
                    value+=probability*v[(size_t)t*128+(h/7)*64+d];
                }
                out[(size_t)pos*D+h*64+d]=cc_bf(value);
            }
        }
        free(scores);
    }
}
int main(int argc,char **argv){
    if(argc!=4)die("usage: same_input ARCHIVE FIXTURE OUTPUT");
    if(fesetround(FE_TONEAREST))die("rounding");omp_set_dynamic(0);omp_set_num_threads(6);cc_load(argv[1]);
    size_t bytes;uint8_t *raw=read_file(argv[2],&bytes);
    if(bytes<16 || memcmp(raw,"M286IN01",8) || u32le(raw+8)!=24 || u32le(raw+12)!=147 || bytes!=16+(size_t)24*147*3968*4)die("fixture binding");
    FILE *file=fopen(argv[3],"wb");if(!file)die("output open");cc_write(file,"M286OP01",8);
    uint32_t dims[2]={24,147};cc_write(file,dims,8);
    for(int li=0;li<24;li++){
        const float *cursor=(const float*)(raw+16)+(size_t)li*147*3968;
        const float *x=cursor;cursor+=147*896;const float *norm=cursor;cursor+=147*896;
        const float *q=cursor;cursor+=147*896;const float *k=cursor;cursor+=147*128;
        const float *postq=cursor;cursor+=147*896;const float *postk=cursor;cursor+=147*128;const float *v=cursor;
        float *outs=calloc((size_t)147*4864,4);if(!outs)die("output allocation");float *p=outs;
        float *on=p;p+=147*896;float *oq=p;p+=147*896;float *ok=p;p+=147*128;float *ov=p;p+=147*128;
        float *rq=p;p+=147*896;float *rk=p;p+=147*128;float *ac=p;p+=147*896;float *ab=p;
        const CcLayer *layer=&cc_layers[li];
        for(int pos=0;pos<147;pos++){
            cc_norm(x+(size_t)pos*D,layer->in_norm,on+(size_t)pos*D);
            cc_mat(layer->q,norm+(size_t)pos*D,layer->qb,oq+(size_t)pos*D,D,D);
            cc_mat(layer->k,norm+(size_t)pos*D,layer->kb,ok+(size_t)pos*128,128,D);
            cc_mat(layer->v,norm+(size_t)pos*D,layer->vb,ov+(size_t)pos*128,128,D);
            float co[32],si[32];for(int j=0;j<32;j++){
                float phase=(float)pos*(1.0f/powf(1000000.0f,(float)(2*j)/64.0f));co[j]=cc_bf(cosf(phase));si[j]=cc_bf(sinf(phase));
            }
            memcpy(rq+(size_t)pos*D,q+(size_t)pos*D,D*4);memcpy(rk+(size_t)pos*128,k+(size_t)pos*128,128*4);
            cc_rope(rq+(size_t)pos*D,14,co,si);cc_rope(rk+(size_t)pos*128,2,co,si);
        }
        attention_probe(postq,postk,v,ac,147,0);attention_probe(postq,postk,v,ab,147,1);
        cc_write(file,outs,(size_t)147*4864*4);free(outs);
    }
    if(fclose(file))die("output close");free(raw);puts("{\"layers\":24,\"positions_per_layer\":147,\"diagnostic_only\":true}");return 0;
}
