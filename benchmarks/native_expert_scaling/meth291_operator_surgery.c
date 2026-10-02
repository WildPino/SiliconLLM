// Nondeployable causal assay; ORIGINAL285 CPU equations on actual inputs.
#include "meth285_model_operator.h"
__declspec(dllexport) int m291_load(const char *path){
    if(fesetround(FE_TONEAREST) || !__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))die("rounding/ISA");
    omp_set_dynamic(0);omp_set_num_threads(6);integer_edge_guards();cc_load(path);
    int used=0;for(int i=0;i<CC_FIELD_COUNT;i++)used+=cc_used[i]!=0;return used;
}
__declspec(dllexport) void m291_norm(int li,int kind,int n,const float *x,float *out){
    const float *weight=kind==2?cc_final_norm:kind==1?cc_layers[li].post_norm:cc_layers[li].in_norm;
    for(int t=0;t<n;t++)cc_norm(x+(size_t)t*D,weight,out+(size_t)t*D);
}
__declspec(dllexport) void m291_mat(int li,int kind,int n,const float *x,float *out){
    const CcLayer *l=&cc_layers[li];const uint16_t *w=kind==0?l->q:kind==1?l->k:kind==2?l->v:kind==3?l->o:cc_embedding;
    const float *bias=kind==0?l->qb:kind==1?l->kb:kind==2?l->vb:NULL;
    int rows=kind==1 || kind==2?128:kind==4?CC_V:D;
    for(int t=0;t<n;t++)cc_mat(w,x+(size_t)t*D,bias,out+(size_t)t*rows,rows,D);
}
__declspec(dllexport) void m291_ffn(int li,int n,const float *x,float *out){
    for(int t=0;t<n;t++){
        uint16_t bits[D];for(int d=0;d<D;d++)bits[d]=cc_bf_bits(x[(size_t)t*D+d]);
        row_forward(li,bits,out+(size_t)t*D);for(int d=0;d<D;d++)out[(size_t)t*D+d]=cc_bf(out[(size_t)t*D+d]);
    }
}
__declspec(dllexport) void m291_attention(int n,const float *q,const float *k,const float *v,float *out){
    #pragma omp parallel for schedule(static)
    for(int h=0;h<14;h++){
        float *scores=malloc((size_t)n*4);if(!scores)die("scores allocation");
        for(int pos=0;pos<n;pos++){
            float maximum=-INFINITY;
            for(int t=0;t<=pos;t++){
                const float *key=k+(size_t)t*128+(h/7)*64;float value=cc_dot(key,q+(size_t)pos*D+h*64,64)*.125f;
                scores[t]=value;if(value>maximum)maximum=value;
            }
            float total=0;for(int t=0;t<=pos;t++){scores[t]=expf(scores[t]-maximum);total+=scores[t];}
            for(int d=0;d<64;d++){
                float value=0;for(int t=0;t<=pos;t++)value+=(scores[t]/total)*v[(size_t)t*128+(h/7)*64+d];
                out[(size_t)pos*D+h*64+d]=cc_bf(value);
            }
        }
        free(scores);
    }
}
